"""Measure a phase-free spectral source-pipeline hypothesis.

This is a research harness, not a SynthID detector. It classifies complete
generation/export pipelines whose renderer, output geometry, encoding, and
watermark state may all be confounded. A watermark claim requires a causal
marked/clean contrast and an independent oracle.

The input directory contains one subdirectory per class. Matching filename
stems are treated as one prompt group and always stay in the same fold.

Example:
    uv run python scripts/spectral_pipeline_probe.py DATASET \
        --class-dir gemini --class-dir openai --class-dir flux \
        --attack original --attack crop2 --attack resize95 \
        --attack jpeg75 --attack blur \
        --report-out .local-eval/spectral-pipeline-report.json
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import click
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
ATTACKS = ("original", "crop2", "resize95", "jpeg75", "blur")

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SpectralGeometry:
    """Precomputed coordinates for one phase-free spectral representation."""

    size: int
    grid: int
    radius: NDArray[np.int32]
    radius_counts: IntArray
    passband: BoolArray


@dataclass(frozen=True)
class FoldModel:
    """One standardized dual-ridge model and its held-out rows."""

    test_rows: BoolArray
    mean: FloatArray
    scale: FloatArray
    active: BoolArray
    weights: FloatArray


def make_geometry(size: int = 256, grid: int = 32) -> SpectralGeometry:
    """Build the radial detrending and pooling geometry."""
    if size < 16 or grid < 2 or size % grid:
        raise ValueError("size must be at least 16 and divisible by grid")
    yy, xx = np.mgrid[:size, :size]
    radius = np.rint(np.hypot(yy - size // 2, xx - size // 2)).astype(np.int32)
    radius_counts = np.bincount(radius.ravel())
    passband = (radius >= 4) & (radius <= size * 0.48)
    return SpectralGeometry(size, grid, radius, radius_counts, passband)


def phase_free_log_spectrum(channel: FloatArray) -> FloatArray:
    """Return centered log FFT magnitude, deliberately discarding phase."""
    if channel.ndim != 2:
        raise ValueError("channel must be a two-dimensional array")
    centered = np.asarray(channel, dtype=np.float64) - float(np.mean(channel))
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(centered))))


def spectral_feature(pixels: FloatArray, geometry: SpectralGeometry) -> FloatArray:
    """Extract pooled radial-residual spectra from luma and opponent color."""
    if pixels.shape != (geometry.size, geometry.size, 3):
        raise ValueError(f"pixels must have shape {(geometry.size, geometry.size, 3)}, got {pixels.shape}")
    red, green, blue = np.moveaxis(np.asarray(pixels, dtype=np.float64), 2, 0)
    channels = (
        0.299 * red + 0.587 * green + 0.114 * blue,
        red - green,
        blue - (red + green) / 2.0,
    )
    block = geometry.size // geometry.grid
    features: list[FloatArray] = []
    for channel in channels:
        spectrum = phase_free_log_spectrum(channel)
        radial_sum = np.bincount(geometry.radius.ravel(), weights=spectrum.ravel())
        radial_mean = radial_sum / geometry.radius_counts
        residual = np.where(
            geometry.passband,
            spectrum - radial_mean[geometry.radius],
            0.0,
        )
        pooled = residual.reshape(geometry.grid, block, geometry.grid, block).mean(axis=(1, 3))
        features.append(pooled.ravel())
    return np.concatenate(features)


def _apply_attack(image: Image.Image, attack: str) -> Image.Image:
    """Apply one declared robustness transform before canonical resizing."""
    if attack == "original":
        return image
    if attack == "crop2":
        if image.width <= 4 or image.height <= 4:
            raise ValueError("crop2 requires both image dimensions to exceed four pixels")
        return image.crop((2, 2, image.width, image.height))
    if attack == "resize95":
        return image.resize(
            (max(1, round(image.width * 0.95)), max(1, round(image.height * 0.95))),
            Image.Resampling.LANCZOS,
        )
    if attack == "jpeg75":
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=75)
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            return decoded.convert("RGB")
    if attack == "blur":
        return image.filter(ImageFilter.GaussianBlur(0.7))
    raise ValueError(f"unsupported attack: {attack}")


def feature_from_path(path: Path, geometry: SpectralGeometry, attack: str) -> FloatArray:
    """Decode one path, apply ATTACK, and return its phase-free feature."""
    with Image.open(path) as source:
        return feature_from_image(source.convert("RGB"), geometry, attack)


def feature_from_image(image: Image.Image, geometry: SpectralGeometry, attack: str) -> FloatArray:
    """Extract one attacked feature from an already decoded RGB image."""
    attacked = _apply_attack(image, attack)
    canonical = attacked.resize((geometry.size, geometry.size), Image.Resampling.LANCZOS)
    pixels = np.asarray(canonical, dtype=np.float64) / 255.0
    return spectral_feature(pixels, geometry)


def discover_grouped_paths(
    root: Path,
    class_dirs: tuple[str, ...],
) -> tuple[list[Path], IntArray, IntArray, list[str]]:
    """Return class-major paths over filename stems shared by every class."""
    if len(class_dirs) < 2 or len(set(class_dirs)) != len(class_dirs):
        raise ValueError("at least two unique class directories are required")
    per_class: list[dict[str, Path]] = []
    for class_dir in class_dirs:
        directory = root / class_dir
        if not directory.is_dir():
            raise ValueError(f"class directory does not exist: {directory}")
        mapping: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                if path.stem in mapping:
                    raise ValueError(f"duplicate filename stem in {directory}: {path.stem}")
                mapping[path.stem] = path
        per_class.append(mapping)
    common_names: set[str] = set(per_class[0])
    for mapping in per_class[1:]:
        common_names.intersection_update(mapping)
    group_names = sorted(common_names)
    if not group_names:
        raise ValueError("class directories have no shared filename stems")
    paths: list[Path] = []
    labels: list[int] = []
    groups: list[int] = []
    for class_index, mapping in enumerate(per_class):
        for group_index, group_name in enumerate(group_names):
            paths.append(mapping[group_name])
            labels.append(class_index)
            groups.append(group_index)
    return paths, np.asarray(labels), np.asarray(groups), group_names


def fit_grouped_models(
    features: FloatArray,
    labels: IntArray,
    groups: IntArray,
    *,
    class_count: int,
    folds: int = 5,
    ridge: float = 1000.0,
) -> list[FoldModel]:
    """Fit prompt-grouped dual ridge models without test-fold leakage."""
    if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(groups):
        raise ValueError("features, labels, and groups must describe the same rows")
    if folds < 2 or len(np.unique(groups)) < folds:
        raise ValueError("fold count must be between two and the number of groups")
    if class_count < 2 or ridge <= 0:
        raise ValueError("class_count must be at least two and ridge must be positive")
    targets = np.eye(class_count, dtype=np.float64)[labels]
    models: list[FoldModel] = []
    for fold in range(folds):
        test_rows = groups % folds == fold
        train_rows = ~test_rows
        train = features[train_rows]
        mean = np.mean(train, axis=0)
        scale = np.std(train, axis=0)
        active = scale > 1e-8
        train_features = (train[:, active] - mean[active]) / scale[active]
        kernel = train_features @ train_features.T
        dual_weights = cast(
            "FloatArray",
            np.linalg.solve(
                kernel + ridge * np.eye(len(kernel)),
                targets[train_rows],
            ),
        )
        weights = train_features.T @ dual_weights
        models.append(FoldModel(test_rows, mean, scale, active, weights))
    return models


def predict_grouped(models: list[FoldModel], features: FloatArray) -> IntArray:
    """Predict every row only with the model that held out its group."""
    predictions = np.full(len(features), -1, dtype=np.int64)
    for model in models:
        test_features = (features[model.test_rows][:, model.active] - model.mean[model.active]) / model.scale[
            model.active
        ]
        scores = test_features @ model.weights
        predictions[model.test_rows] = np.argmax(scores, axis=1)
    if np.any(predictions < 0):
        raise ValueError("fold models did not cover every row")
    return predictions


def confusion_matrix(labels: IntArray, predictions: IntArray, class_count: int) -> IntArray:
    """Count actual rows by predicted class."""
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def _extract_all_attacks(
    paths: list[Path],
    geometry: SpectralGeometry,
    attacks: tuple[str, ...],
) -> dict[str, FloatArray]:
    """Decode each path once and extract every requested attack feature."""
    log.info("Extracting %s attack views from %s images", len(attacks), len(paths))
    rows: dict[str, list[FloatArray]] = {attack: [] for attack in attacks}
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        for attack in attacks:
            rows[attack].append(feature_from_image(image, geometry, attack))
    return {attack: np.asarray(features) for attack, features in rows.items()}


@click.command()
@click.argument("dataset", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "class_dirs",
    "--class-dir",
    multiple=True,
    required=True,
    help="Class subdirectory; repeat in label order.",
)
@click.option(
    "attacks",
    "--attack",
    multiple=True,
    type=click.Choice(ATTACKS),
    default=("original",),
    show_default=True,
)
@click.option("--size", type=click.IntRange(min=16), default=256, show_default=True)
@click.option("--grid", type=click.IntRange(min=2), default=32, show_default=True)
@click.option("--folds", type=click.IntRange(min=2), default=5, show_default=True)
@click.option("--ridge", type=click.FloatRange(min=0.0, min_open=True), default=1000.0, show_default=True)
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(
    dataset: Path,
    class_dirs: tuple[str, ...],
    attacks: tuple[str, ...],
    size: int,
    grid: int,
    folds: int,
    ridge: float,
    report_out: Path,
) -> None:
    """Run grouped spectral pipeline classification over DATASET."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if "original" not in attacks:
        raise click.UsageError("--attack original is required because every model trains on original exports")
    unique_attacks = tuple(dict.fromkeys(attacks))
    try:
        geometry = make_geometry(size, grid)
        paths, labels, groups, group_names = discover_grouped_paths(dataset, class_dirs)
        features_by_attack = _extract_all_attacks(paths, geometry, unique_attacks)
        original = features_by_attack["original"]
        models = fit_grouped_models(
            original,
            labels,
            groups,
            class_count=len(class_dirs),
            folds=folds,
            ridge=ridge,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    results: dict[str, object] = {}
    for attack in unique_attacks:
        features = features_by_attack[attack]
        predictions = predict_grouped(models, features)
        matrix = confusion_matrix(labels, predictions, len(class_dirs))
        results[attack] = {
            "accuracy": float(np.trace(matrix) / np.sum(matrix)),
            "confusion_matrix": matrix.tolist(),
        }

    report = {
        "claim": "phase-free generation/export pipeline classification; not SynthID detection",
        "causal_limit": (
            "Renderer, output geometry, encoding, and watermark state vary together. "
            "Only a marked/clean causal contrast plus an independent oracle can isolate the watermark."
        ),
        "evaluation_limit": (
            "Grouped cross-validation estimates this corpus only. Hyperparameters selected on these folds "
            "need an independent locked corpus before they support a generalization claim."
        ),
        "dataset": str(dataset),
        "classes": list(class_dirs),
        "group_count": len(group_names),
        "image_count": len(paths),
        "configuration": {"size": size, "grid": grid, "folds": folds, "ridge": ridge},
        "results": results,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote spectral pipeline report: %s", report_out)


if __name__ == "__main__":
    main()
