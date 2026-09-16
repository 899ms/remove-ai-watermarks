"""Train or run an abstaining source/export-pipeline classifier.

This research tool combines the library's 124-d forensic descriptor with the
phase-free spectral descriptor from ``spectral_pipeline_probe.py``. Its labels
describe complete generation and export pipelines. It does not decode SynthID
or establish that a watermark is present.

Expected training layout::

    train/openai/*.png
    train/google/*.png
    train/unknown/*.png
    calibration/openai/*.png
    calibration/google/*.png
    calibration/unknown/*.png

Examples::

    uv run python scripts/source_pipeline_classifier.py train \
        train calibration --model-out model.npz
    uv run python scripts/source_pipeline_classifier.py predict \
        model.npz image.png
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import click
import numpy as np
import spectral_pipeline_probe as spectral
from numpy.typing import NDArray
from PIL import Image
from watermark_benchmark import sha256_file

from remove_ai_watermarks._internal.forensic_124d import FEATURE_WIDTH, PATCH, image_features

log = logging.getLogger(__name__)

CLASSES = ("openai", "google", "unknown")
UNKNOWN_INDEX = CLASSES.index("unknown")
DEFAULT_GRID = 16
DEFAULT_PENALTY = 100.0
DEFAULT_UNKNOWN_FPR = 0.01
CLAIM = "source/export pipeline classification; not SynthID detection"

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class RidgeModel:
    """One standardized three-class ridge head with an abstention margin."""

    mean: FloatArray
    scale: FloatArray
    weights: FloatArray
    margin: float
    grid: int
    penalty: float


def discover_labeled_paths(root: Path) -> tuple[list[Path], IntArray, list[str]]:
    """Find images under the three fixed label directories."""
    paths: list[Path] = []
    labels: list[int] = []
    hashes: list[str] = []
    for class_index, class_name in enumerate(CLASSES):
        directory = root / class_name
        if not directory.is_dir():
            raise ValueError(f"class directory does not exist: {directory}")
        class_paths = sorted(
            path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in spectral.IMAGE_SUFFIXES
        )
        if not class_paths:
            raise ValueError(f"class directory has no supported images: {directory}")
        for path in class_paths:
            paths.append(path)
            labels.append(class_index)
            hashes.append(sha256_file(path))
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"duplicate image bytes found inside {root}")
    return paths, np.asarray(labels, dtype=np.int64), hashes


def _feature_from_path(path: Path, geometry: spectral.SpectralGeometry) -> FloatArray | None:
    """Return the fused representation for one path and prepared geometry."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    forensic = image_features(np.asarray(image, dtype=np.uint8))
    if forensic is None:
        return None
    phase_free = spectral.feature_from_image(image, geometry, "original")
    return np.concatenate(
        (
            np.asarray(forensic, dtype=np.float64),
            np.asarray(phase_free, dtype=np.float64),
        )
    )


def feature_from_path(path: Path, grid: int) -> FloatArray | None:
    """Return the fused forensic and phase-free spectral representation."""
    return _feature_from_path(path, spectral.make_geometry(size=PATCH, grid=grid))


def extract_features(paths: list[Path], labels: IntArray, grid: int) -> tuple[FloatArray, IntArray, list[Path]]:
    """Extract valid rows and report files whose forensic feature abstains."""
    geometry = spectral.make_geometry(size=PATCH, grid=grid)
    features: list[FloatArray] = []
    kept_labels: list[int] = []
    skipped: list[Path] = []
    for index, (path, label) in enumerate(zip(paths, labels, strict=True), start=1):
        try:
            feature = _feature_from_path(path, geometry)
        except (OSError, ValueError) as error:
            log.warning("Could not extract %s: %s", path, error)
            feature = None
        if feature is None:
            skipped.append(path)
            continue
        features.append(feature)
        kept_labels.append(int(label))
        if index % 100 == 0:
            log.info("Extracted %s/%s images", index, len(paths))
    if not features:
        raise ValueError("no image produced a valid feature vector")
    return np.asarray(features), np.asarray(kept_labels, dtype=np.int64), skipped


def fit_ridge(
    features: FloatArray,
    labels: IntArray,
    penalty: float,
    grid: int = DEFAULT_GRID,
) -> RidgeModel:
    """Fit a class-balanced standardized ridge head before calibration."""
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features and labels must describe the same rows")
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("training data must contain all three classes")
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    mean = features.mean(axis=0)
    raw_scale = features.std(axis=0)
    active = raw_scale > 1e-10
    scale = np.where(active, raw_scale, 1.0)
    normalized = (features[:, active] - mean[active]) / scale[active]
    sample_weight = len(labels) / (len(CLASSES) * counts[labels])
    root_weight = np.sqrt(sample_weight)[:, None]
    weighted = normalized * root_weight
    targets = np.eye(len(CLASSES), dtype=np.float64)[labels] * root_weight
    gram = weighted.T @ weighted + penalty * np.eye(weighted.shape[1])
    active_weights = cast("FloatArray", np.linalg.solve(gram, weighted.T @ targets))
    weights = np.zeros((features.shape[1], len(CLASSES)), dtype=np.float64)
    weights[active] = active_weights
    return RidgeModel(mean, scale, weights, 0.0, grid, penalty)


def scores(features: FloatArray, model: RidgeModel) -> FloatArray:
    """Return the uncalibrated ridge scores."""
    return ((features - model.mean) / model.scale) @ model.weights


def decision_margins(raw_scores: FloatArray) -> FloatArray:
    """Return each row's best-provider advantage over every alternative."""
    target_scores = raw_scores[:, :UNKNOWN_INDEX]
    best = np.argmax(target_scores, axis=1)
    other = 1 - best
    rows = np.arange(len(raw_scores))
    reference = np.maximum(raw_scores[rows, other], raw_scores[:, UNKNOWN_INDEX])
    return target_scores[rows, best] - reference


def calibrate_margin(
    calibration_scores: FloatArray,
    calibration_labels: IntArray,
    unknown_fpr: float,
) -> float:
    """Choose the smallest empirical cut meeting the unknown false-positive target."""
    if not 0 <= unknown_fpr < 1:
        raise ValueError("unknown_fpr must be in [0, 1)")
    unknown_scores = calibration_scores[calibration_labels == UNKNOWN_INDEX]
    if not len(unknown_scores):
        raise ValueError("calibration data must contain unknown rows")
    return max(
        0.0,
        float(
            np.quantile(
                decision_margins(unknown_scores),
                1.0 - unknown_fpr,
                method="higher",
            )
            + 1e-12
        ),
    )


def predict_scores(raw_scores: FloatArray, margin: float) -> IntArray:
    """Return provider labels only when the calibrated margin is met."""
    best = np.argmax(raw_scores[:, :UNKNOWN_INDEX], axis=1)
    accepted = decision_margins(raw_scores) >= margin
    return np.where(accepted, best, UNKNOWN_INDEX).astype(np.int64)


def save_model(path: Path, model: RidgeModel) -> None:
    """Write a pickle-free model artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        classes=np.asarray(CLASSES),
        mean=model.mean,
        scale=model.scale,
        weights=model.weights,
        margin=np.asarray(model.margin),
        spectral_grid=np.asarray(model.grid),
        penalty=np.asarray(model.penalty),
    )


def load_model(path: Path) -> RidgeModel:
    """Load and validate a pickle-free model artifact."""
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "classes",
            "mean",
            "scale",
            "weights",
            "margin",
            "spectral_grid",
            "penalty",
        }
        if set(payload.files) != required:
            raise ValueError(f"model fields must be exactly {sorted(required)}")
        classes = tuple(str(value) for value in payload["classes"])
        mean = np.asarray(payload["mean"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        weights = np.asarray(payload["weights"], dtype=np.float64)
        margin = float(payload["margin"])
        grid = int(payload["spectral_grid"])
        penalty = float(payload["penalty"])
    expected_features = FEATURE_WIDTH + 3 * grid * grid
    if classes != CLASSES:
        raise ValueError(f"model classes must be {CLASSES}")
    if mean.shape != (expected_features,) or scale.shape != mean.shape:
        raise ValueError("model normalization vectors have the wrong shape")
    if weights.shape != (expected_features, len(CLASSES)):
        raise ValueError("model weights have the wrong shape")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("model normalization vectors must be finite")
    if not np.all(np.isfinite(weights)) or np.any(scale <= 0):
        raise ValueError("model weights must be finite and scales positive")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("model margin must be finite and non-negative")
    if grid < 2 or PATCH % grid or not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("model calibration fields are invalid")
    return RidgeModel(mean, scale, weights, margin, grid, penalty)


@click.group()
def main() -> None:
    """Train or run the research-only source-pipeline classifier."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@main.command("train")
@click.argument("train_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("calibration_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--model-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--grid", type=click.IntRange(min=2), default=DEFAULT_GRID, show_default=True)
@click.option("--penalty", type=click.FloatRange(min=0.0, min_open=True), default=DEFAULT_PENALTY, show_default=True)
@click.option(
    "--unknown-fpr",
    type=click.FloatRange(min=0.0, max=1.0, max_open=True),
    default=DEFAULT_UNKNOWN_FPR,
    show_default=True,
)
def train_command(
    train_root: Path,
    calibration_root: Path,
    model_out: Path,
    grid: int,
    penalty: float,
    unknown_fpr: float,
) -> None:
    """Fit on TRAIN_ROOT and calibrate abstention on CALIBRATION_ROOT."""
    if PATCH % grid:
        raise click.UsageError(f"--grid must divide {PATCH}")
    try:
        train_paths, train_labels, train_hashes = discover_labeled_paths(train_root)
        calibration_paths, calibration_labels, calibration_hashes = discover_labeled_paths(calibration_root)
        overlap = set(train_hashes) & set(calibration_hashes)
        if overlap:
            raise ValueError(f"train/calibration byte overlap: {len(overlap)} images")
        train_features, kept_train_labels, train_skipped = extract_features(train_paths, train_labels, grid)
        calibration_features, kept_calibration_labels, calibration_skipped = extract_features(
            calibration_paths, calibration_labels, grid
        )
        provisional = fit_ridge(train_features, kept_train_labels, penalty, grid)
        margin = calibrate_margin(
            scores(calibration_features, provisional),
            kept_calibration_labels,
            unknown_fpr,
        )
        model = replace(provisional, margin=margin)
        save_model(model_out, model)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        json.dumps(
            {
                "model": str(model_out),
                "claim": CLAIM,
                "train_rows": len(train_features),
                "calibration_rows": len(calibration_features),
                "feature_abstentions": len(train_skipped) + len(calibration_skipped),
                "margin": margin,
            },
            indent=2,
        )
    )


@main.command("predict")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
def predict_command(model_path: Path, images: tuple[Path, ...]) -> None:
    """Classify original-export IMAGES with MODEL_PATH."""
    try:
        model = load_model(model_path)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    results: list[dict[str, object]] = []
    geometry = spectral.make_geometry(size=PATCH, grid=model.grid)
    for path in images:
        try:
            feature = _feature_from_path(path, geometry)
        except (OSError, ValueError) as error:
            raise click.ClickException(f"could not extract {path}: {error}") from error
        if feature is None:
            results.append(
                {
                    "path": str(path),
                    "label": "unknown",
                    "reason": "124-d feature unavailable",
                    "claim": CLAIM,
                }
            )
            continue
        raw_scores = scores(feature[None, :], model)
        prediction = int(predict_scores(raw_scores, model.margin)[0])
        results.append(
            {
                "path": str(path),
                "label": CLASSES[prediction],
                "decision_margin": float(decision_margins(raw_scores)[0]),
                "required_margin": model.margin,
                "scores": {name: float(raw_scores[0, index]) for index, name in enumerate(CLASSES)},
                "claim": CLAIM,
            }
        )
    click.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
