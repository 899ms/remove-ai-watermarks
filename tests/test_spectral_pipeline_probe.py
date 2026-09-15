"""Tests for the phase-free spectral pipeline research harness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import spectral_pipeline_probe as probe


def test_log_spectrum_discards_translation_phase() -> None:
    rng = np.random.default_rng(17)
    channel = rng.normal(size=(32, 32))

    original = probe.phase_free_log_spectrum(channel)
    translated = probe.phase_free_log_spectrum(np.roll(channel, shift=(5, -7), axis=(0, 1)))

    assert np.allclose(original, translated, rtol=1e-12, atol=1e-12)


def test_spectral_feature_is_finite_and_has_declared_shape() -> None:
    geometry = probe.make_geometry(size=32, grid=4)
    yy, xx = np.mgrid[:32, :32]
    carrier = np.cos(2 * np.pi * (5 * xx + 3 * yy) / 32)
    pixels = np.stack((carrier, np.roll(carrier, 1, axis=0), np.roll(carrier, 2, axis=1)), axis=2)

    feature = probe.spectral_feature(pixels, geometry)

    assert feature.shape == (3 * 4 * 4,)
    assert np.all(np.isfinite(feature))
    assert np.linalg.norm(feature) > 0


def test_discovery_keeps_only_stems_shared_by_every_class(tmp_path: Path) -> None:
    for class_name in ("first", "second"):
        directory = tmp_path / class_name
        directory.mkdir()
        for stem in ("001", "002"):
            Image.new("RGB", (8, 8), "white").save(directory / f"{stem}.png")
    Image.new("RGB", (8, 8), "black").save(tmp_path / "first" / "unpaired.png")

    paths, labels, groups, names = probe.discover_grouped_paths(tmp_path, ("first", "second"))

    assert names == ["001", "002"]
    assert [path.stem for path in paths] == ["001", "002", "001", "002"]
    assert labels.tolist() == [0, 0, 1, 1]
    assert groups.tolist() == [0, 1, 0, 1]


def test_grouped_ridge_never_trains_on_the_test_group() -> None:
    groups = np.tile(np.arange(14), 3)
    labels = np.repeat(np.arange(3), 14)
    class_centers = np.eye(3)[labels] * 8.0
    group_nuisance = np.column_stack((groups % 2, groups % 3, groups % 5)) * 0.01
    features = class_centers + group_nuisance

    models = probe.fit_grouped_models(features, labels, groups, class_count=3, folds=5, ridge=1.0)
    predictions = probe.predict_grouped(models, features)

    assert np.array_equal(predictions, labels)
    fold_assignment = np.full(len(groups), -1)
    for fold, model in enumerate(models):
        assert not np.any(fold_assignment[model.test_rows] >= 0)
        fold_assignment[model.test_rows] = fold
    assert np.all(fold_assignment >= 0)
    for group in np.unique(groups):
        assert len(np.unique(fold_assignment[groups == group])) == 1


def test_confusion_matrix_uses_actual_rows_and_predicted_columns() -> None:
    labels = np.asarray([0, 0, 1, 1, 2])
    predictions = np.asarray([0, 1, 1, 2, 0])

    matrix = probe.confusion_matrix(labels, predictions, class_count=3)

    assert matrix.tolist() == [[1, 1, 0], [0, 1, 1], [1, 0, 0]]
