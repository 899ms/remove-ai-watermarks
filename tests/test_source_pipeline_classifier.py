"""Tests for the abstaining source-pipeline classifier research tool."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import source_pipeline_classifier as classifier


def test_command_help_exits_cleanly() -> None:
    result = CliRunner().invoke(classifier.main, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "train" in result.output
    assert "predict" in result.output


def _training_rows() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260915)
    rows = np.vstack(
        (
            rng.normal((-3.0, 0.0, 0.0), 0.1, size=(12, 3)),
            rng.normal((3.0, 0.0, 0.0), 0.1, size=(12, 3)),
            rng.normal((0.0, 3.0, 0.0), 0.1, size=(12, 3)),
        )
    )
    labels = np.repeat(np.arange(3), 12)
    return rows, labels


def test_fit_and_abstention_separate_provider_rows() -> None:
    features, labels = _training_rows()
    model = classifier.fit_ridge(features, labels, penalty=1.0)
    raw_scores = classifier.scores(features, model)
    margin = classifier.calibrate_margin(raw_scores, labels, unknown_fpr=0.01)
    predictions = classifier.predict_scores(raw_scores, margin)

    assert np.array_equal(predictions[:24], labels[:24])
    assert np.all(predictions[24:] == classifier.UNKNOWN_INDEX)


def test_model_round_trip_is_pickle_free_and_strict(tmp_path: Path) -> None:
    grid = 4
    feature_count = classifier.FEATURE_WIDTH + 3 * grid * grid
    model = classifier.RidgeModel(
        mean=np.arange(feature_count, dtype=np.float64),
        scale=np.full(feature_count, 2.0),
        weights=np.arange(feature_count * 3, dtype=np.float64).reshape(feature_count, 3),
        margin=0.25,
        grid=grid,
        penalty=100.0,
    )
    path = tmp_path / "model.npz"

    classifier.save_model(path, model)
    loaded = classifier.load_model(path)

    assert loaded.grid == grid
    assert loaded.margin == model.margin
    assert np.array_equal(loaded.mean, model.mean)
    assert np.array_equal(loaded.scale, model.scale)
    assert np.array_equal(loaded.weights, model.weights)

    with np.load(path, allow_pickle=False) as payload:
        assert payload.files


def test_model_loader_rejects_extra_fields(tmp_path: Path) -> None:
    path = tmp_path / "model.npz"
    np.savez(
        path,
        classes=np.asarray(classifier.CLASSES),
        mean=np.zeros(136),
        scale=np.ones(136),
        weights=np.zeros((136, 3)),
        margin=np.asarray(0.1),
        spectral_grid=np.asarray(2),
        penalty=np.asarray(100.0),
        unexpected=np.asarray(1),
    )

    with pytest.raises(ValueError, match="model fields must be exactly"):
        classifier.load_model(path)


def test_model_loader_rejects_nonfinite_margin(tmp_path: Path) -> None:
    grid = 2
    feature_count = classifier.FEATURE_WIDTH + 3 * grid * grid
    path = tmp_path / "model.npz"
    np.savez(
        path,
        classes=np.asarray(classifier.CLASSES),
        mean=np.zeros(feature_count),
        scale=np.ones(feature_count),
        weights=np.zeros((feature_count, 3)),
        margin=np.asarray(np.nan),
        spectral_grid=np.asarray(grid),
        penalty=np.asarray(100.0),
    )

    with pytest.raises(ValueError, match="margin must be finite"):
        classifier.load_model(path)


def test_calibration_requires_unknown_rows() -> None:
    with pytest.raises(ValueError, match="must contain unknown"):
        classifier.calibrate_margin(
            np.zeros((2, 3)),
            np.asarray([0, 1]),
            unknown_fpr=0.01,
        )
