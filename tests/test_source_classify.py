"""Unit contracts for the lightweight source-pipeline runtime."""

from __future__ import annotations

import re
from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np
import pytest

from remove_ai_watermarks import source_classify

if TYPE_CHECKING:
    from pathlib import Path


def test_model_identity_is_immutable() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", source_classify.WEIGHTS_REVISION)
    assert re.fullmatch(r"[0-9a-f]{64}", source_classify.MODEL_SHA256)


def _artifact(path: Path, *, activation: str = "gelu_exact", output_bias: tuple[float, ...] = (0.0, 0.0, 0.0)) -> None:
    np.savez_compressed(
        path,
        classes=np.asarray(source_classify.CLASSES),
        kind=np.asarray("fused"),
        mean=np.zeros(source_classify.FEATURE_WIDTH),
        scale=np.ones(source_classify.FEATURE_WIDTH),
        margins=np.asarray((0.5, 0.5)),
        layer0_weight=np.zeros((128, source_classify.FEATURE_WIDTH), dtype=np.float32),
        layer0_bias=np.zeros(128, dtype=np.float32),
        layer1_weight=np.zeros((64, 128), dtype=np.float32),
        layer1_bias=np.zeros(64, dtype=np.float32),
        output_weight=np.zeros((3, 64), dtype=np.float32),
        output_bias=np.asarray(output_bias, dtype=np.float32),
        activation=np.asarray(activation),
    )


def test_loader_accepts_the_exact_pickle_free_schema(tmp_path: Path) -> None:
    path = tmp_path / source_classify.MODEL_FILE
    _artifact(path)

    model = source_classify._load_model(path, expected_sha256=None)

    assert model.mean.shape == (source_classify.FEATURE_WIDTH,)
    assert model.margins.tolist() == pytest.approx([0.5, 0.5])


def test_loader_rejects_the_mislabeled_activation(tmp_path: Path) -> None:
    path = tmp_path / source_classify.MODEL_FILE
    _artifact(path, activation="gelu_tanh")

    with pytest.raises(ValueError, match="activation must be gelu_exact"):
        source_classify._load_model(path, expected_sha256=None)


def test_provider_threshold_mutation_changes_the_real_decision(tmp_path: Path) -> None:
    path = tmp_path / source_classify.MODEL_FILE
    _artifact(path, output_bias=(2.0, 0.0, 0.0))
    model = source_classify._load_model(path, expected_sha256=None)
    features = np.zeros(source_classify.FEATURE_WIDTH, dtype=np.float32)
    scores = source_classify._scores(features, model)

    accepted = source_classify._result(scores, np.asarray((1.5, 0.5), dtype=np.float32))
    rejected = source_classify._result(scores, np.asarray((2.5, 0.5), dtype=np.float32))

    assert accepted.label == "openai"
    assert accepted.reason == "classified"
    assert rejected.label == "unknown"
    assert rejected.reason == "abstained"


def test_a_float32_threshold_tie_preserves_the_inclusive_training_decision() -> None:
    scores = np.asarray((2.0, 0.0, 0.0), dtype=np.float32)
    result = source_classify._result(scores, np.asarray((2.0, 0.5), dtype=np.float32))

    assert result.label == "openai"
    assert result.reason == "classified"


def test_small_image_returns_honest_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    weights = tmp_path / "weights"
    weights.mkdir()
    _artifact(weights / source_classify.MODEL_FILE)
    monkeypatch.setenv(source_classify.WEIGHTS_ENV, str(weights))
    monkeypatch.setattr(
        source_classify,
        "MODEL_SHA256",
        sha256((weights / source_classify.MODEL_FILE).read_bytes()).hexdigest(),
    )
    image = tmp_path / "small.png"
    Image.new("RGB", (128, 128), "white").save(image)

    result = source_classify.classify_source(image)

    assert result.label == "unknown"
    assert result.reason == "feature_unavailable"
    assert result.scores == {}


def test_model_hash_is_a_runtime_precondition(tmp_path: Path) -> None:
    path = tmp_path / source_classify.MODEL_FILE
    _artifact(path)

    with pytest.raises(ValueError, match="model SHA-256 mismatch"):
        source_classify._load_model(path, expected_sha256="0" * 64)


def test_spectral_feature_is_finite_and_has_the_frozen_width() -> None:
    from PIL import Image

    from remove_ai_watermarks._internal.source_spectral import feature_from_image, make_geometry

    rng = np.random.default_rng(20260916)
    pixels = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    feature = feature_from_image(Image.fromarray(pixels), make_geometry())

    assert feature.shape == (768,)
    assert np.all(np.isfinite(feature))
