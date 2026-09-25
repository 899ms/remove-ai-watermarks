"""Contracts for the portable OpenAI/Google source-pattern runtime."""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from remove_ai_watermarks import source_classify_v2 as model

if TYPE_CHECKING:
    from pathlib import Path


def _synthetic_weights(path: Path, *, gate: float) -> None:
    zero_mean = np.zeros((256, 256, 3), dtype=np.complex64)
    zero_precision = np.zeros((256, 256, 3), dtype=np.float32)
    unit = np.zeros((992, 992), dtype=np.float32)
    unit[0, 0] = 1
    fields: dict[str, np.ndarray] = {"schema": np.asarray(model.MODEL_SCHEMA)}
    for name in model._FILTER_NAMES:
        fields[f"{name}_mean"] = zero_mean
        fields[f"{name}_precision"] = zero_precision
    for route in model.ROUTES:
        fields[f"openai_{route}_unit"] = unit
    limits = dict.fromkeys(model._THRESHOLD_NAMES, 1.0)
    limits["r183_openai_gate"] = gate
    limits.update({f"r163_{group}": -0.5 for group in model.GROUPS})
    fields.update({f"threshold_{name}": np.asarray(value, dtype=np.float64) for name, value in limits.items()})
    np.savez_compressed(path, **fields)


def test_r163_gate_changes_real_prediction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "photo.png"
    Image.new("RGB", (512, 512), (80, 110, 145)).save(image)
    blocked = tmp_path / "blocked.npz"
    allowed = tmp_path / "allowed.npz"
    _synthetic_weights(blocked, gate=0.5)
    _synthetic_weights(allowed, gate=-0.5)

    def predict(path: Path) -> model.ImageSourceClassification:
        monkeypatch.setenv(model.WEIGHTS_ENV, str(path))
        monkeypatch.setattr(model, "MODEL_SHA256", sha256(path.read_bytes()).hexdigest())
        return model.classify_image_source(image)

    no_gate = predict(blocked)
    with_gate = predict(allowed)

    assert no_gate.label == "unknown"
    assert no_gate.reason == "abstained"
    assert with_gate.label == "openai"
    assert with_gate.reason == "classified"
    assert with_gate.watermark_truth == "unknown"


def test_top_level_api_is_lazy_and_points_to_new_runtime() -> None:
    import remove_ai_watermarks as raiw

    assert raiw.classify_image_source is model.classify_image_source
    assert raiw.ImageSourceClassification is model.ImageSourceClassification
    assert len(model.WEIGHTS_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in model.WEIGHTS_REVISION)


def test_metadata_does_not_change_pixel_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / model.MODEL_FILE
    _synthetic_weights(weights, gate=-0.5)
    monkeypatch.setenv(model.WEIGHTS_ENV, str(weights))
    monkeypatch.setattr(model, "MODEL_SHA256", sha256(weights.read_bytes()).hexdigest())
    pixels = Image.new("RGB", (512, 512), (80, 110, 145))
    clean = tmp_path / "clean.png"
    tagged = tmp_path / "tagged.png"
    pixels.save(clean)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Comment", "arbitrary metadata")
    pixels.save(tagged, pnginfo=metadata)

    first = model.classify_image_source(clean)
    second = model.classify_image_source(tagged)

    assert first.label == second.label == "openai"
    assert first.scores == second.scores


def test_rejects_wrong_artifact_hash(tmp_path: Path) -> None:
    weights = tmp_path / model.MODEL_FILE
    _synthetic_weights(weights, gate=0.5)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        model._load_model(weights, "0" * 64)


def test_small_image_abstains_without_watermark_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model, "_model_path", lambda: pytest.fail("small image loaded weights"))
    image = tmp_path / "small.png"
    Image.new("RGB", (128, 128), "white").save(image)

    result = model.classify_image_source(image)

    assert result.label == "unknown"
    assert result.reason == "feature_unavailable"
    assert result.watermark_truth == "unknown"
