"""The public model staging boundary accepts only verified derivatives."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import publish_image_source_hf as publisher


def test_stage_release_rejects_changed_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    card = tmp_path / "card"
    card.mkdir()
    (card / "README.md").write_text("# Public card\n")
    weights = tmp_path / "source"
    weights.mkdir()
    original = b"derivative weights"
    (weights / publisher.MODEL_FILE).write_bytes(original)
    (card / "metrics.json").write_text(json.dumps({"artifact_sha256": hashlib.sha256(original).hexdigest()}))
    monkeypatch.setattr(publisher, "CARD_DIR", card)

    stage = tmp_path / "stage"
    publisher.stage_release(stage, weights)
    assert sorted(item.name for item in stage.iterdir()) == sorted(publisher.PUBLIC_FILES)

    (weights / publisher.MODEL_FILE).write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        publisher.stage_release(tmp_path / "second", weights)
