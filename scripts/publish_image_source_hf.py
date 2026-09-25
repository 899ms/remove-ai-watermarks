#!/usr/bin/env python3
"""Publish the derivative image-source artifact with an explicit allowlist."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from hashlib import file_digest
from pathlib import Path

from hf_publish import publish_model_folder

log = logging.getLogger(__name__)

HUB_REPO = "wiltodelta/openai-google-image-source-classifier"
MODEL_FILE = "openai-google-source-v1.npz"
REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_DIR = REPO_ROOT / "docs" / "image-source-hf"
PUBLIC_FILES = ("README.md", "metrics.json", MODEL_FILE)


def stage_release(dest: Path, src: Path) -> None:
    """Copy only the public card, metrics, and hash-verified derivative weights."""
    weights = src.expanduser().resolve() / MODEL_FILE
    sources = {
        "README.md": CARD_DIR / "README.md",
        "metrics.json": CARD_DIR / "metrics.json",
        MODEL_FILE: weights,
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError(f"missing publication files: {', '.join(missing)}")
    expected = json.loads(sources["metrics.json"].read_text())["artifact_sha256"]
    with weights.open("rb") as source:
        actual = file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("image-source artifact SHA-256 mismatch")
    dest.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        shutil.copy2(path, dest / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--message", default="Publish image-source classifier")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is not set (write role required)")
    with tempfile.TemporaryDirectory(prefix="raiw-image-source-hf-") as tmp:
        stage = Path(tmp)
        stage_release(stage, args.src)
        revision = publish_model_folder(
            stage,
            repo_id=HUB_REPO,
            token=token,
            message=args.message,
            allow_patterns=PUBLIC_FILES,
        )
    log.info("published %s revision=%s", HUB_REPO, revision or "(unknown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
