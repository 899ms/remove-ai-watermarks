"""Synthetic recall/precision sweep for generic_ai_label_engine's detection gate.

SYNTHETIC-ONLY CALIBRATION. There is no captured real "AI生成"-generator screenshot
in this repository to calibrate against (see the confirmed bug report this engine was
built for: the customer's original image bytes do not exist in any repo). This script
instead:

  * builds POSITIVES by compositing the engine's own rendered glyph asset
    (``assets/generic_ai_label_alpha.png``) onto ``render_visible_examples.py``'s
    generated base photos, at several sizes / scale rungs / opacities;
  * builds CLEAN negatives from the same generated base photos (some with unrelated
    Latin/digit text stamped in the corner, the classical false-positive class for
    this detector family);
  * builds RIVAL negatives from the six other bottom-right/br CJK text marks' own
    committed gallery examples (``data/fixtures/visible/<key>/example.png``) -- every
    one of Doubao/Qwen/Baidu/Kling/Yuanbao's marks contains the literal substring
    "AI生成" (Jimeng's "★ 即梦AI" does not and is excluded), so a generic "AI生成"
    template is at real risk of firing a SECOND time on an already brand-attributed
    mark unless the rival margin separates them.

Prints a gate x rival-margin sweep. The value currently shipped in
``generic_ai_label_engine.py`` (``DETECT_NCC_THRESHOLD = 0.50``, at the shared default
0.10 rival margin) was chosen from this sweep's 100%-precision / best-recall corner,
with headroom above 0.45 left for the library's own removal residual (see that
constant's comment); re-run and re-choose after any change to the rendered asset or
the shared tophat gating constants.

Usage::

    uv run python scripts/calibrate_generic_ai_label.py
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import render_visible_examples as rve  # noqa: E402

from remove_ai_watermarks._text_mark_engine import TextMarkEngine  # noqa: E402
from remove_ai_watermarks.generic_ai_label_engine import _CONFIG  # noqa: E402

log = logging.getLogger(__name__)

_GALLERY = _ROOT / "data" / "fixtures" / "visible"
_RIVAL_KEYS = ["doubao", "jimeng", "qwen", "baidu", "kling", "yuanbao", "samsung", "runninghub"]
_SIZES = [(1536, 1152), (2048, 2048), (1080, 1920), (2048, 1536)]
_SCALE_RUNGS = (0.85, 1.0, 1.15)
_OPACITIES = (0.65, 0.8, 1.0)
_N_CLEAN = 40


def _build_positives() -> list[np.ndarray]:
    """The rendered glyph composited at varied size/scale/opacity."""
    positives: list[np.ndarray] = []
    for i, (w, h) in enumerate(_SIZES):
        base = rve.base_photo(w, h, seed=100 + i)
        for scale_mult in _SCALE_RUNGS:
            for alpha_mult in _OPACITIES:
                out = rve._stamp_text_mark("generic_ai_label", base, size_mult=scale_mult, alpha_mult=alpha_mult)
                if out is not None:
                    positives.append(out[0])
    return positives


def _build_clean_negatives(rng: np.random.Generator) -> list[np.ndarray]:
    """Clean generated photos, a third with unrelated corner text (Latin/digits)."""
    negatives: list[np.ndarray] = []
    for i in range(_N_CLEAN):
        w, h = int(rng.uniform(900, 2200)), int(rng.uniform(900, 2200))
        seed = 500 + i
        img = rve.base_photo(w, h, seed=seed)
        if i % 3 == 0:
            r = np.random.default_rng(seed)
            txt = f"{r.integers(1000, 9999)} PM"
            x, y = int(w * 0.75), int(h * 0.90)
            cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, w / 1400, (235, 235, 235), 2, cv2.LINE_AA)
        negatives.append(img)
    return negatives


def _load_rival_examples() -> dict[str, np.ndarray]:
    rivals: dict[str, np.ndarray] = {}
    for key in _RIVAL_KEYS:
        img = cv2.imread(str(_GALLERY / key / "example.png"))
        if img is not None:
            rivals[key] = img
    return rivals


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(42)
    positives = _build_positives()
    clean_negatives = _build_clean_negatives(rng)
    rival_examples = _load_rival_examples()

    log.info(
        "corpus: %d synthetic positives, %d clean negatives, %d rival-mark examples",
        len(positives),
        len(clean_negatives),
        len(rival_examples),
    )
    log.info("%6s %8s %10s %10s %4s %9s %9s", "gate", "margin", "recall", "precision", "TP", "clean_fp", "rival_fp")
    for gate in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        for margin in (0.10, 0.25):
            cfg = replace(_CONFIG, detect_ncc_threshold=gate, rival_margin=margin)
            eng = TextMarkEngine(cfg)
            tp = sum(1 for im in positives if eng.detect(im).detected)
            clean_fp = sum(1 for im in clean_negatives if eng.detect(im).detected)
            rival_fp = [k for k, im in rival_examples.items() if eng.detect(im).detected]
            fp = clean_fp + len(rival_fp)
            recall = tp / len(positives) if positives else float("nan")
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            log.info(
                "%6.2f %8.2f %9.1f%% %9.1f%% %4d %9d %9d  %s",
                gate,
                margin,
                recall * 100,
                precision * 100,
                tp,
                clean_fp,
                len(rival_fp),
                rival_fp,
            )


if __name__ == "__main__":
    main()
