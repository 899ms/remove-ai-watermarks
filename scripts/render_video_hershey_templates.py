"""Render the frozen OpenCV Hershey-font video templates for Dola and Kling.

The video detectors were calibrated against these ``cv2.putText`` silhouettes as
OpenCV 4.x draws them (byte-identical from 4.8 through 4.14). OpenCV 5 reworked
text rendering, so the same call yields smaller, heavier glyphs, and the Dola
gallery clip was then selected as Kling. The package therefore loads committed
PNGs instead of rendering at import time; this script is the recipe that produced
them. It refuses to run on any OpenCV other than 4.x.

It also renders the independent Dola test fixture: the same text in the other
Hershey face at a smaller scale, which tests/test_video.py stamps onto a frame to
check that the detector is not keyed to its own template's exact pixels.

Regenerate with:
    uv run --no-project --with 'opencv-python-headless>=4.8,<5' --with numpy \
        python scripts/render_video_hershey_templates.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_ASSETS = _ROOT / "src" / "remove_ai_watermarks" / "assets"
_FIXTURES = _ROOT / "data" / "fixtures" / "synthetic"

# Asset name -> (text, font, canvas width). Every template is drawn at scale 2.2,
# thickness 3, anti-aliased, with the baseline origin at (2, 72) on a 100 px canvas.
TEMPLATES = {
    "video_dola_hershey_duplex.png": ("Dola AI", cv2.FONT_HERSHEY_DUPLEX, 400),
    "video_kling_ai_hershey_simplex.png": ("KLING AI", cv2.FONT_HERSHEY_SIMPLEX, 500),
    "video_kling_ai_hershey_duplex.png": ("KLING AI", cv2.FONT_HERSHEY_DUPLEX, 500),
    "video_klingai_hershey_simplex.png": ("KlingAI", cv2.FONT_HERSHEY_SIMPLEX, 500),
    "video_klingai_hershey_duplex.png": ("KlingAI", cv2.FONT_HERSHEY_DUPLEX, 500),
}


def render(
    text: str,
    font: int,
    width: int,
    *,
    height: int = 100,
    origin: tuple[int, int] = (2, 72),
    scale: float = 2.2,
    thickness: int = 3,
) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.putText(canvas, text, origin, font, scale, 255, thickness, cv2.LINE_AA)
    ys, xs = np.nonzero(canvas)
    return canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _write(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise SystemExit(f"failed to write {path}")
    log.info("%s: %dx%d", path.relative_to(_ROOT), image.shape[1], image.shape[0])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not cv2.__version__.startswith("4."):
        raise SystemExit(f"OpenCV {cv2.__version__} draws different glyphs; run with OpenCV 4.x")
    for name, (text, font, width) in TEMPLATES.items():
        _write(_ASSETS / name, render(text, font, width))
    fixture = render("Dola AI", cv2.FONT_HERSHEY_SIMPLEX, 150, height=40, origin=(2, 28), scale=0.9, thickness=2)
    _FIXTURES.mkdir(parents=True, exist_ok=True)
    _write(_FIXTURES / "dola_hershey_simplex.png", fixture)


if __name__ == "__main__":
    main()
