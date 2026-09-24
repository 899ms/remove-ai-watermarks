"""Build the PROVISIONAL synthetic OpenArt alpha asset.

Unlike ``visible_alpha_solve.py``, this does NOT recover an alpha map from a
controlled capture: OpenArt (openart.ai) requires an account to generate
images (no anonymous or API-key access, confirmed 2026-09-21 via its own
help center and MCP docs), and creating an account is outside what this
project's tooling is allowed to do on its own. No committed capture exists
under ``data/calibration/openart/``.

This script instead PROCEDURALLY DRAWS a reconstruction of OpenArt's own
brand mark (the bowtie/infinity icon followed by the "OpenArt" wordmark, as
shown on openart.ai's own marketing pages) as a semi-transparent white
overlay, matching the general rendering style (near-white, low-saturation,
anti-aliased) that every other bundled text-mark asset in this module was
solved from. It is a reconstruction from the public logo, not a recovered
watermark alpha map, and ``openart_engine.py`` documents that provenance
gap explicitly. Replace this asset (and re-run the engine's calibration)
the moment a real captured OpenArt export is available under
``data/calibration/openart/``.

The wordmark is a Hershey ``cv2.putText`` render, which OpenCV 5 draws with
different glyphs than the OpenCV 4 build the committed asset came from, so the
script refuses to run on any OpenCV other than 4.x.

Usage::

    uv run --no-project --with 'opencv-python-headless>=4.8,<5' --with numpy \
        python scripts/build_openart_alpha.py
"""

# cv2/numpy boundary: third-party libs ship no usable element types; relax the
# unknown-type rules for this file only (mirrors visible_alpha_solve.py).
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

_ROOT = Path(__file__).resolve().parents[1]
_ASSET = _ROOT / "src" / "remove_ai_watermarks" / "assets" / "openart_alpha.png"

# Canvas sized to comfortably fit "OpenArt" set bold at this height plus the
# icon and inter-glyph gap, at a resolution close to the other bundled
# assets (doubao_alpha.png is 335x83).
_W, _H = 360, 90
_ALPHA_PEAK = 173  # matches doubao_alpha.png's measured peak (~68% opacity)


def _draw_infinity_icon(canvas: np.ndarray, cx: int, cy: int, size: int) -> None:
    """Draw the bowtie/infinity glyph left of the wordmark (two mirrored lobes)."""
    half = size // 2
    # Two overlapping filled triangles meeting at the center, echoing the
    # angular "infinity" mark on openart.ai's own site header.
    left_lobe = np.array(
        [[cx - size, cy - half], [cx - size, cy + half], [cx, cy]],
        dtype=np.int32,
    )
    right_lobe = np.array(
        [[cx + size, cy - half], [cx + size, cy + half], [cx, cy]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, left_lobe, (_ALPHA_PEAK,), lineType=cv2.LINE_AA)
    cv2.fillConvexPoly(canvas, right_lobe, (_ALPHA_PEAK,), lineType=cv2.LINE_AA)


def build() -> np.ndarray:
    canvas = np.zeros((_H, _W), np.uint8)
    icon_cx, icon_cy, icon_size = 34, _H // 2, 22
    _draw_infinity_icon(canvas, icon_cx, icon_cy, icon_size)

    text_x = icon_cx + icon_size + 18
    cv2.putText(
        canvas,
        "OpenArt",
        (text_x, _H // 2 + 16),
        cv2.FONT_HERSHEY_DUPLEX,
        1.3,
        (_ALPHA_PEAK,),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    if not cv2.__version__.startswith("4."):
        raise SystemExit(f"OpenCV {cv2.__version__} draws different glyphs; run with OpenCV 4.x")
    canvas = build()
    _ASSET.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(_ASSET), canvas)
    print(f"Wrote {_ASSET} ({canvas.shape[1]}x{canvas.shape[0]}, peak={int(canvas.max())})")


if __name__ == "__main__":
    main()
