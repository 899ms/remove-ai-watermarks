"""Tests for the OpenArt visible-watermark engine (localize -> fill).

No real captured OpenArt sample is committed (see ``openart_engine.py``'s module
docstring for why), so every fixture here is a synthetic composite of the
bundled, procedurally reconstructed alpha asset -- there is no
``TestRealSample`` analogue to ``test_doubao_engine.py``'s skip-if-absent class.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks.openart_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_NATIVE_WIDTH,
    _ALPHA_WIDTH_FRAC,
    DETECT_NCC_THRESHOLD,
    OpenArtEngine,
    _alpha_template,
    _glyph_silhouette,
    _template_match_score,
)


def _compose(w: int, h: int, bg: float = 100.0):
    """Composite the bundled alpha (scaled to the image's short side) onto a flat
    bg, centered in the frame. Returns ``(watermarked_uint8, mark_bool_mask)``."""
    img = np.full((h, w, 3), bg, np.float32)
    at = _alpha_template()
    base = min(w, h)
    gw, gh = int(_ALPHA_WIDTH_FRAC * base), int(_ALPHA_HEIGHT_FRAC * base)
    ax = (w - gw) // 2
    ay = (h - gh) // 2
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    a3 = amap[:, :, None]
    wm = (a3 * 255.0 + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, amap > 0.2


class TestLocate:
    def test_box_centered_in_frame(self):
        eng = OpenArtEngine()
        img = np.zeros((2048, 2048, 3), np.uint8)
        loc = eng.locate(img)
        cx, cy = loc.x + loc.w / 2, loc.y + loc.h / 2
        assert cx == pytest.approx(2048 / 2, abs=2048 * 0.02)
        assert cy == pytest.approx(2048 / 2, abs=2048 * 0.02)

    def test_box_scales_with_short_side(self):
        eng = OpenArtEngine()
        small = eng.locate(np.zeros((1024, 1024, 3), np.uint8))
        large = eng.locate(np.zeros((2048, 2048, 3), np.uint8))
        assert large.w == pytest.approx(small.w * 2, rel=0.1)

    def test_box_stays_centered_on_portrait(self):
        """The one confirmed carrier is a portrait (1528x2712); the box must stay
        horizontally AND vertically centered, not just anchored to one axis."""
        eng = OpenArtEngine()
        loc = eng.locate(np.zeros((2712, 1528, 3), np.uint8))
        cx, cy = loc.x + loc.w / 2, loc.y + loc.h / 2
        assert cx == pytest.approx(1528 / 2, abs=1528 * 0.02)
        assert cy == pytest.approx(2712 / 2, abs=2712 * 0.02)


class TestDetect:
    def test_clean_gradient_not_detected(self):
        eng = OpenArtEngine()
        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not eng.detect(img).detected

    def test_solid_blob_center_not_detected(self):
        """A bright blob is not the glyph shape -> low correlation, not detected."""
        eng = OpenArtEngine()
        img = np.zeros((1024, 1024, 3), np.uint8)
        x, y, bw, bh = eng.locate(img).bbox
        img[y + bh // 4 : y + bh * 3 // 4, x : x + bw // 2] = 200
        assert not eng.detect(img).detected

    def test_silhouette_loads(self):
        sil = _glyph_silhouette()
        assert sil is not None
        assert set(np.unique(sil)).issubset({0, 255})

    def test_match_score_shape_sensitive(self):
        """The glyph silhouette correlates with itself, not with a filled block."""
        sil = _glyph_silhouette()
        h, w = sil.shape
        box = np.zeros((h + 8, int(w / _ALPHA_WIDTH_FRAC * 0.2) + w), np.uint8)
        box[4 : 4 + h, 4 : 4 + w] = sil
        assert _template_match_score(box, _ALPHA_NATIVE_WIDTH) >= DETECT_NCC_THRESHOLD
        solid = np.full_like(box, 255)
        assert _template_match_score(solid, _ALPHA_NATIVE_WIDTH) < DETECT_NCC_THRESHOLD

    def test_small_image_guarded_from_false_positive(self):
        """Below the shared minimum short side, detection is skipped outright
        (``_MIN_DETECT_SHORT_SIDE`` in ``_text_mark_engine``) regardless of vendor."""
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        eng = OpenArtEngine()
        assert eng.detect(wm).detected  # native: synthetic mark detected
        assert not eng.detect(cv2.resize(wm, (150, 150))).detected  # below guard: suppressed

    def test_random_textured_images_do_not_false_fire(self):
        """A frame-center locate box sits over far more varied content than a
        corner box (see the module docstring's calibration caveat); guard against
        the obvious failure mode on plain photographic texture."""
        eng = OpenArtEngine()
        rng = np.random.default_rng(0)
        for _ in range(15):
            h, w = int(rng.integers(1000, 2200)), int(rng.integers(1000, 2200))
            noise = rng.integers(60, 200, (h, w, 3)).astype(np.uint8)
            img = cv2.GaussianBlur(noise, (0, 0), sigmaX=8)
            assert not eng.detect(img).detected


class TestAlphaAsset:
    def test_alpha_asset_loads(self):
        at = _alpha_template()
        assert at is not None
        assert at.dtype.kind == "f"
        assert float(at.min()) >= 0.0
        assert float(at.max()) <= 1.0


class TestFootprintMaskAndRemoval:
    def test_footprint_mask_near_frame_center(self):
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        mask = OpenArtEngine().footprint_mask(wm)
        assert mask is not None
        assert mask.shape == wm.shape[:2]
        ys, xs = np.where(mask > 0)
        assert ys.mean() == pytest.approx(wm.shape[0] / 2, abs=wm.shape[0] * 0.15)
        assert xs.mean() == pytest.approx(wm.shape[1] / 2, abs=wm.shape[1] * 0.15)

    def test_removes_synthetic_mark(self):
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        assert OpenArtEngine().detect(wm).detected
        out, region = registry.get_mark("openart").remove(wm, backend="cv2")
        assert region is not None
        assert not OpenArtEngine().detect(out).detected

    @pytest.mark.parametrize(
        ("w", "h"),
        [
            (_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH),  # square
            (1528, 2712),  # the one confirmed carrier's exact aspect/size
        ],
    )
    def test_fill_removes_and_leaves_frame_edges(self, w, h):
        """The fill lowers re-detect confidence and leaves the untouched corners exact."""
        wm, mark = _compose(w, h)
        assert float(np.abs(wm.astype(np.float32)[mark] - 100.0).mean()) > 15  # mark visible
        before = OpenArtEngine().detect(wm)
        out, _ = registry.get_mark("openart").remove(wm, backend="cv2")
        assert OpenArtEngine().detect(out).confidence < before.confidence
        edge = max(4, min(w, h) // 8)
        assert np.array_equal(out[:edge, :edge], wm[:edge, :edge])
        assert np.array_equal(out[-edge:, -edge:], wm[-edge:, -edge:])


class TestDegenerateAndChannelInputs:
    """footprint_mask must not crash on degenerate sizes or non-3-channel inputs."""

    @pytest.mark.parametrize(("w", "h"), [(2048, 1), (1, 2048), (2048, 8)])
    def test_wide_short_does_not_raise(self, w, h):
        eng = OpenArtEngine()
        img = np.zeros((h, w, 3), np.uint8)
        mask = eng.footprint_mask(img, force=True)
        assert mask is None or mask.shape == (h, w)

    def test_grayscale_2d_does_not_raise(self):
        eng = OpenArtEngine()
        gray = np.zeros((2048, 2048), np.uint8)
        mask = eng.footprint_mask(gray, force=True)
        assert mask is None or mask.shape == (2048, 2048)

    def test_bgra_4channel_does_not_raise(self):
        eng = OpenArtEngine()
        bgra = np.zeros((2048, 2048, 4), np.uint8)
        mask = eng.footprint_mask(bgra, force=True)
        assert mask is None or mask.shape == (2048, 2048)

    @pytest.mark.parametrize("shape", [(20, 20, 3), (10, 400, 3), (400, 10, 3), (1, 1, 3), (2000, 2000, 3)])
    def test_locate_box_stays_in_bounds(self, shape):
        """locate() must clamp its geometry box inside the image for ANY size/aspect,
        including the frame-center ("cc") corner -- alongside br/bl as in Doubao's test."""
        from remove_ai_watermarks._text_mark_engine import TextMarkEngine
        from remove_ai_watermarks.doubao_engine import _CONFIG as BR_CONFIG
        from remove_ai_watermarks.openart_engine import _CONFIG as CC_CONFIG
        from remove_ai_watermarks.samsung_engine import _CONFIG as BL_CONFIG

        h, w = shape[:2]
        img = np.zeros(shape, np.uint8)
        for cfg in (BR_CONFIG, BL_CONFIG, CC_CONFIG):
            loc = TextMarkEngine(cfg).locate(img)
            assert loc.x >= 0
            assert loc.y >= 0
            assert loc.x + loc.w <= w
            assert loc.y + loc.h <= h
            assert loc.w > 0
            assert loc.h > 0
