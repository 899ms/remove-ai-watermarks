"""Tests for the generic bare "AI生成" text-mark engine (brand-less TC260 fallback).

No real sample of this mark is committed -- the confirmed production bug this engine
was built for (a bottom-right "AI生成" strip with no vendor wordmark, left completely
unremoved because no existing tuned detector matched it) carries no customer image
bytes in any repository. Detection/removal is exercised against a watermark
synthesized from the engine's own font-rendered alpha asset, mirroring every other
text-mark engine's test pattern (see test_doubao_engine.py / test_jimeng_engine.py).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks.generic_ai_label_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_NATIVE_WIDTH,
    _ALPHA_WIDTH_FRAC,
    DETECT_NCC_THRESHOLD,
    GenericAiLabelEngine,
    _alpha_template,
    _glyph_silhouette,
    _template_match_score,
)


def _compose(w: int, h: int, bg: float = 100.0):
    """Composite the rendered alpha (scaled to width ``w``) onto a flat bg.
    Returns ``(watermarked_uint8, mark_bool_mask)``."""
    img = np.full((h, w, 3), bg, np.float32)
    at = _alpha_template()
    gw, gh = int(_ALPHA_WIDTH_FRAC * w), int(_ALPHA_HEIGHT_FRAC * w)
    margin = int(0.015 * w)
    ax = w - margin - gw
    ay = h - margin - gh
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    a3 = amap[:, :, None]
    wm = (a3 * 255.0 + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, amap > 0.2


class TestLocate:
    def test_box_anchored_bottom_right(self):
        eng = GenericAiLabelEngine()
        img = np.zeros((2048, 2048, 3), np.uint8)
        loc = eng.locate(img)
        assert 2048 - (loc.x + loc.w) < int(2048 * 0.05)
        assert 2048 - (loc.y + loc.h) < int(2048 * 0.05)

    def test_box_scales_with_width(self):
        eng = GenericAiLabelEngine()
        small = eng.locate(np.zeros((1024, 1024, 3), np.uint8))
        large = eng.locate(np.zeros((2048, 2048, 3), np.uint8))
        assert large.w == pytest.approx(small.w * 2, rel=0.1)


class TestDetect:
    def test_clean_gradient_not_detected(self):
        eng = GenericAiLabelEngine()
        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not eng.detect(img).detected

    def test_solid_blob_corner_not_detected(self):
        """A bright blob is not the glyph shape -> low correlation, not detected."""
        eng = GenericAiLabelEngine()
        img = np.zeros((1024, 1024, 3), np.uint8)
        x, y, bw, bh = eng.locate(img).bbox
        img[y + bh // 4 : y + bh * 3 // 4, x : x + bw // 2] = 200
        assert not eng.detect(img).detected

    def test_silhouette_loads(self):
        sil = _glyph_silhouette()
        assert sil is not None
        assert set(np.unique(sil)).issubset({0, 255})

    def test_match_score_shape_sensitive(self):
        """The rendered glyph silhouette correlates with itself, not with a filled block."""
        sil = _glyph_silhouette()
        h, w = sil.shape
        box = np.zeros((h + 8, int(w / _ALPHA_WIDTH_FRAC * 0.2) + w), np.uint8)
        box[4 : 4 + h, 4 : 4 + w] = sil
        assert _template_match_score(box, _ALPHA_NATIVE_WIDTH) >= DETECT_NCC_THRESHOLD
        solid = np.full_like(box, 255)
        assert _template_match_score(solid, _ALPHA_NATIVE_WIDTH) < DETECT_NCC_THRESHOLD

    def test_detects_synthetic_mark(self):
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        eng = GenericAiLabelEngine()
        det = eng.detect(wm)
        assert det.detected
        assert det.confidence >= DETECT_NCC_THRESHOLD

    def test_small_image_guarded_from_false_positive(self):
        """Below the shared minimum short side, detection is skipped outright (the
        same small-image NCC-noise guard every text mark inherits from
        TextMarkEngine._scan)."""
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        eng = GenericAiLabelEngine()
        assert eng.detect(wm).detected  # native: real mark detected
        assert not eng.detect(cv2.resize(wm, (150, 150))).detected  # below guard: suppressed


class TestRivalMarks:
    """The generic template must not double-fire on a mark already attributed to its
    own tuned brand engine -- every rival below contains the literal substring
    "AI生成" in its own glyphs, which is exactly the collision the ``rivals`` guard
    exists to prevent. See generic_ai_label_engine's DETECT_NCC_THRESHOLD comment for
    the calibration this pins."""

    @pytest.mark.parametrize("key", ["doubao", "qwen", "baidu", "kling", "yuanbao"])
    def test_does_not_fire_on_rival_brand_mark(self, key: str):
        from remove_ai_watermarks import watermark_registry as wr

        marked = wr.get_mark(key)
        # Build the rival's own canonical composite via the shared example generator,
        # so this test does not need its own copy of each mark's geometry.
        from scripts.render_visible_examples import base_photo, stamp_image_mark

        base = base_photo(1536, 1152, seed=13)
        out = stamp_image_mark(key, base)
        assert out is not None, key
        img, _box = out
        assert marked.detect(img).detected, f"{key}: sanity check -- its own engine should still fire"
        assert not GenericAiLabelEngine().detect(img).detected, f"generic engine double-fired on {key}"


class TestAlphaAsset:
    def test_alpha_asset_loads(self):
        at = _alpha_template()
        assert at is not None
        assert at.dtype.kind == "f"
        assert float(at.min()) >= 0.0
        assert float(at.max()) <= 1.0


class TestFootprintMaskAndRemoval:
    def test_footprint_mask_in_bottom_right(self):
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        mask = GenericAiLabelEngine().footprint_mask(wm)
        assert mask is not None
        assert mask.shape == wm.shape[:2]
        ys, xs = np.where(mask > 0)
        assert ys.mean() > wm.shape[0] / 2
        assert xs.mean() > wm.shape[1] / 2

    def test_removes_synthetic_mark(self):
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        assert GenericAiLabelEngine().detect(wm).detected
        out, region = registry.get_mark("generic_ai_label").remove(wm, backend="cv2")
        assert region is not None
        assert not GenericAiLabelEngine().detect(out).detected

    def test_far_region_untouched(self):
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        out, _ = registry.get_mark("generic_ai_label").remove(wm, backend="cv2")
        h, w = wm.shape[:2]
        assert np.array_equal(wm[: h // 2, : w // 2], out[: h // 2, : w // 2])


class TestDegenerateAndChannelInputs:
    """footprint_mask must not crash on degenerate sizes or non-3-channel inputs."""

    @pytest.mark.parametrize(("w", "h"), [(2048, 1), (1, 2048), (2048, 8)])
    def test_wide_short_does_not_raise(self, w, h):
        eng = GenericAiLabelEngine()
        img = np.zeros((h, w, 3), np.uint8)
        mask = eng.footprint_mask(img, force=True)
        assert mask is None or mask.shape == (h, w)

    def test_grayscale_2d_does_not_raise(self):
        eng = GenericAiLabelEngine()
        gray = np.zeros((2048, 2048), np.uint8)
        mask = eng.footprint_mask(gray, force=True)
        assert mask is None or mask.shape == (2048, 2048)

    def test_bgra_4channel_does_not_raise(self):
        eng = GenericAiLabelEngine()
        bgra = np.zeros((2048, 2048, 4), np.uint8)
        mask = eng.footprint_mask(bgra, force=True)
        assert mask is None or mask.shape == (2048, 2048)


class TestRegistryWiring:
    """The mark must actually be reachable through the shared registry, the same way
    identify/remove_auto_marks reach every other mark."""

    def test_registered_with_generic_key(self):
        mark = registry.get_mark("generic_ai_label")
        assert mark.key == "generic_ai_label"
        assert mark.in_auto

    def test_participates_in_detect_marks(self):
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        results = registry.detect_marks(wm)
        hit = next(r for r in results if r.key == "generic_ai_label")
        assert hit.detected
