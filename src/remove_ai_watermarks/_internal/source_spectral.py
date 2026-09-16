"""Phase-free spectral features for source/export-pipeline classification."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class SpectralGeometry:
    """Precomputed coordinates for one phase-free spectral representation."""

    size: int
    grid: int
    radius: NDArray[np.int32]
    radius_counts: NDArray[np.int64]
    passband: NDArray[np.bool_]


@lru_cache(maxsize=4)
def make_geometry(size: int = 256, grid: int = 16) -> SpectralGeometry:
    """Build radial detrending and pooling geometry."""
    if size < 16 or grid < 2 or size % grid:
        raise ValueError("size must be at least 16 and divisible by grid")
    yy, xx = np.mgrid[:size, :size]
    radius = np.rint(np.hypot(yy - size // 2, xx - size // 2)).astype(np.int32)
    radius_counts = np.bincount(radius.ravel())
    passband = (radius >= 4) & (radius <= size * 0.48)
    return SpectralGeometry(size, grid, radius, radius_counts, passband)


def phase_free_log_spectrum(channel: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return centered log FFT magnitude, deliberately discarding phase."""
    if channel.ndim != 2:
        raise ValueError("channel must be a two-dimensional array")
    centered = channel - float(np.mean(channel))
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(centered))))


def spectral_feature(pixels: NDArray[np.float64], geometry: SpectralGeometry) -> NDArray[np.float64]:
    """Extract pooled radial-residual spectra from luma and opponent color."""
    expected = (geometry.size, geometry.size, 3)
    if pixels.shape != expected:
        raise ValueError(f"pixels must have shape {expected}, got {pixels.shape}")
    red, green, blue = np.moveaxis(pixels, 2, 0)
    channels = (
        0.299 * red + 0.587 * green + 0.114 * blue,
        red - green,
        blue - (red + green) / 2.0,
    )
    block = geometry.size // geometry.grid
    features: list[NDArray[np.float64]] = []
    for channel in channels:
        spectrum = phase_free_log_spectrum(channel)
        radial_sum = np.bincount(geometry.radius.ravel(), weights=spectrum.ravel())
        radial_mean = radial_sum / geometry.radius_counts
        residual = np.where(
            geometry.passband,
            spectrum - radial_mean[geometry.radius],
            0.0,
        )
        pooled = residual.reshape(geometry.grid, block, geometry.grid, block).mean(axis=(1, 3))
        features.append(pooled.ravel())
    return np.concatenate(features)


def feature_from_image(image: Image.Image, geometry: SpectralGeometry) -> NDArray[np.float64]:
    """Resize an RGB image canonically and extract its phase-free feature."""
    canonical = image.resize((geometry.size, geometry.size), Image.Resampling.LANCZOS)
    pixels = np.asarray(canonical, dtype=np.float64) / 255.0
    return spectral_feature(pixels, geometry)
