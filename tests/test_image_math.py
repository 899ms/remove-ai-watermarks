"""The shared DCT basis must be the orthonormal DCT-II that OpenCV computes."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks._internal.image_math import dct_matrix


@pytest.mark.parametrize("size", [4, 8, 16])
def test_dct_matrix_matches_opencv_dct(size: int) -> None:
    matrix = dct_matrix(np, size)

    # DCT_ROWS transforms each basis vector e_i separately, so row i is column i of the basis.
    np.testing.assert_allclose(matrix, cv2.dct(np.eye(size), flags=cv2.DCT_ROWS).T, atol=1e-12)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(size), atol=1e-12)
