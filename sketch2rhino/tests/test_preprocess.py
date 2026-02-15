import numpy as np
import pytest

pytest.importorskip("cv2")

from sketch2rhino.config import PreprocessConfig
from sketch2rhino.vision.preprocess import preprocess_image


def test_preprocess_image_returns_binary_foreground():
    gray = np.full((128, 128), 255, dtype=np.uint8)
    gray[60:68, 20:108] = 0

    cfg = PreprocessConfig()
    binary = preprocess_image(gray, cfg)

    assert binary.dtype == np.uint8
    assert binary.shape == gray.shape
    assert binary.max() == 1
    assert binary.sum() > 0
