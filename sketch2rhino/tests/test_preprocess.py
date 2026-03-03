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


def test_preprocess_low_ink_recovery_recovers_faint_stroke_when_base_threshold_is_tight():
    gray = np.full((120, 120), 255, dtype=np.uint8)
    gray[60, 15:105] = 245

    cfg_base = PreprocessConfig()
    cfg_base.binarize.otsu_offset = -25.0
    cfg_base.binarize.low_ink_recovery_enable = False
    binary_base = preprocess_image(gray, cfg_base)

    cfg_recover = cfg_base.model_copy(deep=True)
    cfg_recover.binarize.low_ink_recovery_enable = True
    cfg_recover.binarize.low_ink_ratio_trigger = 0.02
    cfg_recover.binarize.low_ink_otsu_boost = 30.0
    cfg_recover.binarize.low_ink_min_component_px = 2
    binary_recover = preprocess_image(gray, cfg_recover)

    assert int(binary_recover.sum()) > int(binary_base.sum())
    assert int(binary_recover.sum()) > 0
