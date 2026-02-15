import numpy as np
import pytest

pytest.importorskip("skimage")

from sketch2rhino.config import SkeletonConfig
from sketch2rhino.vision.skeletonize import skeletonize_image


def test_skeletonize_produces_thin_line():
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[28:36, 10:54] = 1

    cfg = SkeletonConfig(prune_spurs=False)
    skel = skeletonize_image(binary, cfg)

    assert skel.dtype == np.uint8
    assert skel.sum() > 0
    # Skeleton should have fewer foreground pixels than a thick stroke.
    assert skel.sum() < binary.sum()
