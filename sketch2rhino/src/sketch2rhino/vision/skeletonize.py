from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize

from sketch2rhino.config import SkeletonConfig
from sketch2rhino.types import BinaryImage, SkeletonImage

_NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in _NEIGHBORS8:
        nr, nc = r + dr, c + dc
        if _in_bounds(nr, nc, h, w):
            yield nr, nc


def _pixel_degree(skeleton: SkeletonImage, r: int, c: int) -> int:
    h, w = skeleton.shape
    return int(sum(skeleton[nr, nc] for nr, nc in _neighbors8(r, c, h, w)))


def _prune_short_spurs(skeleton: SkeletonImage, min_length_px: int) -> SkeletonImage:
    if min_length_px <= 0:
        return skeleton

    work = skeleton.copy().astype(np.uint8)
    h, w = work.shape

    changed = True
    while changed:
        changed = False
        endpoints = np.argwhere(work == 1)
        for r, c in endpoints:
            if work[r, c] == 0:
                continue
            if _pixel_degree(work, int(r), int(c)) != 1:
                continue

            path: list[tuple[int, int]] = [(int(r), int(c))]
            prev: tuple[int, int] | None = None
            curr = (int(r), int(c))

            for _ in range(min_length_px + 1):
                cr, cc = curr
                candidates = [
                    (nr, nc)
                    for nr, nc in _neighbors8(cr, cc, h, w)
                    if work[nr, nc] == 1 and (prev is None or (nr, nc) != prev)
                ]
                if not candidates:
                    break
                if len(candidates) > 1:
                    break

                nxt = candidates[0]
                path.append(nxt)
                prev, curr = curr, nxt

                degree = _pixel_degree(work, curr[0], curr[1])
                if degree != 2:
                    break

            end_degree = _pixel_degree(work, curr[0], curr[1])
            length = len(path) - 1
            if length >= min_length_px:
                continue

            if end_degree >= 3:
                to_clear = path[:-1]
            else:
                to_clear = path

            if not to_clear:
                continue

            for pr, pc in to_clear:
                work[pr, pc] = 0
            changed = True

    return work


def skeletonize_image(binary: BinaryImage, cfg: SkeletonConfig) -> SkeletonImage:
    skel = sk_skeletonize(binary.astype(bool)).astype(np.uint8)
    if cfg.prune_spurs:
        skel = _prune_short_spurs(skel, int(cfg.spur_min_length_px))
    return skel
