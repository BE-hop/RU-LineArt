from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sketch2rhino.config import PreprocessConfig
from sketch2rhino.types import BinaryImage


def load_grayscale_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return gray


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        # Use channel-average to stay agnostic to RGB/BGR caller convention.
        rgb = image[:, :, :3].astype(np.float32)
        gray = np.mean(rgb, axis=2)
        return np.clip(gray, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _apply_denoise(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if not cfg.denoise.enable:
        return gray

    strength = max(0.1, float(cfg.denoise.strength))
    if cfg.denoise.method == "gaussian":
        k = int(round(2 * strength + 1))
        if k % 2 == 0:
            k += 1
        k = max(k, 3)
        return cv2.GaussianBlur(gray, (k, k), sigmaX=strength)

    if cfg.denoise.method == "bilateral":
        sigma = 30.0 * strength
        return cv2.bilateralFilter(gray, d=5, sigmaColor=sigma, sigmaSpace=sigma)

    return gray


def _remove_small_components(binary_u8: np.ndarray, min_area_px: int) -> np.ndarray:
    min_area = max(0, int(min_area_px))
    if min_area <= 1:
        return binary_u8

    mask = (binary_u8 > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return binary_u8

    keep = np.zeros(n_labels, dtype=np.uint8)
    keep[0] = 0
    for idx in range(1, n_labels):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area:
            keep[idx] = 1

    filtered = (keep[labels] > 0).astype(np.uint8) * 255
    return filtered


def _binarize(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    work = gray
    if not cfg.target_black_on_white:
        work = cv2.bitwise_not(work)

    if cfg.binarize.method == "otsu":
        otsu_t, _ = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = float(np.clip(float(otsu_t) + float(getattr(cfg.binarize, "otsu_offset", 0.0)), 0.0, 255.0))
        _, binary = cv2.threshold(work, t, 255, cv2.THRESH_BINARY_INV)

        if bool(getattr(cfg.binarize, "low_ink_recovery_enable", True)):
            fg_ratio = float(np.count_nonzero(binary)) / float(binary.size)
            trigger = max(0.0, float(getattr(cfg.binarize, "low_ink_ratio_trigger", 0.015)))
            if fg_ratio <= trigger:
                boost = max(0.0, float(getattr(cfg.binarize, "low_ink_otsu_boost", 14.0)))
                t_soft = float(np.clip(t + boost, 0.0, 255.0))
                if t_soft > t:
                    _, binary_soft = cv2.threshold(work, t_soft, 255, cv2.THRESH_BINARY_INV)
                    recovered = cv2.bitwise_and(binary_soft, cv2.bitwise_not(binary))
                    recovered = _remove_small_components(
                        recovered,
                        int(getattr(cfg.binarize, "low_ink_min_component_px", 6)),
                    )
                    binary = cv2.bitwise_or(binary, recovered)
        return binary

    block_size = int(cfg.binarize.block_size)
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1

    return cv2.adaptiveThreshold(
        work,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        cfg.binarize.C,
    )


def _morph_cleanup(binary: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    out = binary.copy()

    if cfg.morph.erode_iter > 0:
        out = cv2.erode(out, kernel, iterations=cfg.morph.erode_iter)
    if cfg.morph.close_iter > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=cfg.morph.close_iter)
    if cfg.morph.open_iter > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=cfg.morph.open_iter)

    return out


def preprocess_image(gray: np.ndarray, cfg: PreprocessConfig) -> BinaryImage:
    gray_in = _ensure_grayscale(gray)
    denoised = _apply_denoise(gray_in, cfg)
    binary = _binarize(denoised, cfg)
    cleaned = _morph_cleanup(binary, cfg)
    # Convention: foreground (ink) = 1, background = 0.
    return (cleaned > 0).astype(np.uint8)
