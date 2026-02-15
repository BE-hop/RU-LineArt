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


def _binarize(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    work = gray
    if not cfg.target_black_on_white:
        work = cv2.bitwise_not(work)

    if cfg.binarize.method == "otsu":
        _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
