"""Numpy-based image preprocessing and sanity checks for NovaSight runtime.

Replicates torchvision B0 evaluation pipeline without requiring torch or torchvision:
  1. PIL Image convert("L") (1-channel grayscale)
  2. Resize to (224, 224) with BILINEAR resampling
  3. Divide by 255.0 to [0, 1] float32
  4. Repeat to 3 channels (channel-first: (3, 224, 224))
  5. Normalize with ImageNet mean/std:
     mean = [0.485, 0.456, 0.406]
     std  = [0.229, 0.224, 0.225]
  6. Return batch tensor (1, 3, 224, 224) float32
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
DEFAULT_RESIZE = 224
MIN_SIDE_DEFAULT = 32
NEAR_CONSTANT_STD_DEFAULT = 1.0


class SanityCheckError(ValueError):
    """Raised when an uploaded image fails application sanity checks."""

    def __init__(self, check: str, message: str) -> None:
        self.check = check
        super().__init__(message)


def load_and_check_app_image_numpy(
    source: Union[bytes, bytearray, str, Path, Image.Image, np.ndarray],
    min_side: int = MIN_SIDE_DEFAULT,
    near_constant_std: float = NEAR_CONSTANT_STD_DEFAULT,
) -> Image.Image:
    """Load image and verify application sanity checks (decodability, min_side, contrast).

    Args:
        source: Image file bytes, path, PIL Image, or numpy RGB array.
        min_side: Minimum allowed side dimension in pixels (default: 32).
        near_constant_std: Grayscale std threshold for near-constant rejection (default: 1.0).

    Returns:
        Validated PIL Image in grayscale ('L') mode.

    Raises:
        SanityCheckError: If image fails decodability, size, or contrast checks.
    """
    try:
        if isinstance(source, Image.Image):
            img = source
        elif isinstance(source, np.ndarray):
            img = Image.fromarray(source.astype(np.uint8))
        elif isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)
        img.load()
    except Exception as exc:
        raise SanityCheckError(
            "undecodable",
            f"Cannot read image file: {exc}. Please upload a valid image (BMP, JPEG, PNG).",
        ) from exc

    w, h = img.size
    if min(w, h) < min_side:
        raise SanityCheckError(
            "min_side",
            f"Image is too small ({w}x{h}). Minimum side length is {min_side} px.",
        )

    gray_arr = np.array(img.convert("L"), dtype=np.float32)
    std = float(gray_arr.std())
    if std < near_constant_std:
        raise SanityCheckError(
            "near_constant",
            f"Image appears near-constant (grayscale std = {std:.3f} < {near_constant_std}). "
            "Please upload an image with visible content.",
        )

    return img.convert("L")


def preprocess_image(
    img: Image.Image,
    resize: int = DEFAULT_RESIZE,
) -> np.ndarray:
    """Apply the B0 evaluation preprocessing pipeline using pure numpy and PIL.

    Args:
        img: Input PIL Image (any mode).
        resize: Target spatial dimensions (width, height).

    Returns:
        Numpy array of shape (1, 3, resize, resize) in float32.
    """
    # 1. Grayscale (L mode)
    g = img.convert("L")
    # 2. Resize to (resize, resize) with bilinear interpolation
    r = g.resize((resize, resize), resample=Image.BILINEAR)
    # 3. Float32 array divided by 255.0
    arr = np.array(r, dtype=np.float32) / 255.0  # (H, W)
    # 4. Repeat to 3 channels: (3, H, W)
    arr3 = np.stack([arr, arr, arr], axis=0)
    # 5. Normalize with ImageNet mean and std
    norm = (arr3 - IMAGENET_MEAN) / IMAGENET_STD
    # 6. Add batch dimension: (1, 3, H, W)
    return np.ascontiguousarray(norm[None, ...], dtype=np.float32)
