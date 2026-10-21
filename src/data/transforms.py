"""NovaSight image preprocessing transforms (Part 4.1).

Pipeline:
  1. Load as grayscale (L mode)
  2. Resize to 224x224 (bilinear)
  3. Repeat to 3 channels
  4. Normalise with backbone-specific stats
     - ResNet (B0/B1): ImageNet mean/std
     - DINOv2 (B2): DINOv2 normalisation (same as ImageNet in practice)

Sanity checks (APPLICATION INPUT ONLY):
  - Undecodable file -> reject
  - Min side < 32 px -> reject
  - Near-constant (grayscale std < 1.0 on 0-255) -> reject
These are NEVER applied to evaluation data.

Augmentation for B1 training (approved_policy):
  - RandomHorizontalFlip
  - RandomVerticalFlip
  - ColorJitter brightness=0.1, contrast=0.1
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from src.config import get_config

logger = logging.getLogger(__name__)

BackboneType = Literal["B0", "B1", "B2"]


def get_normalise_transform(backbone: BackboneType) -> T.Normalize:
    """Return the normalisation transform for the given backbone.

    Args:
        backbone: One of "B0", "B1" (ImageNet), "B2" (DINOv2).

    Returns:
        torchvision Normalize transform.
    """
    cfg = get_config()
    pp = cfg["preprocessing"]
    if backbone in ("B0", "B1"):
        mean = pp["imagenet_mean"]
        std = pp["imagenet_std"]
    else:  # B2 DINOv2
        mean = pp["dinov2_mean"]
        std = pp["dinov2_std"]
    return T.Normalize(mean=mean, std=std)


def get_eval_transform(backbone: BackboneType) -> T.Compose:
    """Return the deterministic evaluation transform (no augmentation).

    Used for: feature extraction, conformal calibration, test evaluation.

    Args:
        backbone: Backbone identifier.

    Returns:
        Composed torchvision transform pipeline.
    """
    cfg = get_config()
    resize = cfg["preprocessing"]["resize"]
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR),
            T.Grayscale(num_output_channels=3),   # repeat channel
            T.ToTensor(),
            get_normalise_transform(backbone),
        ]
    )


def get_train_transform(backbone: BackboneType) -> T.Compose:
    """Return the training transform with augmentation (Part 4.2, approved_policy).

    Augmentation:
      - RandomHorizontalFlip
      - RandomVerticalFlip
      - ColorJitter(brightness=0.1, contrast=0.1)

    Args:
        backbone: Backbone identifier.

    Returns:
        Composed torchvision transform pipeline.
    """
    cfg = get_config()
    ap = cfg["approved_policy"]["b1_finetune"]["augmentation"]
    resize = cfg["preprocessing"]["resize"]

    aug_list = [
        T.Grayscale(num_output_channels=1),
        T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR),
        T.Grayscale(num_output_channels=3),
    ]
    if ap.get("horizontal_flip"):
        aug_list.append(T.RandomHorizontalFlip())
    if ap.get("vertical_flip"):
        aug_list.append(T.RandomVerticalFlip())
    bj = ap.get("brightness_jitter", 0.0)
    cj = ap.get("contrast_jitter", 0.0)
    if bj > 0 or cj > 0:
        aug_list.append(T.ColorJitter(brightness=bj, contrast=cj))
    aug_list.extend([T.ToTensor(), get_normalise_transform(backbone)])
    return T.Compose(aug_list)


class SanityCheckError(ValueError):
    """Raised when an uploaded image fails an application sanity check."""

    def __init__(self, check: str, message: str) -> None:
        """Initialise with check name and user-facing message.

        Args:
            check: Short name of the check that failed.
            message: Human-readable description of the failure.
        """
        self.check = check
        super().__init__(message)


def load_and_check_app_image(
    source: bytes | str | Path,
) -> Image.Image:
    """Load an image and run application sanity checks (Part 4.1).

    This function is for APPLICATION INPUT ONLY.
    NEVER call this on evaluation data.

    Args:
        source: Raw bytes, file path, or PIL image path.

    Returns:
        PIL Image in grayscale ('L') mode if all checks pass.

    Raises:
        SanityCheckError: With a descriptive message if any check fails.
    """
    cfg = get_config()
    sc = cfg["sanity_checks"]
    min_side: int = sc["min_side"]
    near_constant_std: float = sc["near_constant_std_threshold"]

    # Check decodability
    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)
        img.load()
    except Exception as exc:
        raise SanityCheckError(
            "undecodable",
            f"Cannot read image file: {exc}. "
            "Please upload a valid image (BMP, JPEG, PNG).",
        ) from exc

    # Check size
    w, h = img.size
    if min(w, h) < min_side:
        raise SanityCheckError(
            "min_side",
            f"Image is too small ({w}x{h}). "
            f"Minimum side length is {min_side} px.",
        )

    # Check near-constant
    gray_arr = np.array(img.convert("L"), dtype=np.float32)
    std = float(gray_arr.std())
    if std < near_constant_std:
        raise SanityCheckError(
            "near_constant",
            f"Image appears near-constant (grayscale std = {std:.3f} < "
            f"{near_constant_std}). "
            "Please upload an image with visible content.",
        )

    return img.convert("L")


def prepare_tensor(
    img: Image.Image,
    backbone: BackboneType,
) -> torch.Tensor:
    """Apply eval preprocessing pipeline to a PIL image.

    Args:
        img: PIL image (any mode; will be converted to grayscale internally).
        backbone: Backbone identifier for normalisation.

    Returns:
        Float tensor of shape (1, 3, 224, 224).
    """
    tfm = get_eval_transform(backbone)
    return tfm(img).unsqueeze(0)
