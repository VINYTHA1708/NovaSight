"""NovaSight backbone feature extractors (Part 4.2, 4.3).

Backbones:
  B0: torchvision ResNet-18, IMAGENET1K_V1 weights, frozen.
  B1: ResNet-18, same weights, fine-tuned (training done in finetune.py).
      For feature extraction, same code as B0 (frozen after fine-tune).
  B2: DINOv2 ViT-S/14 (torch.hub), frozen.

Feature options (Part 4.3):
  ResNet (B0/B1):
    L4:    layer4 global-average-pooled (D=512)
    L234:  GAP(layer2) 128 + GAP(layer3) 256 + GAP(layer4) 512 = 896
            each block z-scored with TRAIN statistics
  DINOv2 (B2):
    CLS:       CLS token (D=384)
    CLS_patch: CLS + mean patch token (D=768)

Usage:
    extractor = ResNetExtractor("B0", feature="L4")
    features = extractor.extract(image_tensor)  # (N, 512)
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.config import get_config, REPO_ROOT

logger = logging.getLogger(__name__)

BackboneId = Literal["B0", "B1", "B2"]
ResNetFeature = Literal["L4", "L234"]
DINOFeature = Literal["CLS", "CLS_patch"]


# ── Weight hashing ────────────────────────────────────────────────────────────

def _hash_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Compute a deterministic sha256 of a model state dict.

    Args:
        state_dict: Model state dictionary.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode())
        tensor = state_dict[key].cpu().contiguous()
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


# ── ResNet feature extractor ──────────────────────────────────────────────────

class ResNetExtractor(nn.Module):
    """Feature extractor based on ResNet-18.

    Supports L4 (layer4 GAP, D=512) and L234 (layers 2-4 concatenated, D=896).
    Always runs on CPU unless a GPU is present.

    Args:
        backbone_id: "B0" (frozen base) or "B1" (fine-tuned).
        feature: "L4" or "L234".
    """

    def __init__(
        self,
        backbone_id: BackboneId,
        feature: ResNetFeature = "L4",
        weight_path: Path | str | None = None,
        weight_sha256: str | None = None,
    ) -> None:
        """Initialise ResNet extractor.

        Args:
            backbone_id: Backbone identifier ("B0" or "B1").
            feature: Feature extraction mode.
            weight_path: Path to fine-tuned weights for B1 (required for B1).
            weight_sha256: Expected SHA-256 hash of the weight file.
        """
        super().__init__()
        if backbone_id not in ("B0", "B1"):
            raise ValueError(f"ResNetExtractor requires B0 or B1, got {backbone_id}")
        self.backbone_id = backbone_id
        self.feature = feature

        if backbone_id == "B1":
            if weight_path is None:
                raise ValueError(
                    "make_extractor('B1') requires fine-tuned weight_path; silent fallback to ImageNet weights is prohibited."
                )
            p = Path(weight_path)
            if not p.is_file():
                raise FileNotFoundError(f"B1 checkpoint file does not exist: {weight_path}")
            from src.utils.hashing import sha256_file
            actual_hash = sha256_file(p)
            if weight_sha256 and actual_hash != weight_sha256:
                raise ValueError(f"B1 weight SHA-256 mismatch: expected {weight_sha256}, got {actual_hash}")
            self.weight_sha256 = actual_hash
            logger.info("Loading fine-tuned B1 weights from %s (sha256: %s)...", p, self.weight_sha256[:16])
            model = models.resnet18(weights=None)
            state = torch.load(p, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            clean_state = {}
            for k, v in state.items():
                if k.startswith("backbone."):
                    clean_state[k[len("backbone."):]] = v
                elif not k.startswith("fc.") and not k.startswith("head."):
                    clean_state[k] = v
            model.load_state_dict(clean_state, strict=False)
        else:
            logger.info("Loading ResNet-18 (IMAGENET1K_V1 weights)...")
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.weight_sha256 = _hash_state_dict(model.state_dict())
            logger.info("ResNet-18 weight sha256: %s", self.weight_sha256[:16] + "...")

        # Keep sub-modules needed for feature extraction
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.eval()

    @property
    def feature_dim(self) -> int:
        """Return the output feature dimensionality."""
        if self.feature == "L4":
            return 512
        elif self.feature == "L234":
            return 896  # 128 + 256 + 512
        raise ValueError(f"Unknown feature: {self.feature}")

    def _forward_features(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Extract intermediate layer outputs.

        Args:
            x: Input tensor (N, 3, 224, 224).

        Returns:
            Dict of layer name -> GAP-pooled feature (N, D).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        l2 = self.gap(self.layer2(x)).flatten(1)   # (N, 128)
        l3 = self.gap(self.layer3(
            self.layer2(x)
        )).flatten(1)  # (N, 256) -- recompute needed due to sequential deps
        # Re-run sequentially to get correct outputs
        return {}, l2, l3  # placeholder; see forward

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of images.

        Args:
            x: Input tensor (N, 3, 224, 224).

        Returns:
            Feature tensor (N, D) where D = feature_dim.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.feature == "L4":
            return self.gap(x4).flatten(1)   # (N, 512)
        elif self.feature == "L234":
            f2 = self.gap(x2).flatten(1)    # (N, 128)
            f3 = self.gap(x3).flatten(1)    # (N, 256)
            f4 = self.gap(x4).flatten(1)    # (N, 512)
            return torch.cat([f2, f3, f4], dim=1)  # (N, 896)
        raise ValueError(f"Unknown feature: {self.feature}")

    def get_head(self) -> nn.Linear | None:
        """Return None for B0 (no classification head used for features).

        Returns:
            None.
        """
        return None


class ResNetWithHead(nn.Module):
    """ResNet-18 with a classification head, used for B1 fine-tuning.

    The head is a single linear layer. layer4 + head are unfrozen;
    stem through layer3 are frozen. The classification head provides
    logits for baselines (MSP, max-logit, energy).

    Args:
        num_classes: Number of output classes.
        feature: Feature mode for the extractor.
        init_seed: Random seed for head initialisation.
    """

    def __init__(
        self,
        num_classes: int,
        feature: ResNetFeature = "L4",
        init_seed: int = 0,
    ) -> None:
        """Initialise the fine-tuning model.

        Args:
            num_classes: Number of output classes.
            feature: Feature mode ("L4" or "L234").
            init_seed: Seed for head weight initialisation.
        """
        super().__init__()
        self.feature = feature
        self.num_classes = num_classes
        self.init_seed = init_seed

        logger.info("Loading ResNet-18 for B1 fine-tuning (IMAGENET1K_V1)...")
        base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.weight_sha256 = _hash_state_dict(base.state_dict())

        # Build extractor layers
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        feature_dim = 512 if feature == "L4" else 896
        torch.manual_seed(init_seed)
        self.head = nn.Linear(feature_dim, num_classes)

        # Freeze stem through layer3 [APPROVED-POLICY]
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        """Freeze stem through layer3; keep layer4 and head trainable."""
        frozen = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3]
        for module in frozen:
            for param in module.parameters():
                param.requires_grad_(False)
        # layer4 and head are trainable
        for param in self.layer4.parameters():
            param.requires_grad_(True)
        for param in self.head.parameters():
            param.requires_grad_(True)

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """Return parameter groups for the optimizer.

        Args:
            backbone_lr: Learning rate for layer4 parameters.
            head_lr: Learning rate for the classification head.

        Returns:
            List of optimizer parameter group dicts.
        """
        return [
            {"params": self.layer4.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning both features and logits.

        Args:
            x: Input tensor (N, 3, 224, 224).

        Returns:
            Dict with keys "features" (N, D) and "logits" (N, C).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.feature == "L4":
            features = self.gap(x4).flatten(1)
        else:
            f2 = self.gap(x2).flatten(1)
            f3 = self.gap(x3).flatten(1)
            f4 = self.gap(x4).flatten(1)
            features = torch.cat([f2, f3, f4], dim=1)

        logits = self.head(features)
        return {"features": features, "logits": logits}

    def get_feature_extractor(self) -> ResNetExtractor:
        """Export the fine-tuned weights as a ResNetExtractor.

        Returns:
            ResNetExtractor with the fine-tuned weights loaded.
        """
        extractor = ResNetExtractor("B1", feature=self.feature)
        # Transfer weights
        for attr in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]:
            getattr(extractor, attr).load_state_dict(
                getattr(self, attr).state_dict()
            )
        extractor.weight_sha256 = _hash_state_dict(
            {k: v for k, v in self.state_dict().items() if not k.startswith("head")}
        )
        return extractor


# ── DINOv2 feature extractor ──────────────────────────────────────────────────

class DINOv2Extractor(nn.Module):
    """Feature extractor based on DINOv2 ViT-S/14, frozen.

    Requires network access on first run to download weights from torch.hub.
    Subsequent runs use the local cache.

    Supports:
      CLS:       CLS token (D=384)
      CLS_patch: CLS + mean patch token (D=768)

    Args:
        feature: "CLS" or "CLS_patch".
        cache_dir: Directory for the torch.hub cache.
    """

    def __init__(
        self,
        feature: DINOFeature = "CLS",
        cache_dir: Path | None = None,
    ) -> None:
        """Initialise DINOv2 extractor.

        Args:
            feature: Feature extraction mode.
            cache_dir: Optional custom hub cache directory.

        Raises:
            RuntimeError: If the model cannot be loaded and cache is unavailable.
        """
        super().__init__()
        self.feature = feature
        cfg = get_config()

        if cache_dir is None:
            cache_dir = REPO_ROOT / cfg["backbones"]["B2"]["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)

        hub_repo = cfg["backbones"]["B2"]["hub_repo"]
        hub_model = cfg["backbones"]["B2"]["hub_model"]

        logger.info("Loading DINOv2 (%s/%s)...", hub_repo, hub_model)
        t0 = time.time()
        try:
            self._model = torch.hub.load(
                hub_repo,
                hub_model,
                verbose=False,
                source="github",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load DINOv2 weights ({hub_repo}/{hub_model}). "
                f"Ensure network access on first run. Error: {exc}"
            ) from exc

        elapsed = time.time() - t0
        logger.info("DINOv2 loaded in %.1fs", elapsed)

        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)

        self.weight_sha256 = _hash_state_dict(self._model.state_dict())
        logger.info("DINOv2 weight sha256: %s", self.weight_sha256[:16] + "...")

    @property
    def feature_dim(self) -> int:
        """Return the output feature dimensionality."""
        if self.feature == "CLS":
            return 384
        elif self.feature == "CLS_patch":
            return 768
        raise ValueError(f"Unknown feature: {self.feature}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of images.

        Args:
            x: Input tensor (N, 3, 224, 224).

        Returns:
            Feature tensor (N, D).
        """
        # DINOv2 forward_features returns dict with "x_norm_clstoken" and
        # "x_norm_patchtokens" keys
        out = self._model.forward_features(x)
        cls = out["x_norm_clstoken"]  # (N, 384)

        if self.feature == "CLS":
            return cls
        elif self.feature == "CLS_patch":
            patches = out["x_norm_patchtokens"]  # (N, num_patches, 384)
            patch_mean = patches.mean(dim=1)      # (N, 384)
            return torch.cat([cls, patch_mean], dim=1)  # (N, 768)
        raise ValueError(f"Unknown feature: {self.feature}")


# ── Factory function ──────────────────────────────────────────────────────────

def make_extractor(
    backbone_id: BackboneId,
    feature: str,
    weight_path: Path | str | None = None,
    weight_sha256: str | None = None,
) -> nn.Module:
    """Create the appropriate feature extractor.

    Args:
        backbone_id: One of "B0", "B1", "B2".
        feature: Feature mode string.
        weight_path: Optional path to fine-tuned weights (required for B1).
        weight_sha256: Optional expected SHA-256 hash of the weight file.

    Returns:
        Feature extractor module in eval mode.
    """
    if backbone_id in ("B0", "B1"):
        if feature not in ("L4", "L234"):
            raise ValueError(f"ResNet features must be L4 or L234, got: {feature}")
        ext = ResNetExtractor(
            backbone_id,
            feature=feature,
            weight_path=weight_path,
            weight_sha256=weight_sha256,
        )
        ext.eval()
        return ext
    elif backbone_id == "B2":
        raise RuntimeError(
            "B2 (DINOv2) is designated NOT RUN under Decision D9. "
            "To run B2, pin a verified hub commit and weight sha256 under a separate approved change."
        )
    raise ValueError(f"Unknown backbone: {backbone_id}")
