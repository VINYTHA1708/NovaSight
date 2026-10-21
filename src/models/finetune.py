"""NovaSight B1 fine-tuning trainer (Part 4.2, approved_policy).

Fixed configuration [APPROVED-POLICY]:
  - Only layer4 unfrozen (stem-layer3 frozen)
  - backbone LR 1e-4, head LR 1e-3
  - AdamW, weight_decay=1e-4
  - batch_size=32, max_epochs=15, patience=3 on early-stop cross-entropy
  - Restore best epoch checkpoint
  - Augmentation: H/V flip + brightness/contrast jitter 0.1
  - model-init seed from config

Same configuration for every nested rehearsal fine-tune and final fine-tune.
"""
from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.config import get_config, REPO_ROOT
from src.data.transforms import get_train_transform, get_eval_transform
from src.models.backbones import ResNetWithHead

logger = logging.getLogger(__name__)


class NEUDataset(Dataset):
    """PyTorch dataset for NEU-CLS images.

    Args:
        records: List of image records (each has "path" and "prefix").
        class_to_idx: Dict mapping prefix -> integer label.
        transform: torchvision transform to apply.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        class_to_idx: dict[str, int],
        transform,
    ) -> None:
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        path = REPO_ROOT / rec["path"]
        img = Image.open(path).convert("L")
        tensor = self.transform(img)
        label = self.class_to_idx[rec["prefix"]]
        return tensor, label


@dataclass
class FinetuneResult:
    """Results from a B1 fine-tuning run.

    Attributes:
        best_epoch: Epoch with lowest early-stop loss (1-indexed).
        train_losses: Per-epoch train cross-entropy losses.
        val_losses: Per-epoch early-stop cross-entropy losses.
        stopped_early: True if early stopping was triggered.
        elapsed_s: Total training time in seconds.
        weight_sha256: SHA-256 of final model state dict.
    """

    best_epoch: int
    train_losses: list[float]
    val_losses: list[float]
    stopped_early: bool
    elapsed_s: float
    weight_sha256: str


def finetune_b1(
    train_records: list[dict[str, Any]],
    early_stop_records: list[dict[str, Any]],
    class_to_idx: dict[str, int],
    num_classes: int,
    feature: str = "L4",
    init_seed: int = 0,
    device: torch.device | None = None,
) -> tuple[ResNetWithHead, FinetuneResult]:
    """Fine-tune ResNet-18 (B1) with the approved fixed configuration.

    This function is the ONLY place B1 training happens.
    The configuration is read from configs/default.yaml approved_policy
    and asserted to match -- never overridden by caller.

    Args:
        train_records: Image records for training (known classes, train pool).
        early_stop_records: Image records for early stopping (early_stop pool).
        class_to_idx: Mapping from class prefix to integer label.
        num_classes: Number of output classes.
        feature: Feature mode ("L4" or "L234").
        init_seed: Random seed for head initialisation.
        device: Torch device (CPU if None).

    Returns:
        Tuple of (fine-tuned model, FinetuneResult).
    """
    cfg = get_config()
    ap = cfg["approved_policy"]["b1_finetune"]

    # Read fixed approved config
    backbone_lr: float = ap["backbone_lr"]
    head_lr: float = ap["head_lr"]
    weight_decay: float = ap["weight_decay"]
    batch_size: int = ap["batch_size"]
    max_epochs: int = ap["max_epochs"]
    patience: int = ap["patience"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "B1 fine-tune: n_train=%d, n_val=%d, classes=%d, feature=%s, "
        "backbone_lr=%.1e, head_lr=%.1e, max_epochs=%d, patience=%d, device=%s",
        len(train_records), len(early_stop_records), num_classes,
        feature, backbone_lr, head_lr, max_epochs, patience, device,
    )

    # Assertion: verify config matches approved policy
    assert backbone_lr == 1e-4, f"backbone_lr must be 1e-4, got {backbone_lr}"
    assert head_lr == 1e-3, f"head_lr must be 1e-3, got {head_lr}"
    assert weight_decay == 1e-4, f"weight_decay must be 1e-4, got {weight_decay}"
    assert batch_size == 32, f"batch_size must be 32, got {batch_size}"
    assert max_epochs == 15, f"max_epochs must be 15, got {max_epochs}"
    assert patience == 3, f"patience must be 3, got {patience}"

    train_tfm = get_train_transform("B1")
    val_tfm = get_eval_transform("B1")

    torch.manual_seed(init_seed)
    model = ResNetWithHead(
        num_classes=num_classes,
        feature=feature,
        init_seed=init_seed,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.get_param_groups(backbone_lr, head_lr),
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    train_ds = NEUDataset(train_records, class_to_idx, train_tfm)
    val_ds = NEUDataset(early_stop_records, class_to_idx, val_tfm)

    # Data loaders with worker seed for reproducibility
    g = torch.Generator()
    g.manual_seed(init_seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    stopped_early = False

    t0 = time.time()
    model.train()

    for epoch in range(1, max_epochs + 1):
        # Training
        model.train()
        epoch_train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(imgs)
        epoch_train_loss /= len(train_ds)
        train_losses.append(epoch_train_loss)

        # Validation (early_stop pool)
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out["logits"], labels)
                epoch_val_loss += loss.item() * len(imgs)
        epoch_val_loss /= len(val_ds)
        val_losses.append(epoch_val_loss)

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch, max_epochs, epoch_train_loss, epoch_val_loss,
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                stopped_early = True
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.time() - t0
    from src.models.backbones import _hash_state_dict
    sha = _hash_state_dict(model.state_dict())

    result = FinetuneResult(
        best_epoch=best_epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        stopped_early=stopped_early,
        elapsed_s=elapsed,
        weight_sha256=sha,
    )
    logger.info(
        "B1 fine-tune complete: best_epoch=%d, stopped_early=%s, elapsed=%.1fs",
        best_epoch, stopped_early, elapsed,
    )
    return model, result
