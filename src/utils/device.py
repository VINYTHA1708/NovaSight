"""Device configuration and deterministic CUDA settings (Protocol v1.1)."""
from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


def get_device_info(device: torch.device | None = None) -> dict[str, Any]:
    """Return dictionary with detailed device, CUDA, and cuDNN information."""
    use_cuda = torch.cuda.is_available() if device is None else (device.type == "cuda")
    info: dict[str, Any] = {
        "device": "cuda" if use_cuda else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if use_cuda and torch.cuda.is_available():
        info.update({
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
            "deterministic_cuda": True,
        })
    else:
        info.update({
            "device_name": "CPU",
            "cuda_version": None,
            "cudnn_version": None,
            "deterministic_cuda": False,
        })
    return info


def setup_device(
    device_arg: str = "auto",
    return_info: bool = False,
) -> torch.device | tuple[torch.device, dict[str, Any]]:
    """Configure compute device and deterministic CUDA flags if GPU is selected.

    Deterministic settings for CUDA:
        - torch.backends.cudnn.deterministic = True
        - torch.backends.cudnn.benchmark = False
        - torch.use_deterministic_algorithms(True, warn_only=True)
        - CUBLAS_WORKSPACE_CONFIG = :4096:8

    Args:
        device_arg: "auto", "cpu", or "cuda".
        return_info: Whether to return (device, device_info) tuple. Default False (returns device).

    Returns:
        torch.device (or tuple of (torch.device, device_info_dict) if return_info=True).
    """
    device_arg = device_arg.lower()
    if device_arg == "auto":
        use_cuda = torch.cuda.is_available()
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested via --device cuda, but torch.cuda.is_available() is False.")
        use_cuda = True
    elif device_arg == "cpu":
        use_cuda = False
    else:
        raise ValueError(f"Unknown device argument '{device_arg}'. Must be one of: 'auto', 'cpu', 'cuda'.")

    if use_cuda:
        device = torch.device("cuda:0")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            logger.warning("Could not set torch.use_deterministic_algorithms: %s", e)

        device_info = get_device_info(device)
        logger.info(
            "Configured device: CUDA (%s, CUDA %s, cuDNN %s)",
            device_info["device_name"],
            device_info["cuda_version"],
            device_info["cudnn_version"],
        )
    else:
        device = torch.device("cpu")
        device_info = get_device_info(device)
        logger.info("Configured device: CPU")

    if return_info:
        return device, device_info
    return device
