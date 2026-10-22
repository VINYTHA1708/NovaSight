"""Export ResNet-18 (B0, L4) to ONNX with metadata and model card (Stage S1).

This script exports the frozen B0 ResNet-18 feature extractor with IMAGENET1K_V1
weights directly to ONNX format. It evaluates the model directly in eval mode
without adding any extra nn.Module wrappers.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys

import torch
import torchvision
import onnx
import onnxruntime as ort

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.backbones import ResNetExtractor, _hash_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_resnet_b0_l4(
    output_dir: Path | str = "models/onnx",
    opset_version: int = 17,
) -> tuple[Path, Path]:
    """Export ResNetExtractor('B0', 'L4') to ONNX and generate MODEL_CARD.json.

    Args:
        output_dir: Directory to save ONNX model and model card.
        opset_version: Target ONNX opset version (default: 17).

    Returns:
        Tuple of (onnx_model_path, model_card_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "resnet18_b0_l4.onnx"
    card_path = out_dir / "MODEL_CARD.json"

    logger.info("Initializing ResNetExtractor('B0', 'L4') with IMAGENET1K_V1 weights...")
    model = ResNetExtractor("B0", "L4")
    # Call .eval() directly on the extractor itself (do NOT wrap in another nn.Module)
    model.eval()

    state_dict_hash = _hash_state_dict(model.state_dict())
    logger.info("Model state_dict SHA-256: %s", state_dict_hash)

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    logger.info("Exporting ONNX model to %s (opset=%d)...", onnx_path, opset_version)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=opset_version,
        dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["f"],
        dynamic_axes={"x": {0: "batch"}, "f": {0: "batch"}},
    )

    # Compute ONNX file SHA-256
    onnx_bytes = onnx_path.read_bytes()
    onnx_sha256 = hashlib.sha256(onnx_bytes).hexdigest()
    logger.info("ONNX file SHA-256: %s (%d bytes)", onnx_sha256, len(onnx_bytes))

    # Verify ONNX model can be loaded by onnx and onnxruntime
    onnx_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_proto)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_out = session.run(["f"], {"x": dummy_input.numpy()})[0]
    assert test_out.shape == (1, 512), f"Unexpected output shape: {test_out.shape}"
    logger.info("ONNX model validated successfully with onnxruntime.")

    # Write MODEL_CARD.json
    model_card = {
        "model_name": "resnet18_b0_l4",
        "architecture": "ResNet-18",
        "feature_layer": "layer4",
        "feature_dim": 512,
        "pooling": "Global Average Pooling (AdaptiveAvgPool2d(1, 1))",
        "torchvision_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "state_dict_sha256": state_dict_hash,
        "onnx_file": onnx_path.name,
        "onnx_file_sha256": onnx_sha256,
        "onnx_file_bytes": len(onnx_bytes),
        "opset_version": opset_version,
        "input_tensor": {
            "name": "x",
            "shape": ["batch", 3, 224, 224],
            "dtype": "float32",
        },
        "output_tensor": {
            "name": "f",
            "shape": ["batch", 512],
            "dtype": "float32",
        },
        "export_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "python": sys.version,
        },
        "attribution": "torchvision ImageNet weights (IMAGENET1K_V1)",
    }

    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)
    logger.info("Wrote model card to %s", card_path)

    return onnx_path, card_path


if __name__ == "__main__":
    export_resnet_b0_l4()
