"""ONNX-based feature extractor for NovaSight runtime using onnxruntime.

Operates without torch or torchvision. Runs on CPU with configurable threading
and returns features as float64 numpy arrays.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

DEFAULT_ONNX_PATH = Path("models/onnx/resnet18_b0_l4.onnx")


class OnnxFeatureExtractor:
    """CPU feature extractor powered by ONNX Runtime."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_ONNX_PATH,
        num_threads: int | None = None,
    ) -> None:
        """Initialize the ONNX inference session.

        Args:
            model_path: Path to the exported .onnx model file.
            num_threads: Number of intra/inter-op threads (default: from env or 1).
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model file not found: {self.model_path}. "
                "Run scripts/export_onnx.py to export the model first."
            )

        if num_threads is None:
            env_val = os.environ.get("OMP_NUM_THREADS") or os.environ.get("ORT_NUM_THREADS")
            try:
                num_threads = int(env_val) if env_val else 1
            except ValueError:
                num_threads = 1
        self.num_threads = max(1, num_threads)

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.num_threads
        session_options.inter_op_num_threads = self.num_threads
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.feature_dim = 512

    def extract(self, tensor: np.ndarray) -> np.ndarray:
        """Extract features from preprocessed image tensor.

        Args:
            tensor: Input numpy array of shape (N, 3, 224, 224) or (3, 224, 224) float32.

        Returns:
            Extracted feature array in float64.
            If input was a single image (3, 224, 224) or (1, 3, 224, 224), returns (512,).
            If input had batch size N > 1, returns (N, 512).
        """
        if tensor.ndim == 3:
            inp = tensor[None, ...].astype(np.float32)
            single = True
        elif tensor.ndim == 4:
            inp = tensor.astype(np.float32)
            single = (inp.shape[0] == 1)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")

        outputs = self.session.run([self.output_name], {self.input_name: inp})[0]
        # Return float64 feature representations
        feat64 = outputs.astype(np.float64)

        if single:
            return feat64[0]
        return feat64
