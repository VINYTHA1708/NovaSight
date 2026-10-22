"""NovaSight standalone runtime package.

This package provides lightweight, torch-free, torchvision-free, sklearn-free
inference, preprocessing, artifact loading, scoring, gating, and memory lifecycle.
"""
from __future__ import annotations

from src.runtime.preprocess import preprocess_image, load_and_check_app_image_numpy, SanityCheckError
from src.runtime.onnx_extractor import OnnxFeatureExtractor
from src.runtime.artifact import load_runtime_artifact, RuntimeArtifact
from src.runtime.scoring import RuntimeScorer, make_decision
from src.runtime.gate import GateReport, run_gate
from src.runtime.memory import EnrolledClass, RuntimeMemory, replay_events

__all__ = [
    "preprocess_image",
    "load_and_check_app_image_numpy",
    "SanityCheckError",
    "OnnxFeatureExtractor",
    "load_runtime_artifact",
    "RuntimeArtifact",
    "RuntimeScorer",
    "make_decision",
    "GateReport",
    "run_gate",
    "EnrolledClass",
    "RuntimeMemory",
    "replay_events",
]
