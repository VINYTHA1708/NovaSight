"""S4 Tests: memory, enrollment lifecycle, event log, persistence.

Tests:
  - Enrollment/query disjointness
  - Event-log replay determinism
  - Rollback to version k
  - Merge == union of embeddings
  - Known p-value invariance under enroll/remove
  - Gate synthetic cases: outlier->C, known lookalike->N, duplicate->V
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.gaussian import fit_gaussian, GaussianModel
from src.models.conformal import build_conformal_scores, ConformalScores
from src.models.memory import (
    NovaSightMemory, EnrolledClass, replay_events, _apply_event
)
from src.models.gate import run_gate


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def base_gaussian():
    np.random.seed(0)
    D, K, n_c = 32, 3, 40
    class_names = ["A", "B", "C"]
    train_features = {
        c: np.random.randn(n_c, D) + np.array([i * 5] * D)
        for i, c in enumerate(class_names)
    }
    return fit_gaussian(train_features, k=1)


@pytest.fixture(scope="module")
def base_conformal(base_gaussian):
    np.random.seed(1)
    D = base_gaussian.feature_dim
    cal_features = {
        c: np.random.randn(40, D) + np.array([i * 5] * D)
        for i, c in enumerate(base_gaussian.class_names)
    }
    return build_conformal_scores(base_gaussian, cal_features)


@pytest.fixture
def memory(base_gaussian, base_conformal, tmp_path):
    return NovaSightMemory(
        gaussian=base_gaussian,
        conformal=base_conformal,
        inheritance="H1",
        events_path=tmp_path / "events.jsonl",
    )


@pytest.fixture
def dummy_embeddings(base_gaussian):
    np.random.seed(99)
    D = base_gaussian.feature_dim
    # Embeddings far from all known classes -> novel
    return np.random.randn(5, D) + 100.0


# ── Memory / enrollment tests ─────────────────────────────────────────────────

def test_enroll_adds_class(memory, dummy_embeddings):
    """Enrolling a class must add it to memory."""
    sha256s = [f"sha_{i}" for i in range(5)]
    memory.enroll("NewDefect", dummy_embeddings, sha256s, {"outcome": "ACCEPT"})
    assert "NewDefect" in memory.enrolled


def test_enroll_updates_all_class_names(memory, dummy_embeddings):
    """After enrollment, all_class_names must include the new class."""
    sha256s = [f"sha2_{i}" for i in range(5)]
    memory.enroll("AnotherDefect", dummy_embeddings, sha256s, {"outcome": "ACCEPT"})
    assert "AnotherDefect" in memory.all_class_names
    assert set(memory.known_class_names) == {"A", "B", "C"}


def test_remove_enrolled_class(memory, dummy_embeddings):
    """Removing an enrolled class must delete it from memory."""
    sha256s = [f"sha3_{i}" for i in range(5)]
    memory.enroll("ToRemove", dummy_embeddings, sha256s, {"outcome": "ACCEPT"})
    memory.remove("ToRemove")
    assert "ToRemove" not in memory.enrolled


def test_cannot_remove_base_class(memory):
    """Removing a base class must raise ValueError."""
    with pytest.raises(ValueError, match="base class"):
        memory.remove("A")


def test_merge_is_union_of_embeddings(memory, dummy_embeddings, base_gaussian):
    """Merging two classes must pool all 10 embeddings (5+5)."""
    sha_a = [f"sha_a_{i}" for i in range(5)]
    sha_b = [f"sha_b_{i}" for i in range(5)]
    emb_a = dummy_embeddings + 0.1
    emb_b = dummy_embeddings + 0.2
    memory.enroll("ClassA", emb_a, sha_a, {"outcome": "ACCEPT"})
    memory.enroll("ClassB", emb_b, sha_b, {"outcome": "ACCEPT"})
    memory.merge("ClassA", "ClassB", "MergedAB")
    merged = memory.enrolled["MergedAB"]
    assert merged.n == 10
    assert len(merged.embeddings) == 10
    assert len(merged.sha256s) == 10


def test_event_log_replay_deterministic(base_gaussian, base_conformal, tmp_path, dummy_embeddings):
    """Replaying the event log must produce identical state."""
    events_path = tmp_path / "events.jsonl"
    mem1 = NovaSightMemory(
        gaussian=base_gaussian, conformal=base_conformal,
        inheritance="H1", events_path=events_path,
    )
    sha256s = [f"replaysha_{i}" for i in range(5)]
    mem1.enroll("ReplayClass", dummy_embeddings, sha256s, {"outcome": "ACCEPT"})
    mem1.remove("ReplayClass")

    # Replay
    mem2 = replay_events(events_path, base_gaussian, base_conformal, "H1")
    assert set(mem2.enrolled.keys()) == set(mem1.enrolled.keys())


def test_rollback_to_version(base_gaussian, base_conformal, tmp_path, dummy_embeddings):
    """Replaying up_to_version must restore state at that point."""
    events_path = tmp_path / "rollback_events.jsonl"
    mem = NovaSightMemory(
        gaussian=base_gaussian, conformal=base_conformal,
        inheritance="H1", events_path=events_path,
    )
    sha256s = [f"rbsha_{i}" for i in range(5)]
    mem.enroll("RollbackClass", dummy_embeddings, sha256s, {"outcome": "ACCEPT"})
    # Version after enroll = 1. Now remove it.
    mem.remove("RollbackClass")

    # Rollback to version 1 (after enroll, before remove)
    mem_v1 = replay_events(events_path, base_gaussian, base_conformal, "H1", up_to_version=1)
    assert "RollbackClass" in mem_v1.enrolled


def test_known_pvalue_invariant_after_enroll(memory, dummy_embeddings, base_gaussian, base_conformal):
    """Known class p-values must be identical before and after enrollment."""
    f = np.random.RandomState(0).randn(base_gaussian.feature_dim)
    p_before = memory.all_p_values(f)
    sha256s = [f"invsha_{i}" for i in range(5)]
    memory.enroll("InvariantTest", dummy_embeddings, sha256s, {"outcome": "ACCEPT"})
    p_after = memory.all_p_values(f)
    for c in base_gaussian.class_names:
        assert p_before[c] == p_after[c], f"p-value changed for {c}"


def test_h1_inflated_score_formula(base_gaussian, base_conformal, dummy_embeddings):
    """H1 inflated score must be d_new / (1 + 1/n) with n=5."""
    mem = NovaSightMemory(
        gaussian=base_gaussian, conformal=base_conformal, inheritance="H1",
        events_path=Path(tempfile.mktemp()),
    )
    sha256s = [f"h1sha_{i}" for i in range(5)]
    mem.enroll("H1Test", dummy_embeddings, sha256s, {"outcome": "ACCEPT"})
    cls = mem.enrolled["H1Test"]

    f = np.random.RandomState(42).randn(base_gaussian.feature_dim)
    diff = f - cls.mu
    d = float(diff @ base_gaussian.A @ diff)
    expected_dt = d / (1.0 + 1.0 / 5)
    actual_dt = cls.inflated_score(f, base_gaussian.A, base_conformal.tau_hat)
    assert abs(actual_dt - expected_dt) < 1e-9


# ── Gate synthetic tests ──────────────────────────────────────────────────────

def test_gate_duplicate_image_rejected(base_gaussian, base_conformal, dummy_embeddings):
    """Duplicate sha256 (check V) must produce REJECT."""
    sha256s = ["same_hash"] * 5   # all duplicates
    report = run_gate(
        embeddings=dummy_embeddings,
        sha256s=sha256s,
        class_name="DupTest",
        existing_class_names=base_gaussian.class_names,
        gaussian=base_gaussian,
        conformal=base_conformal,
        alpha_cons=0.01,
        q_BH=0.05,
        eps_max=0.10,
    )
    assert report.outcome == "REJECT"
    assert any("Duplicate" in issue for issue in report.validity_issues)


def test_gate_injected_outlier_triggers_consistency(base_gaussian, base_conformal):
    """Injected outlier (very different embedding) must trigger consistency check C."""
    np.random.seed(5)
    D = base_gaussian.feature_dim
    # 4 clean embeddings + 1 very far outlier
    clean = np.random.randn(4, D) + 100.0   # far from known classes
    outlier = np.zeros((1, D)) + 1000.0     # extreme outlier
    embeddings = np.vstack([clean, outlier])
    sha256s = [f"inj_{i}" for i in range(5)]
    report = run_gate(
        embeddings=embeddings,
        sha256s=sha256s,
        class_name="OutlierTest",
        existing_class_names=base_gaussian.class_names,
        gaussian=base_gaussian,
        conformal=base_conformal,
        alpha_cons=0.05,
        q_BH=0.05,
        eps_max=0.10,
    )
    # Should flag consistency (example 5 is unlike others)
    assert len(report.consistency_issues) > 0 or report.outcome in ("REVIEW", "REJECT")


def test_gate_known_lookalike_triggers_novelty(base_gaussian, base_conformal):
    """Known-lookalike (class A embeddings) must trigger novelty check N."""
    np.random.seed(0)
    D = base_gaussian.feature_dim
    # 3 novel embeddings + 2 class A embeddings (near class A mean)
    mu_A = base_gaussian.class_means[0][0]   # class A mean
    novel = np.random.randn(3, D) + 100.0
    lookalike = np.random.randn(2, D) * 0.01 + mu_A  # very close to A
    embeddings = np.vstack([novel, lookalike])
    sha256s = [f"la_{i}" for i in range(5)]
    report = run_gate(
        embeddings=embeddings,
        sha256s=sha256s,
        class_name="LookalikeTest",
        existing_class_names=base_gaussian.class_names,
        gaussian=base_gaussian,
        conformal=base_conformal,
        alpha_cons=0.01,
        q_BH=0.05,
        eps_max=0.10,
    )
    assert len(report.novelty_issues) > 0 or report.outcome in ("REVIEW", "REJECT")
