"""Simple DARTH/LAET-inspired policy for PLAID adaptive budget selection.

This module deliberately contains no qrels logic.  It only consumes signals
available at an approximate checkpoint.  Candidate-generation coverage is
reported separately by the replay script as an offline diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


def _float(row: Mapping[str, object], key: str) -> Optional[float]:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if value == value else None


@dataclass(frozen=True)
class HybridPolicyConfig:
    # DARTH-style confidence proxy: approximate Top-100 overlap with the
    # final exact refinement.  Thresholds are calibrated offline on old only.
    overlap_threshold_400: float = 0.7900
    overlap_threshold_512: float = 0.8594
    # LAET-style local stability guards.  A checkpoint is safe only when the
    # newly admitted tail and score variance are both small.
    new_fraction_threshold_400: float = 0.25
    new_fraction_threshold_512: float = 0.20
    tail_variance_threshold_400: float = 0.0040
    tail_variance_threshold_512: float = 0.0040
    # Missing signals are treated conservatively as unsafe.


def _stable(row: Mapping[str, object], budget: int, cfg: HybridPolicyConfig) -> bool:
    if budget == 400:
        new_limit = cfg.new_fraction_threshold_400
        variance_limit = cfg.tail_variance_threshold_400
    else:
        new_limit = cfg.new_fraction_threshold_512
        variance_limit = cfg.tail_variance_threshold_512
    tail_variance = _float(row, "tail_variance")
    if tail_variance is None:
        return False
    # At the first checkpoint there is no previous checkpoint, so
    # newly_admitted_fraction is mechanically 1.0 and cannot be a stability
    # signal.  From 512 onward it becomes meaningful.
    if budget == 400:
        return tail_variance <= variance_limit
    new_fraction = _float(row, "newly_admitted_fraction")
    return new_fraction is not None and new_fraction <= new_limit and tail_variance <= variance_limit


def checkpoint_safe(row: Mapping[str, object], budget: int,
                    cfg: HybridPolicyConfig = HybridPolicyConfig()) -> bool:
    """Return whether a checkpoint is safe for early termination.

    The overlap is the DARTH-like confidence signal.  The two local guards are
    the LAET-like stability signal.  This is intentionally conservative and
    deployable: it does not inspect qrels or future-budget results.
    """
    overlap = _float(row, "approx_topk_overlap_stage1_vs_refine")
    if overlap is None:
        return False
    threshold = (cfg.overlap_threshold_400 if budget == 400
                 else cfg.overlap_threshold_512)
    return overlap >= threshold and _stable(row, budget, cfg)


def choose_budget(row400: Mapping[str, object], row512: Mapping[str, object],
                  cfg: HybridPolicyConfig = HybridPolicyConfig()) -> int:
    """Choose 400, 512, or 1024 using only checkpoint-local signals."""
    if checkpoint_safe(row400, 400, cfg):
        return 400
    if checkpoint_safe(row512, 512, cfg):
        return 512
    return 1024
