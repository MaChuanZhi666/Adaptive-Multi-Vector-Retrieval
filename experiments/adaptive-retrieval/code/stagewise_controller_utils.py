"""Shared leakage-safe feature and split utilities for staged budget control."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BUDGETS = (400, 512, 768, 1024)
STAGES = (400, 512, 768)
LABEL_COLUMN = "oracle_budget_098"
STATE_FEATURES = (
    "cutoff_margin",
    "score_entropy",
    "score_gain",
    "tail_slope",
    "new_candidate_fraction",
    "candidate_generation_failure",
    "truncation_risk",
)


def read_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def eligible_features(stage: int) -> list[str]:
    """Only query-time and current/past stage signals can be used at `stage`."""
    features = ["query_length"]
    for budget in BUDGETS:
        if budget > stage:
            break
        features.extend(f"{name}_{budget}" for name in STATE_FEATURES)
    return features


def make_query_disjoint_split(frame: pd.DataFrame, seed: int = 20261015) -> dict:
    """Deterministic stratified split, reserving at least one rare-class item in val/test."""
    if frame["query_id"].astype(str).duplicated().any():
        raise ValueError("Expected exactly one row per query_id before splitting")
    if frame[LABEL_COLUMN].isna().any():
        raise ValueError("Every query needs a repeated-latency oracle label")

    rng = random.Random(seed)
    buckets = {"train": [], "validation": [], "test": []}
    labels = frame[LABEL_COLUMN].astype(int)
    for label in sorted(labels.unique()):
        ids = frame.loc[labels == label, "query_id"].astype(str).tolist()
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            raise ValueError(f"Label {label} has only {n} examples; cannot reserve train/validation/test")
        n_val = max(1, round(n * 0.10))
        n_test = max(1, round(n * 0.10))
        if n_val + n_test >= n:
            n_val = n_test = 1
        buckets["validation"].extend(ids[:n_val])
        buckets["test"].extend(ids[n_val:n_val + n_test])
        buckets["train"].extend(ids[n_val + n_test:])

    for split_ids in buckets.values():
        rng.shuffle(split_ids)
    sets = {name: set(ids) for name, ids in buckets.items()}
    if (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
            or sets["validation"] & sets["test"]):
        raise AssertionError("Query IDs overlap across splits")
    if set.union(*sets.values()) != set(frame["query_id"].astype(str)):
        raise AssertionError("Split does not cover every query exactly once")

    label_by_id = dict(zip(frame["query_id"].astype(str), labels))
    return {
        "seed": seed,
        "label_column": LABEL_COLUMN,
        "method": "stratified_by_oracle_label_with_minimum_one_validation_and_test_per_class",
        "query_ids": buckets,
        "counts": {name: len(ids) for name, ids in buckets.items()},
        "label_counts": {
            name: {str(label): count for label, count in sorted(Counter(label_by_id[qid] for qid in ids).items())}
            for name, ids in buckets.items()
        },
        "pairwise_overlap": {"train_validation": 0, "train_test": 0, "validation_test": 0},
    }


def numeric_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = frame.reindex(columns=columns).copy()
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.replace([np.inf, -np.inf], np.nan)


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

