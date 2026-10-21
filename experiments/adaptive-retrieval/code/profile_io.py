"""CSV/JSONL utilities for per-query multi-budget profiles."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from feature_extractor import candidate_overlap, topk_churn
from profiling_schema import BUDGETS, BudgetProfile, CandidateInfo, Metrics, QueryBudgetProfile, SearchState


def _float(value: Any):
    return None if value in (None, "", "nan", "NaN") else float(value)


def _int(value: Any):
    return None if value in (None, "") else int(value)


def _bool(value: Any):
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def profile_from_runtime_row(row: Mapping[str, Any]) -> BudgetProfile:
    """Convert an existing runtime row without inferring unavailable values."""
    raw = row.get("approx_features_json", "")
    try:
        features = json.loads(raw) if isinstance(raw, str) and raw else dict(raw or {})
    except (TypeError, ValueError):
        features = {}
    get = lambda key: row.get(key) if row.get(key) not in (None, "") else features.get(key)
    get_first = lambda *keys: next((get(key) for key in keys if get(key) not in (None, "")), None)
    def ids(key):
        value = get(key)
        if value in (None, ""):
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                return None
        return [int(x) for x in value] if value is not None else None
    return BudgetProfile(
        latency_ms=_float(row.get("total_retrieval_latency_ms", row.get("end_to_end_ms"))),
        metrics=Metrics(_float(row.get("recall_at_10")), _float(row.get("recall_at_100")),
                        _float(row.get("mrr_at_10", row.get("mrr"))), _float(row.get("ndcg"))),
        search_state=SearchState(
            _float(get("candidate_overlap_previous")), _float(get("approx_cutoff_margin")),
            _float(get("tail_slope")), None, _float(get("approx_score_entropy")),
            _float(get("approx_score_gain_vs_previous_budget")),
            _float(get_first("stage2_new_candidate_fraction", "newly_admitted_fraction")),
        ),
        candidate_info=CandidateInfo(_int(get_first("candidate_count", "stage1_count")),
                                     _bool(get("candidate_generation_failure")),
                                     ids("candidate_pids_json"), ids("approx_top100_pids_json")),
    )


def csv_to_profiles(csv_path: str | Path) -> List[QueryBudgetProfile]:
    """Convert runtime CSV rows; missing 768 remains a null placeholder."""
    grouped: Dict[str, QueryBudgetProfile] = {}
    selected_rows: Dict[tuple[str, str], Mapping[str, Any]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qid = str(row.get("query_id", ""))
            profile = grouped.setdefault(qid, QueryBudgetProfile(qid, row.get("query_text", "")))
            if row.get("budget", ""):
                budget = str(row["budget"])
                key = (qid, budget)
                previous = selected_rows.get(key)
                # Prefer rows carrying approximate checkpoint features (the
                # TwoStage rows) over duplicate Full rows with an empty JSON.
                def quality(candidate):
                    scenario = str(candidate.get("scenario", ""))
                    feature_blob = str(candidate.get("approx_features_json", ""))
                    return (scenario.startswith("TwoStage"), len(feature_blob), scenario)
                if previous is None or quality(row) > quality(previous):
                    selected_rows[key] = row
                    profile.budgets[budget] = profile_from_runtime_row(row)
    for profile in grouped.values():
        profile.ensure_budgets()
        for old_budget, new_budget in zip(BUDGETS, BUDGETS[1:]):
            old = profile.budgets[str(old_budget)].candidate_info
            new = profile.budgets[str(new_budget)].candidate_info
            profile.transition_features[f"{old_budget}_to_{new_budget}"] = {
                "topk_churn": topk_churn(old.topk_ids, new.topk_ids),
                "candidate_overlap": candidate_overlap(old.candidate_ids, new.candidate_ids),
            }
    return list(grouped.values())


def write_jsonl(profiles: Iterable[QueryBudgetProfile], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for profile in profiles:
            handle.write(json.dumps(profile.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[QueryBudgetProfile]:
    from profiling_schema import profile_from_dict
    with Path(path).open(encoding="utf-8") as handle:
        return [profile_from_dict(json.loads(line)) for line in handle if line.strip()]


def jsonl_to_dataframe(path: str | Path):
    import pandas as pd
    rows = []
    for profile in read_jsonl(path):
        for budget, value in profile.to_dict()["budgets"].items():
            rows.append({"query_id": profile.query_id, "query_text": profile.query_text,
                         "budget": int(budget), "latency_ms": value["latency_ms"],
                         **value["metrics"],
                         **{f"search_{k}": v for k, v in value["search_state"].items()},
                         **{f"candidate_{k}": v for k, v in value["candidate_info"].items()}})
    return pd.DataFrame(rows)
