"""Build one-row-per-query, no-label-leak controller data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_budget_oracle import analyze
from profile_io import read_jsonl
from profiling_schema import BUDGETS


def _repeated_latency(path: str | Path) -> dict[str, dict[int, float]]:
    grouped = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            runs = [float(x) for x in row.get("runs", [])]
            if len(runs) < 5:
                continue
            grouped.setdefault(str(row["query_id"]), {})[int(row["budget"])] = float(row.get("median_latency", sorted(runs)[len(runs)//2]))
    return {qid: values for qid, values in grouped.items() if set(values) == set(BUDGETS)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="budget_profile_full.jsonl")
    parser.add_argument("--repeat-results", default="latency_repeat_results.jsonl")
    parser.add_argument("--output", default="controller_dataset.parquet")
    parser.add_argument("--manifest", help="pre-registered train/validation and latency-repeat manifest")
    args = parser.parse_args()
    profiles = read_jsonl(args.profile)
    repeated = _repeated_latency(args.repeat_results)
    split_by_id = {}
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        for split in ("train", "validation"):
            for query_id in manifest.get(f"{split}_query_ids", []):
                query_id = str(query_id)
                if query_id in split_by_id:
                    raise ValueError(f"query_id appears in multiple splits: {query_id}")
                split_by_id[query_id] = split
        planned_repeats = {str(qid) for qid in manifest.get("latency_repeat_plan", {}).get("query_ids", [])}
        if not planned_repeats:
            raise ValueError("manifest is missing latency_repeat_plan.query_ids")
        non_train_repeats = sorted(qid for qid in planned_repeats if split_by_id.get(qid) != "train")
        if non_train_repeats:
            raise ValueError(f"repeat sample contains non-training queries: {non_train_repeats[:10]}")
        observed_ids = {profile.query_id for profile in profiles}
        if observed_ids != set(split_by_id):
            raise ValueError("profile query IDs do not exactly match the locked train/validation manifest")
        observed_repeated = set(repeated)
        if observed_repeated != planned_repeats:
            raise ValueError("repeated-latency query IDs do not exactly match the locked manifest")
    oracle_rows = {row["query_id"]: row for row in analyze(profiles, latency_by_query=repeated)}
    rows = []
    for profile in profiles:
        data = profile.to_dict()
        row = {"query_id": profile.query_id,
               "split": split_by_id.get(profile.query_id),
               "query_length": len(profile.query_text.split()) if profile.query_text else None,
               "entity_count": None}
        for budget in BUDGETS:
            item = data["budgets"][str(budget)]
            state = item["search_state"]
            prefix = str(budget)
            for name in ("cutoff_margin", "score_entropy", "score_gain", "tail_slope", "new_candidate_fraction"):
                row[f"{name}_{prefix}"] = state.get(name)
            row[f"recall_at_100_{prefix}"] = item["metrics"].get("recall@100")
            candidate = item["candidate_info"]
            row[f"candidate_generation_failure_{prefix}"] = candidate.get("candidate_generation_failure")
            row[f"truncation_risk_{prefix}"] = None
            row[f"latency_{prefix}"] = repeated.get(profile.query_id, {}).get(budget)
        oracle = oracle_rows.get(profile.query_id, {})
        row["oracle_budget_095"] = oracle.get("oracle_budget_0.95")
        row["oracle_budget_098"] = oracle.get("oracle_budget_0.98")
        rows.append(row)
    frame = pd.DataFrame(rows)
    if Path(args.output).suffix.lower() == ".csv":
        frame.to_csv(args.output, index=False)
    else:
        try:
            frame.to_parquet(args.output, index=False)
        except (ImportError, ValueError) as exc:
            raise RuntimeError("Writing parquet requires pyarrow or fastparquet; use a .csv output or install pyarrow") from exc
    labeled = frame["oracle_budget_098"].notna()
    labels = frame.loc[labeled, "oracle_budget_098"].value_counts().sort_index().to_dict()
    split_counts = frame["split"].value_counts(dropna=False).to_dict() if "split" in frame else {}
    labeled_split_counts = frame.loc[labeled, "split"].value_counts(dropna=False).to_dict() if "split" in frame else {}
    print(json.dumps({"rows": len(frame), "columns": len(frame.columns),
                      "repeated_latency_queries": len(repeated), "labeled_queries": int(labeled.sum()),
                      "oracle_budget_098_distribution": labels,
                      "split_counts": split_counts,
                      "labeled_split_counts": labeled_split_counts,
                      "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
