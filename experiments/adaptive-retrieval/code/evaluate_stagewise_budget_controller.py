"""Evaluate the staged controller and fixed/oracle baselines on held-out queries."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stagewise_controller_utils import LABEL_COLUMN, STAGES, numeric_feature_frame, read_dataset


def summarize(method: str, budgets: list[int], frame: pd.DataFrame, overhead_ms=None) -> dict:
    recalls = [float(frame.loc[qid, f"recall_at_100_{budget}"]) for qid, budget in zip(frame.index, budgets)]
    latencies = [float(frame.loc[qid, f"latency_{budget}"]) for qid, budget in zip(frame.index, budgets)]
    overhead_ms = overhead_ms or [0.0] * len(latencies)
    total = [latency + overhead for latency, overhead in zip(latencies, overhead_ms)]
    ref_quality = frame["recall_at_100_1024"].astype(float).to_numpy()
    ref_latency = frame["latency_1024"].astype(float).to_numpy()
    return {
        "method": method, "query_count": len(frame),
        "mean_recall_at_100": statistics.mean(recalls),
        "median_recall_at_100": statistics.median(recalls),
        "mean_latency_ms_including_controller": statistics.mean(total),
        "median_latency_ms_including_controller": statistics.median(total),
        "p95_latency_ms_including_controller": float(np.percentile(total, 95)),
        "mean_controller_overhead_ms": statistics.mean(overhead_ms),
        "mean_latency_saving_vs_fixed1024_ms": statistics.mean(ref_latency) - statistics.mean(total),
        "median_latency_saving_vs_fixed1024_ms": statistics.median(ref_latency) - statistics.median(total),
        "mean_quality_loss_vs_fixed1024": statistics.mean(ref_quality) - statistics.mean(recalls),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="controller_dataset_repeated_full.csv")
    parser.add_argument("--model", default="models/stagewise_budget_controller_lgbm.pkl")
    parser.add_argument("--split-manifest", default="models/budget_split_manifest.json")
    parser.add_argument("--output", default="stagewise_comparison_results.csv")
    parser.add_argument("--predictions", default="stagewise_test_predictions.csv")
    args = parser.parse_args()

    frame = read_dataset(args.dataset)
    frame["query_id"] = frame["query_id"].astype(str)
    frame = frame.set_index("query_id", drop=False)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    test_ids = [str(qid) for qid in manifest["query_ids"]["test"]]
    test = frame.loc[test_ids].copy()
    with Path(args.model).open("rb") as handle:
        bundle = pickle.load(handle)

    results = []
    for budget in (400, 512, 768, 1024):
        results.append(summarize(f"fixed{budget}", [budget] * len(test), test))

    oracle_budgets = test[LABEL_COLUMN].astype(int).tolist()
    results.append(summarize("oracle_adaptive", oracle_budgets, test))

    selected = []
    overheads = []
    trace_rows = []
    for qid, row in test.iterrows():
        selected_budget = 1024
        total_predict_ms = 0.0
        decisions = {}
        for stage in STAGES:
            stage_bundle = bundle["models"][stage]
            x = numeric_feature_frame(pd.DataFrame([row]), stage_bundle["feature_columns"])
            start = time.perf_counter()
            continue_search = int(float(stage_bundle["model"].predict(x)[0]) >= stage_bundle["threshold"])
            total_predict_ms += (time.perf_counter() - start) * 1000.0
            decisions[str(stage)] = "continue" if continue_search else "stop"
            if not continue_search:
                selected_budget = stage
                break
        selected.append(selected_budget)
        overheads.append(total_predict_ms)
        trace_rows.append({"query_id": qid, "predicted_budget": selected_budget,
                           "oracle_budget_098": int(row[LABEL_COLUMN]),
                           "controller_overhead_ms": total_predict_ms,
                           **{f"decision_after_{stage}": decisions.get(str(stage), "not_reached") for stage in STAGES}})
    results.append(summarize("learned_adaptive", selected, test, overheads))

    fields = list(results[0])
    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)
    with Path(args.predictions).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader(); writer.writerows(trace_rows)
    print(json.dumps({"test_queries": len(test), "test_query_ids_from_manifest": True,
                      "results": results, "output": args.output,
                      "predictions": args.predictions}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
