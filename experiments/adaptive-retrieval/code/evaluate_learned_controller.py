"""Evaluate learned budget choices offline against fixed and oracle policies."""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="controller_dataset.parquet")
    parser.add_argument("--model", default="models/budget_controller_lgbm.pkl")
    parser.add_argument("--output", default="comparison_results.csv")
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    frame = pd.read_parquet(args.dataset)
    with Path(args.model).open("rb") as handle:
        bundle = pickle.load(handle)
    features = bundle["feature_columns"]
    _, test = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
        frame[frame["oracle_budget_098"].notna()], test_size=0.2, random_state=args.seed,
        stratify=frame[frame["oracle_budget_098"].notna()]["oracle_budget_098"])
    # The second split reproduces the 10% test partition used by training.
    _, test = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
        test, test_size=0.5, random_state=args.seed, stratify=test["oracle_budget_098"])
    pred = bundle["model"].predict(test[features]).astype(int).ravel()
    rows = []
    ref_quality = test["recall_at_100_1024"].astype(float).median()
    ref_latency = test["latency_1024"].median()
    for b in (400, 512, 768, 1024):
        quality = test[f"recall_at_100_{b}"].astype(float).median()
        rows.append({"method": f"fixed{b}", "recall_at_100": quality,
                     "median_latency_ms": test[f"latency_{b}"].median(),
                     "latency_saving_vs_fixed1024_ms": ref_latency - test[f"latency_{b}"].median(),
                     "quality_loss_vs_fixed1024": ref_quality - quality})
    chosen = pd.Series(pred, index=test.index)
    learned_lat = [test.loc[idx, f"latency_{int(b)}"] for idx, b in chosen.items()]
    oracle = test["oracle_budget_098"].astype(int)
    oracle_lat = [test.loc[idx, f"latency_{int(b)}"] for idx, b in oracle.items()]
    learned_quality = [test.loc[idx, f"recall_at_100_{int(b)}"] for idx, b in chosen.items()]
    oracle_quality = [test.loc[idx, f"recall_at_100_{int(b)}"] for idx, b in oracle.items()]
    rows += [{"method": "oracle_adaptive", "recall_at_100": pd.Series(oracle_quality).median(),
              "median_latency_ms": pd.Series(oracle_lat).median(),
              "latency_saving_vs_fixed1024_ms": ref_latency - pd.Series(oracle_lat).median(),
              "quality_loss_vs_fixed1024": ref_quality - pd.Series(oracle_quality).median()},
             {"method": "learned_adaptive", "recall_at_100": pd.Series(learned_quality).median(),
              "median_latency_ms": pd.Series(learned_lat).median(),
              "latency_saving_vs_fixed1024_ms": ref_latency - pd.Series(learned_lat).median(),
              "quality_loss_vs_fixed1024": ref_quality - pd.Series(learned_quality).median()}]
    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
