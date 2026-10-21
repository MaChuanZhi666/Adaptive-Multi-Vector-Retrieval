"""Train one leakage-safe continue/stop LightGBM model per retrieval stage."""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from stagewise_controller_utils import (
    LABEL_COLUMN, STAGES, eligible_features, make_query_disjoint_split,
    numeric_feature_frame, read_dataset, write_json,
)


def binary_metrics(target, probabilities):
    truth = np.asarray(target, dtype=int)
    pred = (np.asarray(probabilities) >= 0.5).astype(int)
    tn = int(((truth == 0) & (pred == 0)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    tp = int(((truth == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": float((pred == truth).mean()),
        "continue_f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "continue_recall": float(recall),
        "confusion_matrix_stop_continue": [[tn, fp], [fn, tp]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="controller_dataset_repeated_full.csv")
    parser.add_argument("--model", default="models/stagewise_budget_controller_lgbm.pkl")
    parser.add_argument("--metrics", default="stagewise_controller_metrics.json")
    parser.add_argument("--split-manifest", default="models/budget_split_manifest.json")
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required: python -m pip install --user lightgbm") from exc

    frame = read_dataset(args.dataset)
    frame["query_id"] = frame["query_id"].astype(str)
    frame[LABEL_COLUMN] = pd.to_numeric(frame[LABEL_COLUMN], errors="raise").astype(int)
    split = make_query_disjoint_split(frame, args.seed)
    write_json(args.split_manifest, split)
    index = frame.set_index("query_id", drop=False)
    models = {}
    stage_metrics = {}
    feature_manifest = {}

    for stage in STAGES:
        candidates = eligible_features(stage)
        train = index.loc[split["query_ids"]["train"]]
        validation = index.loc[split["query_ids"]["validation"]]
        target_train = (train[LABEL_COLUMN] > stage).astype(int)
        target_val = (validation[LABEL_COLUMN] > stage).astype(int)

        x_train_all = numeric_feature_frame(train, candidates)
        # Drop columns unusable in training only; no test-set inspection is involved.
        feature_cols = [c for c in candidates if x_train_all[c].notna().any() and x_train_all[c].nunique(dropna=True) > 1]
        if not feature_cols:
            raise RuntimeError(f"No usable decision-time features at stage {stage}")
        x_train = x_train_all[feature_cols]
        x_val = numeric_feature_frame(validation, feature_cols)
        positive = int(target_train.sum())
        negative = int(len(target_train) - positive)
        if positive == 0 or negative == 0:
            raise RuntimeError(f"Stage {stage} has a single-class training target")

        train_data = lgb.Dataset(x_train, label=target_train)
        val_data = lgb.Dataset(x_val, label=target_val, reference=train_data)
        model = lgb.train(
            {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.04,
             "max_depth": 3, "num_leaves": 7, "min_data_in_leaf": 8,
             "lambda_l2": 2.0, "scale_pos_weight": negative / positive,
             "verbosity": -1, "seed": args.seed + stage, "num_threads": 1},
            train_data, num_boost_round=80, valid_sets=[val_data],
        )
        train_scores = model.predict(x_train)
        val_scores = model.predict(x_val)
        models[stage] = {"model": model, "feature_columns": feature_cols,
                         "threshold": 0.5, "positive_label": f"oracle_budget > {stage}"}
        feature_manifest[str(stage)] = {
            "decision_time": f"after completing budget {stage}",
            "features": feature_cols,
            "maximum_observed_stage": stage,
            "contains_future_stage_feature": any(
                any(column.endswith(f"_{future}") for future in (400, 512, 768, 1024) if future > stage)
                for column in feature_cols
            ),
            "dropped_all_null_or_constant": [c for c in candidates if c not in feature_cols],
        }
        stage_metrics[str(stage)] = {
            "train_stop_continue_counts": {"stop": negative, "continue": positive},
            "validation_stop_continue_counts": {
                "stop": int((target_val == 0).sum()), "continue": int((target_val == 1).sum())},
            **{f"train_{key}": value for key, value in binary_metrics(target_train, train_scores).items()},
            **{f"validation_{key}": value for key, value in binary_metrics(target_val, val_scores).items()},
        }

    bundle = {"models": models, "budgets": (400, 512, 768, 1024),
              "oracle_label": LABEL_COLUMN, "threshold": 0.5, "seed": args.seed,
              "feature_manifest": feature_manifest, "split_manifest": str(args.split_manifest)}
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.model).open("wb") as handle:
        pickle.dump(bundle, handle)
    report = {
        "dataset": str(args.dataset), "query_count": len(frame),
        "label_distribution": {str(k): int(v) for k, v in sorted(Counter(frame[LABEL_COLUMN]).items())},
        "split_counts": split["counts"], "split_label_counts": split["label_counts"],
        "split_pairwise_overlap": split["pairwise_overlap"],
        "feature_manifest": feature_manifest, "stage_metrics": stage_metrics,
        "model_path": args.model, "split_manifest_path": args.split_manifest,
        "warning": "The 768->1024 continue model has only four positive training queries and one positive test query; treat its estimates as exploratory.",
    }
    write_json(args.metrics, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

