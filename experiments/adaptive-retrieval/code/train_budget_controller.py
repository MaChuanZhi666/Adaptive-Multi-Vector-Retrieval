"""Train the first no-leak LightGBM budget classifier."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="controller_dataset.parquet")
    parser.add_argument("--model", default="models/budget_controller_lgbm.pkl")
    parser.add_argument("--metrics", default="controller_metrics.json")
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required: python -m pip install --user lightgbm") from exc
    frame = pd.read_parquet(args.dataset)
    forbidden = {"oracle_budget_095", "oracle_budget_098"}
    forbidden_prefixes = ("latency_", "recall_at_", "mrr_", "ndcg_")
    feature_cols = [c for c in frame.columns if c not in forbidden and c != "query_id" and not c.startswith(forbidden_prefixes)]
    usable = frame[frame["oracle_budget_098"].notna()].copy()
    X = usable[feature_cols].copy()
    y = usable["oracle_budget_098"].astype(int)
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(float)
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    x_train, x_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=args.seed, stratify=y_tmp)
    model = lgb.LGBMClassifier(objective="multiclass", num_class=4, n_estimators=200,
                               learning_rate=0.05, num_leaves=15, random_state=args.seed,
                               verbosity=-1)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])
    pred = model.predict(x_test).astype(int).ravel()
    metrics = {"feature_columns": feature_cols, "seed": args.seed,
               "train_size": len(x_train), "validation_size": len(x_val), "test_size": len(x_test),
               "accuracy": float(accuracy_score(y_test, pred)),
               "macro_f1": float(f1_score(y_test, pred, average="macro")),
               "confusion_matrix": confusion_matrix(y_test, pred, labels=[400, 512, 768, 1024]).tolist()}
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.model).open("wb") as handle:
        pickle.dump({"model": model, "feature_columns": feature_cols, "seed": args.seed}, handle)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
