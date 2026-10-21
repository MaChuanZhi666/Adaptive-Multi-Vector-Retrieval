"""Repeat fixed-budget CPU retrieval measurements without changing retrieval code."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import tempfile
from pathlib import Path

from profiling_schema import BUDGETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/profile_config.yaml")
    parser.add_argument("--profile", default="budget_profile_full.jsonl")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", default="latency_repeat_results.jsonl")
    parser.add_argument("--summary", default="latency_summary.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest", help="locked sampling manifest; use its repeat query IDs verbatim")
    args = parser.parse_args()
    os.environ["PLAID_DEVICE"] = "cpu"
    from budget_profile_runner import load_config
    import server_two_stage_runtime as runtime

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    qas = runtime.queries(Path(dataset["query_path"]))
    profile_ids = [str(json.loads(line)["query_id"]) for line in Path(args.profile).open(encoding="utf-8") if line.strip()]
    available = [int(qid) for qid in profile_ids if int(qid) in qas]
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        repeat_plan = manifest.get("latency_repeat_plan", {})
        planned = [int(qid) for qid in repeat_plan.get("query_ids", [])]
        if not planned:
            parser.error("manifest has no latency_repeat_plan.query_ids")
        missing_profile = sorted(set(planned) - set(available))
        if missing_profile:
            parser.error(f"{len(missing_profile)} planned IDs are absent from the profile; first={missing_profile[:10]}")
        selected = sorted(planned)
        if len(selected) != len(set(selected)):
            parser.error("manifest contains duplicate latency-repeat query IDs")
        print(json.dumps({"sampling_manifest": str(args.manifest),
                          "sampling_rule": repeat_plan.get("rule"),
                          "selected_query_count": len(selected)}, ensure_ascii=False), flush=True)
    else:
        rng = random.Random(args.seed)
        selected = sorted(rng.sample(available, min(args.sample_size, len(available))))
    searcher = runtime.create_searcher(Path(dataset["index_path"]), Path(dataset["collection_path"]))
    if selected:
        runtime.warmup(searcher, qas[selected[0]]["text"])

    output_path = Path(args.output)
    existing = {}
    if args.resume and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing[(str(row["query_id"]), int(row["budget"]))] = row
    rows = list(existing.values())
    for qid in selected:
        info = qas[qid]
        # Rotate budget order each repetition to reduce systematic order/cache bias.
        for repeat in range(args.repeats):
            order = list(BUDGETS)
            random.Random(args.seed + repeat).shuffle(order)
            for budget in order:
                key = (str(qid), int(budget))
                row = existing.get(key, {"query_id": str(qid), "budget": int(budget), "runs": []})
                runs = row["runs"]
                if len(runs) > repeat:
                    continue
                result = runtime.run_full(searcher, info["text"], info["positives"],
                                          dataset.get("name", "profile"), qid, "repeat", repeat, budget)
                runs.append(float(result["total_retrieval_latency_ms"]))
                print(json.dumps({"query_id": str(qid), "budget": budget,
                                  "repeat": repeat, "latency_ms": runs[-1]}, ensure_ascii=False), flush=True)
                row.update({"mean_latency": statistics.mean(runs),
                            "median_latency": statistics.median(runs),
                            "std_latency": statistics.stdev(runs) if len(runs) > 1 else 0.0})
                existing[key] = row
                rows = list(existing.values())
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fd, temp_name = tempfile.mkstemp(prefix=output_path.name, suffix=".tmp", dir=output_path.parent)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    for saved in rows:
                        handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
                os.replace(temp_name, output_path)
    summary = []
    for budget in BUDGETS:
        vals = [v for row in rows if row["budget"] == budget for v in row["runs"]]
        summary.append({"budget": budget, "mean": statistics.mean(vals) if vals else None,
                        "median": statistics.median(vals) if vals else None,
                        "p50": statistics.median(vals) if vals else None,
                        "p95": (statistics.quantiles(vals, n=20)[18] if len(vals) >= 2 else (vals[0] if vals else None)),
                        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0})
    with Path(args.summary).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["budget", "mean", "median", "p50", "p95", "std"])
        writer.writeheader(); writer.writerows(summary)
    print(json.dumps({"query_count": len(selected), "repeats": args.repeats,
                      "output": args.output, "summary": args.summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
