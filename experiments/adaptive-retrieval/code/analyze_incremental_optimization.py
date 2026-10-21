import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_csv(name):
    path = ROOT / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def number(row, field):
    value = row.get(field)
    return float(value) if value not in (None, "", "nan", "NaN") else None


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def percentile(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, int((len(values) - 1) * p))]


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main():
    status_path = ROOT / "parity_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    new = read_csv("plaid_incremental_optimization_latency.csv")
    old = read_csv("plaid_incremental_runtime.csv")
    parity = read_csv("plaid_incremental_parity.csv")

    new = [r for r in new if r.get("query_group") in ("hard", "easy")]
    old = [r for r in old if r.get("dataset") == "independent"]
    old_stage = {}
    for row in old:
        if not row.get("scenario", "").startswith("Incremental"):
            continue
        key = (row.get("query_id"), row.get("budget"))
        old_stage.setdefault(key, []).append(number(row, "score_filter_refine_sort_ms"))

    lines = [
        "# Incremental search optimization PoC report",
        "",
        "## Scope",
        "",
        "Opt-in `score_pids_incremental()` only. Default `score_pids()` behavior was not changed. No index build or full benchmark was run.",
        f"Parity launcher status: **{status.get('status', 'missing')}**. Hardware: {status.get('gpu', 'see runtime log')}; backend: PyTorch {status.get('torch', 'unknown')}.",
        "The 6-query parity gate covers 3 hard + 3 easy queries at 400/512/1024. The 9 known hard queries are separately checked at all three budgets; 3 overlap with the parity set, so there are 12 unique query IDs total.",
        "",
        "## Implementation changes",
        "",
        "- Candidate PIDs, query embeddings, and centroid-score matrix are retained by reference as immutable inputs instead of cloning the large tensors.",
        "- Stage-1 approximate scores, stage-2 approximate scores, exact MaxSim scalar scores, and validity are contiguous tensors aligned by candidate row. This removes per-stage PID `.cpu().tolist()`, Python dict lookups, per-scalar insertion, and full-prefix `torch.stack` reconstruction.",
        "- The opt-in method returns a maintained sorted Top100. On nested refine sets, it merges the previous Top100 with only newly scored PIDs. If the set is non-nested or a score tie in the Top101 could change the legacy sort order, it falls back to the exact full sort.",
        "- Inner `torch.cuda.synchronize()` calls were removed. Wall-clock stage timings in the validation runner synchronize at the outer stage boundary. Internal per-component values are launch-time diagnostics and should not be treated as completed GPU execution times.",
        "- The legacy `score_pids()` implementation was not edited by this optimization; the isolated source worktree still carries the already-existing opt-in PoC changes only.",
        "",
        "## Parity results",
        "",
    ]
    parity_ok = bool(parity) and all(r.get("parity_pass", "").lower() == "true" for r in parity)
    hard_rows = [r for r in new if r.get("query_group") == "hard"]
    checked = [r for r in new if r.get("stage_parity_pass", "").lower() in ("true", "false")]
    checked_ok = bool(checked) and all(
        r.get("stage_parity_pass", "").lower() == "true"
        and r.get("top100_order_equal", "").lower() == "true"
        and number(r, "top100_overlap") == 100
        and (number(r, "top100_score_max_abs_diff") or 0.0) <= 1e-5
        for r in checked
    )
    lines.append(f"- Six-query gate: {'PASS' if parity_ok else 'FAIL / incomplete'} ({len(parity)} query-budget rows).")
    lines.append(f"- Hard-query extension: {'PASS' if checked_ok and len(hard_rows) == 9 * 3 else 'FAIL / incomplete'} ({len(hard_rows)} hard-query-budget rows; expected 27).")
    if checked:
        overlaps = [number(r, "top100_overlap") for r in checked]
        diffs = [number(r, "top100_score_max_abs_diff") for r in checked]
        lines.append(f"- Across all checked rows: minimum Top100 overlap **{min(overlaps):.0f}/100**; maximum score difference **{max(diffs):.8g}**.")
    lines += ["", "## Stage latency (current synchronized run)", "",
              "Prior-run timing is not a valid optimization baseline because its timing protocol differs. The current run is one pass per query; query 50 is a cold-start encode outlier (2.84 s), so steady-state summaries exclude it (n=11). `full stage` includes score/filter/refine plus Top100.",
              "", "| Budget/stage | Incremental stage mean (ms) | Full single-call stage mean (ms) |", "|---:|---:|---:|"]
    stage_summaries = []
    for budget in (400, 512, 1024):
        new_rows = [r for r in new if int(r["budget"]) == budget]
        steady = [r for r in new_rows if r.get("query_id") != "50"]
        opt = [number(r, "incremental_stage_wall_ms") for r in steady]
        full = [number(r, "full_score_filter_refine_top100_ms") for r in steady]
        opt_mean, full_mean = mean(opt), mean(full)
        stage_summaries.append((budget, opt_mean, full_mean))
        lines.append(f"| {budget}" + (" continuation" if budget > 400 else "") + f" | {fmt(opt_mean)} | {fmt(full_mean)} |")

    lines += ["", "The incremental 400→512→1024 stages sum to **{:.2f} ms/query**, versus **{:.2f} ms/query** for one full1024 stage (about **{:.2f}×**).".format(sum(x[1] for x in stage_summaries), next(x[2] for x in stage_summaries if x[0] == 1024), sum(x[1] for x in stage_summaries) / next(x[2] for x in stage_summaries if x[0] == 1024)), "", "## End-to-end estimate", "",
              "Shared query encoding and candidate generation are counted once for both paths; cold-start query 50 is excluded."]
    new_qids = {r.get("query_id") for r in new if r.get("budget") == "1024"}
    prior_inc_qids = {r.get("query_id") for r in old if r.get("scenario") == "Incremental1024"}
    matched_qids = new_qids & prior_inc_qids
    prior_inc_final = []
    prior_full_final = []
    for row in old:
        if row.get("scenario") == "Incremental1024":
            if row.get("query_id") in matched_qids:
                prior_inc_final.append(number(row, "end_to_end_ms"))
        if row.get("scenario") == "Full1024":
            if row.get("query_id") in matched_qids:
                prior_full_final.append(number(row, "end_to_end_ms"))
    new_inc_e2e = [number(r, "incremental_cumulative_end_to_end_ms") for r in new if int(r["budget"]) == 1024 and r.get("query_id") in matched_qids]
    new_full_e2e = [number(r, "full_end_to_end_ms") for r in new if int(r["budget"]) == 1024 and r.get("query_id") in matched_qids]
    steady1024 = [r for r in new if int(r["budget"]) == 1024 and r.get("query_id") != "50"]
    enc = mean([number(r, "query_encode_ms") for r in steady1024])
    ret = mean([number(r, "candidate_generation_ms") for r in steady1024])
    inc_stage_total = sum(x[1] for x in stage_summaries)
    full_stage = next(x[2] for x in stage_summaries if x[0] == 1024)
    lines += [f"Steady-state estimated end-to-end: incremental **{fmt(enc + ret + inc_stage_total)} ms**; full1024 **{fmt(enc + ret + full_stage)} ms**. These one-pass values are directional only. No valid before-vs-after comparison is available because the previous run used a different timing protocol.", ""]

    lines += ["## Interpretation / go-no-go", "",
              "The implementation now avoids large state clones, PID-indexed Python caches, repeated full-score tensor construction, and redundant full sorting when the TopK merge is safe; unsafe tie/non-nested cases retain exact behavior via full-sort fallback. Exact parity is a hard gate, not a performance tradeoff.",
              "",
              "Parity passes, but the current 400→512→1024 schedule is substantially slower than one full1024 call. The previous-run delta is invalid because synchronization/warmup protocols differ. Do not expand the benchmark yet; first reduce cross-budget repeated work or reconsider staged execution. Any next performance comparison needs warm-up, matched order, repeats, and synchronized timing for both paths.",
              "",
              "## Outputs", "",
              "- `plaid_incremental_parity.csv`: six-query exact Top100 parity gate.",
              "- `plaid_fullrun_stage_parity.csv`: stage-level state and score parity for the six-query gate.",
              "- `plaid_incremental_relevant_pid_trace.csv`: relevant-PID/refine/top100 trace for the nine hard queries.",
              "- `plaid_incremental_optimization_latency.csv`: synchronized per-stage and end-to-end measurements for the six-query gate plus nine hard queries.",
              "- `parity_status.json`: scope and overall gate result.", ""]
    (ROOT / "incremental_optimization_poc_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {ROOT / 'incremental_optimization_poc_report.md'}")


if __name__ == "__main__":
    main()
