"""Run the complete post-profiling oracle and distribution analysis."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_jsonl")
    parser.add_argument("--output-dir", default="analysis_results")
    args = parser.parse_args()
    profile = Path(args.profile_jsonl)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, "analyze_budget_oracle.py", str(profile),
                    "--output", str(out / "oracle_distribution.csv")], check=True)
    subprocess.run([sys.executable, "analyze_budget_distribution.py", str(profile),
                    "--output-dir", str(out)], check=True)

    summary_path = out / "budget_analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    lines = ["# Budget profiling analysis", "", f"- query 数量：{summary.get('query_count')}",
             f"- Oracle 可分析数量：{summary.get('oracle_query_count')}", ""]
    for alpha in ("0.98", "0.95"):
        lines.append(f"## α={alpha}")
        lines.append(f"- Oracle budget 分布：{summary.get('oracle_distribution_' + alpha)}")
        lines.append(f"- 不需要 1024 的比例：{summary.get('not_need_1024_fraction_' + alpha)}")
    lines += ["", f"- 400 达到 1024 的 95% 质量比例：{summary.get('fraction_400_reaches_95pct_of_1024')}",
              f"- α=0.95 时需要超过 512 的比例：{summary.get('fraction_oracle_budget_over_512_alpha_0.95')}",
              f"- 400→1024 质量收益统计：{summary.get('quality_gain_400_to_1024')}",
              f"- query complexity 相关性：{summary.get('query_complexity_correlation')}（{summary.get('query_complexity_correlation_note', '')}）"]
    if summary.get("plot_warning"):
        lines += ["", f"> 图表警告：{summary['plot_warning']}"]
    (out / "budget_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "files": sorted(p.name for p in out.iterdir())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
