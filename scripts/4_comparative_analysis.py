"""
Comparative analysis: baseline vs. fine-tuned model.
Produces a text report, a CSV table, and a bar-chart PNG.
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = ROOT / "benchmarks"


def load(baseline_file: Path, finetuned_file: Path) -> Tuple[Dict, Dict]:
    with open(baseline_file) as f:
        baseline = json.load(f)
    with open(finetuned_file) as f:
        finetuned = json.load(f)
    return baseline, finetuned


def pct_change(old, new) -> float:
    if old == 0:
        return float("inf")
    return (new - old) / abs(old) * 100


def comparison_table(b: Dict, ft: Dict) -> pd.DataFrame:
    rows = [
        ("BLEU-4", b["avg_bleu_4"], ft["avg_bleu_4"]),
        ("ROUGE-L", b["avg_rouge_l"], ft["avg_rouge_l"]),
        ("Avg gen time (s)", b["avg_generation_time_sec"], ft["avg_generation_time_sec"]),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Baseline", "Fine-Tuned"])
    df["Delta %"] = df.apply(lambda r: f"{pct_change(r['Baseline'], r['Fine-Tuned']):+.1f}%", axis=1)
    return df


def plot(b: Dict, ft: Dict, out_path: Path):
    metrics = ["BLEU-4", "ROUGE-L"]
    base_vals = [b["avg_bleu_4"], b["avg_rouge_l"]]
    ft_vals = [ft["avg_bleu_4"], ft["avg_rouge_l"]]

    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars_b = ax.bar([i - width / 2 for i in x], base_vals, width, label="Baseline", color="#FF6B6B")
    bars_ft = ax.bar([i + width / 2 for i in x], ft_vals, width, label="Fine-Tuned", color="#4ECDC4")

    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_ft:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs. Fine-Tuned: BLEU-4 & ROUGE-L")
    ax.legend()
    ax.set_ylim(0, max(base_vals + ft_vals) * 1.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    logger.info(f"Chart saved → {out_path}")


def text_report(b: Dict, ft: Dict) -> str:
    bleu_delta = pct_change(b["avg_bleu_4"], ft["avg_bleu_4"])
    rouge_delta = pct_change(b["avg_rouge_l"], ft["avg_rouge_l"])
    speed_delta = pct_change(b["avg_generation_time_sec"], ft["avg_generation_time_sec"])

    eg_b = b["examples"][0] if b.get("examples") else {}
    eg_ft = ft["examples"][0] if ft.get("examples") else {}

    return f"""
{'='*70}
CODE COMMENT GENERATION — COMPARATIVE BENCHMARKING REPORT
{'='*70}

PROJECT DETAILS
{'-'*70}
Base Model:             {b['model']}
Fine-Tuned Model:       {ft['model']}
Test Set Size:          {b['num_examples']} examples (same random seed)

PERFORMANCE METRICS
{'-'*70}

BLEU-4:
  Baseline    {b['avg_bleu_4']:.6f}
  Fine-Tuned  {ft['avg_bleu_4']:.6f}
  Change      {bleu_delta:+.1f}%

ROUGE-L:
  Baseline    {b['avg_rouge_l']:.6f}
  Fine-Tuned  {ft['avg_rouge_l']:.6f}
  Change      {rouge_delta:+.1f}%

Generation Speed (avg per example):
  Baseline    {b['avg_generation_time_sec']:.3f}s
  Fine-Tuned  {ft['avg_generation_time_sec']:.3f}s
  Change      {speed_delta:+.1f}%

INTERPRETATION
{'-'*70}

BLEU: {'✓ improved' if bleu_delta > 0 else '✗ regressed'} by {abs(bleu_delta):.1f}%
  Measures n-gram overlap with reference docstrings.

ROUGE-L: {'✓ improved' if rouge_delta > 0 else '✗ regressed'} by {abs(rouge_delta):.1f}%
  Measures longest common subsequence recall against reference.

EXAMPLE OUTPUTS (first example)
{'-'*70}

Baseline:
  Reference  : {eg_b.get('reference', '')[:100]}
  Prediction : {eg_b.get('prediction', '')[:100]}
  BLEU-4     : {eg_b.get('bleu_4', 0):.4f}   ROUGE-L: {eg_b.get('rouge_l', 0):.4f}

Fine-Tuned:
  Reference  : {eg_ft.get('reference', '')[:100]}
  Prediction : {eg_ft.get('prediction', '')[:100]}
  BLEU-4     : {eg_ft.get('bleu_4', 0):.4f}   ROUGE-L: {eg_ft.get('rouge_l', 0):.4f}

{'='*70}
"""


def run():
    baseline_file = BENCHMARK_DIR / "baseline_metrics.json"
    finetuned_file = BENCHMARK_DIR / "finetuned_metrics.json"

    for p in (baseline_file, finetuned_file):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run steps 1 and 3 first.")

    b, ft = load(baseline_file, finetuned_file)

    table = comparison_table(b, ft)
    logger.info("\n" + table.to_string(index=False))
    table.to_csv(BENCHMARK_DIR / "comparison_table.csv", index=False)

    plot(b, ft, BENCHMARK_DIR / "comparison_metrics.png")

    report = text_report(b, ft)
    logger.info(report)
    with open(BENCHMARK_DIR / "comparative_report.txt", "w") as f:
        f.write(report)

    logger.info(f"All outputs saved to {BENCHMARK_DIR}/")


if __name__ == "__main__":
    run()
    print("\n✓ Comparative analysis complete — see benchmarks/")
