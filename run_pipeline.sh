#!/usr/bin/env bash
# Run the full fine-tuning pipeline end-to-end.
# Usage: bash run_pipeline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/ccg_env"
PYTHON="$VENV/bin/python3.12"

# ── 1. Activate venv ──────────────────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
  echo "ERROR: venv not found at $VENV"
  echo "Run: /opt/homebrew/bin/python3.12 -m venv ccg_env && ccg_env/bin/pip install -r requirements.txt"
  exit 1
fi

echo "Using Python: $($PYTHON --version)"

# ── 2. Run steps ──────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

echo ""; echo "▶ Step 0: Data preparation"
"$PYTHON" scripts/0_data_prep.py

echo ""; echo "▶ Step 1: Baseline benchmark"
"$PYTHON" scripts/1_baseline_benchmark.py

echo ""; echo "▶ Step 2: Fine-tuning"
"$PYTHON" scripts/2_finetune.py

echo ""; echo "▶ Step 3: Fine-tuned evaluation"
"$PYTHON" scripts/3_evaluate_finetuned.py

echo ""; echo "▶ Step 4: Comparative analysis"
"$PYTHON" scripts/4_comparative_analysis.py

echo ""
echo "✓ Pipeline complete. Results in benchmarks/"
