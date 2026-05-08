#!/usr/bin/env bash
# Generate all QA pairs for INVITA tasks
# Usage: bash generate_all_qa.sh

set -euo pipefail

PYTHON="${PYTHON:-python3}"
SCRIPT="${SCRIPT:-src/data_processing/qa_generation/qa_generator.py}"
QA_OUTPUT_DIR="${QA_OUTPUT_DIR:-${INVITA_QA_OUTPUT_ROOT:-outputs/qa_generation/generated_qa}}"

echo "========================================"
echo "INVITA QA Pair Generation"
echo "========================================"
echo "Start time: $(date)"
echo ""

"$PYTHON" "$SCRIPT" --output-dir "$QA_OUTPUT_DIR"

echo ""
echo "========================================"
echo "QA Generation Complete!"
echo "End time: $(date)"
echo "========================================"
