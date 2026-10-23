#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

echo "============================================================"
echo "FINAL NMT PROJECT — NON-TRAINING PIPELINE"
echo "Project root: $ROOT"
echo "Python      : $($PYTHON --version 2>&1)"
echo "============================================================"

echo
echo "[1/7] Python syntax check"
$PYTHON -m compileall -q     BiLSTM     preprocess     scripts     tokenizer     evaluate     transformer

echo "Syntax check passed."

echo
echo "[2/7] Data/audit scripts"
$PYTHON scripts/audit_raw_data.py
$PYTHON scripts/audit_char_coverage.py
$PYTHON scripts/sequence_representation.py

echo
echo "[3/7] Tokenizer audit"
# tokenizer/tokenizer.py is not run because it retrains the tokenizer models.
$PYTHON tokenizer/tokenizer_audit.py

echo
echo "[4/7] Common baseline evaluation"
# Evaluates BiLSTM + Baseline Transformer.
$PYTHON evaluate/evaluate_nmt.py

echo
echo "[5/7] Refined Transformer beam tuning"
# $PYTHON transformer/evaluate_transformer.py     --mode tune-beam     --run-name refined_transformer     --checkpoint averaged_5_dev_chrf.pt

echo
echo "[6/7] Refined Transformer final test evaluation"
# $PYTHON transformer/evaluate_transformer.py     --mode test     --run-name refined_transformer     --checkpoint averaged_5_dev_chrf.pt

echo
echo "[7/7] Translation smoke test"
# $PYTHON transformer/translate.py     --checkpoint averaged_5_dev_chrf.pt     --method beam     --text "This is a test sentence."

echo
echo "============================================================"
echo "NON-TRAINING PIPELINE COMPLETE"
echo "============================================================"

echo
echo "Not run automatically:"
echo "  BiLSTM/train_bilstm.py"
echo "  transformer/train_transformer.py"
echo "  tokenizer/tokenizer.py"
echo "  transformer/average_checkpoints.py"
echo "  transformer/web_translate.py"
echo "  transformer/qualitative_analysis.py"
