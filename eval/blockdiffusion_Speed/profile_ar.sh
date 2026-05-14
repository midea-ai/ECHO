#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set your Python executable and model path here
PYTHON_EXEC="python3"
MODEL_DIR="your_ar_model_dir"   # e.g. Qwen/Qwen2.5-VL-7B-Instruct
INPUT_JSON="$SCRIPT_DIR/minic_100.json"
SCRIPT="$SCRIPT_DIR/profile_ar.py"

echo "Profiling AR Baseline..."
$PYTHON_EXEC $SCRIPT \
    --model_dir $MODEL_DIR \
    --input_json $INPUT_JSON \
    --out_dir "$SCRIPT_DIR/output/ar" \
    --max_samples 100 \
    --temperature 0.0
