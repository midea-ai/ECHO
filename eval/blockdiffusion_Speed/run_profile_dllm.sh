#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set your Python executable and model path here
PYTHON_EXEC="python3"
MODEL_DIR="your_dllm_model_dir"   # path to your block-diffusion LLM checkpoint
INPUT_JSON="$SCRIPT_DIR/minic_100.json"
SCRIPT="$SCRIPT_DIR/profile_dllm.py"

echo "Profiling DLLM..."
$PYTHON_EXEC $SCRIPT \
    --model_dir $MODEL_DIR \
    --input_json $INPUT_JSON \
    --out_dir "$SCRIPT_DIR/output/dllm" \
    --remasking_strategy low_confidence_dynamic \
    --block_length 8 \
    --denoising_steps 1 \
    --max_samples 100 \
    --fused_decode \
    --temperature 0.0
