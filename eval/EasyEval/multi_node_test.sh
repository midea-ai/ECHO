#!/usr/bin/env bash
# export NLTK_DATA=/workspace/nltk_data
INPUT_JSON_LIST=(
    "CXRTest_demo/CXRTest/jsons/RexGradient-160k_test_EN_for_infer_normalize.json"
    "CXRTest_demo/CXRTest/jsons/RexGradient-160k_test_ZH_for_infer_normalize.json"
    "CXRTest_demo/CXRTest/jsons/ChexPertPlus_test_EN_for_infer_normalize.json"
    "CXRTest_demo/CXRTest/jsons/ChexPertPlus_test_ZH_for_infer_normalize.json"
    "CXRTest_demo/CXRTest/jsons/mimic_test_EN_for_infer_normalize.json"
    "CXRTest_demo/CXRTest/jsons/mimic_test_ZH_for_infer_normalize.json"
    # Add more JSON paths here...
)
MODEL_PATH="Lingshu-32B"
OUTPUT_DIR="./outputs_lingshu_32B"

cd ./EasyEval
# Loop over each JSON file
total_files=${#INPUT_JSON_LIST[@]}

for ((i=0; i<total_files; i++)); do
    # Task assignment via modulo: this node processes a file only when
    # (file_index % num_nodes) == current_node_rank
    INPUT_JSON_PATH="${INPUT_JSON_LIST[i]}"
    
    echo "========================================"
    echo "[Node ${NODE_RANK}] Processing ($((i+1))/${total_files}): $INPUT_JSON_PATH"
    echo "========================================"
    
    # Derive output path from INPUT_JSON_PATH
    INPUT_BASENAME=$(basename "$INPUT_JSON_PATH" .json)
    RESULT_JSON_PATH="$OUTPUT_DIR/${INPUT_BASENAME}_results/"
    RESULT_JSON_PATH_MERGED="$OUTPUT_DIR/${INPUT_BASENAME}_results_merged.json"
    RESULT_JSON_PATH_METRICS="$OUTPUT_DIR/${INPUT_BASENAME}_results_metrics.json"
    
    
    # 1. infer
    ### Recommended: 8-GPU machine
    # python qwen_infer_llada.py -i "$INPUT_JSON_PATH" -o "$RESULT_JSON_PATH" --gpu_ids 0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7 --model_path "$MODEL_PATH" --prompt_type no_cot
    # python d_qwen_infer.py -i "$INPUT_JSON_PATH" -o "$RESULT_JSON_PATH" --gpu_ids 0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7 --model_path "$MODEL_PATH" --prompt_type no_cot --block_length 4 --denoising_steps 4 --remasking_strategy low_confidence_dynamic --temperature 0.0
    ### Single H20 GPU
    ### python qwen_infer.py -i "$INPUT_JSON_PATH" -o "$RESULT_JSON_PATH" --gpu_ids 0,0,0,0 --model_path "$MODEL_PATH"

    echo "[Node ${NODE_RANK}] Starting vLLM inference..."
    python qwen_infer_vllm.py -i $INPUT_JSON_PATH -o $RESULT_JSON_PATH --gpu_ids 0,1,2,3,4,5,6,7 --batch_size 512 --model_path $MODEL_PATH --prompt_type no_cot 
    
    # # 2. merge and translation
    echo "[Node ${NODE_RANK}] Running merge and translation..."
    python merge_and_translation.py -i "$RESULT_JSON_PATH" -o "$RESULT_JSON_PATH_MERGED" --engine vllm --tensor_parallel_size 8
    
    # 3. eval
    python evaluation.py -s "$RESULT_JSON_PATH_MERGED" -d "$RESULT_JSON_PATH_METRICS"
    
    echo "[Node ${NODE_RANK}] Finished: $INPUT_JSON_PATH"
    echo ""
done

echo "========================================"
echo "[Node ${NODE_RANK}] All assigned tasks completed."
echo "========================================"
