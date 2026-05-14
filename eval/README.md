## ENV Installation

```bash
###java envs
apt-get update
apt-get install openjdk-17-jre -y
conda create -n eval_cxr python=3.10 -y
conda activate eval_cxr
pip install -r requirements.txt
git clone https://github.com/michelecafagna26/cider
cd ./cider
pip install -e .
```

## Evaluation

### `EasyEval/multi_node_test.sh`

Runs **inference → merge/translation → evaluation** in batch: walks each JSON in `INPUT_JSON_LIST` in order (paths such as `CXRTest/jsons/...` must exist).

1. Edit `OUTPUT_DIR`, `INPUT_JSON_LIST`, `MODEL_PATH`, and the inference command as needed (defaults to `qwen_infer_vllm.py` with 8 GPUs).
2. Run from the **repository root** (the script `cd`s into `./EasyEval`):

```bash
bash EasyEval/multi_node_test.sh
```

For multi-node jobs that shard with `NODE_RANK`, let your scheduler inject that variable; on one machine it processes every entry in the list sequentially.

### `blockdiffusion_Speed/run_profile_dllm.sh` / `profile_ar.sh`

Performance/profiling comparison between **DLLM (block diffusion)** and an **AR baseline**; each script `cd`s into `blockdiffusion_Speed`.

1. In the chosen `.sh`, set `PYTHON_EXEC` and `MODEL_DIR` (your checkpoint or Hugging Face model name). Adjust `INPUT_JSON` and `profile_*.py` args if needed.
2. Run from the **repository root**:

```bash
bash blockdiffusion_Speed/run_profile_dllm.sh
bash blockdiffusion_Speed/profile_ar.sh
```

Results are written by default to `blockdiffusion_Speed/output/dllm` and `blockdiffusion_Speed/output/ar` respectively.
