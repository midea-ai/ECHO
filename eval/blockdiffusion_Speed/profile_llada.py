"""
Profile per-stage latency for LLaDA inference; report format matches profile_dllm.

Reuses LLaDA load/input flow from qwen_infer_llada.py and Profiler layout from profile_dllm.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def judge_language_zh_or_eng(text):
    """Return 'zh' if CJK ratio > 10%, else 'eng'."""
    if not text:
        return "eng"
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return "zh" if (chinese_count / len(text)) > 0.1 else "eng"


# --- Profiler (same pattern as profile_dllm) ---
class Profiler:
    def __init__(self):
        self.stats = {}
        self.start_t = 0

    def start(self):
        torch.cuda.synchronize()
        self.start_t = time.time()

    def record(self, name):
        torch.cuda.synchronize()
        dt = time.time() - self.start_t
        if name not in self.stats:
            self.stats[name] = []
        self.stats[name].append(dt)
        self.start_t = time.time()


@torch.no_grad()
def profile_llada_sample(model, tokenizer, image_processor, sample, device, gen_config, prof, prompt_text):
    """
    Run LLaDA on one sample and record stage timings.

    Stages:
    - image_load: load PIL images
    - image_process: process_images
    - tokenize: build conversation and tokenize
    - generate: model.generate()
    """
    model.eval()
    image_list = sample["images"]
    ground_truth = sample.get("ground_truth", {})

    # 1. Load images
    prof.start()
    images = []
    for p in image_list:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")
        img = Image.open(p).convert("RGB")
        images.append(img)
    prof.record("image_load")

    # 2. Vision preprocess
    prof.start()
    from llava.mm_utils import process_images
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    image_tensor = process_images(images, image_processor, model.config)
    image_tensor = [_img.to(dtype=torch.float16, device=device) for _img in image_tensor]
    image_sizes = [img.size for img in images]
    prof.record("image_process")

    # 3. Conversation + tokenize
    prof.start()
    from llava.conversation import conv_templates

    conv_template = "llava_llada"
    question_with_image = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question_with_image)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    from llava.mm_utils import tokenizer_image_token
    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    prompt_length = input_ids.shape[1]
    prof.record("tokenize")

    # 4. Generate
    prof.start()
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        steps=gen_config.get("steps", 128),
        gen_length=gen_config.get("gen_length", 128),
        block_length=gen_config.get("block_length", 128),
        tokenizer=tokenizer,
        stopping_criteria=["<|eot_id|>"],
        prefix_refresh_interval=gen_config.get("prefix_refresh_interval", 32),
        threshold=gen_config.get("threshold", 1),
    )
    prof.record("generate")

    # Decode
    output_text = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    output_ids = cont[0]
    # New tokens only (exclude prompt)
    if output_ids.shape[0] > prompt_length:
        generated_ids = output_ids[prompt_length:]
    else:
        generated_ids = output_ids
    output_token_len = len(generated_ids)

    return {
        "prompt_length": prompt_length,
        "output_token_len": output_token_len,
        "output_text": output_text,
        "ground_truth": ground_truth,
    }


def load_llada_model(model_path, device, use_fast_dllm=False, use_dllm_cache=False, cache_config=None):
    """Load LLaDA tokenizer/model/image_processor (same idea as qwen_infer_llada.init_worker)."""
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from llava.conversation import conv_templates, SeparatorStyle

    model_name = "llava_llada"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path, None, model_name, attn_implementation="sdpa", device_map=device
    )
    model.eval()

    if use_fast_dllm:
        from llava.hooks.fast_dllm_hook import register_fast_dllm_hook
        register_fast_dllm_hook(model)
        print("Fast dLLM hook enabled")
    elif use_dllm_cache:
        from llava.cache import dLLMCache, dLLMCacheConfig
        from llava.hooks import register_cache_LLaDA_V
        from dataclasses import asdict
        cfg = cache_config or {}
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=cfg.get("prompt_interval_steps", 25),
                    gen_interval_steps=cfg.get("gen_interval_steps", 7),
                    transfer_ratio=cfg.get("transfer_ratio", 0.25),
                )
            )
        )
        register_cache_LLaDA_V(model, "model.layers")
        print("dLLM-Cache enabled")
    else:
        print("No caching")

    return tokenizer, model, image_processor


def get_prompt_for_sample(ground_truth, prompt_type="no_cot"):
    """Return prompt text from prompt_type and ground-truth language."""
    try:
        from config import (
            system_prompt_eng,
            eng_instruction,
            system_prompt_zh,
            zh_instruction,
            no_cot_eng_prompt,
            no_cot_zh_prompt,
            system_prompt_zh_with_bbox,
            system_prompt_eng_with_bbox,
        )
    except ImportError:
        # No EasyEval config: use simple defaults (ZH vs EN format)
        findings = ground_truth.get("findings", "")
        has_zh = judge_language_zh_or_eng(findings) == "zh"
        if has_zh:
            return (
                "这是一组胸部X光图像，请生成一份医学报告，包括所见和结论。"
                "以以下格式返回报告：所见：{} 结论：{}。"
            )
        return (
            "This is a set of chest X-ray images. Please generate a medical report including "
            "findings and impression. Return the report in the following format: "
            "Findings: {} Impression: {}."
        )

    findings = ground_truth.get("findings", "")
    impression = ground_truth.get("impression", "")
    is_zh = judge_language_zh_or_eng(findings) == "zh" and judge_language_zh_or_eng(impression) == "zh"

    if prompt_type == "cot":
        if is_zh:
            return zh_instruction  # instruction only (matches qwen_infer_llada)
        return eng_instruction
    elif prompt_type == "no_cot":
        if is_zh:
            return no_cot_zh_prompt
        return no_cot_eng_prompt
    elif prompt_type == "cot_with_bbox":
        if is_zh:
            return zh_instruction
        return eng_instruction
    else:
        return no_cot_eng_prompt if not is_zh else no_cot_zh_prompt


def main():
    parser = argparse.ArgumentParser(description="Profile LLaDA inference stages (latency breakdown)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="your_llada_medv_model_dir",
                        help="Path to LLaDA weights")
    parser.add_argument(
        "-i", "--input_json", type=str, required=True,
        help="Sample list JSON (same schema as qwen_infer_llada)",
    )
    parser.add_argument("--out_dir", type=str, default="./profile_results", help="Output directory for JSON report")
    parser.add_argument("--max_samples", type=int, default=100, help="Max number of samples to profile")

    parser.add_argument(
        "--prompt_type", type=str, default="no_cot",
        choices=["cot", "no_cot", "cot_with_bbox"], help="Prompt style",
    )

    # Generation (aligned with qwen_infer_llada)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=64)
    parser.add_argument("--prefix_refresh_interval", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=1.0)

    # Acceleration
    parser.add_argument("--use_fast_dllm", action="store_true", default=True)
    parser.add_argument("--use_dllm_cache", action="store_true", default=True)
    parser.add_argument("--prompt_interval_steps", type=int, default=25)
    parser.add_argument("--gen_interval_steps", type=int, default=7)
    parser.add_argument("--transfer_ratio", type=float, default=0.25)

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading LLaDA from {args.model_path} ...")
    cache_config = {
        "prompt_interval_steps": args.prompt_interval_steps,
        "gen_interval_steps": args.gen_interval_steps,
        "transfer_ratio": args.transfer_ratio,
    }
    tokenizer, model, image_processor = load_llada_model(
        args.model_path, device,
        use_fast_dllm=args.use_fast_dllm,
        use_dllm_cache=args.use_dllm_cache,
        cache_config=cache_config,
    )

    # Count decoder forwards for "effective tokens per forward"
    # (LLaDA generate() does not use top-level model.forward; hook first transformer layer)
    forward_counter = {"count": 0}

    def _get_decoder_first_layer(model):
        # LLaDA: model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[0]
        # Fallback: model.model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
            return model.model.model.layers[0]
        return None

    first_layer = _get_decoder_first_layer(model)
    if first_layer is not None:
        orig_layer_forward = first_layer.forward

        def counted_layer_forward(*args, **kwargs):
            forward_counter["count"] += 1
            return orig_layer_forward(*args, **kwargs)

        first_layer.forward = counted_layer_forward
        print("Hooked decoder layer[0].forward for forward count")
    else:
        print("WARNING: Could not find decoder layers, Effective Tokens per Forward will be 0")

    with open(args.input_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("input_json must be a JSON list of samples")
    samples = samples[: args.max_samples] if args.max_samples else samples
    print(f"Profiling {len(samples)} samples.")

    gen_config = {
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "prefix_refresh_interval": args.prefix_refresh_interval,
        "threshold": args.threshold,
    }

    agg_stats = {}
    total_input_tokens = 0
    total_output_tokens = 0
    results_list = []

    print("\nStarting LLaDA profiling ...")
    for i, sample in enumerate(tqdm(samples, desc="Profile")):
        try:
            ground_truth = sample.get("ground_truth", {})
            prompt_text = get_prompt_for_sample(ground_truth, args.prompt_type)
            prof = Profiler()
            out = profile_llada_sample(
                model, tokenizer, image_processor,
                sample, device, gen_config, prof, prompt_text,
            )
            prompt_length = out["prompt_length"]
            output_token_len = out["output_token_len"]
            total_input_tokens += prompt_length
            total_output_tokens += output_token_len

            inference_time = sum(sum(v) for v in prof.stats.values())
            result_item = {
                "sample_id": sample.get("sample_id", f"sample_{i}"),
                "images": sample.get("images", []),
                "ground_truth": ground_truth,
                "output": out["output_text"],
                "inference_time": inference_time,
                "input_tokens": prompt_length,
                "output_tokens": output_token_len,
            }
            results_list.append(result_item)

            for k, v in prof.stats.items():
                if k not in agg_stats:
                    agg_stats[k] = []
                agg_stats[k].extend(v)

            print(f"  Sample {i+1}: input_tokens={prompt_length}, output_tokens={output_token_len}, time={inference_time:.4f}s")
        except Exception as e:
            print(f"Skipped sample {i}: {e}")
            import traceback
            traceback.print_exc()

    # Report (same structure as profile_dllm)
    total_profile_time = sum(sum(v) for v in agg_stats.values())
    print("\n" + "=" * 70)
    print("LLaDA INFERENCE BREAKDOWN")
    print(f"Total Input Tokens:  {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print("=" * 70)

    report = {
        "model_path": args.model_path,
        "prompt_type": args.prompt_type,
        "gen_config": gen_config,
        "use_fast_dllm": args.use_fast_dllm,
        "use_dllm_cache": args.use_dllm_cache,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_profile_time_sec": total_profile_time,
        "breakdown": {},
    }

    def print_stat(name, total_time=None):
        vals = agg_stats.get(name, [])
        if not vals:
            return
        count = len(vals)
        total_time_sec = sum(vals)
        avg_ms = np.mean(vals) * 1000
        pct = (total_time_sec / total_time) * 100 if total_time and total_time > 0 else 0
        print(f"{name.ljust(15)}: {avg_ms:6.2f} ms/op * {count:5d} = {total_time_sec:6.2f} s ({pct:5.1f}%)")
        report["breakdown"][name] = {
            "count": count,
            "total_time_sec": total_time_sec,
            "avg_ms_per_op": float(avg_ms),
            "pct_of_total_profile_time": float(pct),
        }

    if agg_stats:
        print_stat("image_load", total_profile_time)
        print_stat("image_process", total_profile_time)
        print_stat("tokenize", total_profile_time)
        print_stat("generate", total_profile_time)
        print("-" * 70)
        print(f"Total Profiled Time: {total_profile_time:.2f} s")

        # Throughput (aligned with profile_dllm)
        prefill_time = total_profile_time - sum(agg_stats.get("generate", []))  # load + preprocess + tokenize
        decode_time = sum(agg_stats.get("generate", []))  # generation only
        throughput_overall = total_output_tokens / total_profile_time if total_profile_time > 0 else 0
        throughput_decode = total_output_tokens / decode_time if decode_time > 0 else 0
        total_forwards = forward_counter["count"]
        tokens_per_forward = total_output_tokens / total_forwards if total_forwards > 0 else 0

        report["prefill_time_sec"] = prefill_time
        report["decode_time_sec"] = decode_time
        report["throughput_overall_tokens_per_sec"] = float(throughput_overall)
        report["throughput_decode_tokens_per_sec"] = float(throughput_decode)
        report["total_forwards"] = total_forwards
        report["effective_tokens_per_forward"] = float(tokens_per_forward)

        print("=" * 70)
        print(f"Throughput (Overall): {throughput_overall:.2f} tokens/s (including prefill)")
        print(f"Throughput (Decode) : {throughput_decode:.2f} tokens/s (generation only)")
        print("-" * 70)
        print(f"Effective Tokens per Forward: {tokens_per_forward:.2f}")
        print("=" * 70)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(
        args.out_dir,
        f"profile_results_llada_{args.prompt_type}_steps{args.steps}_gen{args.gen_length}_blk{args.block_length}.json",
    )
    output_payload = {
        "results": results_list,
        "report": report,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
