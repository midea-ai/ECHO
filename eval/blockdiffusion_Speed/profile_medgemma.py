"""
Profile MedGemma-27B-IT (or medgemma-27bvl-it) for tokens/s throughput.
Uses AutoModelForImageTextToText + AutoProcessor, mimicking profile_ar.py structure.
"""
import argparse
import time
import torch
import json
import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from torch.nn import functional as F

# --- Helpers ---
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def sample(logits, temperature=1.0, top_k=0, top_p=1.0):
    logits = logits.reshape(-1, logits.shape[-1])
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, top_k)
        logits = torch.where(logits < v[:, -1:], float('-inf'), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cum_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False
        logits.masked_fill_(torch.scatter(torch.zeros_like(logits, dtype=torch.bool), -1, sorted_indices, sorted_mask), float('-inf'))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)

# --- Profiler Class ---
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

# --- Core Logic (AR Instrumented for MedGemma) ---
@torch.no_grad()
def profile_autoregressive(model, inputs, processor, args):
    """
    MedGemma uses processor.apply_chat_template which returns dict with
    input_ids, attention_mask, pixel_values (optional).
    """
    model.eval()
    prof = Profiler()

    # MedGemma inputs from apply_chat_template: dict with input_ids, attention_mask, pixel_values
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    all_input_ids = input_ids
    past_key_values = None
    generated_ids = []

    seq_length = input_ids.shape[1]
    cache_position = torch.arange(seq_length, device=model.device)

    # 1. Prefill
    prof.start()
    prefill_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "use_cache": True,
        "return_dict": True,
    }
    # Gemma3/MedGemma may support cache_position; pass only non-None
    prefill_kwargs["cache_position"] = cache_position
    outputs = model(**{k: v for k, v in prefill_kwargs.items() if v is not None})
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    prof.record("prefill")

    # Sample First Token
    prof.start()
    next_token = sample(next_token_logits, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    prof.record("sampling")

    next_token_scalar = next_token.item()
    generated_ids.append(next_token_scalar)

    all_input_ids = torch.cat([all_input_ids, next_token], dim=-1)
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((attention_mask.shape[0], 1), device=model.device, dtype=attention_mask.dtype),
        ], dim=-1)

    current_len = seq_length
    eos_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None) or getattr(model.generation_config, "eos_token_id", None)

    # 2. Decode Loop
    for _ in range(args.gen_length - 1):
        cache_position = torch.tensor([current_len], device=model.device)

        model_inputs = model.prepare_inputs_for_generation(
            all_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            cache_position=cache_position,
        )

        # [B] Forward
        prof.start()
        outputs = model(**model_inputs)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        prof.record("forward")

        # [C] Sampling
        prof.start()
        next_token = sample(next_token_logits, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        prof.record("sampling")

        # [D] Overhead
        prof.start()
        next_token_scalar = next_token.item()
        generated_ids.append(next_token_scalar)

        all_input_ids = torch.cat([all_input_ids, next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=model.device, dtype=attention_mask.dtype),
            ], dim=-1)

        current_len += 1

        if eos_id is not None and next_token_scalar == eos_id:
            prof.record("overhead")
            break
        prof.record("overhead")

    return len(generated_ids), prof.stats, generated_ids


def profile_simple_generate(model, inputs, processor, args):
    """
    Fallback: use model.generate() for simple end-to-end tokens/s measurement.
    """
    model.eval()
    prof = Profiler()

    input_len = inputs["input_ids"].shape[-1]

    prof.start()
    generation = model.generate(**inputs, max_new_tokens=args.gen_length, do_sample=(args.temperature > 0))
    prof.record("total")

    n_tokens = generation.shape[-1] - input_len
    generated_ids = generation[0, input_len:].tolist()

    return n_tokens, {"total": prof.stats["total"]}, generated_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="google/medgemma-27b-it")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./profile_results_medgemma")
    parser.add_argument("--max_samples", type=int, default=5)

    # Gen Params
    parser.add_argument("--gen_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)

    # Use simple generate() instead of manual AR loop (if manual loop fails)
    parser.add_argument("--simple", action="store_true", help="Use model.generate() for throughput (no detailed breakdown)")

    args = parser.parse_args()

    # Load MedGemma
    print(f"Loading {args.model_dir}...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_dir)

    # Data
    with open(args.input_json) as f:
        samples = json.load(f)[:args.max_samples]

    agg_stats = {}
    total_tokens_generated = 0
    results_list = []

    profile_fn = profile_simple_generate if args.simple else profile_autoregressive

    print(f"\nStarting Profiling ({'simple generate' if args.simple else 'AR'} mode)...")
    for i, sample in enumerate(samples):
        try:
            imgs = [Image.open(p).convert("RGB") for p in sample["images"]]
            ground_truth = sample.get("ground_truth", {})
            has_zh = any("\u4e00" <= char <= "\u9fff" for char in ground_truth.get("findings", ""))
            prompt_text = (
                "这是一组胸部X光图像，请生成一份医学报告，包括所见和结论。以以下格式返回报告：所见：{} 结论：{}。"
                if has_zh
                else "Review this chest X-ray and write a report. Use this format: Findings: {}, Impression: {}."
            )

            # MedGemma message format (image first or interleaved)
            content = [{"type": "image", "image": img} for img in imgs] + [{"type": "text", "text": prompt_text}]
            messages = [{"role": "user", "content": content}]

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Move to model device (BatchEncoding has .to())
            inputs = inputs.to(model.device, dtype=torch.bfloat16)

            input_token_len = inputs["input_ids"].shape[1]

            n_tokens, stats, generated_ids = profile_fn(model, inputs, processor, args)
            total_tokens_generated += n_tokens

            output_text = processor.decode(generated_ids, skip_special_tokens=True)
            inference_time = sum(sum(v) for v in stats.values())
            #print(output_text)
            result_item = {
                "sample_id": sample.get("sample_id", f"sample_{i}"),
                "images": sample.get("images", []),
                "ground_truth": sample.get("ground_truth", {}),
                "output": output_text,
                "inference_time": inference_time,
                "input_tokens": input_token_len,
                "output_tokens": n_tokens,
            }
            results_list.append(result_item)

            for k, v in stats.items():
                if k not in agg_stats:
                    agg_stats[k] = []
                agg_stats[k].extend(v)

            print(f"Sample {i+1}: Generated {n_tokens} tokens. Time: {inference_time:.4f}s")

        except Exception as e:
            print(f"Skipped sample {i}: {e}")
            import traceback
            traceback.print_exc()
            if not args.simple:
                print("\nTry running with --simple for basic throughput measurement.")

    # Save Results JSON
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"profile_results_medgemma_{int(time.time())}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {json_path}")

    # Report
    print("\n" + "=" * 70)
    print(f"MedGemma INFERENCE BREAKDOWN (Total Tokens: {total_tokens_generated})")
    print("=" * 70)

    if not agg_stats:
        print("No stats collected.")
        return

    total_profile_time = sum(sum(v) for v in agg_stats.values())

    def print_stat(name):
        vals = agg_stats.get(name, [])
        if not vals:
            return
        count = len(vals)
        total_time_sec = sum(vals)
        avg_ms = np.mean(vals) * 1000
        pct = (total_time_sec / total_profile_time) * 100
        print(f"{name.ljust(15)}: {avg_ms:6.2f} ms/op * {count:5d} = {total_time_sec:6.2f} s ({pct:5.1f}%)")

    if not args.simple:
        print_stat("prefill")
        print("-" * 70)
        print_stat("forward")
        print_stat("sampling")
        print_stat("overhead")
    else:
        print_stat("total")
    print("-" * 70)
    print(f"Total Profiled Time: {total_profile_time:.2f} s")

    # TPS Calculations
    prefill_time = sum(agg_stats.get("prefill", []))
    decode_time = total_profile_time - prefill_time

    avg_tps_overall = total_tokens_generated / total_profile_time if total_profile_time > 0 else 0
    avg_tps_decode = total_tokens_generated / decode_time if decode_time > 0 else avg_tps_overall

    print("=" * 70)
    print(f"Throughput (Overall): {avg_tps_overall:.2f} tokens/s (including prefill)")
    print(f"Throughput (Decode) : {avg_tps_decode:.2f} tokens/s (generation only)")
    print("=" * 70)


if __name__ == "__main__":
    main()
