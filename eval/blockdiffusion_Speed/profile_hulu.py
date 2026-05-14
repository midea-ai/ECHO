"""
Profile Hulu-med for tokens/s throughput.
Uses same inference logic as hulu_infer.py: AutoModelForCausalLM + AutoProcessor
with processor(conversation=..., add_system_prompt=True, add_generation_prompt=True).
"""
import argparse
import time
import torch
import json
import os
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.nn import functional as F

# Import prompt building from hulu_infer
from hulu_infer import build_prompt_and_images

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
        logits.masked_fill_(
            torch.scatter(torch.zeros_like(logits, dtype=torch.bool), -1, sorted_indices, sorted_mask),
            float('-inf'),
        )

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


# --- Core Logic (AR Instrumented for Hulu-med) ---
@torch.no_grad()
def profile_autoregressive(model, inputs, processor, args):
    """
    Hulu-med inputs: dict from processor(conversation=...) with input_ids, pixel_values, attention_mask.
    """
    model.eval()
    prof = Profiler()

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")
    # Hulu-med uses grid_sizes, merge_sizes, modals (not image_grid_thw)
    grid_sizes = inputs.get("grid_sizes")
    merge_sizes = inputs.get("merge_sizes")
    modals = inputs.get("modals")
    if modals is None and grid_sizes is not None:
        # Fallback: assume all images when modals not provided
        n_imgs = grid_sizes.shape[0] if hasattr(grid_sizes, "shape") else len(grid_sizes)
        modals = ["image"] * n_imgs

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
        "grid_sizes": grid_sizes,
        "merge_sizes": merge_sizes,
        "modals": modals,
        "use_cache": True,
        "return_dict": True,
        "cache_position": cache_position,
    }
    prefill_kwargs = {k: v for k, v in prefill_kwargs.items() if v is not None}
    outputs = model(**prefill_kwargs)
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

    # 2. Decode Loop
    for _ in range(args.gen_length - 1):
        cache_position = torch.tensor([current_len], device=model.device)

        # Decode: no pixel_values/grid_sizes/merge_sizes (vision only in prefill)
        model_inputs = model.prepare_inputs_for_generation(
            all_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            attention_mask=attention_mask,
            cache_position=cache_position,
        )

        prof.start()
        outputs = model(**model_inputs)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        prof.record("forward")

        prof.start()
        next_token = sample(next_token_logits, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        prof.record("sampling")

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

        if next_token_scalar == processor.tokenizer.eos_token_id:
            prof.record("overhead")
            break
        prof.record("overhead")

    return len(generated_ids), prof.stats, generated_ids


def profile_simple_generate(model, inputs, processor, args):
    """Fallback: use model.generate() for simple throughput measurement."""
    model.eval()
    prof = Profiler()

    input_len = inputs["input_ids"].shape[-1]

    prof.start()
    output_ids = model.generate(**inputs, max_new_tokens=args.gen_length)
    prof.record("total")

    n_tokens = output_ids.shape[-1] - input_len
    generated_ids = output_ids[0, input_len:].tolist()

    return n_tokens, {"total": prof.stats["total"]}, generated_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="your_hulu_model_dir")
    parser.add_argument("--input_json", type=str, default="minic_100.json")
    parser.add_argument("--out_dir", type=str, default="./profile_results_hulu")
    parser.add_argument("--max_samples", type=int, default=100)

    parser.add_argument("--prompt_type", type=str, default="no_cot", choices=["cot", "no_cot", "cot_with_bbox"])

    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--simple", action="store_true", help="Use model.generate() for throughput (no detailed breakdown)")

    args = parser.parse_args()

    # Load Hulu-med (same as hulu_infer.py)
    print(f"Loading {args.model_dir}...")
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = processor.tokenizer

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
            image_list = sample["images"]
            ground_truth = sample.get("ground_truth", {})

            # Build messages (same as hulu_infer)
            conversation = build_prompt_and_images(image_list, ground_truth, args.prompt_type, processor)

            # Processor call (same as batch_hulu_infer)
            inputs = processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            # Convert numpy arrays (grid_sizes, merge_sizes) to tensors on device
            for key in ("grid_sizes", "merge_sizes"):
                if key in inputs and inputs[key] is not None and not isinstance(inputs[key], torch.Tensor):
                    inputs[key] = torch.tensor(inputs[key], dtype=torch.long, device=model.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            input_token_len = inputs["input_ids"].shape[1]

            n_tokens, stats, generated_ids = profile_fn(model, inputs, processor, args)
            total_tokens_generated += n_tokens

            output_text = processor.batch_decode(
                [generated_ids],
                skip_special_tokens=True,
                use_think=False,
            )[0].strip()

            inference_time = sum(sum(v) for v in stats.values())

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
    json_path = os.path.join(args.out_dir, f"profile_results_hulu_{int(time.time())}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {json_path}")

    # Report
    print("\n" + "=" * 70)
    print(f"Hulu-med INFERENCE BREAKDOWN (Total Tokens: {total_tokens_generated})")
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
