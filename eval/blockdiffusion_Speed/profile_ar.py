import argparse
import time
import torch
import json
import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.cache_utils import DynamicCache
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
        if name not in self.stats: self.stats[name] = []
        self.stats[name].append(dt)
        self.start_t = time.time()

# --- Core Logic (AR Instrumented) ---
@torch.no_grad()
def profile_autoregressive(model, inputs, tokenizer, args):
    model.eval()
    prof = Profiler()
    
    # Store initial inputs for prepare_inputs_for_generation
    input_ids = inputs.input_ids
    pixel_values = getattr(inputs, 'pixel_values', None)
    image_grid_thw = getattr(inputs, 'image_grid_thw', None)
    attention_mask = getattr(inputs, 'attention_mask', None)
    
    all_input_ids = input_ids
    past_key_values = None
    
    generated_ids = []
    
    # Initialize cache_position
    seq_length = input_ids.shape[1]
    cache_position = torch.arange(seq_length, device=model.device)
    
    # 1. Prefill
    prof.start()
    
    # Note: For Qwen2.5-VL, passing cache_position is good practice if supported
    # But usually prefill works without it if we don't use prepare_inputs_for_generation for prefill
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
        cache_position=cache_position # Explicitly pass cache_position
    )
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    prof.record("prefill")
    
    # Sample First Token
    prof.start()
    next_token = sample(next_token_logits, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    prof.record("sampling")
    
    next_token_scalar = next_token.item()
    generated_ids.append(next_token_scalar)
    
    # Update all_input_ids
    all_input_ids = torch.cat([all_input_ids, next_token], dim=-1)
    
    # Update attention mask if it exists
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=model.device, dtype=attention_mask.dtype)], dim=-1)

    # Update cache_position for next step (single token)
    current_len = seq_length
    
    # 2. Decode Loop
    for _ in range(args.gen_length - 1):
        # Update cache_position to point to the new token
        cache_position = torch.tensor([current_len], device=model.device)
        
        # [A] Prepare Inputs
        model_inputs = model.prepare_inputs_for_generation(
            all_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cache_position=cache_position # Pass it here!
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
             attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=model.device, dtype=attention_mask.dtype)], dim=-1)
        
        current_len += 1
        
        if next_token_scalar == tokenizer.eos_token_id:
            prof.record("overhead")
            break
        prof.record("overhead")

    return len(generated_ids), prof.stats, generated_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./profile_results")
    parser.add_argument("--max_samples", type=int, default=5)
    
    # Gen Params
    parser.add_argument("--gen_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Load (Handle Qwen2.5-VL Config)
    print(f"Loading {args.model_dir}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    except ValueError as e:
        if "Qwen2_5_VL" in str(e):
            print("Using Qwen2_5_VLForConditionalGeneration...")
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
        else:
            raise e
            
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    
    # Ensure generation config is loaded
    if model.generation_config is None:
        model.generation_config = GenerationConfig.from_pretrained(args.model_dir)
    
    # Data
    with open(args.input_json) as f: samples = json.load(f)[:args.max_samples]
    
    agg_stats = {}
    total_tokens_generated = 0
    results_list = []
    
    print("\nStarting Profiling (AR)...")
    for i, sample in enumerate(samples):
        try:
            imgs = [Image.open(p).convert("RGB") for p in sample['images']]
            # image_content = []
            # for image_path in sample['images']:
            #     image_content.append({
            #         "type": "image",
            #         "image": {"image_path": image_path},
            #     })

            ground_truth = sample["ground_truth"]
            has_zh = any('\u4e00' <= char <= '\u9fff' for char in ground_truth.get("findings", ""))
            prompt_text = "这是一组胸部X光图像，请生成一份医学报告，包括所见和结论。以以下格式返回报告：所见：{} 结论：{}。" if has_zh else "Review this chest X-ray and write a report. Use this format: Findings: {}, Impression: {}."
            #prompt_text = "OCR:"
            
            messages = [{"role": "user", "content": [{"type": "image"}]*len(imgs) + [{"type": "text", "text": prompt_text}]}]
            #messages = [{"role": "user", "content": image_content + [{"type": "text", "text": prompt_text}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=imgs, max_pixels=2250000, padding=True, return_tensors="pt").to("cuda")
            # inputs = processor(text=[text], images=imgs, padding=True, return_tensors="pt").to("cuda")
            # inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")
            
            input_token_len = inputs['input_ids'].shape[1]

            n_tokens, stats, generated_ids = profile_autoregressive(model, inputs, tokenizer, args)
            total_tokens_generated += n_tokens
            
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            inference_time = sum(sum(v) for v in stats.values())
            
            # Save Result
            result_item = {
                "sample_id": sample.get("sample_id", f"sample_{i}"),
                "images": sample.get("images", []),
                "ground_truth": sample.get("gt", {}),
                "output": output_text,
                "inference_time": inference_time,
                "input_tokens": input_token_len,
                "output_tokens": n_tokens
            }
            results_list.append(result_item)

            # Aggregate
            for k, v in stats.items():
                if k not in agg_stats: agg_stats[k] = []
                agg_stats[k].extend(v)
                
            print(f"Sample {i+1}: Generated {n_tokens} tokens. Time: {inference_time:.4f}s")
            
        except Exception as e:
            print(f"Skipped sample {i}: {e}")
            import traceback
            traceback.print_exc()

    # Save Results JSON
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"profile_results_ar_{int(time.time())}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {json_path}")

    # Report
    print("\n" + "="*70)
    print(f"AR (BASELINE) INFERENCE BREAKDOWN (Total Tokens: {total_tokens_generated})")
    print("="*70)
    
    if not agg_stats:
        print("No stats collected.")
        return

    total_profile_time = sum(sum(v) for v in agg_stats.values())
    
    def print_stat(name):
        vals = agg_stats.get(name, [])
        if not vals: return
        
        count = len(vals)
        total_time_sec = sum(vals)
        avg_ms = np.mean(vals) * 1000
        pct = (total_time_sec / total_profile_time) * 100
        
        # Format: Name : Avg ms/op * Count = Total Time (Pct)
        print(f"{name.ljust(15)}: {avg_ms:6.2f} ms/op * {count:5d} = {total_time_sec:6.2f} s ({pct:5.1f}%)")

    print_stat("prefill")
    print("-" * 70)
    print_stat("forward")
    print_stat("sampling")
    print_stat("overhead")
    print("-" * 70)
    print(f"Total Profiled Time: {total_profile_time:.2f} s")
    
    # TPS Calculations
    prefill_time = sum(agg_stats.get("prefill", []))
    decode_time = total_profile_time - prefill_time
    
    avg_tps_overall = total_tokens_generated / total_profile_time if total_profile_time > 0 else 0
    avg_tps_decode = total_tokens_generated / decode_time if decode_time > 0 else 0
    
    print("=" * 70)
    print(f"Throughput (Overall): {avg_tps_overall:.2f} tokens/s (including prefill)")
    print(f"Throughput (Decode) : {avg_tps_decode:.2f} tokens/s (generation only)")
    print("=" * 70)

if __name__ == "__main__":
    main()
