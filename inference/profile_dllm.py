import argparse
import time
import torch
import json
import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from torch.nn import functional as F
import shutil

# --- Helpers ---
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def create_block_diffusion_mask(num_blocks, block_length, device):
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    full_mask = block_mask.repeat_interleave(block_length, dim=0)\
                          .repeat_interleave(block_length, dim=1)
    return full_mask[None, None, :, :]

def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens

def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]
    vocab_size = logits.shape[-1]
    logits_2d = logits.reshape(-1, vocab_size)

    if temperature == 0:
        tokens = torch.argmax(logits_2d, dim=-1, keepdim=True)
        probs = F.softmax(logits_2d, dim=-1)
        token_probs = torch.gather(probs, -1, tokens)
    else:
        logits_scaled = logits_2d / temperature
        if top_k > 0:
            values, _ = torch.topk(logits_scaled, top_k)
            min_values = values[:, -1:]
            logits_scaled = torch.where(logits_scaled < min_values, float('-inf'), logits_scaled)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits_scaled, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            mask_indices = torch.scatter(torch.zeros_like(logits_scaled, dtype=torch.bool), -1, sorted_indices, sorted_mask)
            logits_scaled = logits_scaled.masked_fill(mask_indices, float('-inf'))
        
        probs = F.softmax(logits_scaled, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1)
        token_probs = torch.gather(probs, -1, tokens)

    return tokens.view(batch_size, seq_len), token_probs.view(batch_size, seq_len)

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

# --- Core Logic (Instrumented) ---
@torch.no_grad()
def profile_block_diffusion(model, inputs, mask_id, tokenizer, args):
    model.eval()
    prof = Profiler()
    
    # 1. Setup Inputs
    input_ids = inputs['input_ids']
    pixel_values = inputs.get('pixel_values', None)
    image_grid_thw = inputs.get('image_grid_thw', None)
    pixel_values_videos = inputs.get('pixel_values_videos', None)
    video_grid_thw = inputs.get('video_grid_thw', None)

    prompt_length = input_ids.shape[1]
    
    # Align total length to block size
    num_blocks = (prompt_length + args.gen_length + args.block_length - 1) // args.block_length
    total_length = num_blocks * args.block_length

    # Initialize Canvas
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids

    # Setup KV Cache
    kv_cache = DynamicCache() if args.use_cache else None
    
    prof.start()
    global_attn_mask = create_block_diffusion_mask(num_blocks, args.block_length, model.device)
    prof.record("init_overhead")

    # 2. Prefill Stage
    if args.use_cache:
        prof.start()
        print(f"Prefilling {prompt_length} tokens...")
        prefill_mask = global_attn_mask[:, :, :prompt_length, :prompt_length]
        model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=prefill_mask,
            past_key_values=kv_cache,
            use_cache=True,
            store_kv=True
        )
        
        # KV Cache Alignment / Rewind
        prefill_blocks = prompt_length // args.block_length
        aligned_prefill_len = prefill_blocks * args.block_length
        if aligned_prefill_len < prompt_length:
            print(f"Rewinding KV Cache from {prompt_length} to {aligned_prefill_len}.")
            kv_cache.crop(aligned_prefill_len)
        prof.record("prefill")
    else:
        prefill_blocks = prompt_length // args.block_length

    # 3. Decode Stage
    print("Starting Block Diffusion Generation...")
    num_transfer_tokens = get_num_transfer_tokens(args.block_length, args.denoising_steps)
    
    generated_tokens = 0
    final_output_len = prompt_length

    # --- Fused-decode path (step=1 + cache only) ---
    # Instead of: denoising_forward(masked, store_kv=False) + kv_update(x0, store_kv=True)
    # We do:      first_block: denoising_forward(masked, store_kv=False)
    #             subsequent:  combined_forward([prev_x0 | cur_masked], store_kv=True, store_kv_len=block_length)
    #                          which simultaneously writes prev block's KV and denoises cur block.
    use_fused_decode = (
        args.fused_decode
        and args.use_cache
        and args.denoising_steps == 1
    )

    if use_fused_decode:
        prev_finalized_x = None  # finalized tokens of the previous generation block

        for num_block in range(prefill_blocks, num_blocks):
            block_start = num_block * args.block_length
            block_end = (num_block + 1) * args.block_length

            if block_start >= total_length:
                break

            cur_x = x[:, block_start:block_end].clone()
            cache_position = torch.arange(
                block_start, block_end, device=model.device, dtype=torch.long
            )
            mask_index = (cur_x == mask_id)

            # Prompt-only block (no masks): do a normal kv_update and carry forward
            if mask_index.sum() == 0:
                prof.start()
                cur_attn_mask = global_attn_mask[:, :, block_start:block_end, :block_end]
                model(
                    input_ids=cur_x,
                    attention_mask=cur_attn_mask,
                    cache_position=cache_position,
                    past_key_values=kv_cache,
                    use_cache=True,
                    store_kv=True,
                )
                prof.record("kv_update")
                prev_finalized_x = cur_x
                x[:, block_start:block_end] = cur_x
                final_output_len = block_end
                continue

            # --- Forward pass ---
            prof.start()
            if prev_finalized_x is not None:
                # Combined forward: [prev_finalized | cur_masked]
                # - stores KV for prev_finalized (store_kv_len = block_length)
                # - returns logits for cur_masked (last block_length positions)
                prev_block_start = block_start - args.block_length
                combined_input = torch.cat([prev_finalized_x, cur_x], dim=1)
                combined_cache_pos = torch.arange(
                    prev_block_start, block_end, device=model.device, dtype=torch.long
                )
                combined_attn_mask = global_attn_mask[:, :, prev_block_start:block_end, :block_end]
                outputs = model(
                    input_ids=combined_input,
                    attention_mask=combined_attn_mask,
                    cache_position=combined_cache_pos,
                    past_key_values=kv_cache,
                    use_cache=True,
                    store_kv=True,
                    store_kv_len=args.block_length,
                )
                logits = outputs.logits[:, -args.block_length:]
            else:
                # First generation block: regular denoising (no prev block to fold in)
                cur_attn_mask = global_attn_mask[:, :, block_start:block_end, :block_end]
                outputs = model(
                    input_ids=cur_x,
                    attention_mask=cur_attn_mask,
                    cache_position=cache_position,
                    past_key_values=kv_cache,
                    use_cache=True,
                    store_kv=False,
                )
                logits = outputs.logits
            prof.record("forward")

            # Sampling (step=1 → transfer all masked positions at once)
            prof.start()
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            prof.record("sampling")

            # Remasking
            prof.start()
            transfer_index = torch.zeros(
                cur_x.shape[0], cur_x.shape[1], device=cur_x.device, dtype=torch.bool
            )
            if args.remasking_strategy == 'sequential':
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        mask_positions = mask_index[j].nonzero(as_tuple=True)[0]
                        num_to_select = min(num_transfer_tokens[0], len(mask_positions))
                        transfer_index[j, mask_positions[:num_to_select]] = True
            elif args.remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, torch.tensor(-torch.inf, device=cur_x.device))
                for j in range(cur_x.shape[0]):
                    num_masks = mask_index[j].sum().item()
                    k = min(num_transfer_tokens[0], num_masks)
                    if k > 0 and not torch.all(torch.isinf(confidence[j])):
                        _, idx = torch.topk(confidence[j], k)
                        transfer_index[j, idx] = True
            elif args.remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, torch.tensor(-torch.inf, device=cur_x.device))
                for j in range(cur_x.shape[0]):
                    high_conf_mask = confidence[j] > args.confidence_threshold
                    num_high_confidence = high_conf_mask.sum().item()
                    if num_high_confidence >= num_transfer_tokens[0]:
                        transfer_index[j] = high_conf_mask
                    else:
                        num_masks = mask_index[j].sum().item()
                        k = min(num_transfer_tokens[0], num_masks)
                        if k > 0:
                            _, idx = torch.topk(confidence[j], k)
                            transfer_index[j, idx] = True
            else:
                raise ValueError(f"Unknown remasking strategy: {args.remasking_strategy}")

            retained = transfer_index.sum().item()
            generated_tokens += retained
            cur_x[transfer_index] = x0[transfer_index]
            prof.record("remask_logic")

            prev_finalized_x = cur_x.clone()
            x[:, block_start:block_end] = cur_x
            final_output_len = block_end

            # EOS check
            valid_start_idx = max(0, prompt_length - block_start)
            if valid_start_idx < cur_x.shape[1]:
                new_tokens = cur_x[:, valid_start_idx:]
                if tokenizer.eos_token_id is not None and (new_tokens == tokenizer.eos_token_id).any():
                    break

    else:
        # --- Original multi-step decode path ---
        for num_block in range(prefill_blocks, num_blocks):
            block_start = num_block * args.block_length
            block_end = (num_block + 1) * args.block_length

            if block_start >= total_length:
                break

            cur_x = x[:, block_start:block_end].clone()

            # Manually construct cache_position
            cache_position = torch.arange(
                block_start, block_end, device=model.device, dtype=torch.long
            )

            # Denoising Steps
            for step in range(args.denoising_steps + 1):
                mask_index = (cur_x == mask_id)

                # --- [A] Finalization Step (No masks left) ---
                if mask_index.sum() == 0:
                    if args.use_cache:
                        prof.start()
                        # Update KV Cache with finalized block
                        cur_attn_mask = global_attn_mask[:, :, block_start:block_end, :block_end]
                        model(
                            input_ids=cur_x,
                            attention_mask=cur_attn_mask,
                            cache_position=cache_position,
                            past_key_values=kv_cache,
                            use_cache=True,
                            store_kv=True
                        )
                        prof.record("kv_update")
                    break

                # --- [B] Forward Pass (Denoising) ---
                prof.start()
                if args.use_cache:
                    # With Cache: Only forward current block
                    cur_attn_mask = global_attn_mask[:, :, block_start:block_end, :block_end]
                    outputs = model(
                        input_ids=cur_x,
                        attention_mask=cur_attn_mask,
                        cache_position=cache_position,
                        past_key_values=kv_cache,
                        use_cache=True,
                        store_kv=False
                    )
                    logits = outputs.logits
                else:
                    # Without Cache: Recompute Full Context
                    full_input = x[:, :block_end].clone()
                    full_input[:, block_start:block_end] = cur_x
                    context_mask = global_attn_mask[:, :, :block_end, :block_end]
                    outputs = model(
                        input_ids=full_input,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=video_grid_thw,
                        attention_mask=context_mask,
                        past_key_values=None,
                        use_cache=False,
                        store_kv=False
                    )
                    logits = outputs.logits[:, block_start:block_end]
                prof.record("forward")

                # Sampling
                prof.start()
                x0, x0_p = sample_with_temperature_topk_topp(
                    logits,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                prof.record("sampling")

                # Remasking Strategy
                prof.start()
                transfer_index = torch.zeros(cur_x.shape[0], cur_x.shape[1], device=cur_x.device, dtype=torch.bool)
                
                if args.remasking_strategy == 'sequential':
                    for j in range(cur_x.shape[0]):
                        if mask_index[j].any():
                            mask_positions = mask_index[j].nonzero(as_tuple=True)[0]
                            num_to_select = min(num_transfer_tokens[step], len(mask_positions))
                            selected_positions = mask_positions[:num_to_select]
                            transfer_index[j, selected_positions] = True
                            
                elif args.remasking_strategy == 'low_confidence_static':
                    confidence = torch.where(mask_index, x0_p, torch.tensor(-torch.inf, device=cur_x.device))
                    for j in range(cur_x.shape[0]):
                        num_masks = mask_index[j].sum().item()
                        k = min(num_transfer_tokens[step], num_masks)
                        if k > 0 and not torch.all(torch.isinf(confidence[j])):
                            _, idx = torch.topk(confidence[j], k)
                            transfer_index[j, idx] = True
                            
                elif args.remasking_strategy == 'low_confidence_dynamic':
                    confidence = torch.where(mask_index, x0_p, torch.tensor(-torch.inf, device=cur_x.device))
                    for j in range(cur_x.shape[0]):
                        high_conf_mask = confidence[j] > args.confidence_threshold
                        num_high_confidence = high_conf_mask.sum().item()
                        if num_high_confidence >= num_transfer_tokens[step]:
                            transfer_index[j] = high_conf_mask
                        else:
                            num_masks = mask_index[j].sum().item()
                            k = min(num_transfer_tokens[step], num_masks)
                            if k > 0:
                                _, idx = torch.topk(confidence[j], k)
                                transfer_index[j, idx] = True
                else:
                    raise ValueError(f"Unknown remasking strategy: {args.remasking_strategy}")
                
                retained = transfer_index.sum().item()
                generated_tokens += retained
                
                # Update Canvas
                cur_x[transfer_index] = x0[transfer_index]
                prof.record("remask_logic")

            # Update global canvas with finalized block
            x[:, block_start:block_end] = cur_x
            final_output_len = block_end

            # Check stopping criteria
            # Fix: Only check generated tokens, ignore prompt tokens in the current block
            valid_start_idx = 0
            if block_start < prompt_length:
                valid_start_idx = prompt_length - block_start
            
            if valid_start_idx < cur_x.shape[1]:
                new_tokens = cur_x[:, valid_start_idx:]
                if tokenizer.eos_token_id is not None:
                    if (new_tokens == tokenizer.eos_token_id).any():
                        # Truncate at EOS if found
                        # Find exact position in global x
                        eos_pos_in_block = (new_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[1][0]
                        # block_start + valid_start_idx + eos_pos_in_block
                        # But actually we just want to know where to cut
                        
                        # Just mark break, we can post-process truncation
                        break

    # Truncate x to valid length
    # Note: loop might have broken early or finished
    # We'll use the final state of x. 
    # But x is initialized with MASKs.
    # We should return the whole x and let caller truncate at EOS or max length
    
    return generated_tokens, prof.stats, x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./profile_results")
    parser.add_argument("--max_samples", type=int, default=None)
    
    # DLLM Params
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=8)
    parser.add_argument("--denoising_steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.85)
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence_dynamic",
                        choices=["sequential", "low_confidence_dynamic", "low_confidence_static", "entropy_bounded"])
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--fused_decode", action="store_true", default=False,
                        help="Fused kv_update+denoising for step=1: fold prev block's kv_update "
                             "into the next block's denoising forward (combined forward). "
                             "Only effective when denoising_steps=1 and use_cache=True.")
    
    args = parser.parse_args()
    
    # Load
    set_seed(args.seed)

    print(f"Loading {args.model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    
    # Determine mask_id
    if hasattr(model.config, 'mask_token_id'): mask_id = model.config.mask_token_id
    elif hasattr(tokenizer, 'mask_token_id'): mask_id = tokenizer.mask_token_id
    else: mask_id = tokenizer("<|MASK|>", add_special_tokens=False)['input_ids'][0]
    
    # Data
    with open(args.input_json) as f: samples = json.load(f)[:args.max_samples]
    
    agg_stats = {}
    total_tokens_generated = 0
    results_list = []
    
    print(f"\nStarting Profiling (Strategy: {args.remasking_strategy})...")
    for i, sample in enumerate(samples):
        try:
            imgs = [Image.open(p).convert("RGB") for p in sample['images']]
            
            ground_truth = sample["ground_truth"]
            has_zh = any('\u4e00' <= char <= '\u9fff' for char in ground_truth.get("findings", ""))
            prompt_text = "这是一组胸部X光图像，请生成一份医学报告，包括所见和结论。以以下格式返回报告：所见：{} 结论：{}。" if has_zh else "This is a set of chest X-ray images. Please generate a medical report including findings and impression. Return the report in the following format: Findings: {} Impression: {}."
            #prompt_text = "Extract the text content from this image."
            
            messages = [{"role": "user", "content": [{"type": "image"}] * len(imgs) + [{"type": "text", "text": prompt_text}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=imgs,
                max_pixels=2250000,
                padding=True,
                return_tensors="pt"
            ).to("cuda")
            
            input_token_len = inputs['input_ids'].shape[1]
            
            n_tokens, stats, final_x = profile_block_diffusion(model, inputs, mask_id, tokenizer, args)
            
            # Post-process output
            generated_ids = final_x[0, input_token_len:]
            # Truncate at EOS
            if tokenizer.eos_token_id in generated_ids:
                eos_idx = (generated_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                generated_ids = generated_ids[:eos_idx]
                # Also remove padding/mask tokens if any remain (though they shouldn't if we stop at EOS)
            
            # Remove any trailing MASK tokens if EOS wasn't hit but generation stopped
            if mask_id in generated_ids:
                 mask_idx = (generated_ids == mask_id).nonzero(as_tuple=True)[0][0]
                 generated_ids = generated_ids[:mask_idx]
            
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            inference_time = sum(sum(v) for v in stats.values())
            
            total_tokens_generated += n_tokens
            
            # Save Result
            result_item = {
                "sample_id": sample.get("sample_id", f"sample_{i}"),
                "images": sample.get("images", []),
                "ground_truth": sample.get("gt", {}),
                "output": output_text,
                "inference_time": inference_time,
                "input_tokens": input_token_len,
                "output_tokens": len(generated_ids) # This is length of valid generated sequence
            }
            results_list.append(result_item)
            
            # Aggregate Stats
            for k, v in stats.items():
                if k not in agg_stats: agg_stats[k] = []
                agg_stats[k].extend(v)
                
            print(f"Sample {i+1}: Generated {n_tokens} retained tokens, {len(generated_ids)} valid tokens. Time: {inference_time:.4f}s")
            
        except Exception as e:
            print(f"Skipped sample {i}: {e}")
            import traceback
            traceback.print_exc()

    # Report
    print("\n" + "="*70)
    print(f"DLLM INFERENCE BREAKDOWN (Strategy: {args.remasking_strategy})")
    print(f"Total Tokens (Retained): {total_tokens_generated}")
    print("="*70)

    report = {
        "strategy": args.remasking_strategy,
        "total_tokens_retained": total_tokens_generated,
        "breakdown": {}
    }

    if not agg_stats:
        print("No stats collected.")
    else:
        total_profile_time = sum(sum(v) for v in agg_stats.values())
        report["total_profile_time_sec"] = total_profile_time

    def print_stat(name, total_profile_time=None):
        vals = agg_stats.get(name, [])
        if not vals:
            return
        
        count = len(vals)
        total_time_sec = sum(vals)
        avg_ms = np.mean(vals) * 1000
        pct = (total_time_sec / total_profile_time) * 100 if total_profile_time and total_profile_time > 0 else 0
        
        # Format: Name : Avg ms/op * Count = Total Time (Pct)
        print(f"{name.ljust(15)}: {avg_ms:6.2f} ms/op * {count:5d} = {total_time_sec:6.2f} s ({pct:5.1f}%)")
        report["breakdown"][name] = {
            "count": count,
            "total_time_sec": total_time_sec,
            "avg_ms_per_op": float(avg_ms),
            "pct_of_total_profile_time": float(pct),
        }

    if agg_stats:
        print_stat("prefill", total_profile_time)
        print("-" * 70)
        print_stat("forward", total_profile_time)
        print_stat("sampling", total_profile_time)
        print_stat("remask_logic", total_profile_time)
        print_stat("kv_update", total_profile_time)
        print("-" * 70)
        print(f"Total Profiled Time: {total_profile_time:.2f} s")
        
        # TPS Calculations
        prefill_time = sum(agg_stats.get("prefill", []))
        decode_time = total_profile_time - prefill_time
        
        avg_tps_overall = total_tokens_generated / total_profile_time if total_profile_time > 0 else 0
        avg_tps_decode = total_tokens_generated / decode_time if decode_time > 0 else 0
        
        # DLLM Specific: Effective Tokens per Forward
        total_forwards = len(agg_stats.get("forward", []))
        tokens_per_forward = total_tokens_generated / total_forwards if total_forwards > 0 else 0
        
        report["prefill_time_sec"] = prefill_time
        report["decode_time_sec"] = decode_time
        report["throughput_overall_tokens_per_sec"] = float(avg_tps_overall)
        report["throughput_decode_tokens_per_sec"] = float(avg_tps_decode)
        report["total_forwards"] = total_forwards
        report["effective_tokens_per_forward"] = float(tokens_per_forward)
        
        print("=" * 70)
        print(f"Throughput (Overall): {avg_tps_overall:.2f} tokens/s (including prefill)")
        print(f"Throughput (Decode) : {avg_tps_decode:.2f} tokens/s (generation only)")
        print("-" * 70)
        print(f"Effective Tokens per Forward: {tokens_per_forward:.2f}")
        print("=" * 70)

    # Save Results JSON (with report)
    os.makedirs(args.out_dir, exist_ok=True)
    fused_suffix = "_fused" if (args.fused_decode and args.use_cache and args.denoising_steps == 1) else ""
    json_path = os.path.join(args.out_dir, f"profile_results_dllm_{args.remasking_strategy}{args.confidence_threshold}_blk{args.block_length}_step{args.denoising_steps}_t{args.temperature}{fused_suffix}.json")
    output_payload = {
        "results": results_list,
        "report": report,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {json_path}")

if __name__ == "__main__":
    main()
