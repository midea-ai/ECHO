# pip install transformers==4.55.4
import argparse
import sys
import os
import shutil
import torch
from torch.nn import functional as F
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from PIL import Image


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Sample tokens with temperature, top-k, and top-p."""
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]
    vocab_size = logits.shape[-1]

    logits_2d = logits.reshape(-1, vocab_size)

    if temperature == 0:
        # Greedy sampling
        tokens = torch.argmax(logits_2d, dim=-1, keepdim=True)
        probs = F.softmax(logits_2d, dim=-1)
        token_probs = torch.gather(probs, -1, tokens)
    else:
        # Apply temperature
        logits_scaled = logits_2d / temperature

        # Apply top-k
        if top_k > 0:
            values, _ = torch.topk(logits_scaled, top_k)
            min_values = values[:, -1:]
            logits_scaled = torch.where(
                logits_scaled < min_values, float('-inf'), logits_scaled)

        # Apply top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                logits_scaled, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            mask_indices = torch.scatter(
                torch.zeros_like(logits_scaled, dtype=torch.bool),
                -1, sorted_indices, sorted_mask
            )
            logits_scaled = logits_scaled.masked_fill(
                mask_indices, float('-inf'))

        probs = F.softmax(logits_scaled, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1)
        token_probs = torch.gather(probs, -1, tokens)

    return tokens.view(batch_size, seq_len), token_probs.view(batch_size, seq_len)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


def create_block_diffusion_mask(num_blocks, block_length, device):
    """Creates a block-diagonal lower triangular mask."""
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    full_mask = block_mask.repeat_interleave(block_length, dim=0)\
                          .repeat_interleave(block_length, dim=1)
    return full_mask[None, None, :, :]  # (1, 1, seq_len, seq_len)


@torch.no_grad()
def block_diffusion_generate_vl(
    model,
    inputs,
    mask_id,
    gen_length=128,
    block_length=8,
    denoising_steps=8,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    remasking_strategy='low_confidence_dynamic',
    confidence_threshold=0.85,
    stopping_criteria_idx=None,
    use_cache=True,  # ⚠️ Added flag
    tokenizer=None,
):

    model.eval()

    # 1. Setup Inputs
    input_ids = inputs['input_ids']
    pixel_values = inputs.get('pixel_values', None)
    image_grid_thw = inputs.get('image_grid_thw', None)
    pixel_values_videos = inputs.get('pixel_values_videos', None)
    video_grid_thw = inputs.get('video_grid_thw', None)

    prompt_length = input_ids.shape[1]

    # Align total length to block size
    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Initialize Canvas
    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids

    # 2. Setup KV Cache (Only if enabled)
    past_key_values = DynamicCache() if use_cache else None

    # 3. Construct Global Attention Mask
    global_attn_mask = create_block_diffusion_mask(
        num_blocks, block_length, model.device)

    if use_cache:
        # 4. Prefill Stage
        #print(f"Prefilling {prompt_length} tokens (Use Cache: {use_cache})...")

        prefill_mask = global_attn_mask[:, :, :prompt_length, :prompt_length]
        model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=prefill_mask,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True
        )
        # KV Cache Alignment / Rewind
        prefill_blocks = prompt_length // block_length
        aligned_prefill_len = prefill_blocks * block_length
        if aligned_prefill_len < prompt_length:
            print(
                f"Rewinding KV Cache from {prompt_length} to {aligned_prefill_len}.")
            past_key_values.crop(aligned_prefill_len)
    else:
        # Without KV cache, we don't need explicit prefill forward pass
        # unless we want to sanity check. We will compute everything in the loop.
        prefill_blocks = prompt_length // block_length

    # 6. Decode Stage
    #print("Starting Block Diffusion Generation...")
    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length

        if block_start >= total_length:
            break

        cur_x = x[:, block_start:block_end].clone()
        
        # if tokenizer is not None:
        #      print(f"\n[Block {num_block}] Initial cur_x IDs: {cur_x[0].tolist()}")
        #      print(f"[Block {num_block}] Initial cur_x Text: {repr(tokenizer.decode(cur_x[0], skip_special_tokens=False))}")

        # Manually construct cache_position
        cache_position = torch.arange(
            block_start, block_end, device=model.device, dtype=torch.long
        )

        # Denoising Steps
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)

            # --- [A] Finalization Step (No masks left) ---
            if mask_index.sum() == 0:
                if use_cache:
                    # Update KV Cache with finalized block
                    cur_attn_mask = global_attn_mask[:, :,
                                                     block_start:block_end, :block_end]
                    model(
                        input_ids=cur_x,
                        attention_mask=cur_attn_mask,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True
                    )
                else:
                    # No cache update needed, just update global canvas x (done at end of loop)
                    pass
                break

            # --- [B] Forward Pass (Denoising) ---
            if use_cache:
                # With Cache: Only forward current block
                cur_attn_mask = global_attn_mask[:, :,
                                                 block_start:block_end, :block_end]
                outputs = model(
                    input_ids=cur_x,
                    attention_mask=cur_attn_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False
                )
                logits = outputs.logits
            else:
                # Without Cache: Recompute Full Context
                full_input = x[:, :block_end].clone()
                # Update current block part
                full_input[:, block_start:block_end] = cur_x

                # Full mask up to block_end
                context_mask = global_attn_mask[:, :, :block_end, :block_end]

                # We need to re-pass vision inputs every time
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
                # Extract logits corresponding to the current block
                # Output shape: (1, block_end, vocab) -> Slice [block_start:block_end]
                logits = outputs.logits[:, block_start:block_end]

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )


            # Remasking Strategy
            transfer_index = torch.zeros(
                cur_x.shape[0], cur_x.shape[1], device=cur_x.device, dtype=torch.bool)
            if remasking_strategy == 'sequential':
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        mask_positions = mask_index[j].nonzero(as_tuple=True)[
                            0]
                        num_to_select = min(
                            num_transfer_tokens[step], len(mask_positions))
                        selected_positions = mask_positions[:num_to_select]
                        transfer_index[j, selected_positions] = True

            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(
                    mask_index, x0_p, torch.tensor(-torch.inf, device=cur_x.device))
                for j in range(cur_x.shape[0]):
                    num_masks = mask_index[j].sum().item()
                    k = min(num_transfer_tokens[step], num_masks)
                    if k > 0 and not torch.all(torch.isinf(confidence[j])):
                        _, idx = torch.topk(confidence[j], k)
                        transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(
                    mask_index, x0_p, torch.tensor(-torch.inf, device=cur_x.device))
                for j in range(cur_x.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
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
                raise ValueError(
                    f"Unknown remasking strategy: {remasking_strategy}")

                
            # Update Canvas
            cur_x[transfer_index] = x0[transfer_index]

            # Update global canvas with finalized block
            # if tokenizer is not None:
            #     print(f"  [Step {step}] Sampled block{num_block} IDs: {cur_x[0].tolist()}")
                
            #     annotated_tokens = []
            #     for idx in range(cur_x.shape[1]):
            #         # Get predicted token from x0
            #         x0_val = x0[0, idx].item()
            #         x0_str = tokenizer.decode([x0_val], skip_special_tokens=False)
                    
            #         # If this position was a mask
            #         if mask_index[0, idx]:
            #             # And it was selected for update -> Show accepted token
            #             if transfer_index[0, idx]:
            #                 token_str = x0_str
            #             # And it was NOT selected -> Show dropped token prediction
            #             else:
            #                 token_str = f"{{DROP: {x0_str}}}"
            #         else:
            #             # Existing context (not a mask in this step)
            #             cur_val = cur_x[0, idx].item()
            #             token_str = tokenizer.decode([cur_val], skip_special_tokens=False)
                    
            #         annotated_tokens.append(token_str)
                
            #     full_text = "".join(annotated_tokens)
            #     print(f"  [Step {step}] Sampled block{num_block} Text: {repr(full_text)}")

        x[:, block_start:block_end] = cur_x

        # Check stopping criteria
        if stopping_criteria_idx is not None:
            # Fix: Only check generated tokens, ignore prompt tokens in the current block
            # Calculate where the "new generation" starts within this block
            valid_start_idx = 0
            if block_start < prompt_length:
                valid_start_idx = prompt_length - block_start
            
            # Extract only the generated part of cur_x for checking
            if valid_start_idx < cur_x.shape[1]:
                new_tokens = cur_x[:, valid_start_idx:]
                if any(stop_idx in new_tokens for stop_idx in stopping_criteria_idx):
                    break

    return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="../model_weights/ECHO_Base_block8",
                        help="Path to the pretrained model directory or HuggingFace repo id")
    parser.add_argument("--image_path", type=str, default="../eval/CXRTest_demo/CXRTest/mimic/files/p10/p10032725/s50331901/687754ce-7420bfd3-0a19911f-a27a3916-9019cd53.jpg",
                        help="Image path or URL")
    parser.add_argument("--prompt_text", type=str, default="这是一组胸部X光图像，请生成一份医学报告，包括所见和结论。以以下格式返回报告：所见：{} 结论：{}。",
                        help="User prompt")
    # Block Diffusion Args
    parser.add_argument("--mask_id", type=int, default=None)
    parser.add_argument("--gen_length", type=int, default=1024)
    parser.add_argument("--block_length", type=int, default=8)
    parser.add_argument("--denoising_steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. Use 0 for greedy/deterministic sampling.")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-K sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-P sampling probability threshold")
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence_dynamic",
                        choices=["sequential", "low_confidence_dynamic",
                                 "low_confidence_static", "entropy_bounded"],
                        help="Strategy for remasking tokens")
    parser.add_argument("--confidence_threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "float32", "bfloat16"])

    # ⚠️ New Flag
    parser.add_argument("--use_cache", action="store_true",
                        default=True, help="Enable KV Cache")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    if args.temperature <= 0:
        print("Using GREEDY sampling (deterministic)")
    else:
        print(f"Using stochastic sampling with temperature={args.temperature}")

    print(f"Loading {args.model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=args.dtype,
        device_map=args.device,
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True)

    if args.mask_id is None:
        # 1. Try reading from model config (Loaded from config.json)
        if hasattr(model.config, 'mask_token_id') and model.config.mask_token_id is not None:
            args.mask_id = model.config.mask_token_id
            print(f"Found mask_token_id in model config: {args.mask_id}")

        # 2. Fallback to Tokenizer (if not in model config)
        elif hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
            args.mask_id = tokenizer.mask_token_id
            print(f"Found mask_token_id in tokenizer: {args.mask_id}")

        # 3. Last resort: manual lookup
        else:
            print(
                "Warning: mask_token_id not found in config or tokenizer. Using <|MASK|> lookup.")
            mask_ids = tokenizer("<|MASK|>", add_special_tokens=False)[
                'input_ids']
            if len(mask_ids) > 0:
                args.mask_id = mask_ids[0]
            else:
                raise ValueError(
                    "Could not determine mask_id. Please provide --mask_id.")

    print(f"Using mask_id: {args.mask_id}")

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": args.prompt_text}
        ]}
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print(f"Prompt ends with: {repr(text)}")

    inputs = processor(
        text=[text],
        images=[Image.open(args.image_path).convert("RGB")],
        max_pixels=2250000,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate
    output_ids = block_diffusion_generate_vl(
        model,
        inputs=inputs,
        mask_id=args.mask_id,
        gen_length=args.gen_length,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        stopping_criteria_idx=[tokenizer.eos_token_id],
        use_cache=args.use_cache,  # ⚠️ Pass Flag
        tokenizer=tokenizer
    )

    generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\nGenerated Output:")
    print(output_text)


if __name__ == "__main__":
    main()
