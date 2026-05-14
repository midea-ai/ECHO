# pip install transformers==4.55.4
import argparse
import torch
from torch.nn import functional as F
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_block_diffusion_mask(num_blocks, block_length, device):
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    full_mask = (block_mask
                 .repeat_interleave(block_length, dim=0)
                 .repeat_interleave(block_length, dim=1))
    return full_mask[None, None, :, :]


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    batch_size, seq_len, vocab_size = logits.shape
    logits_2d = logits.reshape(-1, vocab_size)

    if temperature == 0:
        tokens = torch.argmax(logits_2d, dim=-1, keepdim=True)
        probs = F.softmax(logits_2d, dim=-1)
        token_probs = torch.gather(probs, -1, tokens)
    else:
        logits_scaled = logits_2d / temperature
        if top_k > 0:
            values, _ = torch.topk(logits_scaled, top_k)
            logits_scaled = torch.where(
                logits_scaled < values[:, -1:], float('-inf'), logits_scaled)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits_scaled, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            mask_indices = torch.scatter(
                torch.zeros_like(logits_scaled, dtype=torch.bool),
                -1, sorted_indices, sorted_mask)
            logits_scaled = logits_scaled.masked_fill(mask_indices, float('-inf'))
        probs = F.softmax(logits_scaled, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1)
        token_probs = torch.gather(probs, -1, tokens)

    return tokens.view(batch_size, seq_len), token_probs.view(batch_size, seq_len)


@torch.no_grad()
def echo_generate(
    model,
    inputs,
    mask_id,
    tokenizer,
    gen_length=512,
    block_length=4,
    denoising_steps=1,
    temperature=0.0,
    top_k=0,
    top_p=1.0,
    remasking_strategy='low_confidence_dynamic',
    confidence_threshold=0.85,
    fused_decode=True,
):
    """
    Block-diffusion generation for ECHO (distilled one-step model).

    When denoising_steps=1 and fused_decode=True the fused-decode path is used:
    each iteration simultaneously writes the previous block's KV cache and
    denoises the current block in a single forward pass, halving the number of
    model calls compared to the standard two-pass approach.
    """
    model.eval()

    input_ids = inputs['input_ids']
    pixel_values = inputs.get('pixel_values', None)
    image_grid_thw = inputs.get('image_grid_thw', None)
    pixel_values_videos = inputs.get('pixel_values_videos', None)
    video_grid_thw = inputs.get('video_grid_thw', None)

    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids

    kv_cache = DynamicCache()
    global_attn_mask = create_block_diffusion_mask(num_blocks, block_length, model.device)

    # Prefill
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
        store_kv=True,
    )
    prefill_blocks = prompt_length // block_length
    aligned_prefill_len = prefill_blocks * block_length
    if aligned_prefill_len < prompt_length:
        kv_cache.crop(aligned_prefill_len)

    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)
    use_fused = fused_decode and denoising_steps == 1

    # ── Fused single-step decode ───────────────────────────────────────────────
    if use_fused:
        prev_finalized_x = None

        for num_block in range(prefill_blocks, num_blocks):
            block_start = num_block * block_length
            block_end = (num_block + 1) * block_length
            if block_start >= total_length:
                break

            cur_x = x[:, block_start:block_end].clone()
            cache_position = torch.arange(block_start, block_end, device=model.device, dtype=torch.long)
            mask_index = (cur_x == mask_id)

            # Prompt-only block: just update KV cache and move on
            if mask_index.sum() == 0:
                cur_attn_mask = global_attn_mask[:, :, block_start:block_end, :block_end]
                model(
                    input_ids=cur_x,
                    attention_mask=cur_attn_mask,
                    cache_position=cache_position,
                    past_key_values=kv_cache,
                    use_cache=True,
                    store_kv=True,
                )
                prev_finalized_x = cur_x
                x[:, block_start:block_end] = cur_x
                continue

            if prev_finalized_x is not None:
                # Fused forward: write prev block's KV and denoise cur block together
                prev_block_start = block_start - block_length
                combined_input = torch.cat([prev_finalized_x, cur_x], dim=1)
                combined_cache_pos = torch.arange(
                    prev_block_start, block_end, device=model.device, dtype=torch.long)
                combined_attn_mask = global_attn_mask[:, :, prev_block_start:block_end, :block_end]
                outputs = model(
                    input_ids=combined_input,
                    attention_mask=combined_attn_mask,
                    cache_position=combined_cache_pos,
                    past_key_values=kv_cache,
                    use_cache=True,
                    store_kv=True,
                    store_kv_len=block_length,
                )
                logits = outputs.logits[:, -block_length:]
            else:
                # First generation block: regular denoising forward
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

            x0, x0_p = sample_with_temperature_topk_topp(
                logits, temperature=temperature, top_k=top_k, top_p=top_p)

            transfer_index = _remask(
                cur_x, mask_index, x0, x0_p,
                num_transfer_tokens[0], remasking_strategy, confidence_threshold)
            cur_x[transfer_index] = x0[transfer_index]

            prev_finalized_x = cur_x.clone()
            x[:, block_start:block_end] = cur_x

            # EOS check
            valid_start = max(0, prompt_length - block_start)
            if valid_start < cur_x.shape[1]:
                if tokenizer.eos_token_id is not None:
                    if (cur_x[:, valid_start:] == tokenizer.eos_token_id).any():
                        break

    # ── Standard multi-step decode ─────────────────────────────────────────────
    else:
        for num_block in range(prefill_blocks, num_blocks):
            block_start = num_block * block_length
            block_end = (num_block + 1) * block_length
            if block_start >= total_length:
                break

            cur_x = x[:, block_start:block_end].clone()
            cache_position = torch.arange(block_start, block_end, device=model.device, dtype=torch.long)

            for step in range(denoising_steps + 1):
                mask_index = (cur_x == mask_id)

                if mask_index.sum() == 0:
                    cur_attn_mask = global_attn_mask[:, :, block_start:block_end, :block_end]
                    model(
                        input_ids=cur_x,
                        attention_mask=cur_attn_mask,
                        cache_position=cache_position,
                        past_key_values=kv_cache,
                        use_cache=True,
                        store_kv=True,
                    )
                    break

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

                x0, x0_p = sample_with_temperature_topk_topp(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p)

                transfer_index = _remask(
                    cur_x, mask_index, x0, x0_p,
                    num_transfer_tokens[step], remasking_strategy, confidence_threshold)
                cur_x[transfer_index] = x0[transfer_index]

            x[:, block_start:block_end] = cur_x

            valid_start = max(0, prompt_length - block_start)
            if valid_start < cur_x.shape[1]:
                if tokenizer.eos_token_id is not None:
                    if (cur_x[:, valid_start:] == tokenizer.eos_token_id).any():
                        break

    return x


def _remask(cur_x, mask_index, x0, x0_p, n_transfer, strategy, threshold):
    """Select which masked positions to reveal this step."""
    transfer_index = torch.zeros_like(cur_x, dtype=torch.bool)

    for j in range(cur_x.shape[0]):
        if not mask_index[j].any():
            continue

        if strategy == 'sequential':
            mask_positions = mask_index[j].nonzero(as_tuple=True)[0]
            k = min(n_transfer, len(mask_positions))
            transfer_index[j, mask_positions[:k]] = True

        elif strategy == 'low_confidence_static':
            confidence = torch.where(
                mask_index[j], x0_p[j], torch.tensor(-torch.inf, device=cur_x.device))
            num_masks = mask_index[j].sum().item()
            k = min(n_transfer, num_masks)
            if k > 0 and not torch.all(torch.isinf(confidence)):
                _, idx = torch.topk(confidence, k)
                transfer_index[j, idx] = True

        elif strategy == 'low_confidence_dynamic':
            confidence = torch.where(
                mask_index[j], x0_p[j], torch.tensor(-torch.inf, device=cur_x.device))
            high_conf = confidence > threshold
            if high_conf.sum() >= n_transfer:
                transfer_index[j] = high_conf
            else:
                num_masks = mask_index[j].sum().item()
                k = min(n_transfer, num_masks)
                if k > 0:
                    _, idx = torch.topk(confidence, k)
                    transfer_index[j, idx] = True

        else:
            raise ValueError(f"Unknown remasking strategy: {strategy}")

    return transfer_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-sample inference for ECHO (distilled one-step block diffusion VLM).")
    parser.add_argument("--model_dir", type=str, default="../model_weights/ECHO_block8",
                        help="Path to the model checkpoint or HuggingFace repo id")
    parser.add_argument("--image_path", type=str,
                        default="../eval/CXRTest_demo/CXRTest/mimic/files/p10/p10032725/s50331901/687754ce-7420bfd3-0a19911f-a27a3916-9019cd53.jpg",
                        help="Path to the input chest X-ray image")
    # parser.add_argument("--prompt_text", type=str,
    #                     default="Review this chest X-ray and write a report. "
    #                             "Use this format: Findings: {}, Impression: {}.",
    #                     help="User prompt")
    parser.add_argument("--prompt_text", type=str,
                        default="这是一组胸部X光图像，请生成一份医学报告，包括所见和结论。"
                                "以以下格式返回报告：所见：{} 结论：{}。",
                        help="User prompt")
    # Block diffusion parameters
    parser.add_argument("--gen_length", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--block_length", type=int, default=8,
                        help="Number of tokens per diffusion block (must match model)")
    parser.add_argument("--denoising_steps", type=int, default=1,
                        help="Denoising steps per block (1 = single-step, i.e. ECHO mode)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence_dynamic",
                        choices=["sequential", "low_confidence_dynamic", "low_confidence_static"])
    parser.add_argument("--confidence_threshold", type=float, default=0.85)
    parser.add_argument("--fused_decode", action="store_true", default=True,
                        help="Fuse prev-block KV update with cur-block denoising forward "
                             "(only active when denoising_steps=1, reduces forward calls by ~half)")
    parser.add_argument("--no_fused_decode", dest="fused_decode", action="store_false")
    parser.add_argument("--mask_id", type=int, default=None,
                        help="Override mask token id (auto-detected from config if not set)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "float32", "bfloat16"])
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model from {args.model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch_dtype, device_map=args.device, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Resolve mask token id
    if args.mask_id is not None:
        mask_id = args.mask_id
    elif hasattr(model.config, 'mask_token_id') and model.config.mask_token_id is not None:
        mask_id = model.config.mask_token_id
    elif hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        mask_id = tokenizer.mask_token_id
    else:
        ids = tokenizer("<|MASK|>", add_special_tokens=False)['input_ids']
        if not ids:
            raise ValueError("Cannot determine mask_id. Pass --mask_id explicitly.")
        mask_id = ids[0]
    print(f"mask_id: {mask_id}")

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": args.prompt_text},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image = Image.open(args.image_path).convert("RGB")
    inputs = processor(
        text=[text],
        images=[image],
        max_pixels=2250000,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")
    print(f"denoising_steps={args.denoising_steps}, block_length={args.block_length}, "
          f"fused_decode={args.fused_decode and args.denoising_steps == 1}")

    output_ids = echo_generate(
        model=model,
        inputs=inputs,
        mask_id=mask_id,
        tokenizer=tokenizer,
        gen_length=args.gen_length,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        fused_decode=args.fused_decode,
    )

    generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]

    # Truncate at EOS
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in generated_ids:
        eos_idx = (generated_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
        generated_ids = generated_ids[:eos_idx]
    # Remove any trailing MASK tokens
    if mask_id in generated_ids:
        mask_idx = (generated_ids == mask_id).nonzero(as_tuple=True)[0][0]
        generated_ids = generated_ids[:mask_idx]

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\n--- Generated Report ---")
    print(output_text)


if __name__ == "__main__":
    main()
