import math
import os
import json
import argparse
import time
import copy
from pathlib import Path
from typing import Dict, List
from multiprocessing import Pool

import torch
from PIL import Image
from tqdm import tqdm

# Global model instance (multiprocessing)
_global_model = None
_global_tokenizer = None
_global_image_processor = None
_global_device = None
_global_use_fast_dllm = None
_global_use_dllm_cache = None
_global_cache_config = None

def smart_resize(height: int, width: int, factor: int = 28,
                 min_pixels: int = 56 * 56, max_pixels: int = 1500*1500):
    """Rescales the image for Qwen2.5-VL"""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def judge_language_zh_or_eng(text):
    """
    Return "zh" if CJK ratio > 10%, else "eng".
    """
    if not text:
        return "eng"
    
    chinese_count = 0
    total_count = len(text)
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_count += 1
    
    chinese_ratio = chinese_count / total_count
    
    if chinese_ratio > 0.1:
        return "zh"
    else:
        return "eng"


def init_worker(gpu_id, model_path, use_fast_dllm=True, use_dllm_cache=False, cache_config=None):
    """
    Init worker: load LLaDA model.

    Args:
        gpu_id: GPU id as string.
        model_path: Path to model weights.
        use_fast_dllm: Enable Fast dLLM.
        use_dllm_cache: Enable dLLM-Cache.
        cache_config: Cache config dict.
    """
    global _global_model, _global_tokenizer, _global_image_processor, _global_device
    global _global_use_fast_dllm, _global_use_dllm_cache, _global_cache_config
    
    print(f"Initializing worker on GPU {gpu_id}, model_path={model_path}")
    
    # Pin this process to the chosen GPU
    if gpu_id != 'cpu' and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    
    _global_device = device
    _global_use_fast_dllm = use_fast_dllm
    _global_use_dllm_cache = use_dllm_cache
    _global_cache_config = cache_config
    
    # LLaDA imports
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.cache import dLLMCache, dLLMCacheConfig
    from llava.hooks import register_cache_LLaDA_V
    from dataclasses import asdict
    from llava.hooks.fast_dllm_hook import register_fast_dllm_hook, unregister_fast_dllm_hook
    
    model_name = "llava_llada"
    device_map = device
    
    _global_tokenizer, _global_model, _global_image_processor, max_length = load_pretrained_model(
        model_path, None, model_name, attn_implementation="sdpa", device_map=device_map
    )
    
    _global_model.eval()
    
    # Register acceleration hooks
    if use_fast_dllm:
        register_fast_dllm_hook(_global_model)
        print(f"[Worker {gpu_id}] Fast dLLM hook enabled")
    elif use_dllm_cache:
        from dataclasses import asdict
        _global_cache_config = cache_config or {}
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=_global_cache_config.get('prompt_interval_steps', 25),
                    gen_interval_steps=_global_cache_config.get('gen_interval_steps', 7),
                    transfer_ratio=_global_cache_config.get('transfer_ratio', 0.25),
                )
            )
        )
        register_cache_LLaDA_V(_global_model, "model.layers")
        print(f"[Worker {gpu_id}] dLLM-Cache enabled")
    else:
        print(f"[Worker {gpu_id}] No caching")
    
    print(f"Worker ready (GPU {gpu_id})")


def process_single_sample(args_tuple):
    """
    Worker: process one sample (multiprocessing).
    
    Args:
        args_tuple: (sample_id, image_list, ground_truth, prompt_type, output_dir, gen_config)
    
    Returns:
        (sample_id, success, error_msg)
    """
    sample_id, image_list, ground_truth, prompt_type, output_dir, gen_config = args_tuple
    
    try:
        global _global_model, _global_tokenizer, _global_image_processor, _global_device
        
        # Use global model
        if _global_model is None or _global_tokenizer is None:
            return (sample_id, False, "Model not initialized")
        
        # LLaDA imports
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        
        # Validate image paths
        for image_path in image_list:
            if not os.path.exists(image_path):
                return (sample_id, False, f"File not found: {image_path}")
        
        # Output path
        output_file = os.path.join(output_dir, f"{sample_id}.json")
        
        # Skip if already done
        if os.path.exists(output_file):
            return (sample_id, True, "already_exists")
        
        # Prompts from config
        from config import system_prompt_eng, eng_instruction,system_prompt_zh,zh_instruction,no_cot_eng_prompt,no_cot_zh_prompt,system_prompt_zh_with_bbox,system_prompt_eng_with_bbox
        is_zh = judge_language_zh_or_eng(ground_truth["findings"]) =="zh" and judge_language_zh_or_eng(ground_truth["impression"])=="zh"
        
        if prompt_type == "cot":
            if is_zh:
                system_prompt = system_prompt_zh
                instruction = zh_instruction
            else:
                system_prompt = system_prompt_eng
                instruction = eng_instruction
            question = instruction
        elif prompt_type == "no_cot":
            if is_zh:
                instruction = no_cot_zh_prompt
            else:
                instruction = no_cot_eng_prompt
            question = instruction
        elif prompt_type == "cot_with_bbox":
            if is_zh:
                system_prompt = system_prompt_zh_with_bbox
                instruction = zh_instruction
            else:
                system_prompt = system_prompt_eng_with_bbox
                instruction = eng_instruction
            question = instruction
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        
        # Load images
        images = []
        for p in image_list:
            img = Image.open(p).convert("RGB")
            images.append(img)
        
        # Process images
        image_tensor = process_images(images, _global_image_processor, _global_model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=_global_device) for _image in image_tensor]
        image_sizes = [img.size for img in images]
        
        # Build conversation
        conv_template = "llava_llada"
        question_with_image = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question_with_image)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        # Build inputs
        input_ids = tokenizer_image_token(prompt_question, _global_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(_global_device)
        
        # Forward / generate
        start_time = time.time()
        with torch.inference_mode():
            cont = _global_model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                steps=gen_config.get('steps', 128),
                gen_length=gen_config.get('gen_length', 128),
                block_length=gen_config.get('block_length', 128),
                tokenizer=_global_tokenizer,
                stopping_criteria=['<|eot_id|>'],
                prefix_refresh_interval=gen_config.get('prefix_refresh_interval', 32),
                threshold=gen_config.get('threshold', 1),
            )
        inference_time = time.time() - start_time
        
        # Decode
        output_text = _global_tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        
        # Build result dict
        result = {
            "sample_id": sample_id,
            "images": image_list,
            "ground_truth": ground_truth,
            "output": output_text,
            "inference_time": inference_time,
        }
        
        # Save JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return (sample_id, True, None)
        
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        return (sample_id, False, error_detail)


def read_sample_list(input_json: str) -> List[Dict]:
    """
    Load sample list from JSON.
    
    Args:
        input_json: Path to JSON list of samples.
    
    Returns:
        samples: Parsed list of dicts.
    """
    with open(input_json, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    if not isinstance(samples, list):
        raise ValueError(f"Input JSON must be a list, got {type(samples)}")
    
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each sample must be a dict")
        if "sample_id" not in sample:
            raise ValueError(f"Missing sample_id: {sample}")
        if "images" not in sample:
            raise ValueError(f"Missing images: {sample}")
        if not isinstance(sample["images"], list):
            raise ValueError(f"images must be a list: {sample}")
        if "ground_truth" not in sample:
            raise ValueError(f"Missing ground_truth: {sample}")
    
    return samples

    


def main():
    global MODEL_PATH
    
    # Multiprocessing: spawn (CUDA)
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    parser = argparse.ArgumentParser(
        description='LLaDA-MedV batch inference (multi-GPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU
  python %(prog)s -i samples.json -o output_dir
  
  # Multi-GPU: 2 GPUs, 1 process per GPU
  python %(prog)s -i samples.json -o output_dir --gpu_ids 0,1
  
  # Multi-GPU: 2 GPUs, 2 processes per GPU
  python %(prog)s -i samples.json -o output_dir --gpu_ids 0,0,1,1
  
  # CPU
  python %(prog)s -i samples.json -o output_dir --gpu_ids cpu

Input JSON format:
  [
    {
      "sample_id": "sample_001",
      "images": ["path/to/image1.png", "path/to/image2.png"],
      "ground_truth": "label or reference text"
    },
    {
      "sample_id": "sample_002",
      "images": ["path/to/image3.png", "path/to/image4.png"],
      "ground_truth": "label or reference text"
    }
  ]
        """
    )
    
    parser.add_argument('-i', '--input', type=str, default="EasyEval/samples.json",
                        help='Path to input JSON file')
    parser.add_argument('-o', '--output', type=str, default="EasyEval/samples_test2",
                        help='Output directory for per-sample JSON files')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='Comma-separated GPU ids, e.g. 0,0,1,1 for two processes on GPU0 and GPU1 (default: 0)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip samples that already have an output file')
    parser.add_argument('--model_path', type=str, default="LLaDA-MedV",
                        help='Model path (default: LLaDA-MedV)')
    parser.add_argument('--prompt_type', type=str, default='no_cot',
                        help='Prompt type: cot, no_cot, cot_with_bbox (default: no_cot)')
    
    # LLaDA generation
    parser.add_argument('--steps', type=int, default=256,
                        help='Generation steps (default: 128)')
    parser.add_argument('--gen_length', type=int, default=256,
                        help='Generation length (default: 128)')
    parser.add_argument('--block_length', type=int, default=64,
                        help='Block length (default: 128)')
    parser.add_argument('--prefix_refresh_interval', type=int, default=32,
                        help='Prefix refresh interval (default: 32)')
    parser.add_argument('--threshold', type=float, default=1,
                        help='Generation threshold (default: 1)')
    
    # Acceleration hooks
    parser.add_argument('--use_fast_dllm', action='store_true', default=True,
                        help='Use Fast dLLM (default: True)')
    parser.add_argument('--use_dllm_cache', action='store_true', default=True,
                        help='Use dLLM-Cache (default: False)')
    parser.add_argument('--prompt_interval_steps', type=int, default=25,
                        help='dLLM-Cache prompt interval steps (default: 25)')
    parser.add_argument('--gen_interval_steps', type=int, default=7,
                        help='dLLM-Cache gen interval steps (default: 7)')
    parser.add_argument('--transfer_ratio', type=float, default=0.25,
                        help='dLLM-Cache transfer ratio (default: 0.25)')
    
    args = parser.parse_args()
    
    print(f"model_path={args.model_path}")
    print(f"gen: steps={args.steps}, gen_length={args.gen_length}, block_length={args.block_length}")
    print(f"accel: use_fast_dllm={args.use_fast_dllm}, use_dllm_cache={args.use_dllm_cache}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load samples
    print("Loading sample list...")
    try:
        samples = read_sample_list(args.input)
    except Exception as e:
        print(f"Error reading input JSON: {e}")
        return
    
    if not samples:
        print(f"Error: no samples in {args.input}")
        return
    
    print(f"Loaded {len(samples)} samples")
    
    # Parse GPU ids
    gpu_ids = [gid.strip() for gid in args.gpu_ids.split(',')]
    n_workers = len(gpu_ids)
    
    print(f"Using {n_workers} parallel workers")
    print(f"GPU assignment: {gpu_ids}")
    
    # Build generation config
    gen_config = {
        'steps': args.steps,
        'gen_length': args.gen_length,
        'block_length': args.block_length,
        'prefix_refresh_interval': args.prefix_refresh_interval,
        'threshold': args.threshold,
    }
    
    cache_config = {
        'prompt_interval_steps': args.prompt_interval_steps,
        'gen_interval_steps': args.gen_interval_steps,
        'transfer_ratio': args.transfer_ratio,
    }
    
    # Build task list
    tasks = []
    for sample in samples:
        sample_id = sample["sample_id"]
        image_list = sample["images"]
        ground_truth = sample["ground_truth"]
        output_file = os.path.join(args.output, f"{sample_id}.json")
        
        # Respect --skip_existing
        if args.skip_existing and os.path.exists(output_file):
            continue
        
        tasks.append((sample_id, image_list, ground_truth, args.prompt_type, args.output, gen_config))
    
    if args.skip_existing:
        print(f"Skipped {len(samples) - len(tasks)} existing outputs")
    
    if not tasks:
        print("All samples already processed")
        return
    
    print(f"Tasks to run: {len(tasks)}")
    print("-" * 60)
    
    # Assign GPUs to workers
    gpu_assignments = [gpu_ids[i % n_workers] for i in range(n_workers)]
    
    # Multiprocessing
    start_time = time.time()
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # Track failures
    failed_samples = []
    error_log_path = os.path.join(args.output, 'failed_samples.log')
    
    if n_workers == 1:
        # Single-process (debug)
        print("Single-process mode")
        # Initialize model
        init_worker(gpu_ids[0], args.model_path, args.use_fast_dllm, args.use_dllm_cache, cache_config)
        
        for task in tqdm(tasks, desc="progress"):
            sample_id, success, error_msg = process_single_sample(task)
            if success:
                if error_msg == "already_exists":
                    skip_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
                failed_samples.append((sample_id, error_msg))
                print(f"\nError: {sample_id}")
                error_first_line = error_msg.split('\n')[0] if error_msg else "Unknown error"
                print(f"  {error_first_line}")
    else:
        # Multi-process
        print(f"Multi-process mode ({n_workers} workers)")
        print("Initializing worker models...")
        
        # One single-process Pool per GPU
        pools = []
        for i in range(n_workers):
            pool = Pool(
                processes=1,
                initializer=init_worker,
                initargs=(gpu_assignments[i], args.model_path, args.use_fast_dllm, args.use_dllm_cache, cache_config)
            )
            pools.append(pool)
        
        try:
            # Round-robin across pools
            results = []
            for i, task in enumerate(tasks):
                pool_idx = i % n_workers
                result = pools[pool_idx].apply_async(process_single_sample, (task,))
                results.append(result)
            
            # Collect results
            for result in tqdm(results, desc="progress"):
                sample_id, success, error_msg = result.get()
                if success:
                    if error_msg == "already_exists":
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
                    failed_samples.append((sample_id, error_msg))
                    print(f"\nError: {sample_id}")
                    error_first_line = error_msg.split('\n')[0] if error_msg else "Unknown error"
                    print(f"  {error_first_line}")
        finally:
            # Shutdown pools
            for pool in pools:
                pool.close()
                pool.join()
    
    elapsed_time = time.time() - start_time
    
    # Write failure log
    if failed_samples:
        try:
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write(f"# Failed samples ({len(failed_samples)})\n")
                f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for sample_id, error_msg in failed_samples:
                    f.write(f"{sample_id}\n")
                    error_lines = error_msg.split('\n')
                    for line in error_lines[:3]:
                        if line.strip():
                            f.write(f"# {line}\n")
                    f.write("\n")
            print(f"\nFailed sample log: {error_log_path}")
        except Exception as e:
            print(f"\nWarning: could not write failure log: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    print(f"Total tasks: {len(tasks)}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed: {fail_count}")
    if fail_count > 0:
        print(f"Fail rate: {fail_count/len(tasks)*100:.2f}%")
    print(f"Elapsed: {elapsed_time:.2f}s")
    if success_count > 0:
        print(f"Throughput: {success_count / elapsed_time:.2f} samples/s")
    print(f"Output dir: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
