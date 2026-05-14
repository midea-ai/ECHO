import math
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List
from multiprocessing import Pool

import torch
from PIL import Image
# LLaVA-Med-v1.5
from transformers import LlavaForConditionalGeneration, AutoProcessor
# Original Qwen / other model (commented out)
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# from transformers import AutoModelForImageTextToText, AutoModelForCausalLM

# Global model instance (multiprocessing)
_global_model = None
_global_processor = None

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


def init_worker(gpu_id, model_path):
    """
    Init worker: load model.

    Args:
        gpu_id: GPU id as string ('0', '1', 'cpu').
        model_path: Path to model weights.
    """
    global _global_model, _global_processor
    
    print(f"Initializing worker on GPU {gpu_id}, model_path={model_path}")
    
    # Pin this process to the chosen GPU
    if gpu_id != 'cpu' and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    
    # LLaVA-Med-v1.5
    _global_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if device != "cpu" else None,
        device_map="auto",
    )
    _global_processor = AutoProcessor.from_pretrained(model_path)

    # Original Qwen / other model (commented out)
    # _global_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2" if device != "cpu" else None,
    #     device_map=device,
    # )
    # _global_model = AutoModelForImageTextToText.from_pretrained(
    #         model_path,
    #         torch_dtype=torch.bfloat16,
    #         attn_implementation="flash_attention_2" if device != "cpu" else None,
    #         device_map="auto",
    #         trust_remote_code=True
    #     )
    # _global_model = AutoModelForCausalLM.from_pretrained(
    #         model_path,
    #         torch_dtype=torch.bfloat16,
    #         attn_implementation="flash_attention_2" if device != "cpu" else None,
    #         device_map="auto",
    #         trust_remote_code=True
    #     )
    # _global_processor = AutoProcessor.from_pretrained(model_path,  trust_remote_code=True)
    
    _global_model.eval()
    print(f"Worker ready (GPU {gpu_id})")


def process_single_sample(args_tuple):
    """
    Worker: process one sample (multiprocessing).
    
    Args:
        args_tuple: (sample_id, image_list, ground_truth, output_dir)
    
    Returns:
        (sample_id, success, error_msg)
    """
    sample_id, image_list, ground_truth,prompt_type,output_dir = args_tuple
    
    try:
        global _global_model, _global_processor
        
        # Use global model
        if _global_model is None or _global_processor is None:
            return (sample_id, False, "Model not initialized")
        
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
            
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # Append images
            for image_path in image_list:
                messages[1]["content"].append({
                    "type": "image",
                    "image": image_path,
                    "max_pixels": 1500**2,
                    "min_pixels": 256*28*28,
                })
            
            # Append instruction text
            messages[1]["content"].append({"type": "text", "text": instruction})
        elif prompt_type == "no_cot":
            if is_zh:
                instruction = no_cot_zh_prompt
            else:
                instruction = no_cot_eng_prompt
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            for image_path in image_list:
                messages[0]["content"].append({
                    "type": "image",
                    "image": image_path,
                    "max_pixels": 1500**2,
                    "min_pixels": 256*28*28,
                })
            messages[0]["content"].append({"type": "text", "text": instruction})
        elif prompt_type == "cot_with_bbox":
            if is_zh:
                system_prompt = system_prompt_zh_with_bbox
                instruction = zh_instruction
            else:
                system_prompt = system_prompt_eng_with_bbox
                instruction = eng_instruction
            
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # Append images
            for image_path in image_list:
                messages[1]["content"].append({
                    "type": "image",
                    "image": image_path,
                    "max_pixels": 1500**2,
                    "min_pixels": 256*28*28,
                })
            
            # Append instruction text
            messages[1]["content"].append({"type": "text", "text": instruction})
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
            # Build messages
        # messages[1]["content"].append({"type": "text", "text": "..."})  # example prompt

        # LLaVA-Med-v1.5 inference
        # Load images as PIL
        pil_images = []
        for p in image_list:
            img = Image.open(p).convert("RGB")
            h, w = img.size[1], img.size[0]
            new_h, new_w = smart_resize(h, w)
            img = img.resize((new_w, new_h))
            pil_images.append(img)
            
        # Build LLaVA-style messages (image + text)
        full_instruction = instruction
        if prompt_type in ("cot", "cot_with_bbox"):
            full_instruction = system_prompt + "\n\n" + instruction
        llava_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_instruction},
                ],
            }
        ]
        chat_template_path = "llava-med-v1.5-hf/chat_template.json"
        with open(chat_template_path, "r", encoding="utf-8") as f:
            chat_template = json.load(f)["chat_template"]
        prompt = _global_processor.tokenizer.apply_chat_template(
            llava_messages,chat_template=chat_template, tokenize=False, add_generation_prompt=True
        )
        inputs = _global_processor(
            images=pil_images,
            text=prompt,
            return_tensors="pt",
        ).to(_global_model.device, torch.bfloat16)

        # Legacy Qwen path (commented)
        # text = _global_processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        # image_inputs, video_inputs = process_vision_info(messages)
        # inputs = _global_processor(
        #     text=[text],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        # inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # if "pixel_values" in inputs:
        #     inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        # device = next(_global_model.parameters()).device

        # Generation config
        generate_config = {
            "max_new_tokens": 2048,
        }

        # Forward / generate
        start_time = time.time()
        with torch.inference_mode():
            generated_ids = _global_model.generate(**inputs, **generate_config)

        # LLaVA: strip prompt; decode new tokens only
        input_length = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_length:]
        output_text = _global_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )

        # Legacy Qwen decode (commented)
        # output_text = _global_processor.batch_decode(
        #     generated_ids, skip_special_tokens=True, use_think=False
        # )
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # output_text = _global_processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False, use_think=False
        # )
        
        inference_time = time.time() - start_time
        #print(output_text[0])
        # Build result dict
        result = {
            "sample_id": sample_id,
            "images": image_list,
            "ground_truth": ground_truth,
            "output": output_text[0].strip(),
            "inference_time": inference_time,
            # "input_tokens": inputs.input_ids.shape[1],
            # "output_tokens": generated_ids_trimmed[0].shape[0]
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
        description='LLaVA-Med-v1.5 / Qwen2.5-VL batch inference (multi-GPU)',
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
    parser.add_argument('--model_path', type=str, default="LLaVA-Med-v1.5",
                        help='Model path')
    parser.add_argument('--prompt_type', type=str, default='no_cot',
                        help='Prompt type: cot or no_cot')
    
    args = parser.parse_args()
    
    print(f"model_path={args.model_path}")
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
        
        tasks.append((sample_id, image_list, ground_truth, args.prompt_type, args.output))
    
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
        init_worker(gpu_ids[0], args.model_path)
        
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
                initargs=(gpu_assignments[i], args.model_path)
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