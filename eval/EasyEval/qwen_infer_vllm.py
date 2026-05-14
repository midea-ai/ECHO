import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Process, Queue
from PIL import Image

from tqdm import tqdm


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


def build_prompt_and_images(
    image_list: List[str], 
    ground_truth: Dict, 
    prompt_type: str,
    processor
) -> Tuple[str, List]:
    """
    Build vLLM multimodal prompt and image inputs.
    
    Args:
        image_list: list of image paths
        ground_truth: dict with findings and impression
        prompt_type: cot, no_cot, or cot_with_bbox
        processor: Qwen processor for chat template
    
    Returns:
        prompt: formatted prompt string
        image_inputs: processed vision inputs
    """
    from config import (
        no_cot_eng_prompt, no_cot_zh_prompt,
    )
    from qwen_vl_utils import process_vision_info
    
    if 'qa' in prompt_type:
        is_zh = judge_language_zh_or_eng(ground_truth.get("question", "")) == "zh" 
    else:
        is_zh = (judge_language_zh_or_eng(ground_truth.get("findings", "")) == "zh" and 
                 judge_language_zh_or_eng(ground_truth.get("impression", "")) == "zh")
    
    # Build image entries (paths, pixel limits)
    image_content = []
    for image_path in image_list:
        image_content.append({
            "type": "image",
            "image": image_path,
            "max_pixels": 1500**2,
            "min_pixels": 256*28*28,
        })
    
    if prompt_type == "no_cot":
        if is_zh:
            instruction = no_cot_zh_prompt
        else:
            instruction = no_cot_eng_prompt
        
        messages = [
            {"role": "user", "content": image_content + [{"type": "text", "text": instruction}]}
        ]
        
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    # Apply chat template
    prompt = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # process_vision_info (respects max/min pixels)
    image_inputs, video_inputs = process_vision_info(messages)
    
    return prompt, image_inputs


def read_sample_list(input_json: str) -> List[Dict]:
    """
    Load sample list from JSON.
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


def prepare_batch_inputs(
    batch_samples: List[Dict],
    prompt_type: str,
    output_dir: str,
    processor
) -> Tuple[List[Dict], List[Dict], int, int, List[Tuple[str, str]]]:
    """
    Prepare batched inputs for vLLM.
    
    Returns:
        batch_inputs: list of vLLM inputs
        batch_info: per-sample metadata
        skip_count: skipped count
        fail_count: failure count
        failed_samples: failed sample ids
    """
    batch_inputs = []
    batch_info = []
    skip_count = 0
    fail_count = 0
    failed_samples = []
    
    for sample in batch_samples:
        sample_id = sample["sample_id"]
        image_list = sample["images"]
        ground_truth = sample["ground_truth"]
        qa_type = sample["qa_type"] if "qa_type" in sample else None
        
        output_file = os.path.join(output_dir, f"{sample_id}.json")
        
        # Skip if done
        if os.path.exists(output_file):
            skip_count += 1
            continue
        
        # Validate image paths
        valid = True
        for image_path in image_list:
            if not os.path.exists(image_path):
                fail_count += 1
                failed_samples.append((sample_id, f"File not found: {image_path}"))
                valid = False
                break
        
        if not valid:
            continue
        
        try:
            prompt, images = build_prompt_and_images(image_list, ground_truth, prompt_type, processor)
            
            # vLLM multimodal input dict
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": images}
            })
            batch_info.append({
                "sample_id": sample_id,
                "image_list": image_list,
                "ground_truth": ground_truth,
                "output_file": output_file,
                "qa_type": qa_type
            })
        except Exception as e:
            import traceback
            fail_count += 1
            failed_samples.append((sample_id, f"{str(e)}\n{traceback.format_exc()}"))
    
    return batch_inputs, batch_info, skip_count, fail_count, failed_samples


def save_results(
    outputs,
    batch_info: List[Dict],
    batch_inference_time: float,
    num_samples: int = 1,
    prompt_type: Optional[str] = None
) -> Tuple[int, int, List[Tuple[str, str]]]:
    """
    Save inference outputs to JSON.
    
    Args:
        outputs: vLLM RequestOutput objects
        batch_info: per-sample metadata
        batch_inference_time: batch time (s)
        num_samples: samples per prompt
        prompt_type: cot / no_cot / ...
    """
    success_count = 0
    fail_count = 0
    failed_samples = []
        
    avg_inference_time = batch_inference_time / len(outputs) if outputs else 0
    
    for output, info in zip(outputs, batch_info):
        try:
            # First output (num_samples==1)
            output_text = output.outputs[0].text
            
            # Token counts from first output
            input_tokens = len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0
            output_tokens = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else 0
            
            result = {
                "sample_id": info["sample_id"],
                "images": info["image_list"],
                "ground_truth": info["ground_truth"],
                "output": output_text,
                "inference_time": avg_inference_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }

            if prompt_type and 'qa' in prompt_type:
                result["qa_type"] = info["qa_type"]
            
            # If num_samples>1, add all samples
            if num_samples > 1:
                output_k_samples = []
                for i, sample_output in enumerate(output.outputs):
                    sample_tokens = len(sample_output.token_ids) if hasattr(sample_output, 'token_ids') else 0
                    output_k_samples.append({
                        "text": sample_output.text,
                        "output_tokens": sample_tokens
                    })
                result["output_k_samples"] = output_k_samples
            
            with open(info["output_file"], 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            success_count += 1
        except Exception as e:
            import traceback
            fail_count += 1
            failed_samples.append((info["sample_id"], f"{str(e)}\n{traceback.format_exc()}")) 
    
    return success_count, fail_count, failed_samples


def run_single_gpu(samples: List[Dict], args):
    """
    Single-GPU mode.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    
    # CUDA_VISIBLE_DEVICES
    gpu_ids = list(dict.fromkeys([gid.strip() for gid in args.gpu_ids.split(',')]))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
    
    print(f"\nSingle-GPU mode, using GPU {gpu_ids[0]}")
    print("Initializing vLLM and processor...")
    
    init_start = time.time()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Init vLLM engine
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 10},
    )
    
    init_time = time.time() - init_start
    print(f"vLLM ready in {init_time:.2f}s")
    
    # Sampling params
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.0,
        n=args.num_samples,
    )
    
    if args.num_samples > 1:
        print(f"num_samples={args.num_samples}")
    
    # Chunked batches to limit memory
    print("\nProcessing in batches...")
    total_success = 0
    total_skip = 0
    total_fail = 0
    all_failed_samples = []

    # Batch loop
    num_batches = (len(samples) + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    print(f"{len(samples)} samples in {num_batches} batches (batch_size={args.batch_size})")
    print("-" * 60)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]

        # Build batch inputs
        batch_inputs, batch_info, skip, fail, failed = prepare_batch_inputs(
            batch_samples, args.prompt_type, args.output, processor
        )

        total_skip += skip
        total_fail += fail
        all_failed_samples.extend(failed)

        if not batch_inputs:
            print(f"Batch {batch_idx + 1}/{num_batches}: no valid inputs, skip")
            continue

        print(f"Batch {batch_idx + 1}/{num_batches}: {len(batch_inputs)} samples")

        # Run batch
        batch_start_time = time.time()
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        batch_inference_time = time.time() - batch_start_time

        # Save batch outputs
        success_count, fail_count, failed = save_results(outputs, batch_info, batch_inference_time, args.num_samples, args.prompt_type)

        total_success += success_count
        total_fail += fail_count
        all_failed_samples.extend(failed)

        print(f"Batch {batch_idx + 1}/{num_batches} done: ok={success_count}, time={batch_inference_time:.2f}s")

    elapsed_time = time.time() - start_time

    print("-" * 60)
    print(f"Totals: success={total_success}, skip={total_skip}, fail={total_fail}")

    return total_success, total_skip, total_fail, all_failed_samples, elapsed_time


def worker_process(
    gpu_id: str,
    worker_id: int,
    samples: List[Dict],
    args,
    result_queue: Queue
):
    """
    Worker: one GPU.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    
    # CUDA_VISIBLE_DEVICES (this process only)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    print(f"[Worker {worker_id}] init vLLM on GPU {gpu_id}, samples={len(samples)}")
    
    try:
        # Load processor
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        
        # Init vLLM engine
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=1,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 10},
        )
        
        # Sampling params
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.0,
            n=args.num_samples,
        )
        
        print(f"[Worker {worker_id}] vLLM ready, num_samples={args.num_samples}")
        
        # Batch loop (limit memory)
        total_success = 0
        total_skip = 0
        total_fail = 0
        all_failed_samples = []
        
        num_batches = (len(samples) + args.batch_size - 1) // args.batch_size
        
        result_queue.put({
            "type": "progress",
            "worker_id": worker_id,
            "message": f"{len(samples)} samples, {num_batches} batches (batch_size={args.batch_size})"
        })
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]
            
            # Build batch
            batch_inputs, batch_info, skip, fail, failed = prepare_batch_inputs(
                batch_samples, args.prompt_type, args.output, processor
            )
            
            total_skip += skip
            total_fail += fail
            all_failed_samples.extend(failed)
            
            if not batch_inputs:
                continue
            
            # Generate
            start_time = time.time()
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
            batch_inference_time = time.time() - start_time
            # Save
            success_count, fail_count, failed = save_results(outputs, batch_info, batch_inference_time, args.num_samples, args.prompt_type)
            
            total_success += success_count
            total_fail += fail_count
            all_failed_samples.extend(failed)
            
            # Progress queue
            result_queue.put({
                "type": "progress",
                "worker_id": worker_id,
                "message": f"batch {batch_idx + 1}/{num_batches} ok={success_count}"
            })
        
        # Final result
        result_queue.put({
            "type": "done",
            "worker_id": worker_id,
            "success": total_success,
            "skip": total_skip,
            "fail": total_fail,
            "failed_samples": all_failed_samples
        })
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        result_queue.put({
            "type": "error",
            "worker_id": worker_id,
            "error": error_msg,
            "failed_samples": [(s["sample_id"], error_msg) for s in samples]
        })


def run_multi_gpu(samples: List[Dict], args):
    """
    Multi-GPU: one vLLM per GPU, data parallel.
    """
    # Deduplicate GPU ids
    gpu_ids = list(dict.fromkeys([gid.strip() for gid in args.gpu_ids.split(',')]))
    n_gpus = len(gpu_ids)
    
    print(f"\nMulti-GPU data parallel: {n_gpus} GPUs {gpu_ids}")
    
    # Shard dataset
    samples_per_gpu = []
    for i in range(n_gpus):
        gpu_samples = samples[i::n_gpus]  # round-robin
        samples_per_gpu.append(gpu_samples)
        print(f"  GPU {gpu_ids[i]}: {len(gpu_samples)} samples")
    
    # Result queue
    result_queue = Queue()
    
    # Spawn workers
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = Process(
            target=worker_process,
            args=(gpu_id, i, samples_per_gpu[i], args, result_queue)
        )
        p.start()
        processes.append(p)
    
    print(f"\nStarted {n_gpus} workers")
    print("-" * 60)
    
    # Collect results
    start_time = time.time()
    total_success = 0
    total_skip = 0
    total_fail = 0
    all_failed_samples = []
    workers_done = 0
    
    while workers_done < n_gpus:
        result = result_queue.get()
        
        if result["type"] == "progress":
            print(f"[Worker {result['worker_id']}] {result.get('message', '')}")
            
        elif result["type"] == "done":
            workers_done += 1
            total_success += result["success"]
            total_skip += result["skip"]
            total_fail += result["fail"]
            all_failed_samples.extend(result["failed_samples"])
            print(f"[Worker {result['worker_id']}] done: ok={result['success']}, skip={result['skip']}, fail={result['fail']}")
            
        elif result["type"] == "error":
            workers_done += 1
            total_fail += len(result.get("failed_samples", []))
            all_failed_samples.extend(result.get("failed_samples", []))
            print(f"[Worker {result['worker_id']}] error: {result['error'][:200]}")
    
    # Join workers
    for p in processes:
        p.join()
    
    elapsed_time = time.time() - start_time
    
    return total_success, total_skip, total_fail, all_failed_samples, elapsed_time


def main():
    parser = argparse.ArgumentParser(
        description='Qwen2.5-VL vLLM batch inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU
  python %(prog)s -i samples.json -o output_dir --gpu_ids 0 --batch_size 32 --model_path /path/to/model
  
  # Multi-GPU data parallel (one vLLM per GPU)
  python %(prog)s -i samples.json -o output_dir --gpu_ids 0,1,2,3 --batch_size 32 --model_path /path/to/model

Input JSON format:
  [
    {
      "sample_id": "sample_001",
      "images": ["path/to/image1.png", "path/to/image2.png"],
      "ground_truth": {"findings": "...", "impression": "..."}
    }
  ]
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input JSON file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='Comma-separated GPU ids, deduplicated (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data prep only; vLLM batches internally (default: 32)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Model path')
    parser.add_argument('--prompt_type', type=str, default='cot',
                        choices=['cot', 'no_cot', 'cot_with_bbox', 'qa_cot', 'qa_no_cot'],
                        help='Prompt type (default: cot)')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Tensor parallel size (single-GPU mode only, default: 1)')
    parser.add_argument('--max_model_len', type=int, default=20480,
                        help='Max model length (default: 8192)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization (default: 0.9)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip samples that already have an output file')
    
    # Sampling params
    parser.add_argument('--max_tokens', type=int, default=4096,
                        help='Max new tokens (default: 4096)')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature (default: 0.0)')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p (default: 0.95)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples per prompt (default: 1)')
    
    args = parser.parse_args()
    
    # Deduplicate GPU ids
    gpu_ids = list(dict.fromkeys([gid.strip() for gid in args.gpu_ids.split(',')]))
    n_gpus = len(gpu_ids)
    
    print("=" * 60)
    print("Qwen2.5-VL vLLM batch inference")
    print("=" * 60)
    print(f"model_path={args.model_path}")
    print(f"GPUs (deduped): {gpu_ids}")
    print(f"prompt_type={args.prompt_type}")
    print("=" * 60)
    
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
    
    # Filter already-done samples
    if args.skip_existing:
        filtered_samples = []
        for sample in samples:
            output_file = os.path.join(args.output, f"{sample['sample_id']}.json")
            if not os.path.exists(output_file):
                filtered_samples.append(sample)
        skipped = len(samples) - len(filtered_samples)
        print(f"Skipped {skipped} existing outputs")
        samples = filtered_samples
    
    if not samples:
        print("All samples already processed")
        return
    
    print(f"Samples to process: {len(samples)}")
    
    # Single vs multi-GPU
    if n_gpus == 1:
        total_success, total_skip, total_fail, all_failed_samples, elapsed_time = run_single_gpu(samples, args)
    else:
        total_success, total_skip, total_fail, all_failed_samples, elapsed_time = run_multi_gpu(samples, args)
    
    # Write failure log
    error_log_path = os.path.join(args.output, 'failed_samples.log')
    if all_failed_samples:
        try:
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write(f"# Failed samples ({len(all_failed_samples)})\n")
                f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for sample_id, error_msg in all_failed_samples:
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
    print(f"Total samples: {len(samples)}")
    print(f"Success: {total_success}")
    print(f"Skipped (exists): {total_skip}")
    print(f"Failed: {total_fail}")
    if total_fail > 0 and len(samples) > 0:
        print(f"Fail rate: {total_fail/len(samples)*100:.2f}%")
    print(f"Elapsed: {elapsed_time:.2f}s")
    if total_success > 0:
        print(f"Throughput: {total_success / elapsed_time:.2f} samples/s")
    print(f"Output dir: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
