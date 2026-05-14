import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Process, Queue
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import torch
import multiprocessing as mp


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
    Args:
        image_list: list of image paths
        ground_truth: dict with findings and impression
        prompt_type: cot, no_cot, or cot_with_bbox
        processor: Qwen processor for chat template

    Returns:
        messages: chat messages for the model
    """
    from config import (
        system_prompt_eng, eng_instruction,
        system_prompt_zh, zh_instruction,
        no_cot_eng_prompt, no_cot_zh_prompt,
        system_prompt_zh_with_bbox, system_prompt_eng_with_bbox
    )
    
    is_zh = (judge_language_zh_or_eng(ground_truth.get("findings", "")) == "zh" and 
             judge_language_zh_or_eng(ground_truth.get("impression", "")) == "zh")
    
    # Build image entries (paths, pixel limits)
    image_content = []
    for image_path in image_list:
        image_content.append({
            "type": "image",
            "image": {"image_path": image_path},
        })
    
    if prompt_type == "cot":
        if is_zh:
            system_prompt = system_prompt_zh
            instruction = zh_instruction
        else:
            system_prompt = system_prompt_eng
            instruction = eng_instruction
        
        messages = [
            {"role": "user", "content": image_content + [{"type": "text", "text": system_prompt + instruction}]}
        ]
        
    elif prompt_type == "no_cot":
        if is_zh:
            instruction = no_cot_zh_prompt
        else:
            instruction = no_cot_eng_prompt
        
        messages = [
            {"role": "user", "content": image_content + [{"type": "text", "text": instruction}]}
        ]
        
    elif prompt_type == "cot_with_bbox":
        if is_zh:
            system_prompt = system_prompt_zh_with_bbox
            instruction = zh_instruction
        else:
            system_prompt = system_prompt_eng_with_bbox
            instruction = eng_instruction
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_content + [{"type": "text", "text": instruction}]}
        ]
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    
    return messages



def prepare_batch_inputs(
    batch_samples: List[Dict],
    prompt_type: str,
    output_dir: str,
    processor
) -> Tuple[List[Dict], List[Dict], int, int, List[Tuple[str, str]]]:
    """
    Build batched multimodal inputs for vLLM.

    Returns:
        batch_inputs, batch_info, skip_count, fail_count, failed_samples
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
        
        output_file = os.path.join(output_dir, f"{sample_id}.json")
        
        # Skip if output exists
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
            messages = build_prompt_and_images(image_list, ground_truth, prompt_type, processor)
            
            batch_inputs.append(messages)
            batch_info.append({
                "sample_id": sample_id,
                "image_list": image_list,
                "ground_truth": ground_truth,
                "output_file": output_file
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
    num_samples: int = 1
) -> Tuple[int, int, List[Tuple[str, str]]]:
    """
    Write inference JSON files.

    Args:
        outputs: vLLM outputs
        batch_info: per-sample metadata
        batch_inference_time: wall time for the batch (seconds)
        num_samples: samples per prompt
    """
    success_count = 0
    fail_count = 0
    failed_samples = []
    
    avg_inference_time = batch_inference_time / len(outputs) if outputs else 0
    
    for output, info in zip(outputs, batch_info):
        try:
            # First output (num_samples==1)
            output_text = outputs[0]
            
            result = {
                "sample_id": info["sample_id"],
                "images": info["image_list"],
                "ground_truth": info["ground_truth"],
                "output": output_text,
                "inference_time": avg_inference_time,
            }
            
            # If num_samples > 1, store all outputs
            if num_samples > 1:
                output_k_samples = []
                for i, sample_output in enumerate(output):
                    output_k_samples.append({
                        "text": sample_output,
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


def batch_hulu_infer(all_inputs, processor, model):
    outputs = []
    for input in tqdm(all_inputs, desc="Processing", unit="sample"):
        conversation = input

        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        output_ids = model.generate(**inputs, max_new_tokens=2048)
        output = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            use_think=False
        )[0].strip()
        outputs.append(output)
    
    return outputs


def run_single_gpu(samples: List[Dict], args):
    
    # CUDA_VISIBLE_DEVICES
    gpu_ids = list(dict.fromkeys([gid.strip() for gid in args.gpu_ids.split(',')]))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
    
    print(f"\nSingle-GPU mode, GPU {gpu_ids[0]}")
    print("Initializing vLLM and processor...")
    
    init_start = time.time()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = processor.tokenizer
    
    # Init time
    init_time = time.time() - init_start
    print(f"Model ready in {init_time:.2f}s")

    # Build inputs
    print("\nPreparing inputs...")
    all_inputs = []
    all_info = []
    total_skip = 0
    total_fail = 0
    all_failed_samples = []
    

    # Chunked prep (memory)
    num_batches = (len(samples) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="prepare"):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]
        
        batch_inputs, batch_info, skip, fail, failed = prepare_batch_inputs(
            batch_samples, args.prompt_type, args.output, processor
        )
        
        all_inputs.extend(batch_inputs)
        all_info.extend(batch_info)
        total_skip += skip
        total_fail += fail
        all_failed_samples.extend(failed)
    
    if not all_inputs:
        print("No samples to process")
        return 0, total_skip, total_fail, all_failed_samples, 0
    
    print(f"\nPrepared {len(all_inputs)} samples, skip={total_skip}, prep_fail={total_fail}")

    # Batch generate
    print(f"\nRunning inference...")
    print("-" * 60)
    
    start_time = time.time()
    
    outputs = batch_hulu_infer(all_inputs, processor, model)
    
    batch_inference_time = time.time() - start_time
    
    print("\nSaving results...")
    success_count, fail_count, failed = save_results(outputs, all_info, batch_inference_time, args.num_samples)
    
    total_fail += fail_count
    all_failed_samples.extend(failed)
    
    elapsed_time = time.time() - start_time
    
    return success_count, total_skip, total_fail, all_failed_samples, elapsed_time


def worker_process(
    gpu_id: str,
    worker_id: int,
    samples: List[Dict],
    args,
    result_queue: Queue
):
    """
    Worker process (one GPU).
    """
    # CUDA_VISIBLE_DEVICES (this process only)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # Torch sees cuda:0 as local GPU
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    except Exception:
        pass
    
    print(f"[Worker {worker_id}] init vLLM on GPU {gpu_id}, samples={len(samples)}")
    try:
        # Load processor
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        tokenizer = processor.tokenizer
        
        # Map model to local cuda:0
        try:
            if torch.cuda.is_available():
                model.to(torch.device('cuda:0'))
        except Exception:
            pass

        # Build inputs
        all_inputs, all_info, total_skip, total_fail, all_failed_samples = prepare_batch_inputs(
            samples, args.prompt_type, args.output, processor
        )
        if not all_inputs:
            result_queue.put({
                "type": "done",
                "worker_id": worker_id,
                "success": 0,
                "skip": total_skip,
                "fail": total_fail,
                "failed_samples": all_failed_samples
            })
            return
        
        # Progress
        result_queue.put({
            "type": "progress",
            "worker_id": worker_id,
            "message": f"ready, {len(all_inputs)} samples pending"
        })
        start_time = time.time()
        outputs = batch_hulu_infer(all_inputs, processor, model)
        batch_inference_time = time.time() - start_time
        
        # Save JSON
        success_count, fail_count, failed = save_results(outputs, all_info, batch_inference_time, args.num_samples)
        
        total_fail += fail_count
        all_failed_samples.extend(failed)
        
        # Done
        result_queue.put({
            "type": "done",
            "worker_id": worker_id,
            "success": success_count,
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
    Multi-GPU: one model instance per GPU, data parallel.
    """
    # Deduplicate GPU ids
    gpu_ids = list(dict.fromkeys([gid.strip() for gid in args.gpu_ids.split(',')]))
    n_gpus = len(gpu_ids)
    
    print(f"\nMulti-GPU data parallel: {n_gpus} GPUs {gpu_ids}")
    # Shard data
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
        
        # Temporarily set CUDA_VISIBLE_DEVICES for child processes
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        p = Process(
            target=worker_process,
            args=(gpu_id, i, samples_per_gpu[i], args, result_queue)
        )
        p.start()
        processes.append(p)
        
        # Restore parent env
        if prev is None:
            try:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            except KeyError:
                pass
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev

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
    
    # Use spawn (CUDA)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Ignore if already set
        pass

    parser = argparse.ArgumentParser(
        description='Hulu-Med vLLM batch inference',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', default="EasyEval/samples.json", type=str,
                        help='Input JSON path')
    parser.add_argument('-o', '--output', default='EasyEval/samples_test', type=str,
                        help='Output directory')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='Comma-separated GPU ids, deduplicated (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for prep only (default: 1)')
    parser.add_argument('--model_path', default = 'hulu-med-32B', type=str,
                        help='Model path')
    # Example paths (internal):
    # Volcengine: .../Hulu-med
    # Gui'an: .../Hulu-med
    parser.add_argument('--prompt_type', type=str, default='no_cot',
                        choices=['cot', 'no_cot', 'cot_with_bbox'],
                        help='Prompt type (default: no_cot)')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Tensor parallel size, single-GPU only (default: 1)')
    parser.add_argument('--max_model_len', type=int, default=20480,
                        help='Max model length (default: 20480)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory fraction (default: 0.9)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip existing output JSON files')
    # Sampling
    parser.add_argument('--max_tokens', type=int, default=4096,
                        help='Max new tokens (default: 4096)')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature (default: 0.6)')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p (default: 0.95)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Samples per prompt (default: 1)')
    args = parser.parse_args()
    
    # Deduplicate GPU ids
    gpu_ids = list(dict.fromkeys([gid.strip() for gid in args.gpu_ids.split(',')]))
    n_gpus = len(gpu_ids)
    
    print("=" * 60)
    print("Hulu-Med vLLM batch inference")
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
    
    # Filter already processed
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
    # Failure log
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
    
if __name__ == "__main__":
    main()