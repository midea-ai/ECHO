import glob
import argparse
import os
import json
from tqdm import tqdm
from chatbot.prompts import PROMPT_EN



def extract_findings_and_impression_from_gt(data):
    findings = data["ground_truth"]["findings"]
    impression = data["ground_truth"]["impression"]
    return findings, impression

def extract_findings_and_impression_from_response(data):
    """
    Extract findings/impression from model response via regex.
    """
    import re
    response = data.get("output", "")
    # findings
    findings_match = re.search(r"<findings>\s*(.*?)\s*</findings>", response, re.DOTALL)
    findings = findings_match.group(1).strip() if findings_match else ""
    # impression
    impression_match = re.search(r"<impression>\s*(.*?)\s*</impression>", response, re.DOTALL)
    impression = impression_match.group(1).strip() if impression_match else ""
    return findings, impression


def extract_findings_and_impression_from_text(text):
    """
    Parse findings/impression from text (e.g. output_k_samples).
    Supports <findings> tags and no_cot layouts.
    """
    import re
    
    # COT: <findings>...</findings> <impression>...</impression>
    findings_match = re.search(r"<findings>\s*(.*?)\s*</findings>", text, re.DOTALL)
    impression_match = re.search(r"<impression>\s*(.*?)\s*</impression>", text, re.DOTALL)
    
    if findings_match and impression_match:
        findings = findings_match.group(1).strip()
        impression = impression_match.group(1).strip()
        return findings, impression
    
    # English no_cot: Findings: ... Impression: ...
    findings_match = re.search(r"Findings:\s*(.*?)\s*Impression:", text, re.DOTALL | re.IGNORECASE)
    impression_match = re.search(r"Impression:\s*(.*?)$", text, re.DOTALL | re.IGNORECASE)
    
    if findings_match and impression_match:
        findings = findings_match.group(1).strip()
        impression = impression_match.group(1).strip()
        return findings, impression
    
    # Chinese no_cot layout (regex captures Chinese section labels)
    findings_match_zh = re.search(r"所见：\s*(.*?)\s*结论：", text, re.DOTALL)
    impression_match_zh = re.search(r"结论：\s*(.*?)$", text, re.DOTALL)
    
    if findings_match_zh and impression_match_zh:
        findings = findings_match_zh.group(1).strip()
        impression = impression_match_zh.group(1).strip()
        return findings, impression
    
    # No match -> empty
    return "", ""


def extract_findings_and_impression_from_response_no_cot(data):
    """
    Extract findings/impression from model response via regex.
    Supports Chinese (所见/结论) and English Findings/Impression formats.
    """
    import re
    response = data.get("output", "")
    
    # English no_cot pattern
    findings_match = re.search(r"Findings:\s*(.*?)\s*Impression:", response, re.DOTALL | re.IGNORECASE)
    impression_match = re.search(r"Impression:\s*(.*?)$", response, re.DOTALL | re.IGNORECASE)
    
    if findings_match and impression_match:
        findings = findings_match.group(1).strip()
        impression = impression_match.group(1).strip()
        return findings, impression
    
    # Chinese no_cot pattern
    findings_match_zh = re.search(r"所见：\s*(.*?)\s*结论：", response, re.DOTALL)
    impression_match_zh = re.search(r"结论：\s*(.*?)$", response, re.DOTALL)
    
    if findings_match_zh and impression_match_zh:
        findings = findings_match_zh.group(1).strip()
        impression = impression_match_zh.group(1).strip()
        return findings, impression    
    # No match -> empty
    return "",""


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


def translation_to_english(data):
    response = data["response"]
    response = baichuan_llm.chat(response)
    return response


def load_and_extract_single_file(json_file):
    """
    Load one JSON file; return (data, need_translation) or None.
    Returns (data_dict, need_translation) or None.
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
            pred_findings, pred_impression = extract_findings_and_impression_from_response(data)

            if pred_findings=="" and pred_impression=="":
                pred_findings, pred_impression = extract_findings_and_impression_from_response_no_cot(data)
            raw_output = data.get("output", "").strip()
            parse_failed = (pred_findings == "" and pred_impression == "" and raw_output != "")

            gt_findings, gt_impression = extract_findings_and_impression_from_gt(data)
            language = judge_language_zh_or_eng(gt_findings+gt_impression)

            result = {
                "sample_id": data["sample_id"],
                "images": data["images"],
                "pred_findings": pred_findings,
                "pred_impression": pred_impression,
                "gt_findings": gt_findings,
                "gt_impression": gt_impression,
                "language": language,
                "output": data["output"],
                "parse_failed": parse_failed,
            }
            
            # Optional output_k_samples
            if "output_k_samples" in data and len(data["output_k_samples"]) > 0:
                k_samples_extracted = []
                for sample in data["output_k_samples"]:
                    sample_text = sample.get("text", "")
                    sample_findings, sample_impression = extract_findings_and_impression_from_text(sample_text)
                    k_samples_extracted.append({
                        "findings": sample_findings,
                        "impression": sample_impression,
                        "text": sample_text,
                    })
                result["k_samples_extracted"] = k_samples_extracted
            
            # Translation flag
            need_translation = (language == "zh")
            return result, need_translation
            
    except Exception as e:
        print(f"Error processing {json_file}: {str(e)}")
        return None, False


def batch_process_files(json_files, baichuan_llm):
    """
    Batch translate / merge with vLLM.
    """
    # Step 1: load
    print("Step 1/3: load JSON files...")
    all_data = []
    zh_indices = []  # indices needing translation
    zh_texts = []    # texts to translate
    
    # Maps sample index to zh_texts slice
    # zh_text_positions: {data_idx: {main_start, main_count, k_start?, k_count?}}
    zh_text_positions = {}
    
    for json_file in tqdm(json_files, desc="load"):
        result, need_translation = load_and_extract_single_file(json_file)
        if result is not None:
            all_data.append(result)
            if need_translation:
                data_idx = len(all_data) - 1
                zh_indices.append(data_idx)
                
                # Record position
                position_info = {"main_start": len(zh_texts), "main_count": 2}
                
                # Predicted and reference text
                # On parse failure, translate raw output
                if result.get("parse_failed", False):
                    report = result["output"]
                else:
                    report = f"所见: {result['pred_findings']} 结论: {result['pred_impression']}"
                golden = f"所见: {result['gt_findings']} 结论: {result['gt_impression']}"
                zh_texts.extend([report, golden])
                
                # k_samples_extracted
                if "k_samples_extracted" in result:
                    position_info["k_start"] = len(zh_texts)
                    position_info["k_count"] = len(result["k_samples_extracted"])
                    for k_sample in result["k_samples_extracted"]:
                        k_report = f"所见: {k_sample['findings']} 结论: {k_sample['impression']}"
                        zh_texts.append(k_report)
                
                zh_text_positions[data_idx] = position_info
    
    print(f"Loaded {len(all_data)} files; {len(zh_indices)} need translation")
    
    # Step 2: translate
    translated_texts = []
    if zh_texts:
        print(f"Step 2/3: translating {len(zh_texts)} texts...")
        # vLLM batch
        translated_texts = baichuan_llm.batch_chat(zh_texts, PROMPT_EN)
    else:
        print("Step 2/3: skip (nothing to translate)")
    
    # Step 3: assemble
    print("Step 3/3: assemble results...")
    final_results = []
    
    for i, data in enumerate(tqdm(all_data, desc="assemble")):
        if i in zh_indices:
            # Translated branch
            pos_info = zh_text_positions[i]
            main_start = pos_info["main_start"]
            
            response_text = translated_texts[main_start]
            golden_text = translated_texts[main_start + 1]
            
            # Parse translated text
            finding, impression = baichuan_llm.extract_info(response_text)
            eng_response = f"Findings: {finding} Impression: {impression}"
            
            finding, impression = baichuan_llm.extract_info(golden_text)
            eng_golden = f"Findings: {finding} Impression: {impression}"
            
            # k_samples translations
            output_k_eng_response = None
            if "k_start" in pos_info:
                k_start = pos_info["k_start"]
                k_count = pos_info["k_count"]
                output_k_eng_response = []
                for k_idx in range(k_count):
                    k_translated = translated_texts[k_start + k_idx]
                    k_finding, k_impression = baichuan_llm.extract_info(k_translated)
                    output_k_eng_response.append(f"Findings: {k_finding} Impression: {k_impression}")
        else:
            # English branch
            eng_response = f"Findings: {data['pred_findings']} Impression: {data['pred_impression']}"
            eng_golden = f"Findings: {data['gt_findings']} Impression: {data['gt_impression']}"
            
            # English k_samples
            output_k_eng_response = None
            if "k_samples_extracted" in data:
                output_k_eng_response = []
                for k_sample in data["k_samples_extracted"]:
                    output_k_eng_response.append(f"Findings: {k_sample['findings']} Impression: {k_sample['impression']}")
        
        if data.get("parse_failed", False):
            response_text = data["output"]
            # Chinese branch: eng_response already English
            # English parse fail: use raw output
            if data.get("language") == "eng":
                eng_response = data["output"]
        else:
            response_text = f"Finding: {data['pred_findings']} Impression: {data['pred_impression']}"

        result = {
            "sample_id": data["sample_id"],
            "images": data["images"],
            "response": response_text,
            "findings": data["gt_findings"],
            "impression": data["gt_impression"],
            "language": data["language"],
            "eng_response": eng_response,
            "eng_golden": eng_golden,
            "output": data["output"],
        }
        
        # Optional k_samples fields
        if "k_samples_extracted" in data:
            output_k_response = []
            for k_sample in data["k_samples_extracted"]:
                output_k_response.append(f"Finding: {k_sample['findings']} Impression: {k_sample['impression']}")
            result["output_k_response"] = output_k_response
            result["output_k_eng_response"] = output_k_eng_response
        
        final_results.append(result)
    
    return final_results





if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i","--input_dir", type=str, required=True)
    argparser.add_argument("-o","--output_file", type=str, required=True)
    argparser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    argparser.add_argument("--batch_size", type=int, default=None, help="vLLM batch size (default: all at once)")
    args = argparser.parse_args()

    from chatbot.vllm_chatbot import ChatBot
    baichuan_path = "Baichuan-M2-32B"  # for vllm_chatbot
    baichuan_llm = ChatBot(model_name=baichuan_path, tensor_parallel_size=args.tensor_parallel_size)

    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    total_files = len(json_files)
    
    print("=" * 60)
    print(f"Processing {total_files} files")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print("Batch mode (avoid multi-thread CUDA issues)")
    print("=" * 60)
    
    # Batch pipeline
    all_datas = batch_process_files(json_files, baichuan_llm)
    
    fail_nums = total_files - len(all_datas)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("Done.")
    print(f"OK: {len(all_datas)} files")
    print(f"Failed: {fail_nums} files")
    print(f"Success rate: {len(all_datas)/total_files*100:.2f}%")
    print("=" * 60)
    
    # Save merged JSON
    print(f"\nSaving: {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(all_datas, f, indent=4, ensure_ascii=False)
    
    print("Done.")