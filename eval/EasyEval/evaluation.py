import argparse
import os
import os.path as osp
import sys
import numpy as np

# Set env and paths before other imports
os.environ['NLTK_DATA'] = 'your_path/nltk_data'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Extend sys.path for local packages
current_dir = os.path.dirname(os.path.abspath(__file__))
rate_score_path = os.path.join(current_dir, "RaTEScore_positive")
semb_score_path = os.path.join(current_dir, "SembScore")

from rouge import Rouge
from rouge_chinese import Rouge as Rouge_zh
from cidereval import cider, ciderD
from nltk.translate.meteor_score import single_meteor_score
import jieba

# RaTEScore (under RaTEScore_positive/RaTEScore/)
sys.path.insert(0, rate_score_path)
from RaTEScore import RaTEScore

# SembScore
sys.path.insert(0, semb_score_path)
from worker import SEMBScoreMetric

from tqdm import tqdm
import json
import ray

# Ray: workers need SembScore / RaTEScore on PYTHONPATH
if not ray.is_initialized():
    # PYTHONPATH for Ray workers
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_pythonpath = f"{semb_score_path}:{rate_score_path}:{current_pythonpath}" if current_pythonpath else f"{semb_score_path}:{rate_score_path}"
    
    ray.init(runtime_env={
        "env_vars": {
            "NLTK_DATA": "your_path/nltk_data",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": new_pythonpath
        }
    })

# RaTEScore models
rate_bert_path = "Angelakeke/RaTE-NER-Deberta"
rate_eval_path = "FremyCompany/BioLORD-2023-C"
rate_scorer = RaTEScore(bert_model=rate_bert_path, eval_model=rate_eval_path,
                   visualization_path="./rate_score.json",
                   batch_size=32)
# SembScore model
semb_model_path = "your_path/sembscore"
semb_scorer = SEMBScoreMetric(model_path=semb_model_path)

# ROUGE scorers
rouge_scorer = Rouge()
rouge_zh_scorer = Rouge_zh()

def chinese_tokenize(text):
    """Tokenize Chinese text with jieba."""
    return list(jieba.cut(text))


def prep_reports_eng(reports):
    """Preprocesses reports"""
    return [list(filter(
        lambda val: val != "", str(elem) \
            .lower().replace(".", " .").split(" "))) for elem in reports]


def prep_reports(reports, lang='zh'):
    """Tokenize reports (zh or en)."""
    processed = []
    for elem in reports:
        text = str(elem).lower()
        if lang == 'zh':
            # Chinese: jieba
            tokens = chinese_tokenize(text)
        else:
            # English: whitespace
            tokens = text.split()
        # Drop empty tokens
        tokens = [token for token in tokens if token.strip()]
        processed.append(tokens)
    return processed


def compute_rouge_meteor_for_single(response, golden, language, rouge_scorer, rouge_zh_scorer):
    """
    ROUGE + METEOR for one response.
    Returns (metrics_dict, rouge_response_for_cider, rouge_golden_for_cider).
    """
    if language == "zh":
        response = response.replace("Finding:", "所见：").replace("Impression:", "结论：")
    
    if language == "zh":
        tokenized_response = prep_reports([response.lower()])[0]
        tokenized_golden = prep_reports([golden.lower()])[0]
    else:
        tokenized_response = prep_reports_eng([response.lower()])[0]
        tokenized_golden = prep_reports_eng([golden.lower()])[0]
    
    metrics = {}
    
    if language == "zh":
        # Chinese: space-join for rouge_chinese
        rouge_response = " ".join(tokenized_response).lower()
        rouge_golden = " ".join(tokenized_golden).lower()
        rouge_scores = rouge_zh_scorer.get_scores(rouge_response, rouge_golden)
    else:
        rouge_response = response.lower()
        rouge_golden = golden.lower()
        rouge_scores = rouge_scorer.get_scores(rouge_response, rouge_golden)
    
    metrics["rouge-1"] = rouge_scores[0]["rouge-1"]["f"]
    metrics["rouge-2"] = rouge_scores[0]["rouge-2"]["f"]
    metrics["rouge-l"] = rouge_scores[0]["rouge-l"]["f"]
    
    meteor_score = single_meteor_score(hypothesis=tokenized_response, reference=tokenized_golden)
    metrics["meteor"] = meteor_score
    
    return metrics, rouge_response, rouge_golden


def aggregate_metrics(metrics_list, mode='max'):
    """
    Aggregate metrics over K samples.
    mode: 'max' or 'mean'
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    keys = metrics_list[0].keys()
    
    for key in keys:
        values = [m[key] for m in metrics_list]
        if mode == 'max':
            aggregated[key] = max(values)
        else:  # mean
            aggregated[key] = sum(values) / len(values)
    
    return aggregated



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-s","--src_json", type=str, required=True)
    # parser.add_argument("-d","--dst_json", type=str, required=True)
    parser.add_argument("-s","--src_json", type=str, default="EasyEval/test_EN_filtered_all_results.json")
    parser.add_argument("-d","--dst_json", type=str, default="EasyEval/test_EN_filtered_all_results_metrics.json")
    parser.add_argument("--k_aggregate_mode", type=str, default="max", choices=["max", "mean"],
                        help="K-sample aggregation: max or mean (default: max)")
    args = parser.parse_args()

    src_path = args.src_json
    dst_path = args.dst_json
    k_aggregate_mode = args.k_aggregate_mode

    with open(src_path, "r", encoding='utf-8') as f:
        datas = json.load(f)

    # Batched lists for metrics
    eng_golden_list = []
    eng_response_list = []
    cider_response_list = []
    cider_reference_list = []
    
    # Per-sample slice into batched lists: [(start_idx, count), ...]
    sample_batch_info = []

    print(f"K-sample aggregation: {k_aggregate_mode}")
    print("=" * 60)

    for sample in tqdm(datas, desc="rouge/meteor"):
        sample["metrics"] = {}
        
        response = sample["response"]
        language = sample["language"]
        findings = sample["findings"]
        impression = sample["impression"]
        
        if response == "":
            sample_batch_info.append((len(eng_response_list), 0))  # empty response
            continue
        
        # Reference string
        if language == "zh":
            golden = f"所见：{findings} 结论：{impression}"
        else:
            golden = f"Findings: {findings} Impression: {impression}."
        
        # K responses?
        has_k_samples = "output_k_response" in sample and len(sample.get("output_k_response", [])) > 0
        
        if has_k_samples:
            # K-response mode
            k_responses = sample["output_k_response"]
            k_eng_responses = sample.get("output_k_eng_response", [])
            
            k_metrics_list = []
            batch_start_idx = len(eng_response_list)
            
            for k_idx, k_response in enumerate(k_responses):
                if k_response == "":
                    continue
                    
                # ROUGE / METEOR
                metrics, rouge_response, rouge_golden = compute_rouge_meteor_for_single(
                    k_response, golden, language, rouge_scorer, rouge_zh_scorer
                )
                k_metrics_list.append(metrics)
                
                # Append for CIDEr, RaTE, SEMB
                cider_response_list.append(rouge_response)
                cider_reference_list.append([rouge_golden])
                
                # English text for RaTE / SEMB
                if k_idx < len(k_eng_responses):
                    eng_response_list.append(k_eng_responses[k_idx])
                else:
                    eng_response_list.append(sample["eng_response"])  # fallback
                eng_golden_list.append(sample["eng_golden"])
                
            # Record batch span
            k_count = len(k_metrics_list)
            sample_batch_info.append((batch_start_idx, k_count))
            
            # Aggregate ROUGE / METEOR
            if k_metrics_list:
                aggregated = aggregate_metrics(k_metrics_list, mode=k_aggregate_mode)
                sample["metrics"].update(aggregated)
                # Optional per-k detail
                sample["k_metrics_detail"] = k_metrics_list
        else:
            # Single response
            metrics, rouge_response, rouge_golden = compute_rouge_meteor_for_single(
                response, golden, language, rouge_scorer, rouge_zh_scorer
            )
            sample["metrics"].update(metrics)
            
            cider_response_list.append(rouge_response)
            cider_reference_list.append([rouge_golden])
            
            eng_golden_list.append(sample["eng_golden"])
            eng_response_list.append(sample["eng_response"])
            
            sample_batch_info.append((len(eng_response_list) - 1, 1))

    # CIDEr
    print("Computing CIDEr...")
    cider_scores = cider(predictions=cider_response_list, references=cider_reference_list)["scores"]

    # RaTE
    print("Computing RaTE...")
    avg_rate_score = rate_scorer.compute_score(eng_golden_list, eng_response_list)
    rate_scores = []
    with open("./rate_score.json", "r", encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rate_scores.append(json.loads(line)["rate_score"])

    # SEMB
    print("Computing SEMB...")
    semb_scores = semb_scorer.compute_rewards(eng_response_list, eng_golden_list)

    # Scatter batch scores back to samples
    print("Merging metrics...")
    for i, sample in enumerate(tqdm(datas, desc="assign")):
        if not sample["metrics"]:  # skip empty
            continue
            
        start_idx, count = sample_batch_info[i]
        
        if count == 0:
            continue
        elif count == 1:
            # Single response
            sample["metrics"]["cider"] = cider_scores[start_idx]
            sample["metrics"]["rate"] = rate_scores[start_idx]
            sample["metrics"]["semb"] = semb_scores[start_idx]
        else:
            # K-response: aggregate CIDEr, RaTE, SEMB
            k_cider = cider_scores[start_idx:start_idx + count]
            k_rate = rate_scores[start_idx:start_idx + count]
            k_semb = semb_scores[start_idx:start_idx + count]
            
            if k_aggregate_mode == "max":
                sample["metrics"]["cider"] = max(k_cider)
                sample["metrics"]["rate"] = max(k_rate)
                sample["metrics"]["semb"] = max(k_semb)
            else:  # mean
                sample["metrics"]["cider"] = sum(k_cider) / len(k_cider)
                sample["metrics"]["rate"] = sum(k_rate) / len(k_rate)
                sample["metrics"]["semb"] = sum(k_semb) / len(k_semb)
            
            # Per-k detail
            if "k_metrics_detail" in sample:
                for k_idx in range(count):
                    sample["k_metrics_detail"][k_idx]["cider"] = k_cider[k_idx]
                    sample["k_metrics_detail"][k_idx]["rate"] = k_rate[k_idx]
                    sample["k_metrics_detail"][k_idx]["semb"] = k_semb[k_idx]
    
    # Dataset means
    metrics_all = {}
    for sample in datas:
        if not sample.get("metrics"):
            continue
        for key, value in sample["metrics"].items():
            if key not in metrics_all:
                metrics_all[key] = []
            metrics_all[key].append(value)
    for key, value in metrics_all.items():
        metrics_all[key] = sum(value) / len(value)
    
    print("=" * 60)
    print(f"K aggregation mode: {k_aggregate_mode}")
    print(metrics_all)
    
    with open(dst_path, "w", encoding='utf-8') as f:
        json.dump({
            "metrics": metrics_all,
            "k_aggregate_mode": k_aggregate_mode,
            "datas": datas
        }, f, ensure_ascii=False, indent=4)
    
    print(f"Wrote: {dst_path}")






    

