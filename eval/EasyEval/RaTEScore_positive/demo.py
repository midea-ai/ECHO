import argparse
import json
from tqdm import tqdm

# RaTEScore
from RaTEScore import RaTEScore


def aggregate_values(values, mode="max"):
    if not values:
        return None
    if mode == "max":
        return max(values)
    return sum(values) / len(values)


# RaTEScore models
rate_bert_path = "Angelakeke/RaTE-NER-Deberta"
rate_eval_path = "FremyCompany/BioLORD-2023-C"
rate_scorer = RaTEScore(
    bert_model=rate_bert_path,
    eval_model=rate_eval_path,
    visualization_path="./rate_score.json",
    batch_size=32
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-s","--src_json", type=str, required=True)
    # parser.add_argument("-d","--dst_json", type=str, required=True)
    parser.add_argument("-s", "--src_json", type=str, default="EasyEval/test_EN_filtered_all_results.json")
    parser.add_argument("-d", "--dst_json", type=str, default="EasyEval/test_EN_filtered_all_results_metrics.json")
    parser.add_argument("--k_aggregate_mode", type=str, default="max", choices=["max", "mean"],
                        help="K-sample aggregation: max or mean (default: max)")
    args = parser.parse_args()

    src_path = args.src_json
    dst_path = args.dst_json
    k_aggregate_mode = args.k_aggregate_mode

    with open(src_path, "r", encoding='utf-8') as f:
        datas = json.load(f)

    # Batched inputs for RaTEScore
    eng_golden_list = []
    eng_response_list = []
    # Per-sample index into batch lists
    sample_batch_info = []  # [(start_idx, count), ...]

    print(f"K aggregation: {k_aggregate_mode}")
    print("=" * 60)

    # Build batch for RaTEScore
    for sample in tqdm(datas, desc="RaTEScore"):
        sample["metrics"] = {}

        response = sample.get("response", "")
        if response == "":
            sample_batch_info.append((len(eng_response_list), 0))
            continue

        has_k_samples = "output_k_response" in sample and len(sample.get("output_k_response", [])) > 0

        if has_k_samples:
            k_eng_responses = sample.get("output_k_eng_response", [])
            batch_start_idx = len(eng_response_list)
            added_count = 0

            for k_idx, k_resp_raw in enumerate(sample.get("output_k_response", [])):
                if k_resp_raw == "":
                    continue
                if k_idx < len(k_eng_responses):
                    eng_response_list.append(k_eng_responses[k_idx])
                else:
                    eng_response_list.append(sample.get("eng_response", ""))
                eng_golden_list.append(sample.get("eng_golden", ""))
                added_count += 1

            sample_batch_info.append((batch_start_idx, added_count))
        else:
            eng_response_list.append(sample.get("eng_response", ""))
            eng_golden_list.append(sample.get("eng_golden", ""))
            sample_batch_info.append((len(eng_response_list) - 1, 1))

    print("Computing RaTE...")
    if eng_response_list:
        _ = rate_scorer.compute_score(eng_golden_list, eng_response_list)
    rate_scores = []
    with open("./rate_score.json", "r", encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rate_scores.append(json.loads(line)["rate_score"])

    print("Merging metrics...")
    for i, sample in enumerate(tqdm(datas, desc="assign")):
        start_idx, count = sample_batch_info[i]
        if count == 0:
            continue

        if count == 1:
            sample["metrics"]["rate"] = rate_scores[start_idx]
        else:
            k_rate = rate_scores[start_idx:start_idx + count]
            sample["metrics"]["rate"] = aggregate_values(k_rate, mode=k_aggregate_mode)

            # Keep per-k detail
            sample["k_metrics_detail"] = sample.get("k_metrics_detail", [{} for _ in range(count)])
            for k_idx in range(min(count, len(sample["k_metrics_detail"]))):
                sample["k_metrics_detail"][k_idx]["rate"] = k_rate[k_idx]

    # Dataset mean
    rate_values = [s["metrics"]["rate"] for s in datas if s.get("metrics") and "rate" in s["metrics"]]
    metrics_all = {"rate": sum(rate_values) / len(rate_values)} if rate_values else {}
    
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


