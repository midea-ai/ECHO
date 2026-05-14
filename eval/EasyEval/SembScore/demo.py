from worker import SEMBScoreMetric
import numpy as np
import json
from tqdm import tqdm

if __name__ == '__main__':

    model_path = "/data/share/250911/models/sembscore"
    src_path = "/data/hanxiao36/projects/RaTEScore/wandong_test/7b_en.json"

    with open(src_path, "r", encoding='utf-8') as f:
        datas = json.load(f)

    reports, preds = [], []

    for sample in tqdm(datas):
        response = sample["response"]
        if response == "":
            continue
        findings = sample["findings"]
        impression = sample["impression"]
        golden = f"Findings: {findings} Impression: {impression}."
        reports.append(golden)
        preds.append(response)

    metric = SEMBScoreMetric(model_path=model_path)
    results = metric.compute_rewards(preds, reports)
    avg = np.mean(np.array(results))
    print(f"SEMBScoreMetric results: {avg:.4f}")
