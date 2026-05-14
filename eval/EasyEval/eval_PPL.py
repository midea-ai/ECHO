"""
Perplexity (PPL) over model responses in a merged JSON.
Loss applies only to response tokens; prompt positions use label -100.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class ResponsePPLScorer:
    """
    Compute response PPL with a causal LM.
    Concatenate prompt + response; mask prompt tokens in labels (-100); mean loss on response only.
    """

    def __init__(
        self,
        model_path: str = "Qwen3-1.7B",
        device: str = "cuda",
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, trust_remote_code=True
            )
            .to(device)
            .eval()
        )
        self._prompt_len_cache: dict[str, int] = {}

    def _get_prompt_len(self, prompt: str) -> int:
        if prompt not in self._prompt_len_cache:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            self._prompt_len_cache[prompt] = ids.shape[1]
        return self._prompt_len_cache[prompt]

    def compute(self, response: str, prompt: str = "") -> tuple[float | None, float | None]:
        """
        Returns (loss, ppl), or (None, None) if response is empty.

        Args:
            response: Text to score
            prompt: Prefix not counted in loss; default "" scores full text
        """
        if not response or not response.strip():
            return None, None

        prompt_len = self._get_prompt_len(prompt)
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100

        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)

        loss = outputs.loss.item()
        ppl = torch.exp(torch.tensor(loss)).item()
        return loss, ppl


def _apply_k_ppl_to_sample(
    sample: dict,
    ppl_scorer: ResponsePPLScorer,
    ppl_prompt: str,
    k_aggregate_mode: str,
) -> None:
    """Write ppl / ppl_loss for one sample (supports output_k_response)."""
    metrics = sample.setdefault("metrics", {})
    response = sample.get("response", "")
    has_k = "output_k_response" in sample and len(sample.get("output_k_response", [])) > 0

    if has_k:
        k_responses = sample["output_k_response"]
        k_ppl_list: list[float] = []
        k_loss_list: list[float] = []

        for k_response in k_responses:
            loss, ppl = ppl_scorer.compute(k_response, prompt=ppl_prompt)
            if ppl is not None:
                k_ppl_list.append(ppl)
                k_loss_list.append(loss)

        if k_ppl_list:
            if k_aggregate_mode == "max":
                metrics["ppl"] = min(k_ppl_list)
                metrics["ppl_loss"] = min(k_loss_list)
            else:
                metrics["ppl"] = sum(k_ppl_list) / len(k_ppl_list)
                metrics["ppl_loss"] = sum(k_loss_list) / len(k_loss_list)

            if "k_metrics_detail" in sample:
                for k_idx, (k_loss, k_ppl) in enumerate(zip(k_loss_list, k_ppl_list)):
                    if k_idx < len(sample["k_metrics_detail"]):
                        sample["k_metrics_detail"][k_idx]["ppl"] = k_ppl
                        sample["k_metrics_detail"][k_idx]["ppl_loss"] = k_loss
        return

    loss, ppl = ppl_scorer.compute(response, prompt=ppl_prompt)
    if ppl is not None:
        metrics["ppl"] = ppl
        metrics["ppl_loss"] = loss


def main() -> None:
    parser = argparse.ArgumentParser(description="PPL on merged JSON (response tokens only)")
    parser.add_argument("-s", "--src_json", type=str, required=True, help="Input merged JSON")
    parser.add_argument("-d", "--dst_json", type=str, required=True, help="Output JSON with metrics")
    parser.add_argument(
        "--k_aggregate_mode",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="K responses: max uses min PPL (best); mean averages",
    )
    parser.add_argument(
        "--ppl_model_path",
        type=str,
        default="Qwen3-1.7B",
        help="Causal LM for PPL",
    )
    parser.add_argument("--ppl_device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    from config import no_cot_eng_prompt, no_cot_zh_prompt

    with open(args.src_json, "r", encoding="utf-8") as f:
        datas = json.load(f)

    print(f"K aggregation: {args.k_aggregate_mode}")
    print(f"Loading PPL model: {args.ppl_model_path}")
    ppl_scorer = ResponsePPLScorer(model_path=args.ppl_model_path, device=args.ppl_device)
    print("PPL model ready.")
    print("=" * 60)

    for sample in tqdm(datas, desc="PPL"):
        language = sample.get("language", "eng")
        ppl_prompt = no_cot_zh_prompt if language == "zh" else no_cot_eng_prompt
        _apply_k_ppl_to_sample(sample, ppl_scorer, ppl_prompt, args.k_aggregate_mode)

    metrics_all: dict[str, list[float]] = {}
    for sample in datas:
        for key, value in sample.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                metrics_all.setdefault(key, []).append(float(value))

    metrics_avg = {k: sum(v) / len(v) for k, v in metrics_all.items()}

    print("=" * 60)
    for k, v in sorted(metrics_avg.items()):
        print(f"  {k}: {v:.4f}")
    print("=" * 60)

    with open(args.dst_json, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics_avg, "k_aggregate_mode": args.k_aggregate_mode, "datas": datas},
            f,
            ensure_ascii=False,
            indent=4,
        )
    print(f"Wrote: {args.dst_json}")


if __name__ == "__main__":
    main()
