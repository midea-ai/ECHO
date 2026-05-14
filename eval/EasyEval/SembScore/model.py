from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class SEMBScorer(torch.nn.Module):
    def __init__(self, batch_size: int, model_path: str = 'hiaoxui/sembscore'):
        super().__init__()
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_device = 'cpu'
        self.model.eval()

    def cuda(self, device: str = 'cuda:0'):
        self.model_device = device
        self.model.to(device)
        return self

    @torch.inference_mode()
    def score(self, hyps: List[str], refs: List[str]) -> List[float]:
        assert len(hyps) == len(refs), "Hypotheses and references must have the same length."
        scores = []
        for i in range(0, len(hyps), self.batch_size):
            batch_hyps = hyps[i:i + self.batch_size]
            batch_refs = refs[i:i + self.batch_size]
            n_sentences = len(batch_hyps)
            sentences = batch_hyps + batch_refs
            inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
            if self.model_device != 'cpu':
                inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            hyp_embs, ref_embs = cls_embs[:n_sentences], cls_embs[n_sentences:]
            # cosine similarity
            cos_sim = (hyp_embs * ref_embs).sum(axis=1) / (
                    np.linalg.norm(hyp_embs, axis=1) * np.linalg.norm(ref_embs, axis=1))
            scores.extend(cos_sim.tolist())
            # extract CLS token
            # Assuming the model returns a score tensor
        return scores
