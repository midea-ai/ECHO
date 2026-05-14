from typing import List, Optional
import math

import torch
import ray


class RewardServer:
    # A server that computes rewards using multiple workers in parallel.
    num_workers: int
    workers: List

    def __init__(self, num_workers: Optional[int] = None, model_path: Optional[str] = None):
        if num_workers is None:
            num_workers = torch.cuda.device_count()
        self.num_workers = num_workers
        self.model_path = model_path
        self.workers = [self.create_worker() for i in range(num_workers)]

    def create_worker(self):
        # Initialize workers if needed. This method can be overridden by subclasses.
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_rewards(self, hyps, refs) -> List:
        assert len(hyps) == len(refs), "hypotheses and references must be same length"
        total = len(hyps)
        batch_size = math.ceil(total / self.num_workers)

        futures = []
        for i in range(self.num_workers):
            start = i * batch_size
            end = min((i + 1) * batch_size, total)
            if start < end:
                futures.append(self.workers[i].compute.remote(hyps[start:end], refs[start:end]))

        results = ray.get(futures)
        all_rewards = [float(reward) for batch in results for reward in batch]
        return all_rewards
