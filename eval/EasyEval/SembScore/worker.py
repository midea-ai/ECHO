import numpy as np
import ray

from model import SEMBScorer
from reward_server import RewardServer


@ray.remote(num_gpus=1)
class SEMBScoreWorker:
    def __init__(self, batch_size=16, model_path='hiaoxui/sembscore'):
        self.batch_size = batch_size
        self.scorer = SEMBScorer(batch_size=batch_size, model_path=model_path).cuda(device='cuda:0')

    def compute(self, hyps, refs):
        f1 = self.scorer.score(hyps=hyps, refs=refs)
        return np.array(f1)


class SEMBScoreMetric(RewardServer):
    def __init__(
            self,
            num_workers=None,
            model_path=None,
            batch_size=16,
    ):
        self.batch_size = batch_size
        self.model_path = model_path

        super().__init__(num_workers=num_workers, model_path=model_path)

    def create_worker(self):
        return SEMBScoreWorker.remote(batch_size=self.batch_size, model_path=self.model_path)
