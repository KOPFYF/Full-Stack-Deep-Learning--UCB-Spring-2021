from typing import Sequence

import pytorch_lightning as pl
import torch
import editdistance


class CharacterErrorRate(pl.metrics.Metric):
    """
    Character error rate metric, computed using Levenshtein distance.

    HW: In your own words, explain how the CharacterErrorRate metric and the greedy_decode method work.

    Levenshtein distance:
    https://en.wikipedia.org/wiki/Levenshtein_distance
     the Levenshtein distance is a string metric for measuring the difference between two sequences. 
     Informally, the Levenshtein distance between two words is the minimum number of single-character edits (insertions, deletions or substitutions) required to change one word into the other. 


    https://pypi.org/project/editdistance/0.3.1/
    distance = editdistance.distance(pred, target)
    
    """

    def __init__(self, ignore_tokens: Sequence[int], *args):
        super().__init__(*args)
        self.ignore_tokens = set(ignore_tokens)
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.error: torch.Tensor
        self.total: torch.Tensor

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        N = preds.shape[0]
        for ind in range(N):
            pred = [_ for _ in preds[ind].tolist() if _ not in self.ignore_tokens]
            target = [_ for _ in targets[ind].tolist() if _ not in self.ignore_tokens]
            distance = editdistance.distance(pred, target)
            error = distance / max(len(pred), len(target))
            self.error = self.error + error
        self.total = self.total + N

    def compute(self) -> torch.Tensor:
        return self.error / self.total


def test_character_error_rate():
    metric = CharacterErrorRate([0, 1])
    X = torch.tensor(  # pylint: disable=not-callable
        [
            [0, 2, 2, 3, 3, 1],  # error will be 0
            [0, 2, 1, 1, 1, 1],  # error will be .75
            [0, 2, 2, 4, 4, 1],  # error will be .5
        ]
    )
    Y = torch.tensor(  # pylint: disable=not-callable
        [
            [0, 2, 2, 3, 3, 1],
            [0, 2, 2, 3, 3, 1],
            [0, 2, 2, 3, 3, 1],
        ]
    )
    metric(X, Y)
    print(metric.compute())
    assert metric.compute() == sum([0, 0.75, 0.5]) / 3


if __name__ == "__main__":
    test_character_error_rate()
