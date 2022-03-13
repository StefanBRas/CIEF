from dataclasses import dataclass
from typing import List, Literal, Union
import numpy as np
from numpy.typing import NDArray


Feature = int
LeafIndex = int
FloatMatrix = NDArray[np.floating]
FloatArray = NDArray[np.floating]
WUpdateFrequency = Literal["each_step", "each_walk", "each_tree", "each_target"]
KeepData = Union[bool, None, Literal["all", "updates"]]


@dataclass
class SplitResult:
    score: float
    split_threshold: float
    below_avg: float
    above_avg: float
    target: Feature
    feature: Feature


@dataclass
class StepNew:
    sample_target: Feature
    splits: List[SplitResult]
    candidates: List[int]

    def get_best_split(self) -> SplitResult:
        return min(self.splits, key=lambda split: split.score)

    def get_w_delta(self, d: int) -> FloatArray:
        w_delta = np.zeros(d)
        for split in self.splits:
            score = split.score
            w_delta[split.feature] = 1.0 / score if score != 0.0 else 0.0
        return w_delta

    def get_w_delta_scorer(self, d: int, scorer) -> FloatArray:
        w_delta = np.zeros(d)
        for split in self.splits:
            score = split.score
            w_delta[split.feature] = 1.0 / score if score != 0.0 else 0.0
        return w_delta

    def get_considered_features(self):
        return [sr.feature for sr in self.splits]
