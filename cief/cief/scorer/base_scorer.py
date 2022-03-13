from typing import Callable, List, Tuple
from cief.types import FloatArray, FloatMatrix, SplitResult
import numpy as np

ScoreFunc = Callable[[FloatArray, FloatArray], float]
Normalizer = Callable[[FloatArray], FloatArray]

def scale_to_0_1(arr: FloatArray) -> FloatArray:
    return arr / np.max(arr)


class Scorer:
    """ """

    def __init__(self, score_func: ScoreFunc, normalizer: Normalizer = None):
        self.score_func = score_func
        self.normalizer = normalizer 
        self.name = score_func.__name__

    @staticmethod
    def _get_avg_split(data: FloatMatrix, split_result) -> Tuple[FloatArray, FloatArray]:
        Z = data[:, split_result.feature]
        Y = data[:, split_result.target]
        below_idx = Z <= split_result.split_threshold
        Y_split = np.zeros(Y.size)
        Y_split[below_idx] = split_result.below_avg
        Y_split[~below_idx] = split_result.above_avg
        return Y, Y_split


    def __call__(self, data: FloatMatrix, split_results: List[SplitResult]) -> FloatArray:
        d = data.shape[1]
        result = np.zeros(d)
        for split_result in split_results:
            Y, Y_split = self._get_avg_split(data, split_result)
            result[split_result.feature] = self.score_func(Y, Y_split)
        if self.normalizer is not None:
            result = self.normalizer(result)
        return result


