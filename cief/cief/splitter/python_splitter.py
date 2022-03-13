from typing import Callable, Iterable, List, Tuple
from cief.types import Feature, FloatArray, FloatMatrix, SplitResult
import numpy as np

ScoreFunc = Callable[[FloatArray, FloatArray], float]
Normalizer = Callable[[FloatArray], FloatArray]


def pick_supremum_or_self(val: float, array: FloatArray) -> float:
    try:
        return min(array[array > val])
    except ValueError:
        return val

def get_avg_split_vector(
    Y: FloatArray, X: FloatArray, split_point: float
    ) -> Tuple[FloatArray, float, float]:
    """ partitions an array and calculates avg in each partition"""
    below_idx = X <= split_point
    above_idx = X > split_point
    below = Y[below_idx]
    above = Y[above_idx]
    Y_split = np.zeros(Y.size)
    if below.size != 0:
        below_avg = float(sum(below) / below.size)
        Y_split[below_idx] = below_avg
    else:
        below_avg = None
    if above.size != 0:
        above_avg = float(sum(above) / above.size)
        Y_split[above_idx] = above_avg
    else:
        above_avg = None
    below_avg = below_avg or above_avg or 0.0 # Type check fix
    above_avg = above_avg or below_avg or 0.0 # type check fix
    return Y_split, below_avg, above_avg

def get_splitter(scorer: ScoreFunc):
    def splitter(
            data: FloatMatrix, target: Feature, features: Iterable[Feature]
    ) -> List[SplitResult]:
        splits: List[SplitResult] = []
        Y: FloatArray = data[:, target]
        for feature in features:
            X: FloatArray = data[:, feature]
            best_score, best_split_threshold, best_below_avg, best_above_avg = -1.0, 0.0, 0.0, 0.0
            for x in X:
                Y_avg, below_avg, above_avg = get_avg_split_vector(Y, X, x)
                score = scorer(Y, Y_avg)
                if best_score < 0 or best_score < score:
                    best_score = score
                    best_split_threshold = x
                    best_below_avg, best_above_avg = below_avg, above_avg
            best_split_threshold_middlepoint = (best_split_threshold + pick_supremum_or_self(best_split_threshold, Y)) / 2
            split_result = SplitResult(
                score=best_score,
                split_threshold=best_split_threshold_middlepoint,
                below_avg=best_below_avg,
                above_avg=best_above_avg,
                target=target,
                feature=feature,
            )
            splits.append(split_result)
        return splits
    return splitter
