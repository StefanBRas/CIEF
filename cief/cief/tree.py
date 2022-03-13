from __future__ import annotations
from cief.walk import Walk

from dataclasses import dataclass
from cief.types import Feature, FloatArray, FloatMatrix
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from numpy import ndarray, add
from numpy.lib.shape_base import apply_along_axis
import logging

logger = logging.getLogger()



@dataclass
class Step():
    j_old: int
    j: int
    scores: Dict[int, float]
    splits: Dict[int, float]
    C: List[int]
    S: List[int]
    s: int
    below_avg: float
    above_avg: float



@dataclass
class Leaf():
    feature: Feature
    split_threshold: float
    below_avg: float
    above_avg: float
    W: ndarray
    walk: Walk
    left: Optional[Leaf] = None
    right: Optional[Leaf] = None
    n: Optional[int] = None

    def predict_single(self, data: FloatMatrix):
        if data[self.feature] <= self.split_threshold:
            if self.left is not None:
                return self.left.predict_single(data)
            else:
                return self.below_avg
        else:
            if self.right is not None:
                return self.right.predict_single(data)
            else:
                return self.above_avg

@dataclass
class Tree():
    target: Feature
    root: Leaf # Optional only for empty data

    def predict_single(self, data: FloatMatrix):
        return self.root.predict_single(data)

    def predict(self, data: FloatMatrix) -> FloatArray:
        predictions = apply_along_axis(self.predict_single, 1, data)
        return predictions




Relation = Literal['left', 'right', 'root']
TreeTraverseFunc = Callable[[Optional[Leaf], Any, Relation], Any] 

def traverse_tree(tree: Tree, func: TreeTraverseFunc):
    def traverse_leaf(leaf: Leaf, func:TreeTraverseFunc, func_res = None):
        left_func_res = func(leaf.left, func_res, 'left')
        if leaf.left is not None:
            traverse_leaf(leaf.left, func, left_func_res)
        right_func_res = func(leaf.right, func_res, 'right')
        if leaf.right is not None:
            traverse_leaf(leaf.right, func, right_func_res)
    if tree.root is not None:
        root_func_res = func(tree.root, None, 'root')
        traverse_leaf(tree.root, func, root_func_res)


def predict_trees(trees: List[Tree], data: ndarray) -> Tuple[ndarray, List[ndarray]]:
    all_predictions = [tree.predict(data) for tree in trees]
    mean_prediction = add.reduce(all_predictions) / len(all_predictions)
    return mean_prediction, all_predictions


