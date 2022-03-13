from cief.steptree.base_step_tree import StepTree
from cief.utils import init_W
from cief.walk import Walk
from cief.types import FloatArray, FloatMatrix
from cief.splitter import Splitter, base_splitter
from typing import Optional, Union
import numpy as np
from cief.tree import Leaf, Feature, Tree
import logging
from copy import deepcopy, copy

logger = logging.getLogger()



class StepTreeRecursive(StepTree):
    """ """


    @staticmethod
    def _make_split(data: FloatMatrix, split_feature: int, split_point: float):
        left_data = data[data[:, split_feature] <= split_point]
        right_data = data[data[:, split_feature] > split_point]
        return left_data, right_data

    def _walk_recursion(self, data: FloatMatrix, depth=0) -> Union[Leaf, None]:
        rows = data.shape[0]
        if rows <= 1:
            return None
        walk = Walk(
            data=data,
            W=self.W,
            step_max=self.step_max,
            splitter=self.splitter,
            scorer=self.scorer,
            tree_sample_size=self.tree_sample_size,
            target=self.target,
            set_size=self.set_size,

        )
        walk.fit()
        best_split = walk.get_best_split()
        leaf = Leaf(
            feature=best_split.feature,
            split_threshold=best_split.split_threshold,
            below_avg=best_split.below_avg,
            above_avg=best_split.above_avg,
            W=self.W.copy(),
            walk=walk,
            n=rows,
        )
        improving = False  # (self.min_improve is not None and self.min_improve < max(scores.values())) # TODO should be max of k-1 row probably # TODO skipping for now
        shallow = self.depth is not None and depth < self.depth
        if improving or shallow:
            left_data, right_data = self._make_split(
                data, leaf.feature, leaf.split_threshold
            )
            leaf.left = self._walk_recursion(left_data, depth + 1)
            leaf.right = self._walk_recursion(right_data, depth + 1)
        return leaf

    def fit(self, data: FloatMatrix, W: Optional[FloatMatrix] = None) -> "StepTree":
        rows, columns = data.shape
        if self.is_fitted():
            logger.info(
                "Model has already been fitted - refitting and overwritting old results. Reusing old W."
            )
            self.tree = None
        else:
            self.W = init_W(columns) if W is None else W
        self.tree_sample_size = rows 
        root = self._walk_recursion(data)
        if root is None:
            raise ValueError("Todo something something empty tree/leaf")
        self.tree = Tree(target=self.target, root=root)
        np.fill_diagonal(self.W, 0)
        self.W = self.W / self.W.max()
        return self

