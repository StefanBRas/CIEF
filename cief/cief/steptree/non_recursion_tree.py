from dataclasses import dataclass, field
import pandas as pd
from cief.scorer.base_scorer import Scorer
from cief.steptree.base_step_tree import StepTree
from cief.utils import init_W, is_empty, melt_mat, update_W_and_clear_queue
from cief.walk import Walk, WalkFitInfo
from cief.types import FloatArray, FloatMatrix, WUpdateFrequency
from cief.splitter import Splitter, base_splitter
from typing import List, Literal, Optional, Union
import numpy as np
from numpy.typing import NDArray
from cief.tree import Leaf, Feature, Tree
import logging
from copy import deepcopy, copy
from queue import SimpleQueue

logger = logging.getLogger()

@dataclass
class TreeFitInfo:
    w_deltas_applied: List[FloatMatrix] = field(default_factory=list)
    walk_fit_infos: List[WalkFitInfo] = field(default_factory=list)

    def get_normalized_w_df(self):
        deltas_applied = not is_empty(self.w_deltas_applied)
        if deltas_applied:
            dfs =  [melt_mat(mat) for mat in self.w_deltas_applied]
        else:
            dfs = [walk_fit_info.get_normalized_w_df() for walk_fit_info in self.walk_fit_infos]
        for i, df in enumerate(dfs):
            df["walk_iteration"] = i
            if deltas_applied:
                df["step_iteration"] = None
        return pd.concat(dfs)


@dataclass
class LeafToBeFitted:
    indexes: NDArray[np.bool_]
    depth: int
    type: Literal["left", "right", "root"]
    parent: Optional[Leaf]


class StepTreeNonRecursive(StepTree):
    """ """

    def __init__(
        self,
        set_size: int,
        target: int,
        splitter: Splitter,
        scorer: Scorer,
        step_max=3,
        depth=None,
        min_improve=None,
        W: Optional[FloatMatrix] = None,
        w_update_frequency: WUpdateFrequency = "each_step",
        W_update_queue: List[FloatMatrix] = None,
        keep_data: bool = False
    ):
        super().__init__(
                set_size=set_size,
                target=target,
                step_max=step_max,
                depth=depth,
                min_improve=min_improve,
                splitter=splitter,
                scorer=scorer,
                W=W,
                w_update_frequency=w_update_frequency,
                W_update_queue=W_update_queue
                )
        self.to_be_fitted: SimpleQueue[LeafToBeFitted] = SimpleQueue()
        self.leafs: List[Leaf] = []
        self.keep_data = keep_data
        if keep_data:
            self.fit_info = TreeFitInfo()

    @staticmethod
    def _make_split(
        data: FloatMatrix,
        indexes: NDArray[np.bool_],
        split_feature: int,
        split_point: float,
        depth: int,
        parent: Leaf,
    ):
        left_indexes = data[:, split_feature] <= split_point
        right_indexes = ~left_indexes
        left_indexes = left_indexes & indexes
        right_indexes = right_indexes & indexes
        left_to_do = LeafToBeFitted(
            indexes=left_indexes, type="left", parent=parent, depth=depth + 1
        )
        right_to_do = LeafToBeFitted(
            indexes=right_indexes, type="right", parent=parent, depth=depth + 1
        )
        return left_to_do, right_to_do

    def _walk_non_recursion(
        self, data: FloatMatrix, to_do: LeafToBeFitted
    ) -> Union[Leaf, None]:
        if all(to_do.indexes):
            data_subset = data
        else:
            data_subset = data[to_do.indexes, :]
        rows = data_subset.shape[0]
        if rows <= 1:
            return None
        walk = Walk(
            data=data_subset,
            W=self.W,
            step_max=self.step_max,
            splitter=self.splitter,
            scorer=self.scorer,
            tree_sample_size=self.tree_sample_size,
            target=self.target,
            set_size=self.set_size,
            w_update_frequency=self.w_update_frequency,
            W_update_queue=self.W_update_queue,
            keep_data=self.keep_data
        )
        walk.fit()
        if self.w_update_frequency == "each_walk":
            logger.info(f"Updating in walk for target {self.target}")
            w_delta = update_W_and_clear_queue(self.W, self.W_update_queue)
            if self.keep_data:
                self.fit_info.w_deltas_applied.append(w_delta)
        if self.keep_data:
            self.fit_info.walk_fit_infos.append(walk.fit_info)
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
        if to_do.parent:
            if to_do.type == "left":
                to_do.parent.left = leaf
            else:
                to_do.parent.right = leaf
        improving = False  # (self.min_improve is not None and self.min_improve < max(scores.values())) # TODO should be max of k-1 row probably # TODO skipping for now
        shallow = self.depth is not None and to_do.depth < self.depth
        if improving or shallow:
            left_to_do, right_to_do = self._make_split(
                data,
                to_do.indexes,
                leaf.feature,
                leaf.split_threshold,
                depth=to_do.depth,
                parent=leaf,
            )
            if any(left_to_do.indexes):
                self.to_be_fitted.put(left_to_do)
            if any(right_to_do.indexes):
                self.to_be_fitted.put(right_to_do)
        self.leafs.append(leaf)
        return leaf

    def fit(
        self, data: FloatMatrix, W: Optional[FloatMatrix] = None
    ) -> "StepTreeNonRecursive":
        rows, columns = data.shape
        if self.is_fitted():
            logger.info(
                "Model has already been fitted - refitting and overwritting old results. Reusing old W."
            )
            self.tree = None
        else:
            self.W = init_W(columns) if W is None else W
        self.tree_sample_size = rows
        root_to_do = LeafToBeFitted(
            type="root", depth=0, parent=None, indexes=np.full((rows,), True)
        )
        root = self._walk_non_recursion(data, root_to_do)
        if root is None:
            raise ValueError("Todo something something empty tree/leaf")
        while not self.to_be_fitted.empty():
            self._walk_non_recursion(data, self.to_be_fitted.get_nowait())
        self.tree = Tree(target=self.target, root=root)
        # np.fill_diagonal(self.W, 0) # TODO looks very wrong
        # self.W = self.W / self.W.max() # TODO this looks wrong?
        return self
