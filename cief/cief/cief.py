from cief.steptree.non_recursion_tree import TreeFitInfo
import pandas as pd
import seaborn as sns
from queue import SimpleQueue
from dataclasses import dataclass, field
from cief.tree import Leaf
from cief.steptree.base_step_tree import StepTree
from cief.splitter import SPLITTERS, Splitter, SplitterType
from cief.scorer import SCORERS, Scorer, ScorerType
from cief.steptree import StepTreeRecursive, StepTreeNonRecursive
from cief.utils import (
    init_W,
    is_empty, melt_mat,
    predict_multiple,
    update_W_and_clear_queue,
)
from cief.types import FloatArray, FloatMatrix, KeepData, WUpdateFrequency
from typing import Callable, List, Literal, Optional, Tuple, Union, Type, Dict
from itertools import cycle
import numpy as np
import logging
from copy import deepcopy, copy


logger = logging.getLogger()

TargetPicker = Callable[[int], Callable[[], int]]  # dim -> target




@dataclass
class CIEFFitInfo:
    has_fit_data: bool
    w_deltas_applied: List[FloatMatrix] = field(default_factory=list)
    tree_fit_infos: List[TreeFitInfo] = field(default_factory=list)
    final_w: List[FloatMatrix] = field(default_factory=list)

    def get_final_w(self):
        df = melt_mat(self.final_w[0])
        return df

    def get_normalized_w_df(self):
        df = self.get_normalized_w_delta_df()
        df["value"] = df.sort_values("iteration_index").groupby(["edge"])["value"].cumsum()
        df["estimator"] = "CIEF"
        return df

    def get_normalized_w_delta_df(self):
        deltas_applied = not is_empty(self.w_deltas_applied)
        if deltas_applied:
            dfs = [melt_mat(mat) for mat in self.w_deltas_applied]
        else:
            dfs = [
                tree_fit_info.get_normalized_w_df()
                for tree_fit_info in self.tree_fit_infos
            ]
        for i, df in enumerate(dfs):
            df["tree_iteration"] = i
            if deltas_applied:
                df["walk_iteration"] = None
                df["step_iteration"] = None
        full_df = pd.concat(dfs)
        full_df["iteration_index"] = (
            full_df.sort_values(["tree_iteration", "walk_iteration", "step_iteration"])
            .groupby(["row", "column"])
            .cumcount()
        )
        return full_df.reset_index()

    def plot(self, ax=None):
        df = self.get_normalized_w_df()
        return sns.lineplot(data=df, x="iteration_index", y="value", units="edge", estimator=None, ax=ax)



class CIEF:
    """

    target_picker can be None, int or function int -> int. If int, a constant target is chosen.
    If none, (uniformly) random target is chosen. If function, new targets are found by calling
    function. TODO should maybe take sequence instead?
    """

    def __init__(
        self,
        set_size: int,
        step_max: int,
        splitter: Union[SplitterType, Splitter],
        scorer: Union[ScorerType, Scorer],
        depth: int,
        w_update_frequency: WUpdateFrequency = "each_step",
        target_picker: Optional[Union[TargetPicker, int]] = None,
        target=None,
        keep_data: bool = False,
        sample_size: Union[float, int] = None
    ):
        self.set_size = set_size
        self.step_max = step_max
        self.depth = depth
        self.trees: List[StepTreeNonRecursive] = []
        self.W = np.ndarray(0)  # Avoid having to do type assertions
        self.w_update_frequency: WUpdateFrequency = w_update_frequency
        self.W_update_queue: List[
            FloatMatrix
        ] = []  # to be able to modify when to apply the W updates
        self.target_picker = target_picker
        self.target: Optional[
            int
        ] = target  # to be able to fix the target for predictions
        self.splitter: Splitter = SPLITTERS[splitter] if type(splitter) == str else splitter  # type: ignore
        self.scorer: Scorer = SCORERS[scorer] if type(scorer) == str else scorer  # type: ignore
        self.keep_data = keep_data
        self.sample_size = sample_size
        self.fit_info = CIEFFitInfo(has_fit_data=keep_data)

    def _get_target_picker(self, columns: int) -> Callable[[], int]:
        if self.target_picker is None:
            if self.w_update_frequency == "each_target":
                targets = cycle(list(range(columns)))

                def cycle_targets():
                    return next(targets)

                return cycle_targets
            else:

                def pick_random():
                    return np.random.randint(0, columns)

                return pick_random
        elif isinstance(self.target_picker, int):
            x = self.target_picker  # tmp var to satisfy typechecker

            def pick_constant() -> int:  # type: ignore
                return x

            return pick_constant
        else:
            return self.target_picker(columns)

    def is_fitted(self):
        x = self.splitter
        return len(self.trees) > 0

    @staticmethod
    def _get_sample_size(data_size: int, sample_size: Union[float, int] = None):
        if sample_size is None:
            return data_size
        if sample_size <= 0:
            raise ValueError("sample size must be positive")
        if 0.0 < sample_size <= 1.0:
            result = int(np.ceil(sample_size * data_size))
            if result <= 2:
                raise ValueError("sample size must be more than 2")
            return result
        return sample_size


    def fit(
        self, data: FloatMatrix, W: Optional[FloatMatrix] = None, max_iterations=3
    ) -> "CIEF":
        if self.is_fitted():
            logger.info(
                "Model has already been fitted - refitting and overwritting old resuts"
            )
            self.trees = []
        rows, columns = data.shape
        self.W = init_W(columns) if W is None else W.copy()
        iterations = 0
        target_picker = self._get_target_picker(columns)
        sample_size = self._get_sample_size(rows, self.sample_size)
        while iterations < max_iterations:
            iterations += 1
            target = target_picker()
            logger.info(f"Fitting Tree {iterations} for target {target}")
            tree = StepTreeNonRecursive(
                set_size=self.set_size,
                target=target,
                depth=self.depth,
                splitter=self.splitter,
                scorer=self.scorer,
                w_update_frequency=self.w_update_frequency,
                W_update_queue=self.W_update_queue,
                keep_data=self.keep_data,
            )
            subset_idxs = np.random.randint(rows, size=sample_size)
            subset = data[subset_idxs,:]
            tree.fit(subset, W=self.W)
            if self.w_update_frequency == "each_tree":
                logger.info(f"Updating W in iteration {iterations} for target {target}")
                w_delta = update_W_and_clear_queue(self.W, self.W_update_queue)
                if self.keep_data:
                    self.fit_info.w_deltas_applied.append(w_delta)
            if self.w_update_frequency == "each_target" and target == (columns - 1):
                logger.info(f"Updating W in iteration {iterations} for target {target}")
                w_delta = update_W_and_clear_queue(self.W, self.W_update_queue)
                if self.keep_data:
                    self.fit_info.w_deltas_applied.append(w_delta)
            self.trees.append(tree)
            if self.keep_data:
                self.fit_info.tree_fit_infos.append(tree.fit_info)
        # if frequency == "each_target", we might end before having update the queue
        # This ensures that it will be added
        if not is_empty(self.W_update_queue):
            w_delta = update_W_and_clear_queue(self.W, self.W_update_queue)
            if self.keep_data:
                self.fit_info.w_deltas_applied.append(w_delta)
        self.fit_info.final_w.append(self.W.copy())
        self.W = self.W / np.max(self.W)
        return self

    def copy(self) -> "CIEF":
        """Useful for creating multiple (almost) SKlearn compatible models from one
        IF by making copies and changing self.target.

        copy should usually be enough, use deepcopy if in doubt.
        """
        return copy(self)

    def deepcopy(self) -> "CIEF":
        """Useful for creating multiple (almost) SKlearn compatible models from one
        IF by making copies and changing self.target


        copy should usually be enough, use deepcopy if in doubt.
        """
        return deepcopy(self)

    def predict(
        self, data: FloatMatrix, target: Optional[int] = None
    ) -> Tuple[FloatArray, List[FloatArray]]:
        if not self.is_fitted():
            raise ValueError("not fitted yet")
        if target is None and self.target is None:
            raise ValueError("Either target or self.target must be set")
        target = target if target is not None else self.target
        trees_with_target = [tree for tree in self.trees if tree.target == target]
        return predict_multiple(trees_with_target, data)

    def get_leaf_sequel(self) -> List[Leaf]:
        leafs = []
        for tree in self.trees:
            if isinstance(tree, StepTreeNonRecursive):
                for leaf in tree.leafs:
                    leafs.append(leaf)
        return leafs

    def results_to_json(self):
        if not self.is_fitted():
            raise ValueError("not fitted yet")
