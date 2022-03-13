from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, TypeVar
from cief.utils import init_W
from cief.types import FloatArray, FloatMatrix, WUpdateFrequency
from cief.splitter import Splitter, base_splitter
from cief.scorer import Scorer
from typing import Literal, Optional
import numpy as np
from numpy.typing import NDArray
from cief.tree import Leaf, Tree
import logging
from copy import deepcopy, copy

logger = logging.getLogger()

@dataclass
class LeafToBeFitted:
    indexes: NDArray[np.bool_]
    depth: int
    type: Literal["left", "right", "root"]
    parent: Optional[Leaf]


TStepTree = TypeVar("TStepTree", bound="StepTree")


class StepTree(ABC):
    """ """

    def __init__(self,
        set_size: int,
        target: int,
        splitter: Splitter,
        scorer: Scorer,
        step_max=3,
        depth=None,
        min_improve=None,
        w_update_frequency: WUpdateFrequency = "each_step",
        W_update_queue: List[FloatMatrix] = None,
        W: Optional[FloatMatrix] = None,
        keep_data=True,
    ):
        if depth is None and min_improve is None:
            raise ValueError("Atleast one of depth and min_improve must be set")
        self.set_size = set_size
        self.min_improve = min_improve
        self.step_max = step_max
        self.depth = depth
        self.tree: Optional[Tree] = None
        self.W = (
            W if W is not None else init_W(2)
        )  # Will be overwritten - just for type checking
        self.tree_sample_size: int = 0 # Will be overwritten
        self.target = target
        self.splitter = splitter
        self.scorer = scorer
        self.w_update_frequency: WUpdateFrequency = w_update_frequency
        self.W_update_queue: List[FloatMatrix] = W_update_queue if W_update_queue is not None else []

    def is_fitted(self):
        return self.tree is not None

    @abstractmethod
    def fit(self: TStepTree, data: FloatMatrix, W: Optional[FloatMatrix] = None) -> TStepTree:
        data, W
        return self # TODO

    def copy(self: TStepTree) -> TStepTree:
        """Useful for creating multiple (almost) SKlearn compatible models from one
        IF by making copies and changing self.target.

        copy should usually be enough, use deepcopy if in doubt.
        """
        return copy(self)

    def deepcopy(self: TStepTree) -> TStepTree:
        """Useful for creating multiple (almost) SKlearn compatible models from one
        IF by making copies and changing self.target

        copy should usually be enough, use deepcopy if in doubt.
        """
        return deepcopy(self)

    def predict(self, data: FloatMatrix) -> FloatArray:
        if self.tree is None:
            raise NotImplementedError  # TODO correct error
        return self.tree.predict(data)
