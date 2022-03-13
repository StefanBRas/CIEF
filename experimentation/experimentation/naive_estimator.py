from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression
from forestpathwalk.splitter import SPLITTERS, Splitter, SplitterType
from forestpathwalk.scorer import SCORERS, Scorer, ScorerType
from forestpathwalk.utils import (
    melt_mat,
)
from forestpathwalk.types import FloatMatrix
from typing import Callable, List, Union
import numpy as np
import logging
from copy import deepcopy, copy


logger = logging.getLogger()


@dataclass
class NaiveFitInfo:
    final_w: List[FloatMatrix] = field(default_factory=list)
    has_fit_data = False 

    def get_final_w(self):
        df = melt_mat(self.final_w[0])
        return df



class NaiveEstimator:
    """
    Naively classifies by looking at one variable at a time
    """

    def __init__(
        self,
        scorer: Union[ScorerType, Scorer] = "explained_variance",
        sample_size: Union[float, int] = None
    ):
        self.W = np.ndarray(0)  # Avoid having to do type assertions
        self.scorer: Scorer = SCORERS[scorer] if type(scorer) == str else scorer  # type: ignore
        self.sample_size = sample_size
        self.fit_info = NaiveFitInfo()


    def is_fitted(self):
        return self.W.shape != ()

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


    def copy(self) -> "NaiveEstimator":
        """Useful for creating multiple (almost) SKlearn compatible models from one
        IF by making copies and changing self.target.

        copy should usually be enough, use deepcopy if in doubt.
        """
        return copy(self)

    def deepcopy(self) -> "NaiveEstimator":
        """Useful for creating multiple (almost) SKlearn compatible models from one
        IF by making copies and changing self.target


        copy should usually be enough, use deepcopy if in doubt.
        """
        return deepcopy(self)


class NaiveEstimatorRegression(NaiveEstimator):
    """
    Naively classifies by looking at one variable at a time and doing lin regression
    """

    def fit(
        self, data: FloatMatrix, W: None = None, max_iterations=3
    ) -> "NaiveEstimator":
        """
        We keep some redunant arguments for pararity with StepForest
        """
        if self.is_fitted():
            logger.info(
                "Model has already been fitted - refitting and overwritting old resuts"
            )
        rows, columns = data.shape
        self.W = np.zeros((columns, columns)).astype(float)
        sample_size = self._get_sample_size(rows, self.sample_size)
        subset_idxs = np.random.randint(rows, size=sample_size)
        subset = data[subset_idxs,:]
        for target in range(columns):
            variables = set(range(columns)) - set([target])
            update = np.zeros(columns)
            for variable in variables:
                y=subset[:,target]
                X=subset[:,variable].reshape(-1,1)
                mod = LinearRegression()
                mod.fit(y=y, X=X)
                predicted = mod.predict(X)
                update[variable] = self.scorer.score_func(y, predicted)
            w_delta = np.zeros((columns, columns))
            w_delta[target, :] += update
            w_delta[:, target] += update  # assure symmetry
            self.W += w_delta
        self.fit_info.final_w.append(self.W.copy())
        return self


class NaiveEstimatorTree(NaiveEstimator):
    """
    Naively classifies by looking at one variable at a time and doing splits
    """
    def __init__(
        self,
        splitter: Union[SplitterType, Splitter] = "explained_variance_splitter_python",
        scorer: Union[ScorerType, Scorer] = "explained_variance",
        sample_size: Union[float, int] = None
    ):
        super().__init__(sample_size = sample_size, scorer=scorer)
        self.splitter: Splitter = SPLITTERS[splitter] if type(splitter) == str else splitter  # type: ignore

    def fit(
        self, data: FloatMatrix, W: None = None, max_iterations=3
    ) -> "NaiveEstimator":
        """
        We keep some redunant arguments for pararity with StepForest
        """
        if self.is_fitted():
            logger.info(
                "Model has already been fitted - refitting and overwritting old resuts"
            )
        rows, columns = data.shape
        self.W = np.zeros((columns, columns)).astype(float)
        sample_size = self._get_sample_size(rows, self.sample_size)
        subset_idxs = np.random.randint(rows, size=sample_size)
        subset = data[subset_idxs,:]
        for target in range(columns):
            variables = set(range(columns)) - set([target])
            split_results = self.splitter(subset, target, variables)
            update = self.scorer(data, split_results)
            w_delta = np.zeros((columns, columns))
            w_delta[target, :] += update
            w_delta[:, target] += update  # assure symmetry
            self.W += w_delta
        self.fit_info.final_w.append(self.W.copy())
        return self


