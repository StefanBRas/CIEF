from cief.types import Feature, FloatArray, FloatMatrix
from typing import List, Literal, Tuple
import numpy as np
from numpy.random import multivariate_normal
from sklearn.linear_model import LinearRegression
import pandas as pd
from math import floor
from typing import get_args
from typing import Protocol
from logging import getLogger

logger = getLogger()

def init_W(dim: int) -> FloatMatrix:
    mat = np.ones((dim, dim))
    np.fill_diagonal(mat, 0)
    return mat

def get_IB(B):
    d = B.shape[0]
    return np.linalg.inv(np.eye(d) - B)


MATRIX_TYPES = Literal[
    "no_dependencies",
    "fully_connected",
    "pairwise",
    "groups",
    "pairwise_directed",
    "groups_directed",
    "chain",
    "tree",
    "tree_mirrored"
]


class AdjacencyMatrices:
    types: List[MATRIX_TYPES] = list(get_args(MATRIX_TYPES))

    @staticmethod
    def no_dependencies(dim=3):
        return np.zeros((dim, dim))

    @staticmethod
    def fully_connected(dim=3):
        return np.ones((dim, dim))

    @staticmethod
    def pairwise(dim=3):
        blocks = floor(dim / 2)
        if not dim % 2 == 0:
            raise ValueError(f"only even numbers for now - change dim to {dim+1}")
        pair = np.ones((2, 2))
        non_pair = np.zeros((2, 2))
        mat_blocks = [
            [non_pair.repeat(i, axis=1), pair, non_pair.repeat(blocks - i - 1, axis=1)]
            for i in range(blocks)
        ]
        return np.block(mat_blocks)

    @staticmethod
    def pairwise_directed(dim=3):
        blocks = floor(dim / 2)
        if not dim % 2 == 0:
            raise ValueError(f"only even numbers for now - change dim to {dim+1}")
        pair = np.array([[1, 0], [1, 1]])
        non_pair = np.zeros((2, 2))
        mat_blocks = [
            [non_pair.repeat(i, axis=1), pair, non_pair.repeat(blocks - i - 1, axis=1)]
            for i in range(blocks)
        ]
        return np.block(mat_blocks)

    @staticmethod
    def groups_directed(dim=4, groups=2):
        if not dim % groups == 0:
            raise ValueError(f"Only works for groups with even size")
        block_size = dim // groups
        block_shape = (block_size, block_size)
        mat_blocks = []
        for i in range(groups):
            blocks = []
            for j in range(groups):
                if i == j:
                    blocks.append(np.tril(np.ones(block_shape)))
                else:
                    blocks.append(np.zeros(block_shape))
            mat_blocks.append(blocks)
        return np.block(mat_blocks)

    @staticmethod
    def groups(dim=4, groups=2):
        if not dim % groups == 0:
            raise ValueError(f"Only works for groups with even size")
        block_size = dim // groups
        block_shape = (block_size, block_size)
        mat_blocks = []
        for i in range(groups):
            blocks = []
            for j in range(groups):
                if i == j:
                    blocks.append(np.ones(block_shape))
                else:
                    blocks.append(np.zeros(block_shape))
            mat_blocks.append(blocks)
        return np.block(mat_blocks)


    @classmethod
    def get(cls, type: MATRIX_TYPES, dim=4, diag=0, **kwargs):
        mat_func = getattr(cls, type)
        mat = mat_func(dim=dim, **kwargs)
        np.fill_diagonal(mat, diag)
        return mat

    @classmethod
    def sample(cls, type: MATRIX_TYPES, size=1000, dim=4, diag=0, **kwargs):
        cov = cls.get(type, dim, diag=diag, **kwargs)
        return multivariate_normal(
            mean=np.zeros(dim), cov=cov, size=size, check_valid="warn"
        )


def get_lin_regs(data):
    dim = 4
    linregs = np.reshape(np.zeros(dim * dim), (dim, dim))
    for x in range(dim):
        X = data[:, [k for k in range(dim) if k != x]]
        y = data[:, x]
        lr = LinearRegression().fit(X, y)
        for y in range(dim):
            if y == x:
                linregs[x, x] = 0
            elif y > x:
                linregs[y, x] = lr.coef_[y - 1]  # type: ignore
            else:
                linregs[y, x] = lr.coef_[y]  # type: ignore

    [lr.coef_ for lr in linregs]


def flatten(l):
    return [item for sublist in l for item in sublist]


def sample_wrt_w(
    W: FloatMatrix, options: List[Feature], sample_target: Feature, set_size: int
) -> List[int]:
    size = min(set_size, len(options))
    if size <= 0:
        return []
    sample_weights = W[options, sample_target]
    p = sample_weights / sum(sample_weights)
    S = np.random.choice(options, size=size, p=p, replace=False)
    return S


def w_to_df(data: FloatMatrix, **kwargs):
    data = data.copy()
    idxs = np.triu_indices_from(data)
    data[idxs] = None
    df = pd.DataFrame(data)
    melted = df.melt(ignore_index=False)
    melted = melted.loc[~np.isnan(melted["value"])]
    melted["rows"] = melted.index.astype(str)
    melted["columns"] = melted["variable"].astype(str)
    melted_no_diag = melted.loc[melted["rows"] != melted["columns"]]
    melted_no_diag["name"] = melted_no_diag["rows"] + "-" + melted_no_diag["columns"]
    melted_no_diag.drop(["variable", "rows", "columns"], axis=1, inplace=True)
    for key, val in kwargs.items():
        melted_no_diag[key] = val
    return melted_no_diag


class PredictionModel(Protocol):
    def predict(self, data: FloatMatrix) -> FloatArray:
        self, data
        return np.array(1)


def predict_multiple(
    models: List[PredictionModel], data: np.ndarray
) -> Tuple[FloatArray, List[FloatArray]]:
    all_predictions = [tree.predict(data) for tree in models]
    mean_prediction = np.add.reduce(all_predictions) / len(all_predictions)
    return mean_prediction, all_predictions


def lagged_zip(obj):
    # creates a zipped iterator such the for
    # lagged_zip([1,2,3]) = zip([1,2,3], [2,3])
    return zip(obj, obj[1:])


def get_avg_split_vector(
    Z: FloatArray, Y: FloatArray, split_point: float
) -> FloatArray:
    """partitions an array and calculates avg in each partition"""
    below_idx = Z <= split_point
    above_idx = Z > split_point
    below = Y[below_idx]
    above = Y[above_idx]
    Y_split = np.zeros(Y.size)
    if below.size != 0:
        below_avg = sum(below) / below.size
        Y_split[below_idx] = below_avg
    if above.size != 0:
        above_avg = sum(above) / above.size
        Y_split[above_idx] = above_avg
    return Y_split


def update_W_and_clear_queue(W: FloatMatrix, W_update_queue: List[FloatMatrix]):
    w_delta = np.sum(W_update_queue, axis=0)  # type: ignore
    W += w_delta
    W_update_queue.clear()
    return w_delta


def is_empty(x: List):
    return len(x) == 0


def melt_mat(mat: FloatMatrix) -> pd.DataFrame:
    def get_identifier(x):
        if x["row"] < x["column"]:
            return str(x["row"]) + "-" + str(x["column"])
        else:
            return str(x["column"]) + "-" + str(x["row"])

    df = pd.DataFrame(mat).reset_index().melt("index")
    df.columns = ["row", "column", "value"]
    df["edge"] = df.apply(get_identifier, axis=1)
    return df.drop_duplicates("edge")


class AdjacencyMatricesNew:
    types: List[MATRIX_TYPES] = list(get_args(MATRIX_TYPES))

    @staticmethod
    def no_dependencies(dim=3):
        return np.zeros((dim, dim))

    @staticmethod
    def fully_connected(dim=3):
        return np.ones((dim, dim))

    @staticmethod
    def pairwise(dim=3):
        blocks = floor(dim / 2)
        if not dim % 2 == 0:
            raise ValueError(f"only even numbers for now - change dim to {dim+1}")
        pair = np.ones((2, 2))
        non_pair = np.zeros((2, 2))
        mat_blocks = [
            [non_pair.repeat(i, axis=1), pair, non_pair.repeat(blocks - i - 1, axis=1)]
            for i in range(blocks)
        ]
        return np.block(mat_blocks)

    @staticmethod
    def pairwise_directed(dim=3):
        blocks = floor(dim / 2)
        if not dim % 2 == 0:
            raise ValueError(f"only even numbers for now - change dim to {dim+1}")
        pair = np.array([[1, 0], [1, 1]])
        non_pair = np.zeros((2, 2))
        mat_blocks = [
            [non_pair.repeat(i, axis=1), pair, non_pair.repeat(blocks - i - 1, axis=1)]
            for i in range(blocks)
        ]
        return np.block(mat_blocks)

    @staticmethod
    def tree_mirrored(dim=5):
        if dim % 2 == 0:
             logger.warning(f"Only works for odd dimension, recieved {dim} - using {dim + 1} instead")
             dim = dim + 1
        mat = np.zeros((dim, dim))
        middle = np.floor(dim / 2)
        for i in range(dim):
            for j in range(dim):
                if i == middle:
                    if j < middle:
                        mat[i,j] = 1.0
                if i > middle:
                    if j == middle:
                        mat[i,j] = 1.0
        return mat

    @staticmethod
    def tree(dim=5):
        mat = np.zeros((dim, dim))
        mat[range(1,dim),0] = 1
        return mat

    @staticmethod
    def chain(dim=3):
        mat = np.zeros((dim,dim))
        for d in range(dim)[:-1]:
            mat[d+1,d] = 1.0
        return mat

    @staticmethod
    def groups_directed(dim=4, groups=2):
        if not dim % groups == 0:
            raise ValueError(f"Only works for groups with even size")
        block_size = dim // groups
        block_shape = (block_size, block_size)
        mat_blocks = []
        for i in range(groups):
            blocks = []
            for j in range(groups):
                if i == j:
                    blocks.append(np.tril(np.ones(block_shape)))
                else:
                    blocks.append(np.zeros(block_shape))
            mat_blocks.append(blocks)
        return np.block(mat_blocks)

    @staticmethod
    def groups(dim=4, groups=2):
        if not dim % groups == 0:
            raise ValueError(f"Only works for groups with even size")
        block_size = dim // groups
        block_shape = (block_size, block_size)
        mat_blocks = []
        for i in range(groups):
            blocks = []
            for j in range(groups):
                if i == j:
                    blocks.append(np.ones(block_shape))
                else:
                    blocks.append(np.zeros(block_shape))
            mat_blocks.append(blocks)
        return np.block(mat_blocks)

    @classmethod
    def get(cls, type: MATRIX_TYPES, dim=4, diag=0, **kwargs):
        mat_func = getattr(cls, type)
        mat = mat_func(dim=dim, **kwargs)
        np.fill_diagonal(mat, diag)
        return mat

    @classmethod
    def get_covariance_matrix(cls, type: MATRIX_TYPES, dim=4, diag=0.0, **kwargs):
        B = cls.get(type, dim=dim, diag=diag, **kwargs)
        IB = get_IB(B)
        return np.matmul(IB, np.transpose(IB))

    @classmethod
    def get_precision_matrix(cls, type: MATRIX_TYPES, dim=4, diag=0.0, **kwargs):
        covariance_matrix = cls.get_covariance_matrix(type, dim=dim, diag=diag, **kwargs)
        return np.linalg.inv(covariance_matrix)

    @classmethod
    def sample(cls, type: MATRIX_TYPES, size=1000, dim=4, diag=0.0, **kwargs):
        B = cls.get(type, dim, diag=0.0, **kwargs)
        IB = get_IB(B)
        Z = np.random.normal(size = dim * size).reshape((dim,size))
        return np.transpose(np.matmul(IB, Z))
        # cov = cls.get_covariance_matrix(type, dim, diag=diag, **kwargs)
        # return multivariate_normal(
        #     mean=np.zeros(dim), cov=cov, size=size, check_valid="warn"
        # )
