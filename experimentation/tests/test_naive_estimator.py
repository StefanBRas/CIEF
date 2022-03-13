from experimentation.dataset import sample_new
from experimentation.naive_estimator import NaiveEstimatorRegression, NaiveEstimatorTree
import numpy as np
import pytest

def assert_array_equal_not(A,B):
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(A,B)


@pytest.mark.parametrize("estimator", [NaiveEstimatorTree, NaiveEstimatorRegression])
def test_naive_estimator(estimator):
    data = sample_new(rows=100,
            columns=5,
            data_type = "gaus", # type: ignore
            data_args= {"type": "chain", "diag": 0.0})
    model = estimator()
    initial_W = model.W.copy()
    model.fit(data)
    result_W = model.W.copy()
    assert_array_equal_not(initial_W, result_W)
