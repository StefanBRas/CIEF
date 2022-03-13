from cief.splitter.python_splitter import get_splitter
from cief.types import FloatArray
from cief.scorer.base_scorer import Scorer
from cief.cief import CIEF
import numpy as np

X = np.random.standard_normal(100).reshape((10,10))
cief = CIEF(
        set_size=2,
        step_max=2,
        splitter="explained_variance_splitter_python",
        scorer="explained_variance",
        depth = 3
        )
cief.fit(X, max_iterations = 20)
cief.predict(X,2)
print(cief.W[0:2, 0:2])
mean_predictions, predictions_per_tree = cief.predict(X,0)

def unit_interval(Y,X) -> float:
    dist = np.abs(Y - X)
    close = dist < 1
    return(np.sum(1 - dist[close]))
unit_scorer = Scorer(score_func = unit_interval)
unit_splitter = get_splitter(unit_interval)
cief = CIEF(
        set_size=2,
        step_max=2,
        splitter=unit_splitter,
        scorer= unit_scorer,
        depth = 3,
        )
cief.fit(X, max_iterations=20)
print(cief.W[0:2, 0:2])
mean_predictions, predictions_per_tree = cief.predict(X,0)

