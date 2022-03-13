from sklearn.metrics import explained_variance_score as _explained_variance_score
import numpy as np

def mse_scorer(X, Y) -> float:
    mse = ((X - Y) ** 2).mean()
    if mse == 0.0:
        return 1 / (mse + 0.00001)
    else:
        return 1 / mse

def explained_variance_scorer(X,Y) -> float:
    score = _explained_variance_score(X,Y)
    return np.max([score, 0.0]) # type: ignore
