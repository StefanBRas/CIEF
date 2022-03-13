from typing import Dict, Literal
from cief.scorer.python_score_functions import mse_scorer as _mse_scorer, explained_variance_scorer as _explained_variance_scorer
from .base_scorer import Scorer, scale_to_0_1 


ScorerType = Literal["mse", "mse_normalized", "explained_variance"]
SCORERS: Dict[ScorerType, Scorer] = {
        "mse": Scorer(score_func = _mse_scorer), 
        "mse_normalized": Scorer(score_func = _mse_scorer, normalizer = scale_to_0_1), 
        "explained_variance": Scorer(score_func = _explained_variance_scorer)
        }
