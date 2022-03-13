from cief.scorer.python_score_functions import explained_variance_scorer, mse_scorer
from cief.types import Feature, FloatMatrix, SplitResult
from typing import Callable, Dict, Iterable, List, Literal
from .python_splitter import get_splitter


Splitter = Callable[[FloatMatrix, Feature, Iterable[Feature]], List[SplitResult]]


SplitterType = Literal[
        "mse_splitter_python",
        "explained_variance_splitter_python",
        ]

SPLITTERS: Dict[SplitterType, Splitter] = {
        "mse_splitter_python": get_splitter(mse_scorer),
       "explained_variance_splitter_python": get_splitter(explained_variance_scorer),
        }

