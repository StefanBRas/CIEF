from dataclasses import dataclass
from experimentation.dataset import DataSetArgs, DataSetType
from pydantic.json import pydantic_encoder
from typing import Dict, Union
import json

from forestpathwalk.step_forest import ScorerType, SplitterType, WUpdateFrequency



@dataclass
class ModelSpecification:
    splitter: SplitterType
    scorer: ScorerType
    w_update_frequency: WUpdateFrequency
    max_iterations: int
    step_max: int
    set_size: int
    depth: int
    sample_size: Union[float, int]


@dataclass
class DataSpecification:
    rows: int
    columns: int
    type: DataSetType # something norm gaus
    args: DataSetArgs


@dataclass
class ExperimentSpecification:
    name: str
    model_specification: ModelSpecification 
    data_specification: DataSpecification
    seed: int
    keep_data: bool = True

    def save_json(self, path: str):
        with open(path, 'w') as file:
            json.dump(self, file, default=pydantic_encoder)

    @classmethod
    def load_json(cls, path: str):
        with open(path) as file:
            content = json.load(file)
        return cls.from_dict(content)

    @classmethod
    def from_dict(cls, dict_: Dict):
        content = dict_.copy()
        content["model_specification"] = ModelSpecification(**content["model_specification"])
        content["data_specification"] = DataSpecification(**content["data_specification"])
        return cls(**content) # type: ignore


