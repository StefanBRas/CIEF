from experimentation.experiment import Experiment
from typing import List
from forestpathwalk.step_forest import WUpdateFrequency
from experimentation.specifications import (
    ExperimentSpecification,
    ModelSpecification,
    DataSpecification,
)
from forestpathwalk import __version__
import os
from itertools import product

SPEC_PATH = "data/inputs"
OUTPUT_PATH = "data/outputs"
PARQUET_INFO_PATH = os.path.join(OUTPUT_PATH, "full_info.parquet")
PARQUET_RESULT_PATH = os.path.join(OUTPUT_PATH, "results.parquet")

data_set = {
        "standard": DataSpecification(rows=100, columns=6, type="gaus", args={"type": "pairwise", "diag": 1.1}),
        "small_dim": DataSpecification(rows=100, columns=30, type="gaus", args={"type": "pairwise", "diag": 1.1}),
        "small_dim_group": DataSpecification(rows=100, columns=30, type="gaus", args={"type": "groups", "diag": 1.1, "groups": 2}),
        "large_dim": DataSpecification(rows=300, columns=30, type="gaus", args={"type": "pairwise", "diag": 1.1}),
        "large_dim_group": DataSpecification(rows=300, columns=30, type="gaus", args={"type": "groups", "diag": 1.1, "groups": 2})
        }
frequencies: List[WUpdateFrequency] = ["each_step", "each_walk", "each_tree", "each_target"]

experiments = [
        ExperimentSpecification(
            name=f"keepdata_{keep_data}_rows_{rows}_data_{data_type}_frequency_{frequency}_iterations_{iterations}_sample_size_{sample_size}",
            seed=123,
            keep_data = keep_data,
            model_specification = ModelSpecification(
                splitter="explained_variance_splitter_python",
                scorer="explained_variance",
                w_update_frequency=frequency,
                max_iterations=iterations,
                step_max=3,
                set_size=3,
                depth=3,
                sample_size=sample_size
                ),
            data_specification = DataSpecification(
                rows=rows,
                columns=15,
                type="gaus",
                args={"type": data_type, "diag": 0.0}
                )
            )
        for iterations, rows, frequency, data_type, keep_data, sample_size in product(
            [500],
            [500],
            ["each_step", "each_target"],
            ["tree", "tree_mirrored", "chain"],
            [True,False],
            [100]
            )
        ]

test_experiments = [
        ExperimentSpecification(
            name=f"test_rows_{rows}_data_{data_type}_frequency_{frequency}_iterations_{iterations}_keepdata_{keep_data}_sample_size_{sample_size}",
            seed=123,
            keep_data = keep_data,
            model_specification = ModelSpecification(
                splitter="explained_variance_splitter_python",
                scorer="explained_variance",
                w_update_frequency=frequency,
                max_iterations=iterations,
                step_max=3,
                set_size=3,
                depth=3,
                sample_size=sample_size
                ),
            data_specification = DataSpecification(
                rows=rows,
                columns=15,
                type="gaus",
                args={"type": data_type, "diag": 0.0}
                )
            )
        for iterations, rows, frequency, data_type, keep_data, sample_size in product(
            [10],
            [100],
            ["each_step", "each_walk", "each_tree", "each_target"],
            ["tree", "tree_mirrored", "chain"],
            [True, False],
            [100]
            )
        ]


def create_specs():
    os.makedirs(SPEC_PATH, exist_ok=True)
    names = []
    for experiment in [*experiments, *test_experiments]:
        print(experiment.name)
        if experiment.name in names:
            print("experiment has already been seen - perhaps the name is not unique")
        names.append(experiment.name)
        path = os.path.join(SPEC_PATH, f"{experiment.name}.json")
        if os.path.exists(path):
            saved_spec = ExperimentSpecification.load_json(path)
            if saved_spec != experiment:
                print("found old version of spec, updating")
                experiment.save_json(path)
        else:
            print("found no old version of spec, saving")
            experiment.save_json(path)

if __name__ == "__main__":
    # create_specs()
    import numpy as np
