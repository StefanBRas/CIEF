import sys
import os
from experimentation import OUTPUT_PATH
from experimentation.experiment import Experiment
from experimentation.specifications import (
    ExperimentSpecification,
)
from forestpathwalk import __version__

def run_experiment(spec_path: str):
    spec = ExperimentSpecification.load_json(spec_path)
    experiment = Experiment.from_spec(spec)
    experiment.fit()
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    if experiment.keep_data:
        fig = experiment.get_full_plot()
        fig.savefig(os.path.join(OUTPUT_PATH, f"{experiment.name}.pdf"))
        experiment.get_full_W_df().to_parquet(os.path.join(OUTPUT_PATH, f"{experiment.name}_fit_info.parquet"))
    experiment.get_final_W_df().to_parquet(os.path.join(OUTPUT_PATH, f"{experiment.name}_result.parquet"))


if __name__ == "__main__":
    run_experiment(sys.argv[1])
