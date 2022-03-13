from experimentation.dataset import get_precision_mat_as_edge_dict
from experimentation.experiment import Experiment
from experimentation.specifications import DataSpecification, ExperimentSpecification, ModelSpecification
import pytest

@pytest.fixture
def experiment_spec():
        return ExperimentSpecification(
            name=f"new_experiment_{2}",
            seed=123,
            model_specification = ModelSpecification(
                splitter="explained_variance_splitter_python",
                scorer="explained_variance",
                w_update_frequency="each_step",
                max_iterations=10,
                step_max=3,
                set_size=3,
                depth=3,
                ),
            data_specification = DataSpecification(
                rows=100,
                columns=8,
                type="gaus",
                args={"type": "chain", "diag": 0.0}
                )
            )

@pytest.fixture
def experiment(experiment_spec):
    return Experiment.from_spec(experiment_spec)

def test_cov_connected_dict(experiment_spec: ExperimentSpecification):
    data_spec = experiment_spec.data_specification
    edge_dict = get_precision_mat_as_edge_dict(
            data_type = "gaus",
            columns = 3,
            data_args = {"type": "chain", "diag": 0.0}
            )
    expected = {
            '0-0': True,
            '0-1': True,
            '0-2': False,
            '1-1': True,
            '1-2': True,
            '2-2': True 
            }
    assert edge_dict == expected

