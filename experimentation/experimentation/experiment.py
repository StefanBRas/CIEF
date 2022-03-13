
from experimentation import naive_estimator
from experimentation.naive_estimator import NaiveEstimatorRegression, NaiveEstimatorTree
from typing import Any, Literal, Mapping, Optional, Union
from matplotlib.figure import Figure
from experimentation.specifications import ExperimentSpecification 
import matplotlib.pyplot as plt
import time
from experimentation.dataset import (DataSetArgs, DataSetType, get_adj_mat_as_edge_dict, get_cov_mat_as_edge_dict, get_precision_mat_as_edge_dict, sample_new, get_adj_mat_new,
        get_cov_mat, get_precision_mat
        )
from forestpathwalk import __version__
import seaborn as sns
from forestpathwalk.utils import w_to_df
from forestpathwalk.step_forest import ScorerType, SplitterType, StepForest
from forestpathwalk.types import WUpdateFrequency
import networkx as nx
import numpy as np
import pandas as pd
import os


SPEC_PATH = "data/inputs"
OUTPUT_PATH = "data/outputs"
PARQUET_INFO_PATH = os.path.join(OUTPUT_PATH, "full_info.parquet")
HAS_GRAPHVIZ = True
output_path = os.path.join(os.path.dirname(__file__), "outputs")
EstimatorType = Literal["CIEF", "NaiveEstimatorTree", "NaiveEstimatorRegression"]
ESTIMATORS: Mapping[EstimatorType, Any] = {"CIEF": StepForest, 
        "NaiveEstimatorTree": NaiveEstimatorTree,
        "NaiveEstimatorRegression": NaiveEstimatorRegression}



class Experiment:

    def __init__(self, 
            splitter: SplitterType,
            scorer: ScorerType,
            w_update_frequency: WUpdateFrequency,
            max_iterations: int,
            step_max: int,
            set_size: int,
            depth: int,
            data_rows: int,
            data_columns: int,
            data_type: DataSetType,
            data_args: DataSetArgs,
            name: str,
            sample_size: Union[float, int],
            seed: int = None,
            keep_data = True,
            estimator: EstimatorType = "CIEF"
            ):
        if data_columns % 2 == 0:
            raise ValueError("only use odd dimensions because of the graph types")
        self.name = name
        self.splitter  = splitter
        self.scorer  = scorer
        self.max_iterations  = max_iterations
        self.step_max  = step_max
        self.set_size  = set_size
        self.w_update_frequency: WUpdateFrequency = w_update_frequency
        self.depth= depth
        self.seed = seed if seed is not None else 123
        self.data_rows = data_rows
        self.data_columns = data_columns
        self.data_type = data_type
        self.data_args = data_args 
        self.sample_size = sample_size
        self.step_forest = StepForest(
                splitter=self.splitter,
                scorer=self.scorer,
                step_max=self.step_max,
                set_size=self.set_size,
                depth=self.depth,
                keep_data=keep_data,
                w_update_frequency = w_update_frequency,
                sample_size=sample_size
                )
        self.naive_estimator_tree = NaiveEstimatorTree(
                splitter=self.splitter,
                scorer=self.scorer,
                sample_size=sample_size
                )
        self.naive_estimator_linreg = NaiveEstimatorRegression(
                scorer=self.scorer,
                sample_size=sample_size
                )
        self.fit_duration: float = 0.0
        self.keep_data = keep_data

    def fit(self):
        np.random.seed(self.seed)
        start = time.perf_counter()
        data = sample_new(rows=self.data_rows,
                columns=self.data_columns,
                data_type = self.data_type, # type: ignore
                data_args=self.data_args)
        self.step_forest.fit(data, max_iterations=self.max_iterations)
        end = time.perf_counter()
        self.fit_duration = end - start
        self.naive_estimator_linreg.fit(data)
        self.naive_estimator_tree.fit(data)

    def _enrich_df(self, df):
        connected_dict = get_cov_mat_as_edge_dict(
                data_type=self.data_type, # type: ignore
                columns = self.data_columns,
                data_args=self.data_args)
        df["connected_cov"] = df["edge"].replace(connected_dict)
        connected_dict = get_precision_mat_as_edge_dict(
                data_type=self.data_type, # type: ignore
                columns = self.data_columns,
                data_args=self.data_args)
        df["connected_precision"] = df["edge"].replace(connected_dict)
        df["splitter"] = self.splitter
        df["scorer"] = self.scorer
        df["step_max"] = self.step_max
        df["set_size"] = self.set_size
        df["depth"] = self.depth
        df["experiment_name"] = self.name
        df["data_rows"] = self.data_rows
        df["data_columns"] = self.data_columns
        df["graph_type"] = self.data_args.get("type", "UNSPECIFIED")
        df["fit_duration"] = self.fit_duration
        df["w_update_frequency"] = self.w_update_frequency
        df["sample_size"] = self.sample_size
        df["max_iterations"] = self.max_iterations
        df["max_iterations"] = self.max_iterations


    def get_final_W_df(self):
        estimators = {"CIEF": self.step_forest,
        "NaiveEstimatorTree": self.naive_estimator_tree,
        "NaiveEstimatorRegression": self.naive_estimator_linreg
        }
        dfs = []
        for name, model in estimators.items():
            model_df = model.fit_info.get_final_w()
            model_df["estimator"] = name
            dfs.append(model_df)
        df = pd.concat(dfs)
        self._enrich_df(df)
        return df


    def get_full_W_df(self):
        df = self.step_forest.fit_info.get_normalized_w_df()
        self._enrich_df(df)
        return df

    def save_parquet(self, path: str = None):
        full_path = os.path.join(OUTPUT_PATH, f"{self.name}.parquet") if path is None else path
        self.get_full_W_df().to_parquet(path)


    def get_full_plot(self):
        fig: Figure  = plt.figure(figsize=(12.8, 16))
        table_ax = fig.add_subplot(4,1,1)
        graph_ax = fig.add_subplot(4,2,3)
        adjaceny_mat_ax = fig.add_subplot(4,2,4)
        covariance_mat_ax = fig.add_subplot(4,2,5)
        precision_mat_ax = fig.add_subplot(4,2,6)
        w_plot_ax = fig.add_subplot(4,1,4)
        self.add_info(fig)
        self.add_table(table_ax)
        self.add_graph(graph_ax)
        self.add_sample_matrix(adjaceny_mat_ax)
        self.add_covariance_matrix(covariance_mat_ax)
        self.add_precision_matrix(precision_mat_ax)
        self.add_plot_seaborn(w_plot_ax)
        return fig

    def _get_adj_mat_new(self):
         return get_adj_mat_new(
                data_type=self.data_type, # type: ignore
                columns = self.data_columns,
                data_args=self.data_args)

    def _get_cov_mat(self):
         return get_cov_mat(
                data_type=self.data_type, # type: ignore
                columns = self.data_columns,
                data_args=self.data_args)

    def _get_precision_mat(self):
         return get_precision_mat(
                data_type=self.data_type, # type: ignore
                columns = self.data_columns,
                data_args=self.data_args)


    def add_graph(self, ax):
        mat_original = self._get_adj_mat_new()
        if mat_original is None:
            raise ValueError("no adjaceny matrix found in data")
        mat = np.transpose(mat_original.copy())
        np.fill_diagonal(mat, 0)
        G = nx.from_numpy_matrix(mat, create_using=nx.DiGraph) # type: ignore
        if HAS_GRAPHVIZ:
            layout = nx.drawing.nx_pydot.graphviz_layout(G)
        else:
            layout = nx.spring_layout(G)
        nx.draw_networkx(G, ax=ax, pos=layout)
        ax.set_title("Data generating graph")
        return ax

    def add_sample_matrix(self, ax):
        mat = self._get_adj_mat_new()
        if mat is None:
            raise ValueError("no adjaceny matrix found in data")
        sns.heatmap(mat, ax=ax, 
                xticklabels=True,
                yticklabels=True,
                annot=True,
                cbar=False)
        ax.set_title("A, X=AZ, Z ~ N(0,1)")
        return ax

    def add_covariance_matrix(self, ax):
        mat = self._get_cov_mat()
        if mat is None:
            raise ValueError("no adjaceny matrix found in data")
        sns.heatmap(mat, ax=ax, 
                xticklabels=True,
                yticklabels=True,
                annot=True,
                cbar=False)
        ax.set_title("Covariance Matrix")
        return ax

    def add_precision_matrix(self, ax):
        mat = self._get_precision_mat()
        if mat is None:
            raise ValueError("no adjaceny matrix found in data")
        sns.heatmap(mat, ax=ax, 
                xticklabels=True,
                yticklabels=True,
                annot=True,
                cbar=False)
        ax.set_title("Precision Matrix")
        return ax


    def add_plot_seaborn(self, ax):
        df = self.get_full_W_df()
        def apply_connected(x):
            if x['connected_cov'] and x['connected_precision']:
                return "cov_and_precision"
            elif x['connected_cov']:
                return "cov"
            elif x['connected_precision']:
                return "precision"
            else:
                return "not_connected"
        df["connected"] = df.apply(apply_connected, axis=1)
        data_not_connected = df[df['connected'] == "not_connected"]
        data_connected = df[df['connected'] != "not_connected"]
        data_faded = df[
            (df['connected'] == "not_connected") |
            (df['connected'] == "cov")
        ]
        data_non_faded = df[
            (df['connected'] == "cov_and_precision") |
            (df['connected'] == "precision")
        ]
        data_connected = df[df['connected'] != "not_connected"]
        sns.lineplot(data=data_non_faded,
                hue="connected",
                x="iteration_index", y="value", units="edge",
                estimator= None, ax = ax)
        sns.lineplot(data=data_faded,
                hue="connected",
                x="iteration_index", y="value", units="edge",
                alpha= 0.4,
                estimator= None, ax = ax)
        return ax

    def add_info(self, fig):
        fig.suptitle(self.name)
        return fig

    def add_table(self, ax):
        ax.axis("off")
        cells = [
                [
                f"Splitter: {self.splitter}",
                f"Scorer: {self.scorer}",
                f"Max iterations: {self.max_iterations}"],
                [
                f"Step max: {self.step_max}",
                f"Set size: {self.set_size}",
                f"Depth: {self.depth}"],
                [
                f"size: ({self.data_rows}, {self.data_columns})",
                f"size: {self.w_update_frequency}",
                f"duration: {self.fit_duration}",
                ]
                ]
        ax.table(cellText=cells,
                cellLoc ='center', 
                fontsize=100.0,
    loc ='upper left')
        return ax

    @classmethod
    def from_spec(cls, spec: ExperimentSpecification):
        model = spec.model_specification
        data = spec.data_specification

        return cls(
            splitter = model.splitter,
            scorer = model.scorer,
            w_update_frequency=model.w_update_frequency,
            max_iterations = model.max_iterations,
            step_max = model.step_max,
            set_size = model.set_size,
            depth= model.depth,
            data_rows=data.rows,
            data_columns=data.columns,
            data_type= data.type,
            data_args= data.args,
            name = spec.name,
            sample_size = model.sample_size,
            seed= spec.seed,
            keep_data=spec.keep_data)

    @classmethod
    def load_spec(cls, path):
        spec = ExperimentSpecification.load_json(path)
        return cls.from_spec(spec)

    def save_result(self, path: str):
        fig = self.get_full_plot()
        full_path = os.path.join(path, f"{self.name}.pdf")
        fig.savefig(full_path)

