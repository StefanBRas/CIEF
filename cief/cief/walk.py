from dataclasses import dataclass, field
from cief.scorer.base_scorer import Scorer
from itertools import product
import pandas as pd
from pydot import Dot
from cief.utils import init_W, is_empty, lagged_zip, melt_mat, sample_wrt_w, update_W_and_clear_queue
from cief.splitter import Splitter 
from cief.scorer import Scorer
from typing import List, Literal, Optional, Tuple
from cief.types import Feature, FloatArray, FloatMatrix, KeepData, StepNew, WUpdateFrequency
from logging import getLogger
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import orjson


HAS_GRAPHVIZ=True

logger = getLogger()

@dataclass
class WalkFitInfo:
    steps: List[StepNew] = field(default_factory=list)
    w_deltas_applied: List[FloatMatrix] = field(default_factory=list)

    def get_normalized_w_df(self):
        deltas_applied = not is_empty(self.w_deltas_applied)
        if deltas_applied:
            dfs =  [melt_mat(mat) for mat in self.w_deltas_applied]
            for i, df in enumerate(dfs):
                df["step_iteration"] = i
            return pd.concat(dfs)
        else:
            raise ValueError("No deltas applied in walk")

class Walk:
    def __init__(
        self,
        data: FloatMatrix,
        W: FloatMatrix,
        target: int,
        set_size: int,
        tree_sample_size: int,
        splitter: Splitter,
        scorer: Scorer,
        step_max=3,
        W_update_queue: List[FloatMatrix] = None,
        w_update_frequency: WUpdateFrequency = "each_step",
        keep_data: KeepData = False
    ):
        self.data = data
        self.candidates = set(range(data.shape[1])) - set([target])
        self.W = W
        self.tree_sample_size = tree_sample_size
        self.target = target
        self.set_size = set_size
        self.step_max = step_max
        self.splitter = splitter
        self.scorer = scorer
        self.steps: List[StepNew] = []
        self.stopping_criteria = lambda x: x > 3
        self.w_update_frequency: WUpdateFrequency = w_update_frequency
        self.W_update_queue: List[FloatMatrix] = W_update_queue if W_update_queue is not None else []
        self.keep_data = keep_data
        if keep_data:
            self.fit_info = WalkFitInfo()
            self.w_deltas: List[FloatMatrix] = []
            self.W_initial = W.copy()
            self.w_deltas_array: List[FloatArray] = []

    def _sample_candidates(
        self, sample_target: int
    ) -> Tuple[List[Feature], List[Feature]]:
        prev_candidates = list(self.candidates)  # we want to log them for visulisation
        selected = sample_wrt_w(self.W, prev_candidates, sample_target, self.set_size)
        self.candidates = self.candidates - set(selected)
        return selected, prev_candidates

    def _update_w(self, step: StepNew):
        rows, columns = self.data.shape
        w_delta_vector = self.scorer(self.data, step.splits)
        update = (rows / self.tree_sample_size) * w_delta_vector
        w_delta = np.zeros((columns, columns))
        w_delta[self.target, :] += update
        w_delta[:, self.target] += update  # assure symmetry
        self.W_update_queue.append(w_delta)
        if self.keep_data: # TODO remove, legacy fitinfo
            self.w_deltas_array.append(update)
            w_delta = np.zeros((columns, columns))
            w_delta[self.target, :] += update
            w_delta[:, self.target] += update  # assure symmetry
            self.w_deltas.append(w_delta)
            self.w_deltas_array.append(update)


    def _take_step(self, sample_target) -> StepNew:
        selected, prev_candidates = self._sample_candidates(sample_target)
        splits = self.splitter(self.data, self.target, selected)
        step = StepNew(
            sample_target=sample_target, candidates=prev_candidates, splits=splits
        )
        self.steps.append(step)
        self._update_w(step)
        return step

    def fit(self) -> "Walk":
        logger.info("Starting walk")
        sample_target = self.target
        iteration = 0
        while not self.stopping_criteria(len(self.steps)) and len(self.candidates) > 0:
            logger.info("Taking step")
            step = self._take_step(sample_target)
            sample_target = step.get_best_split().feature
            if self.w_update_frequency == "each_step":
                logger.info(f"Updating in step for target {self.target}")
                w_delta = update_W_and_clear_queue(self.W, self.W_update_queue)
                if self.keep_data:
                    self.fit_info.w_deltas_applied.append(w_delta)
            if self.keep_data:
                self.fit_info.steps.append(step)
        return self

    def get_best_split(self):
        best_splits_by_step = [step.get_best_split() for step in self.steps]
        best_split = min(best_splits_by_step, key=lambda split: split.score)
        return best_split

    def _get_step_graph(
        self, step: StepNew, W: Optional[FloatMatrix] = None, d: Optional[int] = None
    ) -> nx.Graph:
        if W is None:
            if d is not None:
                W = init_W(d)
            else:
                raise ValueError("eiter W or d must be set")
        _W = np.round(W.copy(), decimals=3)
        G: nx.Graph = nx.from_numpy_array(_W)  # type: ignore
        for node, data in G.nodes.items():
            if node not in [*step.candidates, step.sample_target]:
                data["color"] = "grey90"
            elif node in step.candidates:
                data["color"] = "black"
                data["fillcolor"] = "yellow"
                data["style"] = "filled"
        for edge, data in G.edges.items():
            if step.sample_target in edge:
                if any([x in edge for x in step.candidates]):
                    pass
        return G

    def _get_walk_graph(
        self, d: int,
        type_: Literal["dot", "circo"] = "dot"
    ) -> nx.DiGraph:
        G = nx.DiGraph()
        for node in range(d): # TODO Maybe only considered nodes?
            G.add_node(node)
        for i, step in enumerate(self.steps):
            if i != 0:
                cur_step = self.steps[i-1]
                next_step = step
                # G.edges[(cur_step.sample_target, next_step.sample_target)]['color'] = 'black'
                G.add_edge(cur_step.sample_target, next_step.sample_target)
                for feature in cur_step.get_considered_features():
                    if feature != next_step.sample_target:
                        G.add_edge(cur_step.sample_target, feature, color="grey")
        layout = nx.drawing.layout.circular_layout(G)
        for node, data in G.nodes.items():
                data["color"] = "black"
                data["fillcolor"] = "yellow"
                data["style"] = "filled"
                data["pos"] = layout[node]
                data["shape"] = "circle" if node != self.target else "square"
        if type_ == "circo":
            for start, end in product(list(G.nodes), list(G.nodes)):
                if (start != end) and (start,end) not in G.edges and (end, start) not in G.edges:
                    G.add_edge(start, end, color="#00000000")
        return G

    def vis(self):
        f, axs = plt.subplots(len(self.steps), 3)
        W = self.W_initial.copy()
        for i, step in enumerate(self.steps):
            w_delta = self.w_deltas[i]
            w_normalized = W / W.max()
            sns.heatmap(w_normalized, annot=True, linewidths=0.5, ax=axs[i, 0])
            G = self._get_step_graph(step, W)
            layout = nx.drawing.layout.circular_layout(G)
            nx.draw_networkx(G, ax=axs[i, 1], pos=layout)
            w_delta_array = w_delta[step.sample_target, :].reshape((W.shape[0], 1))
            sns.heatmap(w_delta_array, annot=True, linewidths=0.5, ax=axs[i, 2])
            W += w_delta * 10
        return f

    def vis_new(self, layout_type: Literal["dot", "circo"] = "dot") -> Dot:
        G = self._get_walk_graph(10, layout_type)
        pydot = nx.nx_pydot.to_pydot(G)
        return pydot

    def _add_graph_to_plot(self, ax):
        G = nx.DiGraph()
        d = self.data.shape[1]
        for node in range(d): # TODO Maybe only considered nodes?
            G.add_node(node)
        step_edges = []
        sample_edges = []
        for step, next_step in lagged_zip(self.steps):
            # G.edges[(cur_step.sample_target, next_step.sample_target)]['color'] = 'black'
            G.add_edge(step.sample_target, next_step.sample_target)
            step_edges.append((step.sample_target, next_step.sample_target))
            for feature in step.get_considered_features():
                if feature != next_step.sample_target:
                    G.add_edge(step.sample_target, feature, color="grey")
                    sample_edges.append((step.sample_target, feature))
        for feature in next_step.get_considered_features():
            if feature == next_step.get_best_split().feature:
                G.add_edge(next_step.sample_target, feature)
                step_edges.append((next_step.sample_target, feature))
            else:
                G.add_edge(next_step.sample_target, feature)
                sample_edges.append((next_step.sample_target, feature))
        # for i, step in enumerate(self.steps):
        #     if i == 0:
        #         step_edges.append((cur_step.sample_target, next_step.sample_target))
        #     if i != 0:
        #         cur_step = self.steps[i-1]
        #         next_step = step
        #         # G.edges[(cur_step.sample_target, next_step.sample_target)]['color'] = 'black'
        #         G.add_edge(cur_step.sample_target, next_step.sample_target)
        #         step_edges.append((cur_step.sample_target, next_step.sample_target))
        #         for feature in cur_step.get_considered_features():
        #             if feature != next_step.sample_target:
        #                 G.add_edge(cur_step.sample_target, feature, color="grey")
        #                 sample_edges.append((cur_step.sample_target, feature))
        if HAS_GRAPHVIZ:
            layout = nx.drawing.nx_pydot.graphviz_layout(G)
        else:
            layout = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, ax=ax, pos=layout)
        labels = {node: str(node) for node in G.nodes}
        nx.draw_networkx_edges(G, 
                edgelist = step_edges,
                ax=ax, pos=layout)
        nx.draw_networkx_edges(G, 
                edgelist = sample_edges,
                alpha=0.2,
                ax=ax, pos=layout)
        ax.set_title("Walk")
        nx.draw_networkx_labels(G, ax=ax, pos=layout, labels = labels, font_color="black")
        return ax

    def _add_W_initial_to_plot(self, ax):
        sns.heatmap(self.W_initial, ax=ax, 
                xticklabels=True,
                yticklabels=True,
                annot=True,
                cbar=False)
        ax.set_title("W at walk start")

    def _add_W_final_to_plot(self, ax):
        sns.heatmap(self.W, ax=ax, 
                xticklabels=True,
                yticklabels=True,
                annot=True,
                cbar=False)
        ax.set_title("W at walk end")

    def _add_sample_value(self, ax):
        ax.axis("off")
        w_deltas = np.add.accumulate([self.W_initial, *self.w_deltas])[1:]
        w_deltas_arrays = [w_deltas[i][:, step.sample_target] for i, step in enumerate(self.steps)]
        mat = np.round(w_deltas_arrays, decimals=3).astype(str)
        cells = list(np.transpose(mat))
        columns = [f"{x.sample_target}" for x in self.steps]
        row_labels=[str(x) for x in range(self.data.shape[1])]
        ax.table(cellText=cells,
        colLabels=columns,
        colColours=["red" for x in columns],
        rowLabels=row_labels,
        rowColours=["red" for x in row_labels],
                cellLoc ='center', 
            loc ='upper left')
        ax.set_title("Sample Probablity Vector ")
        return ax

    def _add_step_score(self, ax):
        ax.axis("off")
        cells = list(np.transpose([np.round(w_delta,decimals = 3).astype(str) for w_delta in self.w_deltas_array])) # type: ignore
        columns = [f"{x.sample_target}" for x in self.steps]
        row_labels=[str(x) for x in range(self.data.shape[1])]
        ax.table(cellText=cells,
        colLabels=columns,
        colColours=["red" for x in columns],
        rowLabels=row_labels,
        rowColours=["red" for x in row_labels],
                cellLoc ='center', 
    loc ='upper left')
        return ax

    def get_plot(self):
        fig  = plt.figure(figsize=(6.4, 12))
        w_start_ax = fig.add_subplot(3,2,1)
        w_end_ax = fig.add_subplot(3,2,2)
        graph_ax = fig.add_subplot(3,1,2)
        step_score_ax = fig.add_subplot(3,2,5)
        sample_prob_ax = fig.add_subplot(3,2,6)
        # self.add_info(fig)
        self._add_W_initial_to_plot(w_start_ax)
        self._add_W_final_to_plot(w_end_ax)
        self._add_graph_to_plot(graph_ax)
        self._add_step_score(step_score_ax)
        self._add_sample_value(sample_prob_ax)
        return fig

    def to_json(self) -> bytes:
        json_dict = {
                "W": self.W.tolist(),
            "target": self.target,
            "keep_data": self.keep_data,
            }
        if self.keep_data:
            extra_data = {
                    "w_deltas": [w.tolist() for  w in self.w_deltas],
            "W_initial": self.W_initial.tolist(),
            "w_deltas_array": [w.tolist() for  w in self.w_deltas_array]
                    }
            json_dict["extra_data"] = extra_data
        return orjson.dumps(json_dict)

