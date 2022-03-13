from forestpathwalk.utils import AdjacencyMatrices, AdjacencyMatricesNew, melt_mat
from typing import Literal, Mapping, Union
import pandas as pd


DataSetType = Literal["gaus"]

Primitive = Union[int, float, str, None, bool] 
DataSetArgs = Mapping[str, Primitive]

def sample(
            data_type: DataSetType,
            columns: int,
            rows: int,
            data_args: DataSetArgs
            ):
    if data_type == "gaus":
        return AdjacencyMatrices.sample(size = rows,
                dim = columns,  **data_args)
    else:
        raise ValueError(f"Data type: {data_type} not supported")

def get_adj_mat(
            data_type: DataSetType,
            columns: int,
            data_args: DataSetArgs
            ):
    if data_type == "gaus":
        return AdjacencyMatrices.get(dim = columns, **data_args)
    else:
        raise ValueError(f"Data type: {data_type} not supported")

def get_adj_mat_as_edge_dict(
            data_type: DataSetType,
            columns: int,
            data_args: DataSetArgs
            ):
    if data_type == "gaus":
        adj_mat = get_adj_mat(data_type, columns, data_args)
        melted = melt_mat(adj_mat)
        melted["connected"] = melted["value"] == 1.0
        return dict(zip(melted["edge"], melted["connected"]))
    else:
        raise ValueError(f"Data type: {data_type} not supported")

def sample_new(
            data_type: DataSetType,
            columns: int,
            rows: int,
            data_args: DataSetArgs
            ):
    if data_type == "gaus":
        return AdjacencyMatricesNew.sample(size = rows,
                dim = columns,  **data_args)
    else:
        raise ValueError(f"Data type: {data_type} not supported")

def _get_mat(attr: str):
    def get_mat(
                data_type: DataSetType,
                columns: int,
                data_args: DataSetArgs
                ):
        if data_type == "gaus":
            func = getattr(AdjacencyMatricesNew, attr)
            return func(dim = columns, **data_args)
        else:
            raise ValueError(f"Data type: {data_type} not supported")
    return get_mat

get_adj_mat_new = _get_mat("get")
get_cov_mat = _get_mat("get_covariance_matrix")
get_precision_mat = _get_mat("get_precision_matrix")

def get_precision_mat_as_edge_dict(
            data_type: DataSetType,
            columns: int,
            data_args: DataSetArgs
            ):
    if data_type == "gaus":
        adj_mat = get_precision_mat(data_type, columns, data_args)
        melted = melt_mat(adj_mat)
        melted["connected"] = abs(melted["value"]) > 0.001
        result = dict(zip(melted["edge"], melted["connected"]))
        return result
    else:
        raise ValueError(f"Data type: {data_type} not supported")

def get_cov_mat_as_edge_dict(
            data_type: DataSetType,
            columns: int,
            data_args: DataSetArgs
            ):
    if data_type == "gaus":
        adj_mat = get_cov_mat(data_type, columns, data_args)
        melted = melt_mat(adj_mat)
        melted["connected"] = abs(melted["value"]) > 0.001
        return dict(zip(melted["edge"], melted["connected"]))
    else:
        raise ValueError(f"Data type: {data_type} not supported")

        
if __name__ == "__main__":
    print(sample("gaus", 6, 100, {"type": "pairwise", "diag": 1.1}))

