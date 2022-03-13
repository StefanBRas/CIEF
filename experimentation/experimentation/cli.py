from typing import List
from experimentation import OUTPUT_PATH, PARQUET_INFO_PATH, PARQUET_RESULT_PATH, SPEC_PATH, create_specs as _create_specs
from experimentation.run_experiments import run_experiment
import typer
import time
import os
import pandas as pd

app = typer.Typer()

@app.command()
def create_specs():
    _create_specs()

@app.command()
def run(spec_paths: List[str]):
    for spec_path in spec_paths:
        run_experiment(spec_path)



@app.command()
def run_thread(spec_paths: List[str]):
    start = time.perf_counter()
    if spec_paths[0] == "all":
        file_names = os.listdir(SPEC_PATH)
        spec_paths = [os.path.join(SPEC_PATH, name) for name in file_names if name.startswith("keepdata_True")]
    if spec_paths[0] == "all_fast":
        file_names = os.listdir(SPEC_PATH)
        spec_paths = [os.path.join(SPEC_PATH, name) for name in file_names if name.startswith("keepdata_False")]
    if spec_paths[0] == "test":
        file_names = os.listdir(SPEC_PATH)
        spec_paths = [os.path.join(SPEC_PATH, name) for name in file_names if name.startswith("test")]
    from multiprocessing import Pool
    print("running experiment:")
    print(spec_paths)
    with Pool(10) as p:
        p.map(run_experiment, spec_paths)
    end = time.perf_counter()
    print(f"time: {end - start}")
    collect_parquets()


def _collect_parquets(ending: str, output_path):
    def collect_parquets():
        if os.path.exists(output_path):
            os.remove(output_path)
        paths = [path for path in os.listdir(OUTPUT_PATH) if path.endswith(f"{ending}.parquet")]
        if paths == []:
            print(f"no files ending in {ending} found")
            return
        parquets_loaded = [pd.read_parquet(os.path.join(OUTPUT_PATH, path)) for path in paths]
        full_df = pd.concat(parquets_loaded)
        full_df.to_parquet(output_path)
    return collect_parquets

collect_final_results = _collect_parquets("result", PARQUET_RESULT_PATH)
collect_fit_info = _collect_parquets("fit_info", PARQUET_INFO_PATH)

@app.command()
def collect_parquets():
    collect_final_results()
    collect_fit_info()

@app.command()
def collect_results():
    collect_final_results()

if __name__ == "__main__":
    app()
