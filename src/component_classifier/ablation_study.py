import json
import math
from collections import defaultdict
from importlib.resources import files

import holoviews as hv
import hvplot.pandas  # noqa F401
import mlflow
import pandas as pd
import panel as pn
from tqdm.auto import tqdm

from component_classifier.data_utils import (
    LABEL_COLS,
    ImageDataset,
    df_to_error_plot,
    get_metadata_df,
)
from component_classifier.main import loader_from_ds, start_training

try:
    hv.notebook_extension("bokeh")
except Exception:
    hv.extension("bokeh")

pn.extension()

ABLATION_RUNS_PATH = files("component_classifier") / "runs/ablation_study.json"
SUBSAMPLING_RUNS_PATH = files("component_classifier") / "runs/subsampling_study.json"


def train_dataset_size_ablation():
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set("Subsampling")
    meta_df = get_metadata_df()

    µs = [0, 7]
    subsamplings = [150, 500, 1000]  # Start with the smallest so it fails early if too small

    result = defaultdict(list)
    for subsampling in tqdm(subsamplings):
        for µ in µs:
            for _ in range(5):
                default_train_dev_unlabeled_loader = [
                    loader_from_ds(
                        ImageDataset.from_metadata_df(
                            meta_df, LABEL_COLS, split=split, n_cache=int(1e6), µ=µ, n_train=subsampling
                        )
                    )
                    for split in ["train", "test", "unlabeled"]
                ]
                train_size = len(default_train_dev_unlabeled_loader[0].dataset)
                assert math.isclose(subsampling, train_size, abs_tol=len(LABEL_COLS)), train_size  # Rounded down
                override_params = {"subsampling": subsampling, "µ": µ, "dataset_name": "engines"}
                run_id = start_training(*default_train_dev_unlabeled_loader, **override_params)
                result[f"{subsampling}_{µ}"].append(run_id)

    SUBSAMPLING_RUNS_PATH.write_text(json.dumps(result, indent=4))


def plot_subsampling():
    runs = json.load(open(files("component_classifier") / "runs/subsampling_study.json"))

    plot_data = []
    for run_name, run_ids in runs.items():
        subsampling, _, mu = run_name.rpartition("_")
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            plot_data.append(
                {"Error rate": 100 * (1 - run.data.metrics["dev MulticlassAccuracy"])}
                | {"supervised labels": int(subsampling), "mu": int(mu)}
            )

    df = pd.DataFrame(plot_data).replace({0: "µ0", 7: "µ7"}).assign(x="x")

    pn.serve(
        df_to_error_plot(
            df,
            x="supervised labels",
            y="Error rate",
            by="mu",
            groupby="x",
            by_color_map={"µ0": "blue", "µ7": "red"},
            override_params={"width": 300},
        )[0].opts(title=""),
        title="subsampling",
    )

    # plot = df.hvplot.box(
    #     by=["subsampling", "µ"],
    #     y="Error rate",
    #     grid=True,
    #     width=550,
    #     height=300,
    #     yformatter="%.0f%%",
    #     shared_axes=False,
    #     fontsize={"labels": 18, "xticks": 14, "yticks": 14},
    # )
    # plot
    # pn.serve(plot, title="subsampling_study")

    # plots["µ"].opts(xlabel="µ (unlabeled/labeled ratio)")
    # plots["λ"].opts(xlabel="λ (unlabeled loss ratio)")
    # plots["τ"].opts(xlabel="τ (pseudo-label threshold)")
    # plots["n_strong_aug"].opts(xlabel="#Strong augmentations")
    # # pn.serve(hv.Layout(plots).cols(2), title='ablation_study')
    # pn.serve(pn.Column(*[plots[v] for v in ["µ", "λ", "τ", "n_strong_aug"]]), title="ablation_study")


def plot_ablation():
    runs = json.load(open(files("component_classifier") / "runs/ablation_study.json"))

    plot_data = []
    for run_name, run_ids in runs.items():
        param, _, value = run_name.rpartition("_")
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            plot_data.append(
                {"Error rate": 100 * (1 - run.data.metrics["dev MulticlassAccuracy"])}
                | {"param": param, "value": float(value)}
            )

    df = pd.DataFrame(plot_data).rename(columns={"value": "magnitude"}).assign(none="")
    pn.serve(
        df_to_error_plot(
            df,
            x="magnitude",
            y="Error rate",
            by="none",
            groupby="param",
            by_color_map={"": "blue"},
            override_params={"ylim": (7, 29)},
        ),
        title="ablation",
    )

    # plots = df.hvplot.box(
    #     by="value",
    #     y="Error rate",
    #     groupby="param",
    #     grid=True,
    #     width=350,
    #     height=300,
    #     yformatter="%.0f%%",
    #     ylim=(15, 30),
    #     shared_axes=False,
    #     fontsize={"labels": 18, "xticks": 14, "yticks": 14},
    # )
    # plots = df.hvplot.scatter(
    #     x="value",
    #     y="Error rate",
    #     groupby="param",
    #     grid=True,
    #     width=350,
    #     height=300,
    #     yformatter="%.0f%%",
    #     ylim=(15, 30),
    #     shared_axes=False,
    #     fontsize={"labels": 18, "xticks": 14, "yticks": 14},
    # )

    # pn.serve(plots["µ"].opts(xlabel="µ (unlabeled/labeled ratio)"))
    # plots["λ"].opts(xlabel="λ (unlabeled loss ratio)")
    # plots["τ"].opts(xlabel="τ (pseudo-label threshold)")
    # plots["n_strong_aug"].opts(xlabel="#Strong augmentations")
    # # pn.serve(hv.Layout(plots).cols(2), title='ablation_study')
    # pn.serve(pn.Column(*[plots[v] for v in ["µ", "λ", "τ", "n_strong_aug"]]), title="ablation_study")


def perform_ablation_study():
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set("Ablation")

    n_train = 500
    meta_df = get_metadata_df()

    searches = {
        "n_strong_aug": [0, 1, 2, 4, 8],
        "µ": [0, 1, 5, 10, 20],
        "λ": [0.5, 0.75, 1, 1.5, 2],
        "τ": [0.5, 0.75, 0.95, 0.99],
    }

    result = defaultdict(list)
    for _ in range(5):
        cached_ds = [
            ImageDataset.from_metadata_df(meta_df, LABEL_COLS, split=split, n_cache=int(1e6), µ=7, n_train=n_train)
            for split in ["train", "test", "unlabeled"]
        ]

        for param, values in searches.items():
            for value in tqdm(values, desc=f'Optimising "{param}"', leave=True):
                ds = (
                    [
                        ImageDataset.from_metadata_df(
                            meta_df, LABEL_COLS, split=split, n_cache=int(1e6), µ=value, n_train=n_train
                        )
                        for split in ["train", "test", "unlabeled"]
                    ]
                    if param == "µ"
                    else cached_ds
                )
                override_params = {param: value, "subsampling": n_train, "dataset_name": "engines"}
                run_id = start_training(*map(loader_from_ds, ds), **override_params)
                result[f"{param}_{value}"].append(run_id)

    ABLATION_RUNS_PATH.write_text(json.dumps(result, indent=4))


def n_aug_dev_set():
    runs = json.load(open(files("component_classifier") / "runs/n_aug_study.json"))

    plot_data = []
    for tag, run_id in runs.items():
        run = mlflow.get_run(run_id)
        _, _, value = tag.rpartition("_")
        plot_data.append({"Error rate": 1 - run.data.metrics["dev MulticlassAccuracy"]} | {"n_aug": value})

    df = pd.DataFrame(plot_data)
    aug_plot = df.hvplot.scatter(
        x="n_aug",
        y="Error rate",
        grid=True,
        width=350,
        height=300,
        s=80,
        fontsize={"labels": 18, "xticks": 14, "yticks": 14},
    )
    pn.serve(pn.Column(aug_plot), title="n_aug")


if __name__ == "__main__":
    # perform_ablation_study()
    train_dataset_size_ablation()

    # plot_ablation()
