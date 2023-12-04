import json
from collections import defaultdict
from importlib.resources import files

import holoviews as hv
import hvplot.pandas  # noqa F401
import mlflow
import pandas as pd
import panel as pn
import torch
from tqdm.auto import tqdm

from component_classifier.data_utils import df_to_error_plot
from component_classifier.main import ImageDataset, loader_from_ds, start_training
from component_classifier.train_utils import get_model

try:
    hv.notebook_extension("bokeh")
except Exception:
    hv.extension("bokeh")

pn.extension()

DATASET_RUNS_PATH = files("component_classifier") / "runs/dataset_study.json"

ds_sizes = {
    # CIFAR100: This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class.
    # This means there are 600 * 100 = 60,000 training images. This makes it impossible to have (10_000 training + 10_000*7) unlabeled images
    "CIFAR10": [40, 250, 4000],
    "CIFAR100": [400, 2500],
    "SVHN": [40, 250, 1000],
    # STL10: 500 training images, 800 test images per class -> 1300*10 = 13_000 training images total
    "STL10": [1000],
}


def start_5_fold_training(ds_name, train_size, override_params):
    with mlflow.start_run() as parent_run:
        for _ in range(5):
            train, test, unlabeled = [
                ds.cache_n_files(int(1e6))  # cache all
                for ds in ImageDataset.from_torch_dataset(ds_name, n_samples=train_size, µ=override_params["µ"])
            ]

            start_training(loader_from_ds(train), loader_from_ds(test), loader_from_ds(unlabeled), **override_params)
        parent_run_id = parent_run.info.run_id
    return parent_run_id


def perform_dataset_study():
    µs = [0, 7]
    for ds_name, sizes in ds_sizes.items():
        for train_size in sizes:
            for µ in µs:
                train, test, unlabeled = ImageDataset.from_torch_dataset(ds_name, n_samples=train_size, µ=µ)
                assert µ <= int(len(unlabeled) / len(train)), int(len(unlabeled) / len(train))

    result = defaultdict(list)
    try:
        for ds_name, sizes in ds_sizes.items():
            mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(ds_name)
            for train_size in sizes:
                for µ in µs:
                    override_params = {
                        "eval_split": 5,
                        "fine_tune": "final_layer",
                        "subsampling": train_size,
                        "µ": µ,
                        "model_name": "resnet50_imagenet",
                        "dataset_name": ds_name,
                        "w_decay": 1e-3 if ds_name == "CIFAR100" else 5e-4,
                    }
                    run_id = start_5_fold_training(ds_name, train_size, override_params)
                    result[f"{ds_name}_{train_size}_{µ}"] = run_id
    finally:
        DATASET_RUNS_PATH.write_text(json.dumps(result, indent=4))


def display_dataset_study():
    runs = json.loads(DATASET_RUNS_PATH.read_text())

    runs_df = mlflow.search_runs(experiment_names=list(ds_sizes))
    child_runs = runs_df.groupby("tags.mlflow.parentRunId").run_id.agg(list)

    plot_data = []
    for run_name, parent_run_id in runs.items():
        ds, subsampling, mu = run_name.split("_")
        for run_id in child_runs[parent_run_id]:
            run = mlflow.get_run(run_id)
            plot_data.append(
                {"error_rate": (1 - run.data.metrics["dev MulticlassAccuracy"])}
                | {"Dataset": ds, "size": int(subsampling), "µ": int(mu)}
            )

    df = pd.DataFrame(plot_data)
    df["size"] = df["size"].apply(lambda x: f"{x} labels")
    groups = ["Dataset", "size", "µ"]
    agg_df = df.groupby(groups).agg({"mean"}).droplevel(1, axis=1)
    # format the error rate as pct
    agg_df["error_rate_delta"] = df.groupby(groups).error_rate.apply(
        lambda x: max(x.max() - x.mean(), x.mean() - x.min())
    )
    agg_df["error_summary"] = (
        agg_df["error_rate"].apply(lambda x: f"${x:.2%}")
        + agg_df["error_rate_delta"].apply(lambda x: f"_{{\\pm{x:.2%}}}$")
    ).str.replace("%", "")
    agg_df = agg_df.unstack("µ")
    idx = pd.MultiIndex.from_tuples(
        [
            ("CIFAR10", "40 labels"),
            ("CIFAR10", "250 labels"),
            ("CIFAR10", "4000 labels"),
            ("CIFAR100", "400 labels"),
            ("CIFAR100", "2500 labels"),
            ("CIFAR100", "10000 labels"),
            ("SVHN", "40 labels"),
            ("SVHN", "250 labels"),
            ("SVHN", "1000 labels"),
            ("STL10", "1000 labels"),
        ]
    )
    agg_df = agg_df.reindex(idx)
    agg_df.loc[idx, "paper_ra_error_rate"] = [
        "$13.81_{\pm3.37}$",
        "$5.07_{\pm0.65}$",
        "$4.26_{\pm0.05}$",
        "$48.85_{\pm1.75}$",
        "$28.29_{\pm0.11}$",
        "$22.60_{\pm0.12}$",
        "$3.96_{\pm2.17}$",
        "$2.48_{\pm0.38}$",
        "$2.28_{\pm0.11}$",
        "$7.98_{\pm1.50}$",
    ]
    agg_df.loc[idx, "paper_cta_error_rate"] = [
        "$11.39_{\pm3.35}$",
        "$5.07_{\pm0.33}$",
        "$4.31_{\pm0.15}$",
        "$49.95_{\pm3.01}$",
        "$28.64_{\pm0.24}$",
        "$23.18_{\pm0.11}$",
        "$7.65_{\pm7.65}$",
        "$2.64_{\pm0.64}$",
        "$2.36_{\pm0.19}$",
        "$5.17_{\pm0.63}$",
    ]
    formatted_df = agg_df[["error_summary", "paper_ra_error_rate", "paper_cta_error_rate"]].T
    formatted_df.index = formatted_df.index.to_flat_index()
    print(formatted_df.to_latex(index=None, escape=False))

    df = retrieve_runs().rename(
        columns={"params.µ": "µ", "params.subsampling": "supervised labels", "error_rate": "Error rate"}
    )
    pn.serve(
        df_to_error_plot(
            df,
            x="supervised labels",
            y="Error rate",
            by="µ",
            groupby="experiment_id",
            by_color_map={"µ0": "green", "µ7": "orange"},
        ),
        title="dataset_study",
    )


def retrieve_runs():
    xid_to_xname = {
        "801031181522110214": "STL10",
        "966423010245122331": "SVHN",
        "509633626869058711": "CIFAR100",
        "732631612698781478": "CIFAR10",
    }

    runs_df = mlflow.search_runs(experiment_names=list(ds_sizes))
    subset_df = (
        runs_df.query("`tags.mlflow.parentRunId`.notna()")[
            [
                "experiment_id",
                "params.µ",
                "params.subsampling",
                "start_time",
                "run_id",
                "metrics.dev MulticlassAccuracy",
                "metrics.dev loss",
            ]
        ]
        .assign(start_time=lambda x: x.start_time.dt.strftime("%Y-%m-%d %H:%M:%S"))
        .sort_values("start_time", ascending=False)
        .head(90)
        .assign(start_time=lambda x: pd.to_datetime(x.start_time))
        .query('start_time > "2023-11-24"')
        .replace({**xid_to_xname, "0": "µ0", "7": "µ7"})
        .assign(error_rate=lambda x: 100 * (1 - x["metrics.dev MulticlassAccuracy"]))
        .astype({"params.subsampling": int})
        .sort_values("params.subsampling")
    )
    return subset_df


def _old_display():
    subset_df = retrieve_runs()
    grouping = ["experiment_id", "params.µ", "params.subsampling"]
    plots = subset_df.hvplot.box(
        y="error_rate",
        by=grouping,
        # groupby="experiment_id",
        height=750,
        ylim=(0, 100),
        grid=True,
        ylabel="Error rate",
        xlabel="",
        yformatter="%.0f%%",
        fontsize={"labels": 18, "xticks": 14, "yticks": 14},
        legend=False,
    )
    plots
    plot = (
        hv.Layout(
            [
                plots["CIFAR10"].opts(width=300),
                plots["CIFAR100"].opts(width=130, yaxis=None),
                plots["SVHN"].opts(width=190, yaxis=None),
                plots["STL10"].opts(width=65, yaxis=None),
            ]
        )
        .cols(4)
        .opts(fontsize={"labels": 18, "xticks": 14, "yticks": 14}, shared_axes=False)
    )
    pn.serve(plot, title="dataset_study")


def get_embedding_variance():
    model = get_model("resnet50_imagenet", 1)
    model.fc = torch.nn.Identity()
    model.eval()
    model = model.to("cuda")
    for ds_name in ["SVHN", "CIFAR10", "CIFAR100", "STL10"]:
        train, _, _ = ImageDataset.from_torch_dataset(ds_name, µ=0, n_samples=1_000)
        dl = loader_from_ds(train)
        embed = [model(batch[0].to("cuda")).detach().cpu() for batch in tqdm(dl)]
        print(ds_name, torch.cat(embed).std(dim=0).mean().item())


if __name__ == "__main__":
    perform_dataset_study()
    display_dataset_study()
    get_embedding_variance()
