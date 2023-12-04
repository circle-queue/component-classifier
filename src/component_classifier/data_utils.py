import functools
import itertools
import math
import random
import warnings
from importlib.resources import files
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.environment_variables
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, STL10, SVHN
from torchvision.models import ResNet50_Weights
from torchvision.transforms import Compose
from tqdm.auto import tqdm

tqdm.pandas()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOCAL_IMG_FOLDER = files("component_classifier") / "data" / "images"
MLFLOW_DIR = files("component_classifier") / "data" / "mlruns"

LABEL_COLS = [
    "Piston Ring Overview",  # 2300
    "Liner",  # 1800
    "Single Piston Ring",  # 1500
    "topland",  # 650
    "piston top",  # 330
    "skirt",  # 330
    "scavange box",  # 250
    "piston rod",  # 45 images
    # "scavange port",  # UNK imgs
]

mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
mlflow.environment_variables.MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR.set("False")
assert not mlflow.environment_variables.MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR.get()


def add_local_path(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        local_path=df.apply(lambda x: (LOCAL_IMG_FOLDER / f"{x.id}{Path(x.image_path).suffix}"), axis=1),
        torch_path=df.apply(lambda x: (LOCAL_IMG_FOLDER / f"{x.id}.pt"), axis=1),
        features_path=df.apply(lambda x: (LOCAL_IMG_FOLDER / f"{x.id}.features"), axis=1),
        clip_features_path=df.apply(lambda x: (LOCAL_IMG_FOLDER / f"{x.id}.clip_features"), axis=1),
    )


@functools.cache
def get_unlabeled_target_df():
    return (
        pd.read_csv(files("component_classifier") / "data/annotations.csv")
        .rename(columns={"id": "id", "annot": "Y"})
        .replace({"label": 1, "no-label": 0})
    )


@functools.cache
def get_metadata_df(filter_label_cols: bool = True):
    f"""
    Important columns:
    - id: unique identifier
    - label: the index of the label with respect to {LABEL_COLS} (-1 for unlabeled)
    - image_path: path to image
    - torch_path: path to torch tensor (transformed/normed image)
    - features_path: path to torch tensor (features from resnet)
    - clip_features_path: path to torch tensor (features from CLIP)
    """
    metadata_df = (
        pd.read_csv(files("component_classifier") / "data/images_db_mariel.tsv", sep="\t", encoding="latin1")
        .reset_index(drop=True)
        .merge(
            pd.read_csv(files("component_classifier") / "data/images_split.tsv", sep="\t", encoding="latin1"),
            on="id",
            how="outer",
        )
        # .query('split != "test"')  # Remove test to avoid accidentally using it
        .sort_values("id")
        .pipe(add_local_path)
        .assign(
            type_dict=lambda df: df.type.str.replace("'", '"').apply(lambda x: eval(x) if pd.notna(x) else {}),
            exists=lambda df: df.torch_path.apply(Path.exists),
        )
        .query("exists")
    )
    avoid = ["cylinder nr visable", "cylinder nr", "rotate"]
    labels_df = pd.json_normalize(metadata_df["type_dict"]).drop(columns=avoid)
    metadata_df[labels_df.columns] = labels_df.fillna(0).astype(float).values  # must be float for pytorch

    if filter_label_cols:
        # We may skip this step for doing analytics on the full dataset
        metadata_df = metadata_df[(metadata_df[LABEL_COLS].sum(axis=1) == 1) | (metadata_df.split == "unlabeled")]

    metadata_df["label"] = np.where(metadata_df.split == "unlabeled", -1, metadata_df[LABEL_COLS].values.argmax(axis=1))
    n_smallest_class = metadata_df.query('split != "test"').groupby("label").id.count().min()
    assert n_smallest_class == 45

    # Split the data into 45 bins, which each preserve the label distribution
    # Each bin will have at least one image of each label
    # We dedicate 9/45 = 20% of the data to a dev split
    # This leaves 36/45 = 80% of the data for training
    kfold = StratifiedKFold(n_splits=n_smallest_class, shuffle=True, random_state=42)
    train_df = metadata_df.query('split == "train"')
    for split_num, (_, train_split_idx) in enumerate(kfold.split(train_df, train_df.label)):
        # We cannot directly assign to metadata_df, because the idx matches those of train_df
        metadata_split_idx = train_df.iloc[train_split_idx].index
        metadata_df.loc[metadata_split_idx, "split_idx"] = split_num
    assert metadata_df.query('split == "train"').split_idx.notna().all(), "split_num is not fully populated"

    # We assign 6/30 = 20% of the data to a default dev split
    metadata_df.loc[metadata_df.split_idx.values >= round(0.8 * n_smallest_class), "split"] = "dev"
    assert math.isclose(0.20, len(metadata_df.query('split == "dev"')) / len(train_df), abs_tol=0.01)
    return metadata_df


def load_tensor(path: Path):
    return torch.load(path, map_location="cpu")  # Must be CPU to allow for memory pinning


class ImageDataset(Dataset):
    @classmethod
    def from_metadata_df(
        cls,
        meta_df: pd.DataFrame,
        cols: list[str],
        split: Literal["train", "dev", "test", "unlabeled"],
        n_cache: int = 0,
        n_train: int = 4247,
        µ: int = None,
    ):
        df = meta_df.query(f"split == '{split}'")
        if split != "unlabeled":
            # Ensure we have exactly one label
            df = df[lambda df: df[cols].sum(axis=1) == 1]

        X_id = df[["torch_path", "id"]].values
        # 1d labels perform better https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        label = df[cols].values.argmax(axis=1)

        n = n_train if split == "train" else max(1, int(n_train * µ)) if split == "unlabeled" else min(n_train, len(df))
        X_id, label = cls.stratified_sample(X_id, label, n)

        return ImageDataset(
            torch.from_numpy(X_id[:, 1].astype(int)),
            X_id[:, 0],
            torch.from_numpy(label),
            pipeline=Compose([load_tensor]),
        ).cache_n_files(n_cache)

    @staticmethod
    def to_ds(x, y) -> "ImageDataset":
        return ImageDataset(
            np.arange(len(x)),
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.int64),
            pipeline=ResNet50_Weights.IMAGENET1K_V2.transforms(),
        )

    @staticmethod
    def stratified_sample(x, y, n_samples: int):
        """Sample n_samples from x, y such that each label is represented with the same ratio as in the original data"""
        data = list(zip(y, x))
        np.random.shuffle(data)
        data.sort(key=lambda y_x: y_x[0])

        data_by_label = [list(group) for _, group in itertools.groupby(data, key=lambda y_x: y_x[0])]

        downsample_frac = n_samples / sum(map(len, data_by_label))
        n_by_label = [int(len(label_data) * downsample_frac) for label_data in data_by_label]
        stratified_sample = [y_x for label_data, n in zip(data_by_label, n_by_label) for y_x in list(label_data)[:n]]
        Y, X = map(np.array, zip(*stratified_sample))
        assert np.array(np.unique(Y) == np.unique(y)).all(), f"Must not lose any labels, {np.unique(Y)=}, {np.unique(y)=}"
        return X, Y

    @staticmethod
    def balanced_sample(x, y, n_samples: int, n_labels: int):
        """Sample n_samples from x, y such that each label is represented n_samples/n_labels times"""
        data = list(zip(y, x))
        np.random.shuffle(data)
        data.sort(key=lambda y_x: y_x[0])

        data_by_label = [list(group) for _, group in itertools.groupby(data, key=lambda y_x: y_x[0])]
        balanced_sample = [y_x for label_data in data_by_label for y_x in list(label_data)[: n_samples // n_labels]]
        Y, X = map(np.array, zip(*balanced_sample))
        return X, Y

    @classmethod
    def from_torch_dataset(
        cls,
        dataset_name: Literal["SVHN", "STL10", "CIFAR10", "CIFAR100"],
        n_samples: int,
        µ: int,
    ):
        src = files("component_classifier") / "data"

        if not dataset_name.startswith("CIFAR"):
            if dataset_name == "SVHN":
                ds_train = SVHN(root=src / "svhn_train", split="train", download=True)
                ds_test = SVHN(root=src / "svhn_test", split="test", download=True)
                ds_unlabeled = SVHN(root=src / "svhn_unlabeled", split="extra", download=True)
            elif dataset_name == "STL10":
                ds_train = STL10(root=src / "stl_train", split="train", download=True)
                ds_test = STL10(root=src / "stl_test", split="test", download=True)
                ds_unlabeled = STL10(root=src / "stl_unlabeled", split="unlabeled", download=True)

            X_train, y_train = cls.balanced_sample(ds_train.data, ds_train.labels, n_samples, n_labels=10)
            X_test, y_test = cls.balanced_sample(ds_test.data, ds_test.labels, n_samples, n_labels=10)

            X_unlabeled_idxs = np.random.choice(len(ds_unlabeled.data), max(µ * n_samples, 1), replace=False)
            X_unlabeled = ds_unlabeled.data[X_unlabeled_idxs]
            y_unlabeled = np.full(max(µ * n_samples, 1), -1)
        else:
            if dataset_name == "CIFAR10":  # train=True -> train set. train=False -> test set
                ds = CIFAR10(root=src / "cifar10_train", train=True, download=True)
            elif dataset_name == "CIFAR100":  # train=True -> train set. train=False -> test set
                ds = CIFAR100(root=src / "cifar100_train", train=True, download=True)
            else:
                raise AttributeError(f"'{dataset_name}' is not a valid dataset.")

            labels = np.array(ds.targets)
            data = np.array(ds.data).transpose(0, 3, 1, 2)  # from shape (50000, 32, 32, 3) -> (50000, 3, 32, 32)

            n_samples_per_label, remainder = divmod(n_samples, len(np.unique(labels)))
            assert not remainder, "n_samples must be a multiple of the number of labels"

            labeled_idx, unlabeled_idx, test_idx = cls.sample_indices(labels, n_samples_per_label)
            X_train = data[labeled_idx]
            y_train = labels[labeled_idx]
            X_unlabeled = data[unlabeled_idx]
            y_unlabeled = labels[unlabeled_idx]
            X_test = data[test_idx]
            y_test = labels[test_idx]

        print(f"{dataset_name=} {n_samples=} {len(X_train)=} {len(X_unlabeled)=} {len(X_test)=}")
        assert µ <= int(X_unlabeled.shape[0] / X_train.shape[0])
        return [cls.to_ds(x, y) for x, y in [(X_train, y_train), (X_test, y_test), (X_unlabeled, y_unlabeled)]]

    def __init__(self, ids: torch.Tensor, X: list[Path], Y: torch.Tensor, pipeline: Compose):
        assert len(ids) == len(X) == len(Y)
        self.ids, self.X, self.Y = ids, X, Y
        self.pipeline = pipeline
        self._cache = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if cached := self._cache.get(idx):
            return cached  # avoid IO

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tensor = self.pipeline(self.X[idx])

        return (
            tensor,
            self.Y[idx],
            self.ids[idx],
        )  # The feature "tensor" must be the first argument for torch.optim.swa_utils.update_bn

    def cache_n_files(self, n_files: int):
        """Supports method chaining (ds = ds.cache_n_files(foobar))"""
        self._cache = {
            idx: self[idx]
            for idx in tqdm(
                np.random.choice(len(self), size=min(n_files, len(self)), replace=False), desc="Caching", leave=False
            )
        }
        return self

    @staticmethod
    def sample_indices(labels, n_samples_per_label):
        """Sample indices from each label. Returns a list of indices for labeled and unlabeled data"""
        label_indices = {}
        for idx, label in enumerate(labels):
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(idx)

        selected_labeled = []
        selected_unlabeled = []
        random.seed(42)
        for label, indices in label_indices.items():
            if len(indices) < n_samples_per_label * 8:
                print(f"{n_samples_per_label=}, {len(indices)=}, {len(label_indices)=}")
                raise RuntimeError("Too few labeled")
            samples_from_label = random.sample(indices, n_samples_per_label * 8)  # taking 8*N samples

            selected_labeled.extend(samples_from_label[:n_samples_per_label])
            selected_unlabeled.extend(samples_from_label[n_samples_per_label:])

        # Test set is at most size of unlabeled dataset
        selected_test = list(set(range(len(labels))) - set(selected_labeled) - set(selected_unlabeled))[
            : len(selected_labeled)
        ]

        return selected_labeled, selected_unlabeled, selected_test


def load_model(run_id: str) -> nn.Module:
    """
    >>> run_id = "997af1749be44280a804aa2c6e4ea0c7"
    """
    return mlflow.pytorch.load_model(f"runs:/{run_id}/model")


def load_preds_df(run_id: str) -> pd.DataFrame:
    """
    >>> run_id = "997af1749be44280a804aa2c6e4ea0c7"
    """
    # Each epoch writes new preds. By picking "last", we get the final epoch
    return mlflow.load_table("dev_preds.csv", [run_id]).groupby("ids").agg("last")


def df_to_error_plot(df, x, y, by, groupby, by_color_map, override_params={}):
    import holoviews as hv
    import hvplot.pandas  # noqa

    data = (
        df.groupby([x, by, groupby])[y]
        .describe()
        .reset_index()
        .assign(
            yerr1=lambda x: x["mean"] - x["min"],
            yerr2=lambda x: x["max"] - x["mean"],
            color=lambda x: x[by].map(by_color_map),
        )
        .rename(columns={"mean": y})
        .astype({x: str})
    )

    axes = dict(shared_axes=False) | override_params
    plot_kwargs = (
        dict(
            x=x,
            y=y,
            by=by,
            groupby=groupby,
            c="color",
            yformatter="%.0f%%",
            width=200,
            height=300,
            line_color="color",
            grid=True,
        )
        | axes
    )
    plots = data.hvplot.errorbars(yerr1="yerr1", yerr2="yerr2", line_width=3, alpha=0.5, **plot_kwargs) * data.astype(
        {x: str}
    ).hvplot.scatter(**plot_kwargs)
    groups = sorted(data[groupby].unique())
    return hv.Layout([plots[g].opts(**axes, legend_position="bottom", title=g) for g in groups]).cols(len(groups))


if __name__ == "__main__":
    train_ds, test_ds, unlabeled_ds = ImageDataset.from_torch_dataset("STL10", n_samples=1000, µ=7)
    df = pd.DataFrame(
        [
            train_ds.Y.numpy(),
            test_ds.Y.numpy(),
            unlabeled_ds.Y.numpy(),
        ]
    )
