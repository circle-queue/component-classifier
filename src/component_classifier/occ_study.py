# results This file tries to classify images as either "good" or "bad" based on the distance to the mean of the training set.
# Files with a very low/negative score are considered "bad".
import json
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import holoviews as hv
import hvplot.pandas  # noqa F401
import mlflow
import numpy as np
import pandas as pd
import panel as pn
import torch
from PIL import Image
from pyod.models.ecod import ECOD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.svm import OneClassSVM
from torchmetrics import Accuracy, Precision, Recall
from tqdm.auto import tqdm

from component_classifier.data_utils import (
    LABEL_COLS,
    get_metadata_df,
    get_unlabeled_target_df,
)

hv.notebook_extension("bokeh")

FALSE_NEGATIVE_RATIO = 0.05  # How many false negatives do we want to allow?

RUNS_PATH: Path = files("component_classifier") / "runs/one_class_study.json"


class OneClassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, false_negative_ratio: float):
        self.false_negative_ratio = false_negative_ratio

    def predict(self, X: pd.DataFrame):
        """returns 1 if the image is good, 0 if it is bad"""
        return np.array(self._score(X) > self.threshold)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Override this method to fit the model"""
        self.threshold: float = ...
        return self

    def _score(self, X: pd.DataFrame):
        """The larger the value, the more likely the image is in-class"""
        ...


class MeanClassifier(OneClassClassifier):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.mean = X.mean(axis=0)
        self.threshold = self._score(X).quantile(self.false_negative_ratio)
        return self

    def _score(self, X: pd.DataFrame):
        # Negative distance to mean, so that the further away, the more negative
        return -(((X - self.mean) ** 2).sum(axis=1) ** 0.5)


class ClassMeanClassifier(OneClassClassifier):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.means_train = X.groupby(y).mean()
        self.threshold = self._score(X).quantile(self.false_negative_ratio)
        return self

    def _score(self, X: pd.DataFrame):
        # Negative smallest distance to a class mean, so that the further away, the more negative
        # Sort of like single linkage
        return pd.DataFrame(
            [-(((X - mean_train) ** 2).sum(axis=1) ** 0.5) for mean_train in self.means_train.values]
        ).min(axis=0)


class KMeanClassifier(OneClassClassifier):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.means_train = KMeans(n_clusters=20).fit(X).cluster_centers_
        self.threshold = self._score(X).quantile(self.false_negative_ratio)
        return self

    def _score(self, X: pd.DataFrame):
        # Negative smallest distance to a class mean, so that the further away, the more negative
        # Sort of like single linkage
        return pd.DataFrame([-(((X - mean_train) ** 2).sum(axis=1) ** 0.5) for mean_train in self.means_train]).min(
            axis=0
        )


class OSVMClassifier(OneClassClassifier):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.clf = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=self.false_negative_ratio,
            tol=0.001,
            shrinking=True,
        ).fit(X)

        self.threshold = pd.Series(self._score(X)).quantile(self.false_negative_ratio)
        return self

    def _score(self, X: pd.DataFrame):
        return self.clf.decision_function(X)


class ECODClassifier(OneClassClassifier):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.clf = ECOD(contamination=self.false_negative_ratio, n_jobs=-1).fit(X)
        self.threshold = pd.Series(self._score(X)).quantile(self.false_negative_ratio)
        return self

    def _score(self, X: pd.DataFrame):
        return -self.clf.decision_function(X)


def get_X_resnet50(subset_df):
    """Loads pre-calculated resnet50 embeddings for each line in the dataframe"""
    return pd.DataFrame(
        torch.vstack([torch.load(path, map_location="cpu") for path in tqdm(subset_df.features_path)]).detach().numpy(),
        index=subset_df.id,
    )


def get_X_clip(subset_df):
    """Loads pre-calculated clip embeddings for each line in the dataframe"""
    return pd.DataFrame(
        torch.vstack([torch.load(path, map_location="cpu") for path in tqdm(subset_df.clip_features_path)])
        .detach()
        .numpy(),
        index=subset_df.id,
    )


def eval_one_class(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    label: pd.Series,
    id: pd.Series,
    **params: dict,
):
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        df = pd.DataFrame(
            dict(pred=np.array(clf.predict(X)), y=np.array(y), label=np.array(label), id=np.array(id)) | params
        )
        mlflow.log_table(df, artifact_file="predictions.json")

        accuracies = (
            df.groupby(["y", "label"])
            .apply(lambda df: (df.pred == df.y).mean())
            .rename("accuracy")
            .to_frame()
            .reset_index()
        )
        mlflow.log_table(accuracies.assign(**params), artifact_file="accuracies.json")
        mlflow.log_metric("no_class_has_class_macro_accuracy", accuracies.query("label == -1").accuracy.mean())
        mlflow.log_metric("train_macro_TPR", accuracies.query("label != -1").accuracy.mean())
        run_id = run.info.run_id
    return run_id


def get_all_data(embedding_method: Callable):
    meta_df = get_metadata_df()
    meta_df["y_inferred"] = meta_df.label.isin(range(0, 8)).astype(int)

    train_df = meta_df.query('split == "train"')
    dev_df = meta_df.query('split == "dev"')
    test_df = meta_df.merge(get_unlabeled_target_df(), on="id")

    y_all = pd.concat([train_df.y_inferred, dev_df.y_inferred, test_df.Y])
    label_all = pd.concat([train_df.label, dev_df.label, test_df.label])
    id_all = pd.concat([train_df.id, dev_df.id, test_df.id])

    X_train = embedding_method(train_df)
    X_dev = embedding_method(dev_df)
    X_test = embedding_method(test_df)

    X_all = pd.concat([X_train, X_dev, X_test])
    return X_all, y_all, label_all, id_all, X_train, train_df.y_inferred


def run_one_class_study():
    result = {}
    for x_method in [get_X_clip]:  # [get_X_resnet50, get_X_clip]:
        mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(f"OneClass - {x_method.__name__}")

        X_all, y_all, label_all, id_all, X_train, y_train = get_all_data(x_method)

        method_data = result[x_method.__name__] = {}
        for clf_class in [MeanClassifier, ClassMeanClassifier, KMeanClassifier, OSVMClassifier, ECODClassifier]:
            for false_negative_ratio in np.linspace(0.01, 0.50, 20):
                clf = clf_class(false_negative_ratio)
                clf.fit(X_train, y_train)

                params = {"false_negative_ratio": false_negative_ratio, "clf_class": clf_class.__name__}
                method_data[clf_class.__name__, false_negative_ratio] = eval_one_class(
                    clf,
                    X=X_all,
                    y=y_all,
                    label=label_all,
                    id=id_all,
                    **params,
                )

    RUNS_PATH.write_text(json.dumps(result, indent=4))


def plot_embeds_to_2d():
    X_all, y_all, label_all, id_all, X_train, y_train = get_all_data(get_X_clip)

    clf = OSVMClassifier(FALSE_NEGATIVE_RATIO).fit(X_train, y_train)

    tsne = TSNE(n_components=2, n_jobs=8)
    transformed = pd.DataFrame(tsne.fit_transform(X_all), columns=["dim1", "dim2"])
    transformed["label"] = label_all.replace(dict(enumerate(LABEL_COLS))).values
    transformed["y"] = y_all.values
    transformed["y_pred"] = clf.predict(X_all).astype(int)
    transformed["id"] = id_all.values

    kwargs = dict(hover_cols="all", x="dim1", y="dim2", xaxis=None, yaxis=None)
    x_train_tsne_plot = transformed.query("label != -1").hvplot.scatter(
        **kwargs,
        by="label",
        height=400,
        width=500,
        alpha=1,
        legend=False,
    )  # .opts(title='TSNE of "good" images, colored by class')
    pn.serve(x_train_tsne_plot, title="train_tsne.png")

    plots = transformed.query("label == -1").hvplot.scatter(
        **kwargs,
        groupby=["y", "y_pred"],
        hover_cols="all",
        height=400,
        width=500,
        fontsize={"legend": 15},
    )
    x_test_tsne_plot = (
        # ERRORS
        plots[1, 0].opts(marker="square", size=7, color="black").relabel("FN")
        * plots[0, 1].opts(marker="square", size=7, color="darkred").relabel("FP")
        # GOOD
        * plots[1, 1].opts(marker="star", color="gray", size=8).relabel("TP")
        * plots[0, 0].opts(marker="star", color="red", size=8).relabel("TN")
    )  # .opts(title="Carsten's in/out-of-class grouped by correctness of model prediction")
    pn.serve(x_test_tsne_plot, title="test_tsne.png")

    img_id = 106121
    [img_path] = [
        path
        for path in (files("component_classifier") / "data/images/").glob(f"{img_id}*")
        if path.suffix not in [".pt", ".clip_features", ".features"] and path.stem == str(img_id)
    ]
    img = Image.open(img_path).resize((512, 512))
    img

    # TODO:
    # Have Carsten examine the output
    # Test more models / parameters
    # 3d TSNE


def evaluation():
    run_ids = json.loads(RUNS_PATH.read_text())

    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(f"OneClass - {get_X_clip.__name__}")
    pred_df = mlflow.load_table("predictions.json", run_ids=run_ids["get_X_clip"].values())
    acc_df = mlflow.load_table("accuracies.json", run_ids=run_ids["get_X_clip"].values())
    pred_df["pred"] = pred_df.pred.astype(int)

    dev_df = get_metadata_df().query('split == "dev"')
    pred_df["is_dev"] = pred_df.id.isin(dev_df.id)
    dev_acc = (
        pred_df.query("is_dev")
        .groupby("clf_class")
        .apply(lambda df: (df.pred == df.y).mean())
        .rename("accuracy")
        .describe()
    )
    max_delta = max(dev_acc["mean"] - dev_acc["min"], dev_acc["max"] - dev_acc["mean"])
    print(f'dev_fnr = {1-dev_acc["mean"]:.2%} ± {max_delta:.2%}')

    annotated_df = pred_df.query("label == -1")
    print(
        (
            annotated_df.groupby("clf_class")
            .pred.mean()
            .rename("NR")
            .rename_axis(None)
            .sort_values()
            .to_frame()
            .T.assign(true=annotated_df.y.mean())
            .map(lambda x: f"{1 - x - 0.05:.0%}")
        ).to_latex(escape=True)
    )

    test_acc_df = annotated_df.groupby(["y", "clf_class"]).apply(lambda df: 1 - (df.pred == df.y).mean()).rename("error")
    abbrev = {
        x: x.split("Classifier")[0]
        for x in [
            "ClassMeanClassifier",
            "ECODClassifier",
            "KMeanClassifier",
            "MeanClassifier",
            "OSVMClassifier",
        ]
    }
    pn.serve(
        test_acc_df.reset_index()
        .replace({**abbrev, 0: "no-class", 1: "has-class"})
        .hvplot.bar(
            "y",
            "error",
            by="clf_class",
            rot=30,
            width=400,
            c="#6baed6",
            xlabel="",
            ylabel="Test micro error rate",
            fontsize={"labels": 14, "xticks": 11, "yticks": 11},
        ),
        title="has_vs_no_class_accuracy.png",
    )

    dev_acc_df_by_labael = (
        pred_df.query("is_dev")
        .groupby(["label", "clf_class"])
        .apply(lambda df: 1 - (df.pred == df.y).mean())
        .rename("error")
        .reset_index()
        .replace(abbrev)
    )
    pn.serve(
        dev_acc_df_by_labael.hvplot.box(
            "error",
            by="clf_class",
            width=400,
            rot=30,
            xlabel="",
            ylabel="Dev macro error rate",
            fontsize={"labels": 14, "xticks": 14, "yticks": 14},
        ),
        title="train_class_accuracy_boxplot.png",
    )

    res = pred_df.groupby([pred_df.pred, pred_df.y]).apply(
        lambda df: pn.Column(
            f"# y={df.y.iloc[0]} & pred={df.pred.iloc[0]}",
            *df.sample(5)
            .apply(
                lambda x: pn.Column(
                    Image.open(get_metadata_df().query("id == @x.id").local_path.item()).resize((256, 256)),
                ),
                axis=1,
            )
            .values,
        )
    )
    x = pn.Row(*res.values)
    pn.serve(x)


def TODO_AUC_curve():
    X_all, y_all, label_all, id_all, X_train, y_train = get_all_data(get_X_clip)

    data = []
    for expected_train_error in np.linspace(0, 1, 100):
        clf = MeanClassifier(expected_train_error)
        clf.fit(X_train, y_train)
        y_pred_test = torch.tensor(clf.predict(X_test).values)
        y_pred = y_pred_test.numpy()

        [
            [true_negative, false_positive],  # These are in-class
            [false_negative, true_positive],  # These are out-of-class
        ] = confusion_matrix(test_df["Y"], test_df["y_pred"])

        data.append(
            {
                "precision": Precision(task="binary")(y_pred_test, torch.from_numpy(test_df["Y"].values)).item(),
                "recall": Recall(task="binary")(y_pred_test, torch.from_numpy(test_df["Y"].values)).item(),
                "macro_accuracy": Accuracy(task="binary", average="macro")(
                    y_pred_test, torch.from_numpy(test_df["Y"].values)
                ).item(),
                "E(train_error)": expected_train_error,
            }
        )

    pd.DataFrame(data).hvplot.scatter(
        x="recall",
        y="precision",
        c="macro_accuracy",
        hover_cols="all",
        title="Precision-Recall Curve",
        xlim=(0, 1.01),
        ylim=(0, 1.01),
    )

    pd.DataFrame(
        confusion_matrix(test_df["Y"], test_df["y_pred"], normalize="true"),
        columns=["pred_bad", "pred_good"],
        index=["true_bad", "true_good"],
    ).hvplot.heatmap(
        rot=15,
        cmap="Blues",
        width=500,
        height=500,
        title="Confusion Matrix",
    )


if __name__ == "__main__":
    run_one_class_study()
    evaluation()
