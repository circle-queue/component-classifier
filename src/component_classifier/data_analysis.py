import warnings

import holoviews as hv
import hvplot.pandas
import mlflow
import pandas as pd
import panel as pn
import polars as pl
import torch
from IPython.display import display
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision.models import ResNet50_Weights

from component_classifier.data_utils import (
    LABEL_COLS,
    get_metadata_df,
    load_model,
    load_preds_df,
)
from component_classifier.train_utils import DEVICE


def merge_labels(struct: dict) -> int:
    return ", ".join(sorted([k for k, v in struct.items() if bool(v)]))


def display_merged_labels(meta_df):
    meta_df["merged_label"] = pl.from_pandas(meta_df["type_dict"]).apply(merge_labels).to_pandas().fillna("UNK").values
    display(meta_df.groupby("merged_label").count().iloc[0])


def display_label_examples(meta_df):
    examples = {col: meta_df.query(f"`{col}` == 1").image_path.sample(10, replace=True) for col in LABEL_COLS}
    counts = meta_df[LABEL_COLS].stack().droplevel(0)[lambda x: x == 1].index.value_counts()

    rows = pn.Row()
    for col, img_paths in examples.items():
        col = pn.Column(f"# {col} (#{counts[col]})")
        for path in img_paths:
            try:
                img = Image.open(path)
            except:
                continue

            img.thumbnail((250, 250))
            col.append(img)
        rows.append(col)
    pn.serve(rows)


def display_single_labeled_example(meta_df):
    example_row = meta_df.query('split == "train"').sample(1).squeeze()
    label = example_row[LABEL_COLS][lambda x: x == 1].index.item()
    print(f"{example_row.id = } {label = }")

    img = Image.open(example_row.local_path).convert("RGB").resize((224, 224))

    display(img)

    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    processed_img_arr = transform(img)

    kwargs = dict(axis=0, keepdim=True)
    processed_img: torch.Tensor = transform(img).permute(1, 2, 0)
    processed_img -= processed_img.min(**kwargs).values.min(**kwargs).values
    processed_img /= processed_img.max(**kwargs).values.max(**kwargs).values
    processed_img = Image.fromarray(processed_img.mul(255).clamp(0, 255).to(torch.uint8).numpy())
    display(processed_img)

    # Example usage
    result = model(processed_img_arr.to(DEVICE).unsqueeze(0))
    predicted_label = LABEL_COLS[result.argmax(dim=1)]
    print(f"{predicted_label = }")


def display_unlabeled_examples():
    example_imgs = [
        Image.open(path).resize((224, 224)) for path in meta_df.query('split == "unlabeled"').sample(50).local_path
    ]
    pn.GridBox(*example_imgs, ncols=7, width=500, height=1500)


def display_confusion_matrix(run_id: str) -> None:
    pred_df = load_preds_df(run_id)

    confusion_df = pd.DataFrame(
        confusion_matrix(pred_df["y"], pred_df["pred"], normalize="true"),
        columns=LABEL_COLS,
        index=LABEL_COLS,
    )
    confusion_df.hvplot.heatmap(
        rot=15,
        cmap="Blues",
        width=500,
        height=500,
        title="Confusion Matrix",
    )


def double_transform_img():
    example_row = meta_df.query('split == "train"').sample(1).squeeze()
    label = example_row[LABEL_COLS][lambda x: x == 1].index.item()
    print(f"{example_row.id = } {label = }")

    img = Image.open(example_row.local_path).convert("RGB").resize((224, 224))

    stored_arr = torch.load(example_row.torch_path).permute(1, 2, 0)
    stored_arr -= stored_arr.min()
    stored_arr /= stored_arr.max()
    stored_arr = Image.fromarray(stored_arr.mul(255).clamp(0, 255).to(torch.uint8).numpy())

    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    processed_img: torch.Tensor = transform(img).permute(1, 2, 0)
    processed_img -= processed_img.min()
    processed_img /= processed_img.max()
    processed_img = Image.fromarray(processed_img.mul(255).clamp(0, 255).to(torch.uint8).numpy())

    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    processed_img_twice: torch.Tensor = transform(processed_img).permute(1, 2, 0)
    processed_img_twice -= processed_img_twice.min(axis=0).values.min(axis=0).values
    processed_img_twice /= processed_img_twice.max(axis=0).values.max(axis=0).values
    processed_img_twice = Image.fromarray(processed_img_twice.mul(255).clamp(0, 255).to(torch.uint8).numpy())

    display(img, stored_arr, processed_img, processed_img_twice)


if __name__ == "__main__":
    meta_df = get_metadata_df()

    pn.extension()

    hv.notebook_extension("bokeh")

    run_id = "2f63490296834ac38c753641065a174e"
    model = load_model(run_id)
