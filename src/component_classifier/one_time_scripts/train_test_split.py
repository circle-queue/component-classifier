import shutil
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from component_classifier.data_utils import (
    LOCAL_IMG_FOLDER,
    add_local_path,
    get_metadata_df,
)

LOCAL_TEST_FOLDER = files("component_classifier") / "data/test"
LOCAL_TRAIN_FOLDER = files("component_classifier") / "data/train"
LOCAL_UNLABELED_FOLDER = files("component_classifier") / "data/unlabeled"


def split_train_test_unlabeled(metadata_df, labels, test_size):
    metadata_df = metadata_df.copy()

    # add new column, None for no label, 0 for train split, 1 for test split
    metadata_df["split"] = "unlabeled"

    np.random.seed(42)
    for label in labels:
        metadata_label = metadata_df[metadata_df[label] == 1].copy()
        metadata_df.loc[metadata_label.index.to_list(), "split"] = "train"

        # .npceil to at least have 1 test image for classes less than 10
        label_test_size = int(np.ceil(len(metadata_label) * test_size))

        # choose test set, randomly the df of that label
        test_idx = np.random.choice(metadata_label.index.to_list(), size=label_test_size)

        metadata_df.loc[test_idx, "split"] = "test"

    return metadata_df


def copy_test_train_unlabeled(src_split: tuple[Path, int]):
    src, test_split = src_split

    src_img = LOCAL_IMG_FOLDER / src.name
    src_pt = LOCAL_IMG_FOLDER / src.with_suffix(".pt").name

    if test_split == "test":
        dst_img = LOCAL_TEST_FOLDER / src.name
        dst_pt = LOCAL_TEST_FOLDER / src.with_suffix(".pt").name

    elif test_split == "train":
        dst_img = LOCAL_TRAIN_FOLDER / src.name
        dst_pt = LOCAL_TRAIN_FOLDER / src.with_suffix(".pt").name

    elif test_split == "unlabeled":
        dst_img = LOCAL_UNLABELED_FOLDER / src.name
        dst_pt = LOCAL_UNLABELED_FOLDER / src.with_suffix(".pt").name

    # Check if they exist in dst, copy if they don't
    if src_img.exists() and not dst_img.exists():
        shutil.copy(src_img, dst_img)

    if src_pt.exists() and not dst_pt.exists():
        shutil.copy(src_pt, dst_pt)


if __name__ == "__main__":
    LOCAL_TEST_FOLDER.mkdir(exist_ok=True)
    LOCAL_TRAIN_FOLDER.mkdir(exist_ok=True)
    LOCAL_UNLABELED_FOLDER.mkdir(exist_ok=True)

    metadata_df = get_metadata_df()
    metadata_df = add_local_path(metadata_df)
    labels = [
        "Liner",
        "Piston Ring Overview",
        "Single Piston Ring",
        "skirt",
        "topland",
        "piston top",
        "piston rod",
        "scavange box",
        "scavange port",
    ]

    # Perform split - remember to set test size
    metadata_df = split_train_test_unlabeled(metadata_df, labels, test_size=0.2)

    # Copy images
    src_split = metadata_df[["local_path", "split"]].to_records(index=False).tolist()
    process_map(copy_test_train_unlabeled, src_split, chunksize=100)

    # Export id and split to tsv
    metadata_df[["id", "split"]].to_csv(
        files("component_classifier") / "data/images_split.tsv", sep="\t", encoding="latin1", index=False
    )
