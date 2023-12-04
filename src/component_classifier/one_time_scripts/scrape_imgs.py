import shutil
from importlib.resources import files
from pathlib import Path

import pandas as pd
from tqdm.contrib.concurrent import process_map

from component_classifier.data_utils import LOCAL_IMG_FOLDER, add_local_path


def copy_to_local(src_dst: tuple[Path, Path]):
    src, dst = src_dst
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)


if __name__ == "__main__":
    LOCAL_IMG_FOLDER.mkdir(exist_ok=True)

    img_df = pd.read_csv(files("component_classifier") / "data/images_db_mariel.tsv", sep="\t", encoding="latin1")
    assert img_df.id.is_unique
    img_df = add_local_path(img_df)

    src_dst = img_df[["image_path", "local_path"]].applymap(Path).to_records(index=False).tolist()
    process_map(copy_to_local, src_dst, chunksize=100)
