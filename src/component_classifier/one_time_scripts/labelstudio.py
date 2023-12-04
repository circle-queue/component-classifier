import os
import subprocess
import urllib.parse
import webbrowser
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple

import numpy as np
from label_studio_sdk import Client, Project
from PIL import Image, ImageDraw
from requests import ConnectTimeout, HTTPError
from tqdm.auto import tqdm

from component_classifier.data_utils import get_metadata_df


class C:  # Config
    LABELSTUDIO_ROOT = files("component_classifier") / "../../labelstudio"

    LABELSTUDIO_DATA = LABELSTUDIO_ROOT / "labelstudio_data"
    LABELSTUDIO_CONTAINER_DATA = Path("/label-studio/data")

    LABELSTUDIO_IMGS_FOLDER_NAME = "imgs"

    LABELSTUDIO_URL = "http://localhost:8080/"
    LABELSTUDIO_API_KEY = "f0bd81bb642be15b1a6999cf5525a952cee92a74"


class LabelStudioClient(Client):
    def __init__(self):
        """Does all checks to ensure correct initialisation"""
        try:
            env_value = os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"]
            assert env_value == self.posix_path, f"{env_value = } != {self.posix_path = }"
        except (KeyError, AssertionError) as error:
            subprocess.run("setx LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED true", check=True)
            subprocess.run(f"setx LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT {self.posix_path}", check=True)
            raise IOError("New environment variables set. Please restart the terminal.") from error

        url, api_key = C.LABELSTUDIO_URL, C.LABELSTUDIO_API_KEY
        try:
            super().__init__(url=url, api_key=api_key)
            self.check_connection()
            self.get_projects()
        except (ConnectTimeout, HTTPError):
            start_cmd = "textract-start-labelstudio"
            raise ConnectionError(
                f"Could not connect to Label Studio with {url =} and {api_key = }. Try starting it using:\n\t{start_cmd}"
            )

    @property
    def project(self) -> Project:
        projects = self.get_projects()
        if len(projects) > 2:
            if "DELETE" == input("Multiple projects exists. Type DELETE to delete ALL projects and start over: ").strip():
                self.delete_all_projects()
                init_labelstudio()
        return projects[0]

    @property
    def posix_path(self):
        return C.LABELSTUDIO_CONTAINER_DATA.as_posix()

    def _api_key_error(self):
        raise KeyError("Wrong API key?")


def start_labelstudio():
    print(f"Opening browser at {C.LABELSTUDIO_URL}. Keep refreshing until it works. Don't go to 0.0.0.0:8080")
    webbrowser.open(C.LABELSTUDIO_URL)
    subprocess.run(f'py {(C.LABELSTUDIO_ROOT / "start.py").absolute()}', check=True)


def init_labelstudio():
    ls = LabelStudioClient()
    try:
        ls.project
        if "DELETE" == input("A project already exists. Type DELETE to overwrite it: ").strip():
            ls.delete_all_projects()
        else:
            raise RuntimeError("A project already exists")
    except IndexError:
        # Good, project does not exist
        pass

    project = ls.start_project(
        title="no-label-annotation",
        show_instructions=True,
        show_skip_button=True,
        enable_empty_annotation=True,
    )

    import_dir = C.LABELSTUDIO_DATA / C.LABELSTUDIO_IMGS_FOLDER_NAME
    container_import_dir = (
        C.LABELSTUDIO_CONTAINER_DATA / C.LABELSTUDIO_IMGS_FOLDER_NAME
    )  # This is how labelstudio sees the files
    import_dir.mkdir(exist_ok=True)

    meta_df = get_metadata_df().query('split == "unlabeled"')
    for path in tqdm(meta_df.local_path):
        dst = import_dir / path.name
        if dst.exists():
            continue
        preprocessed_img = Image.open(path).resize((512, 512))
        preprocessed_img.save(dst)

    # Hack to allow providing a posix path (docker) on a windows system
    old_isdir = os.path.isdir
    posix_path = container_import_dir.as_posix()
    os.path.isdir = lambda path: True if path == posix_path else old_isdir(path)

    storage_request = project.connect_local_import_storage(posix_path, use_blob_urls=True)
    project.sync_storage(storage_type=storage_request["type"], storage_id=storage_request["id"])


def extract_annotations() -> list[dict]:
    ls = LabelStudioClient()
    project = ls.project
    submissions = project.get_labeled_tasks()
    target_prefix = f"/data/local-files/?d={C.LABELSTUDIO_IMGS_FOLDER_NAME}"
