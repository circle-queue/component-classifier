import os
import subprocess
from pathlib import Path


class C:  # Config
    LABELSTUDIO_ROOT = Path(r"E:\component-classifier\labelstudio")

    LABELSTUDIO_DATA = LABELSTUDIO_ROOT / "labelstudio_data"
    LABELSTUDIO_CONTAINER_DATA = Path("/label-studio/data")

    LABELSTUDIO_IMGS_FOLDER_NAME = "imgs"

    LABELSTUDIO_URL = "http://localhost:8082/"
    LABELSTUDIO_API_KEY = "f0bd81bb642be15b1a6999cf5525a952cee92a74"


start_from = Path(C.LABELSTUDIO_ROOT)
assert start_from.exists()
os.chdir(start_from)
assert Path(os.getcwd()).resolve() == start_from.resolve()

posix_path = C.LABELSTUDIO_CONTAINER_DATA.as_posix()
# User, password and token are not sensitive as long as we run locally
cmd = f"""
docker run -it -p 8082:8082 -v %cd%/{C.LABELSTUDIO_DATA.name}:{posix_path}
--env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
--env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={posix_path}
heartexlabs/label-studio:latest label-studio
--log-level INFO
--username=femewap162@dekaps.com
--password=femewap162@dekaps.com
--user-token f0bd81bb642be15b1a6999cf5525a952cee92a74
""".replace(
    "\n", " "
)

print(cmd)
subprocess.run("setx LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED true", check=True)
subprocess.run(f"setx LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT {posix_path}", check=True)
subprocess.run(cmd, shell=True)
