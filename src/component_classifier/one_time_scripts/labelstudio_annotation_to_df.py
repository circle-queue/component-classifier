import json
from importlib.resources import files
from pathlib import Path

import pandas as pd

path = files("component_classifier") / "data/project-1-at-2023-10-27-11-57-3fff1137.json"
raw_data = json.load(path.open())
data = []
for x in raw_data:
    try:
        data.append(
            {
                "id": int(Path(x["data"]["image"]).stem),
                "annot": x["annotations"][0]["result"][0]["value"]["choices"][0],
            }
        )
    except Exception as e:
        print(e)
pd.DataFrame(data).to_csv(files("component_classifier") / "data/annotations.csv", index=False)
