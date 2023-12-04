from importlib.resources import files
from pathlib import Path

import requests
import torch
from component_classifier.data_utils import get_metadata_df
from component_classifier.train_utils import get_model
from PIL import Image
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import CLIPModel, CLIPProcessor

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


model = get_model("resnet50", 1)
model.fc = torch.nn.Identity()
model.to("cuda")


def tensor_to_features(src: Path):
    dst = src.with_suffix(".features")
    tensor = torch.load(src, map_location="cuda").unsqueeze(0)
    features = model(tensor).squeeze()
    try:
        torch.save(features, dst)
        torch.load(dst)
    except:
        dst.unlink(missing_ok=True)  # Avoid corrupted files
        raise


def img_to_clip_features(src: Path):
    dst = src.with_suffix(".clip_features")
    img = Image.open(src)
    inputs = clip_processor(images=[img], return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
    features = clip_model.get_image_features(**inputs)
    features /= features.norm(dim=-1, keepdim=True)

    try:
        torch.save(features, dst)
        torch.load(dst)
    except:
        dst.unlink(missing_ok=True)  # Avoid corrupted files
        raise


def img_to_clip_features():
    suffix = ".clip_features"

    meta_df = get_metadata_df()
    local_paths = [p for p in meta_df.local_path if not p.with_suffix(suffix).exists()]

    chunk_size = 128
    chunked_paths = [local_paths[i : i + chunk_size] for i in range(0, len(local_paths), chunk_size)]

    with torch.no_grad():
        for chunk in tqdm(chunked_paths, desc="Generating .clip_features files"):
            imgs = [Image.open(p) for p in chunk]
            inputs = clip_processor(images=imgs, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to("cuda", non_blocking=True)
            chunk_features = clip_model.get_image_features(**inputs)
            chunk_features /= chunk_features.norm(dim=-1, keepdim=True)
            chunk_features = chunk_features.detach().cpu()
            inputs["pixel_values"] = inputs["pixel_values"].cpu()

            for features, path in zip(chunk_features, chunk):
                dst = path.with_suffix(suffix)
                try:
                    torch.save(features, dst)
                except:
                    dst.unlink(missing_ok=True)  # Avoid corrupted files
                    raise


if __name__ == "__main__":
    meta_df = get_metadata_df()
    paths = meta_df.torch_path
    img_dir = files("component_classifier") / "data/images"
    local_paths = sorted(set(img_dir.glob("*.pt")))
    local_paths = [p for p in local_paths if not p.with_suffix(".features").exists()]
    # process_map(tensor_to_features, local_paths, chunksize=100, max_workers=4, smoothing=0)

    for p in tqdm(local_paths, desc="Generating .pt files"):
        tensor_to_features(p)

    paths = meta_df.local_path
    src = paths[0]

    for p in meta_df.clip_features_path:
        try:
            torch.load(p).numpy()
        except:
            torch.save(torch.load(p).detach().cpu(), p)
