from importlib.resources import files
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

Image.MAX_IMAGE_PIXELS = 933120000  # DecompressionBombWarning: Image size (133400418 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack

TENSOR_SIZE = (224, 224)  # Accepts any size, but filters are trained on this size.
# https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
"""
Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects.
The images are resized to resize_size=[232] using interpolation=InterpolationMode.BILINEAR, 
followed by a central crop of crop_size=[224]. 
Finally the values are first rescaled to [0.0, 1.0] 
and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
"""
transform = ResNet50_Weights.IMAGENET1K_V2.transforms()


def img_to_tensor(src: Path):
    dst = src.with_suffix(".pt")
    discard_files_with_errors = ["image file is truncated", "Truncated File Read", "cannot identify image file"]
    if not dst.exists():
        try:
            im = Image.open(src)
            tensor = transform(im.convert("RGB"))
            try:
                torch.save(tensor, dst)
                torch.load(dst)
            except:
                dst.unlink(missing_ok=True)  # Avoid corrupted files
                raise
        except Exception as e:
            try:
                im.close()
            except Exception:
                ...  # Never opened

            if any([error in str(e) for error in discard_files_with_errors]):
                print(f"Discarding file with error {src}: {e}")
                src.unlink()
            else:
                raise


def verify_pt_files(img_dir: Path):
    for path in tqdm(list(img_dir.glob("*.pt")), desc="Verifying .pt files"):
        try:
            torch.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}. Deleting. You must re-run the generation script")
            path.unlink()


if __name__ == "__main__":
    allowed_suffixes = {".JPEG", ".JPG", ".PNG", ".gif", ".jpeg", ".jpg", ".png"}
    img_dir = files("component_classifier") / "data/images"
    local_paths = sorted(set(img_dir.glob("*")) - set(img_dir.glob("*.pt")))
    suffixes = {path.suffix for path in local_paths}
    assert suffixes == allowed_suffixes

    local_paths = [p for p in local_paths if not p.with_suffix(".pt").exists()]
    # process_map(img_to_tensor, local_paths, chunksize=25, max_workers=4, smoothing=0)
    for p in tqdm(local_paths, desc="Generating .pt files"):
        img_to_tensor(p)

    verify_pt_files(img_dir)
