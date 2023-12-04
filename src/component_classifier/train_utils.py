import logging
import warnings
from importlib.resources import files

import mlflow
import numpy as np
import pandas as pd
import torch
import torchmetrics
from mlflow.models import infer_signature
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet18, resnet34, resnet50
from tqdm.auto import tqdm

from component_classifier.data_utils import ImageDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name: str, num_classes) -> nn.Module:
    def _get(model, num_classes: int, weights):
        # To not use pretrained weights: model = resnet50(weights=None)
        model = model(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    match model_name:
        case "resnet50_imagenet":
            return _get(resnet50, num_classes, weights=ResNet50_Weights.IMAGENET1K_V2)
        case "resnet50_untrained":
            return _get(resnet50, num_classes, weights=None)
        case "resnet34_untrained":
            return _get(resnet34, num_classes, weights=None)
        case "resnet18_untrained":
            return _get(resnet18, num_classes, weights=None)


def update_progress(progress: tqdm, metrics: list[torchmetrics.Metric], prefix: str, loss: torch.Tensor) -> None:
    desc = f"{prefix} Loss: {loss.item():.5f}"
    for metric in metrics:
        desc += f" {metric.__class__.__name__}={metric.compute().item():.5f}"
    progress.set_description(desc)


def finalize_metrics(metrics: list[torchmetrics.Metric], prefix: str, loss: torch.Tensor, k: int) -> None:
    mlflow.log_metric(key=f"{prefix} loss", value=loss, step=k)
    for metric in metrics:
        mlflow.log_metric(key=f"{prefix} {metric.__class__.__name__}", value=metric.compute().item(), step=k)
        metric.reset()


def log_model(model: nn.Module) -> None:
    with torch.no_grad():
        X = torch.rand(1, 3, 224, 224, device=DEVICE)
        signature = infer_signature(X.cpu().numpy(), model(X).cpu().numpy())

    req_path = str(files("component_classifier").parents[1] / "requirements.txt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        mlflow.pytorch.log_model(model, "model", signature=signature, pip_requirements=req_path)


transform = ResNet50_Weights.IMAGENET1K_V2.transforms()  # Turn into ImageNet format

_weak_augment = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(degrees=0, translate=(0.125, 0.125))]
)
_weak_augment_svhn = transforms.Compose([transforms.RandomAffine(degrees=0, translate=(0.125, 0.125))])


def weak_augment(arr: torch.Tensor, is_svhn: bool) -> torch.Tensor:
    return _weak_augment(arr) if is_svhn else _weak_augment_svhn(arr)


def batches_to_uint8(arr: torch.Tensor) -> torch.Tensor:
    n_batches = arr.shape[0]
    flat = arr.reshape(n_batches, -1)
    _min = flat.min(axis=-1, keepdim=True).values
    _max = flat.max(axis=-1, keepdim=True).values
    flat_uint8_arr = flat.sub_(_min).div_(_max - _min).mul_(255).clamp_(0, 255).to(torch.uint8)
    return flat_uint8_arr.view(arr.shape)


def strong_augment(arr: torch.Tensor, n_strong_aug: int) -> torch.Tensor:
    if not n_strong_aug:
        return arr

    magnitude = np.random.randint(5, 95)  # 5% -> 95%
    _strong_augment = transforms.RandAugment(num_ops=n_strong_aug, magnitude=magnitude, num_magnitude_bins=100)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The default value of the antialias parameter of
        # all the resizing transforms will change from None to True in v0.17
        aug_float_arr = transform(_strong_augment(batches_to_uint8(arr)))
    return aug_float_arr


def macro_loss_weight(train_ds: ImageDataset):
    uniq, y_counts = torch.unique(train_ds.Y, return_counts=True, sorted=True)
    weight = (1 / y_counts).to(DEVICE)
    weight /= weight.sum()
    return weight


def log_preds(preds: pd.DataFrame, prefix: str):
    """Mutes: INFO mlflow.tracking.client: Appending new table to already existing artifact dev_preds.csv"""
    previous_level = logging.root.manager.disable
    logging.disable()
    mlflow.log_table(preds, artifact_file=f"{prefix}_preds.csv")
    logging.disable(previous_level)
