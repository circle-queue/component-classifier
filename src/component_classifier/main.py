import mlflow
import numpy as np
import torch
import torchmetrics
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset

from component_classifier.data_utils import LABEL_COLS, ImageDataset, get_metadata_df
from component_classifier.train_loop_builder import EpochLoopBuilder
from component_classifier.train_utils import DEVICE, get_model, macro_loss_weight

DEFAULT_PARAMS = {
    # These almost match the FixMatch hyperparameters in pg 6
    # (Paper uses K = n_training_steps = 2^20, we do a third)
    "device": DEVICE.type,
    "batch_size": 64,
    "batch_num_workers": 0,  # Don't multi-process the data loading
    "model_name": "resnet50_imagenet",
    "optimiser_name": "SGD",
    "K": 2**20,  # ~1 mil: Total samples to train on (supervised + unsupervised)
    "η": 3e-2,  # Learning rate
    "β": 0.9,  # SGD Momentum
    "µ": 7,  # Number of unlabeled examples per labeled example
    "λ": 1,  # Weight of the unlabeled loss
    "τ": 0.95,  # Threshold for pseudo-labeling to be used for training
    "n_strong_aug": 2,  # Num of RandAugment transforms applied to the unlabeled data
    "w_decay": 5e-4,
    "ema_decay": 0,
    "eval_split": "train_dev",  # Either an int describing a cross validation split, or "train_dev" for a train/dev split
    "fine_tune": "final_layer",  # 'final_layer' or 'all'
    "subsampling": None,  # Metadata about the total size of the dataset when subsampling
    "dataset_name": None,
}


def start_training(trainloader: DataLoader, devloader: DataLoader, unlabeledloader: DataLoader, **override_params):
    params = DEFAULT_PARAMS | override_params
    assert len(params) == len(DEFAULT_PARAMS), f"Unexpected params: {set(override_params) - set(DEFAULT_PARAMS)}"
    assert params["µ"] <= int(len(unlabeledloader.dataset) / len(trainloader.dataset))

    num_classes = np.unique(trainloader.dataset.Y).size
    train_model = get_model(params["model_name"], num_classes).to(DEVICE)

    # EMA supposedly improves the performance of SGD
    # https://pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema
    eval_model = (
        AveragedModel(train_model, multi_avg_fn=get_ema_multi_avg_fn(params["ema_decay"]))
        if params["ema_decay"]
        else train_model
    )

    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # NOTE: The input is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1, in general)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean", weight=macro_loss_weight(trainloader.dataset))

    if params["fine_tune"] == "final_layer":
        for param in train_model.parameters():
            param.requires_grad = False  # Freeze all layers
        train_model.fc.requires_grad_(True)  # Unfreeze the final layer
        parameters = train_model.fc.parameters()
    else:
        assert params["fine_tune"] == "all"
        parameters = train_model.parameters()

    match params["optimiser_name"]:
        case "Adam":
            optimizer = torch.optim.Adam(parameters, params["η"], weight_decay=params["w_decay"])
            assert "β" not in params
        case "SGD":
            optimizer = torch.optim.SGD(
                parameters,
                nesterov=True,
                lr=params["η"],
                momentum=params["β"],
                weight_decay=params["w_decay"],
            )
        case _:
            raise NotImplementedError()

    # The paper uses a cosine learning rate decay, but we don't have time to implement a the tracking of K
    # scheduler = torch.optim.lr_scheduler...

    run_epochs_fn = (
        EpochLoopBuilder()
        .add_optimizer(optimizer)
        # .add_scheduler(scheduler)
        .add_train_dev_split(trainloader, devloader)
    )

    if params["µ"]:
        run_epochs_fn.add_fixmatch(
            train_model,
            unlabeledloader,
            loss_fn,
            params["µ"],
            params["λ"],
            params["τ"],
            params["n_strong_aug"],
            is_svhn=params["dataset_name"] == "SVHN10",
        )

    if params["ema_decay"]:
        run_epochs_fn.add_ema_model(eval_model, train_model)

    run_id = run_epochs_fn(
        params,
        params["K"],
        train_model,
        eval_model,
        loss_fn,
        metrics=[
            torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes, average="macro").to(DEVICE),
        ],
        is_svhn=params["dataset_name"] == "SVHN10",
    )
    return run_id


def replace_dataloader_args(dataloader: DataLoader, **kwargs):
    kwargs = {k: getattr(dataloader, k, None) for k in DataLoader.__init__.__annotations__.keys()} | kwargs
    return DataLoader(**kwargs)


def loader_from_ds(ds: Dataset):
    return DataLoader(
        ds,
        batch_size=DEFAULT_PARAMS["batch_size"],
        num_workers=DEFAULT_PARAMS["batch_num_workers"],
        shuffle=True,
        pin_memory=True,
    )


if __name__ == "__main__":
    meta_df = get_metadata_df()

    default_train_dev_unlabeled_loader = (  # (lazy) Generator expression to only execute if we need it
        loader_from_ds(ImageDataset.from_metadata_df(meta_df, LABEL_COLS, split=split, n_cache=0, µ=7))
        for split in ["train", "dev", "unlabeled"]
    )

    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set("main.py")
    test = "plots"
    match test:
        case "run_all":
            from component_classifier.ablation_study import (
                perform_ablation_study,
                train_dataset_size_ablation,
            )
            from component_classifier.dataset_study import perform_dataset_study

            perform_dataset_study()
            train_dataset_size_ablation()
            perform_ablation_study()
        case "plots":
            from component_classifier.ablation_study import (
                plot_ablation,
                plot_subsampling,
            )
            from component_classifier.dataset_study import display_dataset_study

            display_dataset_study()
            plot_ablation()
            plot_subsampling()

        case "supervised":
            override_params = {"µ": 0, "K": 1_000_000}
            run_id = start_training(*default_train_dev_unlabeled_loader, **override_params)
        case "semisupervised":
            override_params = {"µ": 1, "K": 1}
            run_id = start_training(*default_train_dev_unlabeled_loader, **override_params)
        case "STL10":
            µ = 7
            train_size = 1000
            override_params = {
                "fine_tune": "all",
                "subsampling": train_size,
                "µ": µ,
                "model_name": "resnet18_untrained",
                "dataset_name": "STL10",
            }

            train, test, unlabeled = ImageDataset.from_torch_dataset("STL10", n_samples=train_size, µ=µ)
            assert µ <= int(len(unlabeled) / len(train)), int(len(unlabeled) / len(train))

            override_params = {"fine_tune": "all", "subsampling": train_size, "µ": µ}
            run_id = start_training(*map(loader_from_ds, [train, test, unlabeled]), **override_params)
