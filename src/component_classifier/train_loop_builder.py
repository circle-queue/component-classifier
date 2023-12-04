from dataclasses import dataclass, field
from typing import Any, Callable

import mlflow
import pandas as pd
import torch
import torchmetrics
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from component_classifier.train_utils import (
    DEVICE,
    finalize_metrics,
    log_model,
    log_preds,
    strong_augment,
    update_progress,
    weak_augment,
)

RunId = str


class Hooks(list[Callable]):
    """
    A list of functions to be called at a certain point in the training loop
    Each hook has side effects/mutates
    """

    def __call__(self) -> list[None | Any]:
        return [hook() for hook in self]


@dataclass
class EpochLoopBuilder:
    """
    Simplifies modification of the train loop

    ### Example:
        epoch_fn = TrainLoopBuilder().add_optimizer(optimizer).add_scheduler(scheduler).build()

        run_id = epoch_fn(params, epochs, train_model, eval_model, train_labeled_dataloader, loss_fn, metrics)
    """

    train_model: nn.Module = None
    eval_model: nn.Module = None
    loss_fn: _Loss = None
    metrics: list[torchmetrics.Metric] = None

    k: int = 0  # numer of training steps performed

    train_dataloader: DataLoader = None
    dev_dataloader: DataLoader = None

    # All hooks affect TRAINING only
    pre_inner_loop_hooks: Hooks[Callable] = field(default_factory=Hooks)
    post_inner_loop_hooks: Hooks[Callable] = field(default_factory=Hooks)
    post_loop_hooks: Hooks[Callable] = field(default_factory=Hooks)
    loss_hooks: Hooks[Callable] = field(default_factory=Hooks)

    def __call__(
        self,
        params: dict,
        K: int,
        train_model: nn.Module,
        eval_model: nn.Module,
        loss_fn: _Loss,
        metrics: list[torchmetrics.Metric],
        is_svhn: bool,
    ) -> RunId:
        """Once all the hooks are added, we can execute the main epoch loop"""
        train_epoch, eval_epoch = self._build_train_eval_fn(is_svhn)

        with mlflow.start_run(nested=bool(mlflow.active_run())) as run:
            mlflow.log_params(params)
            with tqdm(total=K, desc="Training samples...") as progress:
                try:
                    best_eval_loss = float("inf")
                    five_epoch_steps = len(self.train_dataloader.dataset) * (params["µ"] + 1) * 5
                    early_stopping_base = early_stopping = max(1000, five_epoch_steps)
                    while self.k < K:
                        _train_loss = train_epoch(train_model, self.train_dataloader, loss_fn, metrics)
                        eval_loss = eval_epoch(eval_model, self.dev_dataloader, loss_fn, metrics)

                        step_k = self.k - progress.n
                        progress.update(step_k)

                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            early_stopping = early_stopping_base
                        elif early_stopping <= 0:
                            break
                        else:
                            early_stopping -= step_k
                finally:
                    log_model(eval_model)
                    mlflow.log_metric("best_eval_loss", best_eval_loss)

                run_id = run.info.run_id
        return run_id

    def _build_train_eval_fn(self, is_svhn: bool):
        def train_epoch(
            model: nn.Module,
            dataloader: DataLoader,
            loss_fn: _Loss,
            metrics: list[torchmetrics.Metric],
        ):
            prefix = "train"
            model.train()

            total_loss = torch.tensor(0.0, device=DEVICE)
            progress = tqdm(dataloader, desc=prefix, leave=False)
            for i, (imgs, y, ids) in enumerate(progress):
                self.k += len(ids)
                imgs = imgs.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                self.pre_inner_loop_hooks()

                imgs = weak_augment(imgs, is_svhn).to(DEVICE, non_blocking=True)
                pred = model(imgs)
                loss = sum([loss_fn(pred, y), *self.loss_hooks()])
                total_loss += loss.detach()
                loss.backward()

                self.post_inner_loop_hooks()

                if i % max((len(progress) // 5), 1) == 0:
                    for metric in metrics:
                        metric(pred, y)

                    update_progress(progress, metrics, prefix, loss)

            self.post_loop_hooks()

            avg_loss = (total_loss / len(dataloader)).item()
            finalize_metrics(metrics, prefix, avg_loss, self.k)
            return avg_loss

        def eval_epoch(
            model: nn.Module | AveragedModel,
            dataloader: DataLoader,
            loss_fn: _Loss,
            metrics: list[torchmetrics.Metric],
        ) -> float:
            prefix = "dev"
            model.eval()

            preds = []
            total_loss = torch.tensor(0.0, device=DEVICE)
            progress = tqdm(dataloader, desc=prefix, leave=False)
            with torch.no_grad():
                for imgs, y, ids in progress:
                    imgs = imgs.to(DEVICE, non_blocking=True)
                    y = y.to(DEVICE, non_blocking=True)

                    pred = model(imgs)
                    loss = loss_fn(pred, y)
                    total_loss += loss.detach()

                    for metric in metrics:
                        metric(pred, y)

                    update_progress(progress, metrics, prefix, loss)
                    preds.append(pd.DataFrame({"ids": ids.cpu(), "y": y.cpu(), "pred": pred.argmax(dim=1).cpu()}))

            log_preds(pd.concat(preds).assign(k=self.k), prefix)

            avg_loss = (total_loss / len(dataloader)).item()
            finalize_metrics(metrics, prefix, avg_loss, self.k)
            return avg_loss

        return train_epoch, eval_epoch

    def add_train_dev_split(self, train_dataloader: DataLoader, dev_dataloader: DataLoader):
        assert self.train_dataloader is None  # Can only assign one dataloader
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        return self

    def add_optimizer(self, optimizer: torch.optim.Optimizer):
        self.pre_inner_loop_hooks.append(lambda: optimizer.zero_grad())
        self.post_inner_loop_hooks.append(lambda: optimizer.step())
        return self

    def add_scheduler(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        self.post_loop_hooks.append(lambda: scheduler.step())
        return self

    def add_ema_model(self, ema_model: AveragedModel, train_model: nn.Module):
        """
        ema_model stores the exponential moving average of the model parameters
        More info: https://pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema
        Paper:
        """
        self.post_inner_loop_hooks.append(lambda: ema_model.update_parameters(train_model))
        self.post_loop_hooks.append(
            lambda: torch.optim.swa_utils.update_bn(self.train_dataloader, ema_model, device=DEVICE)
        )  # This should be done before evaluating the model, so we just do it after training
        return self

    def add_fixmatch(
        self,
        model: nn.Module,
        unlabeled_dataloader: DataLoader,
        loss_fn: _Loss,
        µ: int,
        λ: float,
        τ: float,
        n_strong_aug: int,
        is_svhn: bool,
    ):
        """
        µ is a hyperparameter that determines the relative sizes of X and U. µ = int(len(U) / len(X))
        λ is a fixed scalar hyperparameter denoting the relative weight of the unlabeled loss
        τ is the minimum confidence threshold
        n_strong_aug is the # of RandAugment transforms applied to the unlabeled data
        """

        def set_unlabeled_iter():
            """We want to have this reset every epoch to reset the loading process"""
            self._unlabeled_iter = iter(unlabeled_dataloader)

        set_unlabeled_iter()
        self.post_loop_hooks.append(set_unlabeled_iter)

        def unsupervised_loss():
            losses = []
            n_imgs = 0
            for _ in range(µ):
                try:
                    unlabeled_imgs, _, _ = next(self._unlabeled_iter)
                except StopIteration:
                    break  # Our batch size may make it so a single batch can contain all µ loops of data
                self.k += len(unlabeled_imgs)
                n_imgs += len(unlabeled_imgs)

                weak = weak_augment(unlabeled_imgs, is_svhn).to(DEVICE, non_blocking=True)
                strong = strong_augment(unlabeled_imgs, n_strong_aug).to(DEVICE, non_blocking=True)
                pseudo_preds = model(weak)
                strong_preds = model(strong)

                # We filter the batch, such that we keep only the images
                # where the model is confident enough in its pseudo-label
                keep_img_mask = (torch.softmax(pseudo_preds, dim=1) > τ).max(axis=1).values
                pseudo_y = pseudo_preds[keep_img_mask].argmax(axis=1)
                strong_preds = strong_preds[keep_img_mask]

                mlflow.log_metric(key="n_pseudo_labels", value=len(pseudo_y))
                if len(pseudo_y):
                    losses.append(loss_fn(strong_preds, pseudo_y))

            return λ * sum(losses) / n_imgs

        self.loss_hooks.append(unsupervised_loss)
        return self
