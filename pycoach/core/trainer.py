import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Any
from pycoach.core.engine.loops import run_single_epoch
import time
from pycoach.metrics.generals import *
import pandas as pd


class Trainer:

    def __init__(self, train_dataset: Dataset, model: nn.Module,  device: str = "cpu"):
        self.train_ds = train_dataset
        self.model = model
        self.device = device
        self.optim: optim = None
        self.loss_fn = None  # nn.Module declared after, primarily holds None value
        self.metrics: Dict[str, Metric] = {}
        self.metrics_names: list = []
        self.train_scores: Dict[str, list] = {}
        self.validation_scores: Dict[str, list] = {}
        self.best_loss = float('inf')
        self.train_report = pd.DataFrame()
        self.val_report = pd.DataFrame()

    def compile(self, optimizer: optim, loss_fn: nn.Module, metrics: Dict[str, Metric],
                optimizer_hparams: dict):
        self.optim = optimizer(self.model.parameters(), *optimizer_hparams)
        self.loss_fn = loss_fn
        self.metrics_names = metrics
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> Dict[str, Metric]:
        metric_dict = {}
        for name in self.metrics_names:
            if name == "accuracy":
                metric_dict[name] = Accuracy(self.device)
            elif name == "precision":
                metric_dict[name] = Precision(self.device)
            elif name == "recall":
                metric_dict[name] = Recall(self.device)
            elif name == "f1_score":
                metric_dict[name] = F1Score(self.device)
            elif name == "mse":
                metric_dict[name] = MeanSquaredError(self.device)
            self.train_scores[name] = []
            self.validation_scores[name] = []
        return metric_dict

    def train(self, batch_size: int, num_epochs: int, training_steps: int = -1,
              validation_set: Any = None, logging_index: int = 10,
              validation_steps: int = -1, shuffle: bool = True,
              drop_last_batches: bool = True):

        train_dl, val_dl = self._prepare_data(self.train_ds, validation_set,
                                              batch_size, shuffle, drop_last_batches)
        init = time.time()
        for epoch in range(num_epochs):
            print(f"Epoch : {epoch + 1} :")
            (self.model, self.optim), res_arr = run_single_epoch(train_dl, self.model, self.loss_fn, self.optim,
                                                                 training_steps, self.metrics, val_dl, validation_steps,
                                                                 logging_index, self.device)
            print(f"Training scores : {pd.DataFrame(res_arr[0], index=[0])}")
            self.train_scores = {k: v.append(res_arr[0][k]) for k, v in self.train_scores}
            if validation_set:
                print(f"Validation scores : {pd.DataFrame(res_arr[1], index=[0])}")
                self.validation_scores = {k: v.append(res_arr[0][k]) for k, v in self.validation_scores}
        print(f"Training Completed...")
        print(f"Executed in {round(time.time() - init, 4)} seconds.\n")
        self._prepare_training_report()

    def _prepare_training_report(self):
        self.train_report = pd.DataFrame(self.train_scores)
        self.val_report = pd.DataFrame(self.validation_scores) \
            if len(self.validation_scores[next(iter(self.validation_scores.keys()))]) > 0 else pd.DataFrame()
        print("Prepared training reports...")

    @staticmethod
    def _prepare_data(train_set: Dataset, val_set: Any,
                      batch_size: int, shuffle: bool = True,
                      drop_last: bool = False) -> Tuple[DataLoader, Any]:
        train_dl = DataLoader(train_set, batch_size, shuffle, drop_last=drop_last)
        val_dl = DataLoader(val_set, batch_size, shuffle, drop_last=drop_last) if val_set else None

        return train_dl, val_dl
