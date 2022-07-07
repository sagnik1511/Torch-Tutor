from torch import Tensor
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from pycoach.metrics.__base__ import Metric
import time
from torch.utils.data import DataLoader
import numpy as np


def run_single_batch(batch: Tuple[Tensor, Tensor],
                     model: nn.Module, loss_fn: nn.Module,
                     metrics: Dict[str, Metric],
                     device: str = 'cpu') -> Tuple[nn.Module, Dict[str, float]]:
    x, y = batch
    if device == "cuda":
        x = x.cuda()
        y = y.cuda()
    op = model(x)
    try:
        loss = loss_fn(op, y)
    except:
        loss = loss_fn(op, y.long())
    metric_arr = dict()
    for name, metric in metrics.items():
        metric_arr[name] = metric(y, op)
    metric_arr["loss"] = loss.item()
    return loss, metric_arr


def train_single_epoch(training_loader: DataLoader,
                       model: nn.Module, loss_fn: nn.Module,
                       optimizer: Any, training_steps: int,
                       metrics: Dict[str, Metric], logging_index: int,
                       device: str = "cpu") -> Tuple[Tuple[nn.Module, Any], Dict[str, float]]:

    train_metric_arr = {k: [] for k, _ in metrics.items()}
    train_metric_arr["loss"] = []
    model.train()
    for index, batch in enumerate(training_loader):
        optimizer.zero_grad()
        loss, metric_arr = run_single_batch(batch, model, loss_fn, metrics, device)
        loss.backward()
        optimizer.step()
        metric_arr = {k: round(v, 6) for k, v in metric_arr.items()}
        for k in train_metric_arr.keys():
            train_metric_arr[k].append(metric_arr[k])
        if index % logging_index == 0:
            print(f"[Step {index}] : {metric_arr}")
        if index % training_steps == training_steps - 1:
            break
    train_metric_arr = {k: np.mean(v) for k, v in train_metric_arr.items()}
    return (model, optimizer), train_metric_arr


def validate_single_epoch(val_loader: DataLoader, model: nn.Module,
                          loss_fn: nn.Module, validation_steps: int,
                          metrics: Dict[str, Metric],
                          device: str = "cpu") -> Dict[str, float]:

    val_metric_arr = {k: [] for k, _ in metrics.items()}
    val_metric_arr["loss"] = []
    model.eval()
    for index, batch in enumerate(val_loader):
        _, metric_arr = run_single_batch(batch, model, loss_fn, metrics, device)
        if index % validation_steps == validation_steps - 1:
            break
        for k in val_metric_arr.keys():
            val_metric_arr[k].append(metric_arr[k])
    val_metric_arr = {k: np.mean(v) for k, v in val_metric_arr.items()}
    return val_metric_arr


def run_single_epoch(train_loader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: Any,
                     training_steps: int, metrics: Dict[str, Metric],
                     val_loader: Any, val_steps: int, logging_index: int,
                     device: str = 'cpu') -> Tuple[Tuple[nn.Module, Any], List[Dict[str, float]]]:
    init = time.time()
    res_arr = []
    (model, optimizer), train_res = train_single_epoch(train_loader, model, loss_fn,
                                                       optimizer, training_steps,
                                                       metrics, logging_index, device)
    train_res = {k: round(v, 6) for k, v in train_res.items()}
    res_arr.append(train_res)
    if val_loader:
        val_res = validate_single_epoch(val_loader, model, loss_fn, val_steps, metrics, device)
        val_res = {k: round(v, 6) for k, v in val_res.items()}
        res_arr.append(val_res)

    print(f"Execution Time : {round(time.time() - init, 6)} seconds")
    return (model, optimizer), res_arr
