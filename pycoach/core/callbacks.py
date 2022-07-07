import numpy as np
from typing import List, Dict
from pathlib import Path
from torch import nn as nn
from torch import save
from os import path as osp, mkdir


class CallBack:

    def __init__(self, tracker, stop_epoch: int = 5, save_weights: bool = True,
                 on: str = "training", save_directory: Path = Path("results")):
        assert on in ["training", "validation"]
        self._ascending_tracker_list: List[str] = ["accuracy", "precision",
                                                   "recall", "f1_score"]  # current supporting metrics
        self._descending_tracker_list: List[str] = ["loss", "mse"]  # current supporting metrics
        self.tracker: str = tracker
        self.is_ascending: bool = self._value_init_tracker_value()
        self.tracker_best_value: float = float(-np.inf if self.is_ascending else np.inf)
        self.stop_epoch: int = stop_epoch
        self.save_weights: bool = save_weights
        self.save_directory = save_directory
        self.degrade_epoch: int = 0
        self.epoch_num: int = 0
        self.on: str = on

    def _value_init_tracker_value(self) -> bool:
        if self.tracker not in self._ascending_tracker_list + self._descending_tracker_list:
            raise NotImplementedError
        else:
            return self.tracker in self._ascending_tracker_list

    def _execute(self, curr_value: float, model: nn.Module) -> True:
        if self.is_ascending:
            if self.tracker_best_value < curr_value:
                res = True
            else:
                res = False
        else:
            if self.tracker_best_value > curr_value:
                res = True
            else:
                res = False
        if res:
            print(f"Best score updated. Current Best Score : {round(curr_value, 6)}")
            self.tracker_best_value = curr_value
            if self.save_weights:
                self._save_model(model)
            self.degrade_epoch = 0
            return False
        else:
            self.degrade_epoch += 1
            if self.degrade_epoch < self.stop_epoch:
                print("best score not updated.")
                return False
            else:
                print("Training stopped due to continuous degradation.")
                return True

    def update(self, results: List[Dict[str, List[float]]], model: nn.Module):
        self.epoch_num += 1
        if self.on == "training":
            score = results[0][self.tracker][-1]
        else:
            score = results[1][self.tracker][-1]
        return self._execute(score, model)

    def _save_model(self, model):
        checkpoint = {
            "epoch": self.epoch_num,
            "best_score": self.tracker_best_value,
            "model": model.state_dict(),
        }
        if not osp.isdir(self.save_directory):
            mkdir(self.save_directory)
        filepath = osp.join(self.save_directory, "best_model.pt")
        save(checkpoint, filepath)
