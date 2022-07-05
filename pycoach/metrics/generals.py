from pycoach.metrics.__base__ import Metric
from torch import Tensor
from numpy import argmax
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             r2_score)


class Accuracy(Metric):

    def __call__(self, true: Tensor, pred: Tensor) -> float:
        assert true.shape[0] == pred.shape[0], "Shape Mismatched!"
        true, pred = self.detach_from_device(true, pred)
        pred = argmax(pred, axis=1).reshape(-1)
        return accuracy_score(true, pred)


class Precision(Metric):

    def __call__(self, true: Tensor, pred: Tensor) -> float:
        assert true.shape[0] == pred.shape[0], "Shape Mismatched!"
        true, pred = self.detach_from_device(true, pred)
        pred = argmax(pred, axis=1).reshape(-1)
        return precision_score(true, pred, average="weighted")


class Recall(Metric):

    def __call__(self, true: Tensor, pred: Tensor) -> float:
        assert true.shape[0] == pred.shape[0], "Shape Mismatched!"
        true, pred = self.detach_from_device(true, pred)
        pred = argmax(pred, axis=1).reshape(-1)
        return recall_score(true, pred, average="weighted")


class F1Score(Metric):

    def __call__(self, true: Tensor, pred: Tensor) -> float:
        assert true.shape[0] == pred.shape[0], "Shape Mismatched!"
        true, pred = self.detach_from_device(true, pred)
        pred = argmax(pred, axis=1).reshape(-1)
        return f1_score(true, pred, average="weighted")


class MeanSquaredError(Metric):

    def __call__(self, true: Tensor, pred: Tensor) -> float:
        assert true.shape[0] == pred.shape[0], "Shape Mismatched!"
        true, pred = self.detach_from_device(true, pred)
        return r2_score(true, pred)