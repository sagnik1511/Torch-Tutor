from pycoach.metrics.__base__ import Metric
from torch import Tensor
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             r2_score)


class Accuracy(Metric):

    def __call__(self, true: Tensor, pred: Tensor) -> float:
        assert true.shape[0] == pred.shape[0], "Shape Mismatched!"
        true, pred = self.detach_from_device(true, pred)
        return accuracy_score(true, pred)


class Precision(Metric):

    def __call__(self, true: Tensor, pred: Tensor) -> float:
        assert true.shape[0] == pred.shape[0], "Shape Mismatched!"
        true, pred = self.detach_from_device(true, pred)
        return precision_score(true, pred, average="weighted")


class Recall(Metric):

    def __call__(self, true: Tensor, pred: Tensor) -> float:
        assert true.shape[0] == pred.shape[0], "Shape Mismatched!"
        true, pred = self.detach_from_device(true, pred)
        return recall_score(true, pred, average="weighted")


class F1Score(Metric):

    def __call__(self, true: Tensor, pred: Tensor) -> float:
        assert true.shape[0] == pred.shape[0], "Shape Mismatched!"
        true, pred = self.detach_from_device(true, pred)
        return f1_score(true, pred, average="weighted")


class MeanSquaredError(Metric):

    def __call__(self, true: Tensor, pred: Tensor) -> float:
        assert true.shape[0] == pred.shape[0], "Shape Mismatched!"
        true, pred = self.detach_from_device(true, pred)
        return r2_score(true, pred)