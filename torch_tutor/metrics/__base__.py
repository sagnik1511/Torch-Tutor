import numpy as np
from torch import Tensor
from typing import Tuple


class Metric:

    def __init__(self, device: str):
        self.device = device

    def detach_from_device(self, true: Tensor, pred: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if self.device.startswith("cuda"):
            true = true.cpu()
            pred = pred.cpu()
        true = true.detach().numpy()
        pred = pred.detach().numpy()

        return true, pred

    def __call__(self, true: Tensor, pred: Tensor) -> float:
        _, _ = self.detach_from_device(true, pred)
        return np.random.randn(1)[0]

    @staticmethod
    def div(num1, num2):
        if num2 == 0:
            return 0.0
        else:
            return round(num1 / num2, 6)


if __name__ == "__main__":
    mt = Metric("cpu")
    x = Tensor([1, 2, 43])
    y = Tensor([2, 3, 4, 5])
    print(mt(x, y))
