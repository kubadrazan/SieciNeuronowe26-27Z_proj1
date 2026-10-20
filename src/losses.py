from abc import ABC, abstractmethod
import numpy as np

EPS = 1e-12


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass


class MSE(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n, k = y_pred.shape
        return 2.0 * (y_pred - y_true) / (n * k)


class MAE(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(np.abs(y_pred - y_true)))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n, k = y_pred.shape
        return np.sign(y_pred - y_true) / (n * k)


class BCE(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred_clipped = np.clip(y_pred, EPS, 1.0 - EPS)
        loss = -np.mean(
            y_true * np.log(y_pred_clipped)
            + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
        )
        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred_clipped = np.clip(y_pred, EPS, 1.0 - EPS)
        n, k = y_pred.shape
        grad = (
            (y_pred_clipped - y_true)
            / (y_pred_clipped * (1.0 - y_pred_clipped))
            / (n * k)
        )
        return grad


class CCE(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred_clipped = np.clip(y_pred, EPS, 1.0 - EPS)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_pred.shape[0]
        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred_clipped = np.clip(y_pred, EPS, 1.0 - EPS)
        n = y_pred.shape[0]
        return -y_true / y_pred_clipped / n


LOSS_REGISTRY: dict[str, type[Loss]] = {
    "mse": MSE,
    "mae": MAE,
    "bce": BCE,
    "cce": CCE,
}

AVAILABLE_LOSSES: tuple[str, ...] = tuple(LOSS_REGISTRY.keys())


def get_loss(name: str) -> Loss:
    if not isinstance(name, str):
        raise ValueError(
            f"Nazwa funkcji straty musi być tekstem (str), otrzymano: {type(name).__name__}"
        )

    key = name.strip().lower()
    if key not in LOSS_REGISTRY:
        raise ValueError(
            f"Nieznana funkcja straty: '{name}'. "
            f"Dostępne opcje: {list(AVAILABLE_LOSSES)}"
        )

    return LOSS_REGISTRY[key]()
