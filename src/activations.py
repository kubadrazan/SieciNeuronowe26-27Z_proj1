from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:  # backward
        pass


class Sigmoid(Activation):
    """Funkcja sigmoidalna: f(x) = 1 / (1 + exp(-x))."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        f = self.forward(x)
        return f * (1.0 - f)


class Tanh(Activation):
    """Tangens hiperboliczny: f(x) = tanh(x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        f = np.tanh(x)
        return 1.0 - f**2


class ReLU(Activation):
    """Rectified Linear Unit: f(x) = max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0, 1.0, 0.0)


class Linear(Activation):
    """Funkcja liniowa: f(x) = x."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)  # array samych jedynek o tym samym kształcie co x


class Softmax(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        shift_x = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1.0 - s)


ACTIVATION_REGISTRY: dict[str, type[Activation]] = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "linear": Linear,
    "softmax": Softmax,
}

AVAILABLE_ACTIVATIONS: tuple[str, ...] = tuple(ACTIVATION_REGISTRY.keys())


def get_activation(name: str) -> Activation:
    """Fabryka zwracająca instancję funkcji aktywacji na podstawie jej nazwy."""
    if not isinstance(name, str):
        raise ValueError(
            f"Nazwa funkcji aktywacji musi być tekstem (str), otrzymano: {type(name).__name__}"
        )

    key = name.strip().lower()
    if key not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Nieznana funkcja aktywacji: '{name}'. "
            f"Dostępne opcje: {list(AVAILABLE_ACTIVATIONS)}"
        )

    return ACTIVATION_REGISTRY[key]()
