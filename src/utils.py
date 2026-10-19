import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


class Standardizer:
    def __init__(self):
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)

        self.scale[self.scale == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError(
                "Standaryzator nie został dopasowany (metoda fit nie była wywołana)."
            )
        return (X - self.mean) / self.scale

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Standaryzator nie został dopasowany.")
        return X * self.scale + self.mean


def one_hot(y: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    y = y.astype(int).ravel()
    if num_classes is None:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    res = np.zeros((n, num_classes), dtype=np.float64)
    res[np.arange(n), y] = 1.0
    return res


def load_regression_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    X = data[:, :-1]
    y = data[:, -1:]
    return X, y


def load_classification_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    X = data[:, :-2]
    y = data[:, -1:]
    return X, y


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = make_rng()
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)

    val_size = int(n * val_frac)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def iterate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator
):
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]
