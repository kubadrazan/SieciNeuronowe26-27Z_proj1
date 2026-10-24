import numpy as np


def xavier_uniform(n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=(n_in, n_out))


def he_normal(n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
    std = np.sqrt(2.0 / n_in)
    return rng.normal(0.0, std, size=(n_in, n_out))


def zeros(n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
    return np.zeros((n_in, n_out))


def get_initializer(activation_name: str):
    if not isinstance(activation_name, str):
        raise ValueError(
            f"Nazwa aktywacji musi być tekstem, otrzymano: {type(activation_name).__name__}"
        )
    
    key = activation_name.strip().lower()
    if key == "relu":
        return he_normal
    else:
        return xavier_uniform