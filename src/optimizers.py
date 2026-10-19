import numpy as np


class SGD:

    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocities: list[dict[str, np.ndarray]] | None = None

    def step(self, layers: list) -> None:
        if self.velocities is None:
            self.velocities = []
            for layer in layers:
                v = {"W": np.zeros_like(layer.W)}
                if layer.use_bias and layer.b is not None:
                    v["b"] = np.zeros_like(layer.b)
                self.velocities.append(v)

        for layer, v in zip(layers, self.velocities):
            if layer.dW is not None:
                v["W"] = self.momentum * v["W"] - self.lr * layer.dW
                layer.W += v["W"]

            if layer.use_bias and layer.b is not None and layer.db is not None:
                v["b"] = self.momentum * v["b"] - self.lr * layer.db
                layer.b += v["b"]