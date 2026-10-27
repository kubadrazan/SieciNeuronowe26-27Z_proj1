import numpy as np
from activations import get_activation, Activation
from initializers import get_initializer
from losses import get_loss, Loss
from utils import iterate_minibatches, make_rng


class Dense:
    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: str | Activation,
        use_bias: bool = True,
        rng: np.random.Generator | None = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = use_bias

        if isinstance(activation, str):
            self.activation = get_activation(activation)
            act_name = activation
        else:
            self.activation = activation
            act_name = "custom"

        initializer = get_initializer(act_name)
        self.W = initializer(n_in, n_out, rng)
        self.b = np.zeros((1, n_out), dtype=np.float64) if use_bias else None

        self.a_prev: np.ndarray | None = None
        self.z: np.ndarray | None = None

        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        self.a_prev = a_prev
        if self.use_bias:
            self.z = a_prev @ self.W + self.b
        else:
            self.z = a_prev @ self.W
        return self.activation.forward(self.z)

    def backward(self, da: np.ndarray) -> np.ndarray:
        dz = self.activation.backward(da, self.z)
        self.dW = self.a_prev.T @ dz
        if self.use_bias:
            self.db = dz.sum(axis=0, keepdims=True)
        else:
            self.db = None
        return dz @ self.W.T


class MLP:

    def __init__(
        self,
        layers: list[int],
        activations: list[str],
        loss: str | Loss,
        use_bias: bool = True,
        seed: int | None = None,
    ):
        self.layers = layers
        self.activations_names = activations
        self.use_bias = use_bias
        self.rng = make_rng(seed)

        if isinstance(loss, str):
            self.loss_fn = get_loss(loss)
        else:
            self.loss_fn = loss

        if len(activations) != len(layers) - 1:
            raise ValueError(
                f"Liczba funkcji aktywacji ({len(activations)}) musi być o jeden mniejsza "
                f"niż liczba warstw ({len(layers)})."
            )

        self.dense_layers: list[Dense] = []
        for i in range(len(layers) - 1):
            n_in = layers[i]
            n_out = layers[i + 1]
            act_name = activations[i]

            layer_seed = int(self.rng.integers(0, 2**31 - 1))
            layer_rng = np.random.default_rng(layer_seed)

            dense = Dense(
                n_in=n_in,
                n_out=n_out,
                activation=act_name,
                use_bias=use_bias,
                rng=layer_rng,
            )
            self.dense_layers.append(dense)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.dense_layers:
            out = layer.forward(out)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def get_params(self) -> list[dict[str, np.ndarray]]:
        params = []
        for layer in self.dense_layers:
            p = {"W": layer.W.copy()}
            if layer.use_bias and layer.b is not None:
                p["b"] = layer.b.copy()
            params.append(p)
        return params

    def set_params(self, params: list[dict[str, np.ndarray]]) -> None:
        for layer, p in zip(self.dense_layers, params):
            layer.W = p["W"].copy()
            if layer.use_bias and "b" in p and layer.b is not None:
                layer.b = p["b"].copy()

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        last_act_name = self.activations_names[-1].strip().lower()
        loss_name = type(self.loss_fn).__name__.lower()
        n = y_pred.shape[0]
        k = y_pred.shape[1]

        if last_act_name == "sigmoid" and loss_name == "bce":
            delta = (y_pred - y_true) / (n * k)
        elif last_act_name == "softmax" and loss_name == "cce":
            delta = (y_pred - y_true) / n
        else:
            dL_dypred = self.loss_fn.backward(y_pred, y_true)
            last_layer = self.dense_layers[-1]
            delta = last_layer.activation.backward(dL_dypred, last_layer.z)

        current_delta = delta
        for i in reversed(range(len(self.dense_layers))):
            layer = self.dense_layers[i]
            if i == len(self.dense_layers) - 1:
                layer.dW = layer.a_prev.T @ current_delta
                layer.db = (
                    current_delta.sum(axis=0, keepdims=True)
                    if layer.use_bias and layer.b is not None
                    else None
                )
                current_delta = current_delta @ layer.W.T
            else:
                current_delta = layer.backward(current_delta)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        optimizer,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        record_weights: bool = True,
    ) -> dict:
        history = {
            "train_loss": [],
            "val_loss": [],
            "weights": [] if record_weights else None,
        }

        shuffle_rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))

        for epoch in range(epochs):
            for X_batch, y_batch in iterate_minibatches(X, y, batch_size, shuffle_rng):
                y_pred_batch = self.forward(X_batch)

                self.backward(y_pred_batch, y_batch)

                optimizer.step(self.dense_layers)

            train_preds = self.forward(X)
            train_loss = self.loss_fn.forward(train_preds, y)
            history["train_loss"].append(train_loss)

            if X_val is not None and y_val is not None:
                val_preds = self.forward(X_val)
                val_loss = self.loss_fn.forward(val_preds, y_val)
                history["val_loss"].append(val_loss)
            else:
                history["val_loss"].append(None)

            if record_weights:
                epoch_weights = []
                for layer in self.dense_layers:
                    w_dict = {"W": layer.W.copy()}
                    if layer.use_bias and layer.b is not None:
                        w_dict["b"] = layer.b.copy()
                    epoch_weights.append(w_dict)
                history["weights"].append(epoch_weights)

        return history


def build_model(config: dict) -> MLP:
    net_config = config["network"]
    layers = net_config["layers"]
    activations = net_config["activations"]
    use_bias = net_config.get("use_bias", True)

    train_config = config["training"]
    loss = train_config["loss"]

    seed = config.get("seed", 42)

    return MLP(
        layers=layers,
        activations=activations,
        loss=loss,
        use_bias=use_bias,
        seed=seed,
    )
