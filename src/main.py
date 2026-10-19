from pathlib import Path
import numpy as np

from config_loader import load_config, print_config
from mlp import build_model
from optimizers import SGD
from utils import make_rng, load_regression_csv, Standardizer


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.json"
    config = load_config(config_path)

    seed = config["seed"]
    rng = make_rng(seed)

    # Parametry sieci neuronowej
    layers = config["network"]["layers"]
    activations = config["network"]["activations"]
    use_bias = config["network"]["use_bias"]

    # Parametry uczenia
    loss = config["training"]["loss"]
    learning_rate = config["training"]["learning_rate"]
    momentum = config["training"].get("momentum", 0.0)
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]

    print_config(config)

    # Czytanie danych i trening
    data_cfg = config["data"]
    train_path = Path(__file__).resolve().parent.parent / data_cfg["train_file"]
    test_path = Path(__file__).resolve().parent.parent / data_cfg["test_file"]

    X_train, y_train = load_regression_csv(str(train_path))
    scaler_X, scaler_y = Standardizer(), Standardizer()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    model = build_model(config)
    optimizer = SGD(lr=learning_rate, momentum=momentum)

    print(f"\nRozpoczynanie trenowania ({epochs} epok)...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        record_weights=False,
    )
    print(f"Trening zakończony! Końcowa strata: {history['train_loss'][-1]:.6f}")
