import argparse
from pathlib import Path

import numpy as np

from config_loader import load_config, print_config
from mlp import build_model
from optimizers import SGD
from utils import (
    Standardizer,
    load_classification_csv,
    load_regression_csv,
    one_hot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening i ewaluacja sieci MLP.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "config.json",
        help="Ścieżka do pliku konfiguracyjnego JSON.",
    )
    return parser.parse_args()


def encode_classification_targets(
    y_train: np.ndarray, y_test: np.ndarray, output_size: int
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y_train)
    if classes.size < 2:
        raise ValueError("Klasyfikacja wymaga co najmniej dwóch klas w zbiorze treningowym.")

    unknown_test_classes = np.setdiff1d(np.unique(y_test), classes)
    if unknown_test_classes.size:
        raise ValueError(
            f"Zbiór testowy zawiera klasy nieobecne w treningowym: "
            f"{unknown_test_classes.tolist()}"
        )

    train_indices = np.searchsorted(classes, y_train.ravel())
    test_indices = np.searchsorted(classes, y_test.ravel())

    if output_size == 1 and classes.size == 2:
        return (
            train_indices.reshape(-1, 1).astype(np.float64),
            test_indices.reshape(-1, 1).astype(np.float64),
        )
    if output_size == classes.size:
        return (
            one_hot(train_indices, classes.size),
            one_hot(test_indices, classes.size),
        )

    raise ValueError(
        f"Warstwa wyjściowa ma {output_size} neuronów, ale dla {classes.size} klas "
        "wymaga jednego neuronu dla klasyfikacji binarnej albo jednego neuronu na klasę."
    )


def run(config_path: Path) -> None:
    config = load_config(config_path)
    print_config(config)

    root = Path(__file__).resolve().parent.parent
    data_cfg = config["data"]
    task = data_cfg["task"]
    train_path = root / data_cfg["train_file"]
    test_path = root / data_cfg["test_file"]

    if task == "regression":
        X_train, y_train = load_regression_csv(str(train_path))
        X_test, y_test = load_regression_csv(str(test_path))
        if y_train.shape[1] != 1 or y_test.shape[1] != 1:
            raise ValueError("Regresja R -> R wymaga dokładnie jednej wartości docelowej.")
    else:
        X_train, y_train = load_classification_csv(str(train_path))
        X_test, y_test = load_classification_csv(str(test_path))

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Liczba cech w zbiorach treningowym i testowym musi być taka sama.")

    model = build_model(config)
    if model.layers[0] != X_train.shape[1]:
        raise ValueError(
            f"Sieć oczekuje {model.layers[0]} cech, ale dane zawierają {X_train.shape[1]}."
        )

    if task == "classification":
        y_train, y_test = encode_classification_targets(
            y_train, y_test, model.layers[-1]
        )
    elif model.layers[-1] != y_train.shape[1]:
        raise ValueError(
            f"Sieć ma {model.layers[-1]} neuronów wyjściowych, ale dane regresyjne "
            f"zawierają {y_train.shape[1]} wartości docelowych."
        )

    scaler_X = Standardizer()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    if task == "regression":
        scaler_y = Standardizer()
        y_train = scaler_y.fit_transform(y_train)
        y_test_for_loss = scaler_y.transform(y_test)
    else:
        y_test_for_loss = y_test

    training = config["training"]
    optimizer = SGD(
        lr=training["learning_rate"],
        momentum=training.get("momentum", 0.0),
    )

    print(
        f"\nRozpoczynanie trenowania ({training['epochs']} epok, zadanie: {task})..."
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        optimizer=optimizer,
        record_weights=False,
    )

    y_pred = model.predict(X_test)
    test_loss = model.loss_fn.forward(y_pred, y_test_for_loss)
    print(
        f"Trening zakończony! Końcowa strata treningowa: "
        f"{history['train_loss'][-1]:.6f}"
    )
    print(f"Strata testowa: {test_loss:.6f}")

    if task == "regression":
        y_pred_original = scaler_y.inverse_transform(y_pred)
        rmse = float(np.sqrt(np.mean((y_pred_original - y_test) ** 2)))
        print(f"Test RMSE: {rmse:.6f}")
    else:
        if y_pred.shape[1] == 1:
            predicted_indices = (y_pred[:, 0] >= 0.5).astype(int)
            true_indices = y_test[:, 0].astype(int)
        else:
            predicted_indices = np.argmax(y_pred, axis=1)
            true_indices = np.argmax(y_test, axis=1)
        accuracy = float(np.mean(predicted_indices == true_indices))
        print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    run(parse_args().config)
