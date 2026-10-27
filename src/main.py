import argparse
import time
from pathlib import Path

import numpy as np

from config_loader import load_config, print_config, validate_config
from mlp import build_model
from optimizers import SGD
from utils import (
    Standardizer,
    load_classification_csv,
    load_regression_csv,
    make_rng,
    one_hot,
    train_val_split,
)
from visualization import (
    plot_classification_results,
    plot_loss,
    plot_regression_results,
    plot_weight_evolution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening i ewaluacja sieci MLP.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "config.json",
        help="Ścieżka do pliku konfiguracyjnego JSON.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Katalog zapisu wykresów (domyślnie plots/<nazwa-konfiguracji>).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Pomiń generowanie wykresów.",
    )
    return parser.parse_args()


def encode_classification_targets(
    y_train: np.ndarray, y_test: np.ndarray, output_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.unique(y_train)
    if classes.size < 2:
        raise ValueError(
            "Klasyfikacja wymaga co najmniej dwóch klas w zbiorze treningowym."
        )

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
            classes,
        )
    if output_size == classes.size:
        return (
            one_hot(train_indices, classes.size),
            one_hot(test_indices, classes.size),
            classes,
        )

    raise ValueError(
        f"Warstwa wyjściowa ma {output_size} neuronów, ale dla {classes.size} klas "
        "wymaga jednego neuronu dla klasyfikacji binarnej albo jednego neuronu na klasę."
    )


def _compute_classification_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if y_pred.shape[1] == 1:
        predicted_indices = (y_pred[:, 0] >= 0.5).astype(int)
        true_indices = y_true[:, 0].astype(int)
    else:
        predicted_indices = np.argmax(y_pred, axis=1)
        true_indices = np.argmax(y_true, axis=1)
    return float(np.mean(predicted_indices == true_indices))


def run_from_config(
    config: dict,
    plot_dir: Path | None = None,
    create_plots: bool = False,
    verbose: bool = True,
) -> dict:
    validate_config(config)
    if verbose:
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
            raise ValueError(
                "Regresja R -> R wymaga dokładnie jednej wartości docelowej."
            )
    else:
        X_train, y_train = load_classification_csv(str(train_path))
        X_test, y_test = load_classification_csv(str(test_path))

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Liczba cech w zbiorach treningowym i testowym musi być taka sama."
        )

    model = build_model(config)
    if model.layers[0] != X_train.shape[1]:
        raise ValueError(
            f"Sieć oczekuje {model.layers[0]} cech, ale dane zawierają {X_train.shape[1]}."
        )

    if task == "classification":
        y_train, y_test, classes = encode_classification_targets(
            y_train, y_test, model.layers[-1]
        )
    elif model.layers[-1] != y_train.shape[1]:
        raise ValueError(
            f"Sieć ma {model.layers[-1]} neuronów wyjściowych, ale dane regresyjne "
            f"zawierają {y_train.shape[1]} wartości docelowych."
        )

    X_train_full_raw = X_train.copy()
    X_test_raw = X_test.copy()
    y_test_raw = y_test.copy()

    X_train, y_train, X_val, y_val = train_val_split(
        X_train,
        y_train,
        val_frac=config["training"].get("validation_fraction", 0.2),
        rng=make_rng(config.get("seed", 42)),
    )
    X_train_raw = X_train.copy()
    y_train_raw = y_train.copy()
    X_val_raw = X_val.copy()
    y_val_raw = y_val.copy()

    scaler_X = Standardizer()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    if task == "regression":
        scaler_y = Standardizer()
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        y_test_for_loss = scaler_y.transform(y_test)
    else:
        y_test_for_loss = y_test

    training = config["training"]
    optimizer = SGD(
        lr=training["learning_rate"],
        momentum=training.get("momentum", 0.0),
    )

    if verbose:
        print(
            f"\nRozpoczynanie trenowania ({training['epochs']} epok, zadanie: {task})..."
        )
    start_time = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        optimizer=optimizer,
        X_val=X_val,
        y_val=y_val,
        record_weights=create_plots,
    )
    elapsed_sec = time.perf_counter() - start_time

    y_pred = model.predict(X_test)
    test_loss = float(model.loss_fn.forward(y_pred, y_test_for_loss))
    train_loss_final = float(history["train_loss"][-1])
    val_loss_final = float(history["val_loss"][-1])
    val_losses = [float(v) for v in history["val_loss"] if v is not None]
    val_loss_best = float(np.min(val_losses)) if val_losses else val_loss_final
    best_epoch = (
        int(np.argmin(val_losses) + 1) if val_losses else len(history["train_loss"])
    )

    if verbose:
        print(
            f"Trening zakończony! Końcowa strata treningowa: " f"{train_loss_final:.6f}"
        )
        print(f"Strata testowa: {test_loss:.6f}")
        print(f"Strata walidacyjna: {val_loss_final:.6f}")

    results: dict = {
        "train_loss_final": train_loss_final,
        "val_loss_final": val_loss_final,
        "val_loss_best": val_loss_best,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "time_sec": elapsed_sec,
        "train_acc": None,
        "val_acc": None,
        "test_acc": None,
        "test_rmse": None,
        "test_mae": None,
        "test_rmse_inside": None,
        "test_rmse_outside": None,
    }

    if task == "regression":
        y_pred_original = scaler_y.inverse_transform(y_pred)
        rmse = float(np.sqrt(np.mean((y_pred_original - y_test) ** 2)))
        mae = float(np.mean(np.abs(y_pred_original - y_test)))
        results["test_rmse"] = rmse
        results["test_mae"] = mae
        if verbose:
            print(f"Test RMSE: {rmse:.6f}")
        training_x_min = float(np.min(X_train_full_raw[:, 0]))
        training_x_max = float(np.max(X_train_full_raw[:, 0]))
        inside_training_range = (X_test_raw[:, 0] >= training_x_min) & (
            X_test_raw[:, 0] <= training_x_max
        )
        for is_inside, label in (
            (True, "inside"),
            (False, "outside"),
        ):
            mask = inside_training_range == is_inside
            if np.any(mask):
                range_rmse = float(
                    np.sqrt(np.mean((y_pred_original[mask] - y_test[mask]) ** 2))
                )
                results[f"test_rmse_{label}"] = range_rmse
                if verbose:
                    print(
                        f"Test RMSE {label} training x-range "
                        f"({int(np.sum(mask))} points): {range_rmse:.6f}"
                    )
    else:
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        train_acc = _compute_classification_accuracy(y_train_pred, y_train)
        val_acc = _compute_classification_accuracy(y_val_pred, y_val)
        test_acc = _compute_classification_accuracy(y_pred, y_test)
        results["train_acc"] = train_acc
        results["val_acc"] = val_acc
        results["test_acc"] = test_acc
        if verbose:
            print(f"Test accuracy: {test_acc:.4f}")

    if create_plots:
        if plot_dir is None:
            plot_dir = root / "plots" / "default"
        plot_loss(history, plot_dir / "loss.png")
        plot_weight_evolution(history, plot_dir)
        if task == "regression":
            plot_regression_results(
                X_train_raw,
                y_train_raw,
                X_val_raw,
                y_val_raw,
                X_test_raw,
                y_test_raw,
                scaler_y.inverse_transform(y_pred),
                model,
                scaler_X,
                scaler_y,
                plot_dir / "regression_fit.png",
            )
        else:
            plot_classification_results(
                X_train_raw,
                y_train_raw,
                X_val_raw,
                y_val_raw,
                X_test_raw,
                y_test_raw,
                classes,
                model,
                scaler_X,
                plot_dir / "classification_boundary.png",
            )
        if verbose:
            print(f"Wykresy zapisano w: {plot_dir}")

    return results


def run(
    config_path: Path,
    plot_dir: Path | None = None,
    create_plots: bool = True,
) -> dict:
    config = load_config(config_path)
    root = Path(__file__).resolve().parent.parent
    if create_plots and plot_dir is None:
        plot_dir = root / "plots" / config_path.stem
    return run_from_config(
        config,
        plot_dir=plot_dir,
        create_plots=create_plots,
        verbose=True,
    )


if __name__ == "__main__":
    args = parse_args()
    run(args.config, args.plot_dir, create_plots=not args.no_plots)
