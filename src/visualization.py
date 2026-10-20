from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from utils import Standardizer


def plot_loss(history: dict, output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, history["train_loss"], label="Train")

    val_loss = history.get("val_loss")
    if val_loss and any(value is not None for value in val_loss):
        valid_epochs = [
            (epoch, value)
            for epoch, value in zip(epochs, val_loss)
            if value is not None
        ]
        ax.plot(
            [epoch for epoch, _ in valid_epochs],
            [value for _, value in valid_epochs],
            label="Validation",
        )

    ax.set(title="Loss over epochs", xlabel="Epoch", ylabel="Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, output_path)


def plot_weight_evolution(history: dict, output_dir: Path) -> list[Path]:
    weights = history.get("weights")
    if not weights:
        raise ValueError("Historia treningu nie zawiera ewolucji wag.")

    epochs = np.arange(1, len(weights) + 1)
    output_paths = []
    for layer_index in range(len(weights[0])):
        fig, ax = plt.subplots(figsize=(10, 6))
        layer_weights = np.stack(
            [epoch[layer_index]["W"].ravel() for epoch in weights]
        )
        colors = plt.get_cmap("turbo")(
            np.linspace(0.0, 1.0, layer_weights.shape[1])
        )
        for weight_index in range(layer_weights.shape[1]):
            ax.plot(
                epochs,
                layer_weights[:, weight_index],
                color=colors[weight_index],
                alpha=0.35,
                linewidth=0.7,
            )
        ax.plot(
            epochs,
            np.mean(layer_weights, axis=1),
            color="black",
            linewidth=2,
            label="Mean weight",
        )

        if "b" in weights[0][layer_index]:
            biases = np.stack(
                [epoch[layer_index]["b"].ravel() for epoch in weights]
            )
            for bias_index in range(biases.shape[1]):
                ax.plot(
                    epochs,
                    biases[:, bias_index],
                    linestyle="--",
                    linewidth=1,
                    label="Bias" if bias_index == 0 else None,
                )

        ax.set(
            title=f"Parameter evolution - layer {layer_index + 1}",
            xlabel="Epoch",
            ylabel="Parameter value",
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        output_path = output_dir / f"weight_evolution_layer_{layer_index + 1}.png"
        _save_figure(fig, output_path)
        output_paths.append(output_path)

    return output_paths


def plot_regression_results(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model,
    scaler_X: Standardizer,
    scaler_y: Standardizer,
    output_path: Path,
) -> None:
    if X_train.shape[1] != 1 or y_train.shape[1] != 1:
        raise ValueError("Wykres regresji obsługuje jedną cechę i jedną wartość docelową.")

    x_grid = np.linspace(
        min(float(np.min(X_train)), float(np.min(X_test))),
        max(float(np.max(X_train)), float(np.max(X_test))),
        500,
    ).reshape(-1, 1)
    y_grid = scaler_y.inverse_transform(
        model.predict(scaler_X.transform(x_grid))
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X_train[:, 0], y_train[:, 0], s=18, alpha=0.65, label="Train")
    ax.scatter(X_test[:, 0], y_test[:, 0], s=14, alpha=0.45, label="Test targets")
    ax.scatter(
        X_test[:, 0],
        y_pred[:, 0],
        s=12,
        alpha=0.55,
        label="Test predictions",
    )
    ax.plot(x_grid[:, 0], y_grid[:, 0], color="black", linewidth=2, label="MLP curve")
    ax.set(title="Regression fit", xlabel="x", ylabel="y")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, output_path)


def plot_classification_results(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
    model,
    scaler_X: Standardizer,
    output_path: Path,
) -> None:
    if X_train.shape[1] != 2 or X_test.shape[1] != 2:
        raise ValueError("Granica decyzyjna wymaga dokładnie dwóch cech wejściowych.")

    x_min = min(float(np.min(X_train[:, 0])), float(np.min(X_test[:, 0]))) - 0.5
    x_max = max(float(np.max(X_train[:, 0])), float(np.max(X_test[:, 0]))) + 0.5
    y_min = min(float(np.min(X_train[:, 1])), float(np.min(X_test[:, 1]))) - 0.5
    y_max = max(float(np.max(X_train[:, 1])), float(np.max(X_test[:, 1]))) + 0.5
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    scores = model.predict(scaler_X.transform(grid))
    if scores.shape[1] == 1:
        grid_classes = (scores[:, 0] >= 0.5).astype(int)
    else:
        grid_classes = np.argmax(scores, axis=1)
    regions = grid_classes.reshape(grid_x.shape)

    colors = plt.get_cmap("tab10")(np.arange(max(classes.size, 1)))
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(9, 7))
    regions_plot = ax.contourf(
        grid_x,
        grid_y,
        regions,
        levels=np.arange(-0.5, classes.size + 0.5, 1),
        cmap=cmap,
        alpha=0.25,
    )

    train_classes = _class_indices(y_train)
    test_classes = _class_indices(y_test)
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=train_classes,
        cmap=cmap,
        vmin=0,
        vmax=max(classes.size - 1, 1),
        s=24,
        marker="o",
        edgecolors="black",
        linewidths=0.3,
        label="Train",
    )
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=test_classes,
        cmap=cmap,
        vmin=0,
        vmax=max(classes.size - 1, 1),
        s=28,
        marker="x",
        linewidths=0.8,
        label="Test",
    )
    colorbar = fig.colorbar(regions_plot, ax=ax, ticks=np.arange(classes.size))
    colorbar.ax.set_yticklabels([str(class_label) for class_label in classes])
    colorbar.set_label("Class")
    ax.set(title="Classification decision regions", xlabel="x₁", ylabel="x₂")
    ax.grid(True, alpha=0.2)
    ax.legend()
    _save_figure(fig, output_path)


def _class_indices(targets: np.ndarray) -> np.ndarray:
    if targets.shape[1] == 1:
        return targets[:, 0].astype(int)
    return np.argmax(targets, axis=1)


def _save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
