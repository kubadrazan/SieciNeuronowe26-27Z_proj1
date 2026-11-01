import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import count_parameters
from main import _compute_classification_accuracy, _encode_classification_labels, run_from_config
from mlp import build_model
from optimizers import SGD
from utils import (
    Standardizer,
    load_mnist_idx,
    make_rng,
    subsample_data,
    train_val_split,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

CSV_COLUMNS = [
    "study_name",
    "experiment_id",
    "train_samples",
    "test_samples",
    "seed",
    "num_hidden_layers",
    "hidden_layers",
    "layers_full",
    "total_params",
    "hidden_activation",
    "output_activation",
    "loss",
    "learning_rate",
    "momentum",
    "epochs",
    "batch_size",
    "train_loss_final",
    "val_loss_final",
    "val_loss_best",
    "best_epoch",
    "test_loss",
    "train_acc",
    "val_acc",
    "test_acc",
    "time_sec",
]


def build_mnist_spec(
    study_name: str,
    hidden_layers: list[int],
    hidden_activation: str = "relu",
    loss: str = "cce",
    lr: float = 0.01,
    momentum: float = 0.9,
    epochs: int = 15,
    batch_size: int = 128,
    max_train_samples: int | None = 10000,
    max_test_samples: int | None = 10000,
    seed: int = 42,
) -> dict:
    layers = [784] + list(hidden_layers) + [10]
    activations = [hidden_activation] * len(hidden_layers) + ["softmax"]

    data_cfg: dict = {
        "task": "classification",
        "format": "mnist_idx",
        "train_images": "data/MNIST/train-images.idx3-ubyte",
        "train_labels": "data/MNIST/train-labels.idx1-ubyte",
        "test_images": "data/MNIST/t10k-images.idx3-ubyte",
        "test_labels": "data/MNIST/t10k-labels.idx1-ubyte",
    }
    if max_train_samples is not None and max_train_samples < 60000:
        data_cfg["max_train_samples"] = max_train_samples
    if max_test_samples is not None and max_test_samples < 10000:
        data_cfg["max_test_samples"] = max_test_samples

    config = {
        "seed": seed,
        "network": {
            "layers": layers,
            "activations": activations,
            "use_bias": True,
        },
        "training": {
            "loss": loss,
            "learning_rate": lr,
            "momentum": momentum,
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_fraction": 0.1,
        },
        "data": data_cfg,
    }

    return {
        "study_name": study_name,
        "train_samples": max_train_samples if max_train_samples is not None else 60000,
        "test_samples": max_test_samples if max_test_samples is not None else 10000,
        "hidden_layers": list(hidden_layers),
        "hidden_activation": hidden_activation if hidden_layers else "none",
        "output_activation": "softmax",
        "config": config,
    }


def generate_mnist_specs(seed: int = 42) -> list[dict]:
    specs: list[dict] = []

    # 1. Badanie architektury na podzbiorze N=10000 (0 do 3 warstw ukrytych)
    architectures = [
        [],
        [64],
        [128],
        [256],
        [128, 64],
        [256, 128],
        [256, 128, 64],
    ]
    for arch in architectures:
        specs.append(
            build_mnist_spec(
                study_name="arch_10k",
                hidden_layers=arch,
                hidden_activation="relu",
                loss="cce",
                lr=0.01,
                batch_size=128,
                epochs=15,
                max_train_samples=10000,
                max_test_samples=2000,
                seed=seed,
            )
        )

    # 2. Badanie funkcji aktywacji i funkcji straty na N=10000 (architektura [128, 64])
    act_loss_variants = [
        ("relu", "cce"),
        ("tanh", "cce"),
        ("sigmoid", "cce"),
        ("relu", "mse"),
        ("tanh", "mse"),
    ]
    for act, loss_name in act_loss_variants:
        specs.append(
            build_mnist_spec(
                study_name="act_loss_10k",
                hidden_layers=[128, 64],
                hidden_activation=act,
                loss=loss_name,
                lr=0.01,
                batch_size=128,
                epochs=15,
                max_train_samples=10000,
                max_test_samples=2000,
                seed=seed,
            )
        )

    # 3. Badanie hiperparametrow optymalizacji (learning_rate oraz batch_size) na N=10000 ([128, 64])
    hyperparam_variants = [
        (0.002, 128),
        (0.01, 128),
        (0.05, 128),
        (0.01, 32),
        (0.01, 64),
        (0.01, 256),
    ]
    for lr, bs in hyperparam_variants:
        specs.append(
            build_mnist_spec(
                study_name="hyperparams_10k",
                hidden_layers=[128, 64],
                hidden_activation="relu",
                loss="cce",
                lr=lr,
                batch_size=bs,
                epochs=15,
                max_train_samples=10000,
                max_test_samples=2000,
                seed=seed,
            )
        )

    # 4. Porownanie najlepszych konfiguracji: N=10000 vs pelny zbior N=60000 (pelny test N_test=10000, 20 epok)
    best_configs = [
        ([128, 64], "relu", "cce", 0.01, 64),
        ([256, 128], "relu", "cce", 0.01, 64),
        ([256, 128, 64], "relu", "cce", 0.01, 64),
        ([256, 128], "tanh", "cce", 0.01, 64),
    ]
    for train_n in (10000, 60000):
        study_label = f"best_compare_{train_n // 1000}k"
        for arch, act, loss_name, lr, bs in best_configs:
            specs.append(
                build_mnist_spec(
                    study_name=study_label,
                    hidden_layers=arch,
                    hidden_activation=act,
                    loss=loss_name,
                    lr=lr,
                    batch_size=bs,
                    epochs=20,
                    max_train_samples=train_n,
                    max_test_samples=10000,
                    seed=seed,
                )
            )

    return specs


def _fmt(val: float | None) -> str:
    if val is None:
        return ""
    return f"{val:.6f}"


def run_mnist_experiments(specs: list[dict], output_csv: Path) -> list[dict]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total = len(specs)
    rows: list[dict] = []
    print(f"Rozpoczynanie {total} eksperymentów MNIST. Zapis do: {output_csv}\n")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        f.flush()

        for idx, spec in enumerate(specs, start=1):
            config = spec["config"]
            net = config["network"]
            train = config["training"]

            hidden_layers = spec["hidden_layers"]
            hidden_str = "-".join(str(x) for x in hidden_layers) if hidden_layers else "none"
            layers_full_str = "-".join(str(x) for x in net["layers"])

            metrics = run_from_config(
                config=config,
                create_plots=False,
                verbose=False,
            )

            row = {
                "study_name": spec["study_name"],
                "experiment_id": idx,
                "train_samples": spec["train_samples"],
                "test_samples": spec["test_samples"],
                "seed": config["seed"],
                "num_hidden_layers": len(hidden_layers),
                "hidden_layers": hidden_str,
                "layers_full": layers_full_str,
                "total_params": count_parameters(net["layers"], net["use_bias"]),
                "hidden_activation": spec["hidden_activation"],
                "output_activation": spec["output_activation"],
                "loss": train["loss"],
                "learning_rate": train["learning_rate"],
                "momentum": train["momentum"],
                "epochs": train["epochs"],
                "batch_size": train["batch_size"],
                "train_loss_final": _fmt(metrics["train_loss_final"]),
                "val_loss_final": _fmt(metrics["val_loss_final"]),
                "val_loss_best": _fmt(metrics["val_loss_best"]),
                "best_epoch": metrics["best_epoch"],
                "test_loss": _fmt(metrics["test_loss"]),
                "train_acc": _fmt(metrics["train_acc"]),
                "val_acc": _fmt(metrics["val_acc"]),
                "test_acc": _fmt(metrics["test_acc"]),
                "time_sec": _fmt(metrics["time_sec"]),
            }
            writer.writerow(row)
            f.flush()
            rows.append(row)

            print(
                f"[{idx:2d}/{total:2d}] {spec['study_name']:<17} | "
                f"N={spec['train_samples']:<5} | layers={layers_full_str:<18} | "
                f"act={spec['hidden_activation']:<7} | loss={train['loss']:<3} | "
                f"lr={train['learning_rate']:<5} bs={train['batch_size']:<3} | "
                f"val_acc={metrics['val_acc']:.4f} test_acc={metrics['test_acc']:.4f} "
                f"({metrics['time_sec']:.2f}s)"
            )

    print(f"\nZakończono eksperymenty MNIST! Zapisano w: {output_csv}")
    return rows


def generate_report2_plots(csv_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    X_train_all, y_train_all = load_mnist_idx(
        ROOT_DIR / "data/MNIST/train-images.idx3-ubyte",
        ROOT_DIR / "data/MNIST/train-labels.idx1-ubyte",
    )
    X_test_all, y_test_all = load_mnist_idx(
        ROOT_DIR / "data/MNIST/t10k-images.idx3-ubyte",
        ROOT_DIR / "data/MNIST/t10k-labels.idx1-ubyte",
    )

    # --- Fig 1: Przykładowe cyfry 0-9 ze zbioru MNIST ---
    fig, axes = plt.subplots(2, 10, figsize=(13, 3.2))
    for digit in range(10):
        idxs = np.where(y_train_all.ravel() == digit)[0][:2]
        for row_i in range(2):
            ax = axes[row_i, digit]
            ax.imshow(X_train_all[idxs[row_i]].reshape(28, 28), cmap="gray_r")
            ax.axis("off")
            if row_i == 0:
                ax.set_title(f"Cyfra {digit}", fontsize=10, fontweight="bold")
    fig.suptitle("Przykładowe obrazy cyfr (28×28 pikseli) ze zbioru uczącego MNIST", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_mnist_samples.png", dpi=170)
    plt.close(fig)

    # --- Fig 2: Wyniki strojenia na N=10000 (Architektura oraz Aktywacja/Strata) ---
    arch_rows = [r for r in rows if r["study_name"] == "arch_10k"]
    act_rows = [r for r in rows if r["study_name"] == "act_loss_10k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    labels_arch = [r["hidden_layers"] if r["hidden_layers"] != "none" else "Brak (784-10)" for r in arch_rows]
    val_accs_arch = [float(r["val_acc"]) * 100 for r in arch_rows]
    times_arch = [float(r["time_sec"]) for r in arch_rows]

    x_pos = np.arange(len(labels_arch))
    bars1 = ax1.bar(x_pos, val_accs_arch, width=0.58, color="#2b5c8f", edgecolor="black", alpha=0.88)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels_arch, rotation=22, ha="right", fontsize=9)
    ax1.set_ylim(85, 98.5)
    ax1.set_ylabel("Dokładność walidacyjna [%]")
    ax1.set_title("(a) Wpływ architektury warstw ukrytych (N = 10 000)", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, acc, t_s in zip(bars1, val_accs_arch, times_arch):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{acc:.2f}%\n({t_s:.1f}s)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    labels_act = [f"{r['hidden_activation']} + {r['loss'].upper()}" for r in act_rows]
    val_accs_act = [float(r["val_acc"]) * 100 for r in act_rows]
    colors_act = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    x_pos2 = np.arange(len(labels_act))
    bars2 = ax2.bar(x_pos2, val_accs_act, width=0.55, color=colors_act[: len(labels_act)], edgecolor="black", alpha=0.88)
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(labels_act, rotation=15, ha="right", fontsize=9.5)
    ax2.set_ylim(75, 98.5)
    ax2.set_ylabel("Dokładność walidacyjna [%]")
    ax2.set_title("(b) Aktywacja i funkcja straty dla [128, 64] (N = 10 000)", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars2, val_accs_act):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_mnist_n10000_tuning.png", dpi=170)
    plt.close(fig)

    # --- Trening najlepszego modelu na N=60000 (z zapisem krzywych i predykcji testowych) ---
    best_spec = build_mnist_spec(
        study_name="final_best",
        hidden_layers=[256, 128],
        hidden_activation="relu",
        loss="cce",
        lr=0.01,
        batch_size=64,
        epochs=20,
        max_train_samples=60000,
        max_test_samples=10000,
        seed=42,
    )
    cfg = best_spec["config"]
    classes = np.arange(10)
    y_train_enc = _encode_classification_labels(y_train_all, classes, 10)
    y_test_enc = _encode_classification_labels(y_test_all, classes, 10)

    X_tr, y_tr, X_val, y_val = train_val_split(
        X_train_all, y_train_enc, val_frac=0.1, rng=make_rng(42)
    )
    scaler = Standardizer()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test_all)

    model = build_model(cfg)
    optimizer = SGD(lr=0.01, momentum=0.9)
    history = model.fit(
        X_tr_s,
        y_tr,
        epochs=20,
        batch_size=64,
        optimizer=optimizer,
        X_val=X_val_s,
        y_val=y_val,
        record_weights=False,
    )
    y_test_probs = model.predict(X_test_s)
    y_test_pred_cls = np.argmax(y_test_probs, axis=1)
    y_test_true_cls = y_test_all.ravel().astype(int)

    # --- Fig 3: Porównanie N=10000 vs N=60000 oraz krzywa uczenia najlepszego modelu ---
    rows_10k = [r for r in rows if r["study_name"] == "best_compare_10k"]
    rows_60k = [r for r in rows if r["study_name"] == "best_compare_60k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    cfg_labels = [f"[{r['hidden_layers']}]\n({r['hidden_activation']})" for r in rows_10k]
    acc_10k = [float(r["test_acc"]) * 100 for r in rows_10k]
    acc_60k = [float(r["test_acc"]) * 100 for r in rows_60k]

    x = np.arange(len(cfg_labels))
    w = 0.35
    b1 = ax1.bar(x - w / 2, acc_10k, width=w, label="Podzbiór N = 10 000", color="#6baed6", edgecolor="black")
    b2 = ax1.bar(x + w / 2, acc_60k, width=w, label="Pełny zbiór N = 60 000", color="#08519c", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfg_labels, fontsize=9.5)
    ax1.set_ylim(93.0, 99.2)
    ax1.set_ylabel("Dokładność na zbiorze testowym [%]")
    ax1.set_title("(a) Skuteczność najlepszych modeli: N = 10 000 vs N = 60 000", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12, f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12, f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    epochs_arr = np.arange(1, len(history["train_loss"]) + 1)
    ax2.plot(epochs_arr, history["train_loss"], marker="o", markersize=4, linewidth=2, label="Strata treningowa (Train CCE)", color="#1f77b4")
    ax2.plot(epochs_arr, history["val_loss"], marker="s", markersize=4, linewidth=2, label="Strata walidacyjna (Val CCE)", color="#d62728")
    best_ep = int(np.argmin(history["val_loss"]) + 1)
    best_vl = float(np.min(history["val_loss"]))
    ax2.axvline(best_ep, color="gray", linestyle="--", alpha=0.7, label=f"Najlepsza epoka walidacyjna ({best_ep}, loss={best_vl:.4f})")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Entropia krzyżowa (CCE)")
    ax2.set_title("(b) Przebieg uczenia sieci 784-256-128-10 (N = 60 000)", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_mnist_n10000_vs_n60000.png", dpi=170)
    plt.close(fig)

    # --- Fig 4: Macierz pomyłek 10x10 na zbiorze testowym ---
    conf_mat = np.zeros((10, 10), dtype=int)
    for t_c, p_c in zip(y_test_true_cls, y_test_pred_cls):
        conf_mat[t_c, p_c] += 1

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    im = ax.imshow(conf_mat, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_yticklabels([str(i) for i in range(10)])
    ax.set_xlabel("Klasa przewidziana przez sieć (Predicted)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Prawdziwa klasa (True)", fontsize=10.5, fontweight="bold")
    overall_acc = np.mean(y_test_true_cls == y_test_pred_cls) * 100
    ax.set_title(
        f"Macierz pomyłek na zbiorze testowym MNIST (10 000 próbek, Accuracy = {overall_acc:.2f}%)",
        fontsize=11.5,
        fontweight="bold",
    )

    thresh = conf_mat.max() / 2.0
    for i in range(10):
        for j in range(10):
            val = conf_mat[i, j]
            if val > 0:
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=8.5,
                    fontweight="bold" if i == j else "normal",
                )

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_mnist_confusion_matrix.png", dpi=170)
    plt.close(fig)

    # --- Fig 5: Przykłady poprawnych i błędnych predykcji ---
    correct_idx = np.where(y_test_true_cls == y_test_pred_cls)[0]
    wrong_idx = np.where(y_test_true_cls != y_test_pred_cls)[0]

    # Wybieramy najciekawsze błędne predykcje (o najwyższej pewności błędnej klasy)
    wrong_conf = np.max(y_test_probs[wrong_idx], axis=1)
    top_wrong = wrong_idx[np.argsort(-wrong_conf)[:8]]
    sample_correct = correct_idx[:8]

    fig, axes = plt.subplots(2, 8, figsize=(13.5, 4.2))
    for col_i, idx in enumerate(sample_correct):
        ax = axes[0, col_i]
        ax.imshow(X_test_all[idx].reshape(28, 28), cmap="gray_r")
        ax.axis("off")
        conf = y_test_probs[idx, y_test_pred_cls[idx]] * 100
        ax.set_title(
            f"Prawda: {y_test_true_cls[idx]}\nPred: {y_test_pred_cls[idx]} ({conf:.1f}%)",
            fontsize=8.5,
            color="#1b7837",
            fontweight="bold",
        )

    for col_i, idx in enumerate(top_wrong):
        ax = axes[1, col_i]
        ax.imshow(X_test_all[idx].reshape(28, 28), cmap="gray_r")
        ax.axis("off")
        conf = y_test_probs[idx, y_test_pred_cls[idx]] * 100
        ax.set_title(
            f"Prawda: {y_test_true_cls[idx]}\nPred: {y_test_pred_cls[idx]} ({conf:.1f}%)",
            fontsize=8.5,
            color="#d73027",
            fontweight="bold",
        )

    fig.suptitle(
        "Przykłady predykcji na zbiorze testowym: poprawne (górny rząd) oraz błędne (dolny rząd)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_mnist_predictions.png", dpi=170)
    plt.close(fig)
    print(f"Wszystkie wykresy do raportu 2 zapisano w: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eksperymenty i wykresy dla zbioru MNIST (Część 2).")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "results" / "mnist_experiments.csv",
        help="Ścieżka do pliku wynikowego CSV.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=ROOT_DIR / "plots" / "report2",
        help="Katalog zapisu wykresów do raportu 2.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Wygeneruj tylko wykresy na podstawie istniejącego pliku CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.plots_only:
        specs = generate_mnist_specs(seed=42)
        run_mnist_experiments(specs, args.output)
    generate_report2_plots(args.output, args.plots_dir)

