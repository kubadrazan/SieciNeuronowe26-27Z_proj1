import argparse
import csv
from pathlib import Path

from main import run_from_config

DATASETS_META: dict[str, dict] = {
    "simple": {
        "task": "classification",
        "subdir": "classification",
        "prefix": "data.simple",
        "n_in": 2,
        "n_classes": 2,
    },
    "three_gauss": {
        "task": "classification",
        "subdir": "classification",
        "prefix": "data.three_gauss",
        "n_in": 2,
        "n_classes": 3,
    },
    "activation": {
        "task": "regression",
        "subdir": "regression",
        "prefix": "data.activation",
        "n_in": 1,
        "n_classes": 1,
    },
    "cube": {
        "task": "regression",
        "subdir": "regression",
        "prefix": "data.cube",
        "n_in": 1,
        "n_classes": 1,
    },
}

CSV_COLUMNS: list[str] = [
    "study_name",
    "experiment_id",
    "task",
    "dataset",
    "dataset_size",
    "seed",
    "num_hidden_layers",
    "hidden_layers",
    "layers_full",
    "total_params",
    "use_bias",
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
    "test_rmse",
    "test_mae",
    "test_rmse_inside",
    "test_rmse_outside",
    "time_sec",
]


def count_parameters(layers: list[int], use_bias: bool) -> int:
    total = 0
    for i in range(len(layers) - 1):
        n_in, n_out = layers[i], layers[i + 1]
        total += n_in * n_out
        if use_bias:
            total += n_out
    return total


def build_experiment_spec(
    study_name: str,
    dataset_name: str,
    size: int,
    hidden_layers: list[int],
    hidden_activation: str,
    loss: str,
    output_activation: str | None = None,
    n_out: int | None = None,
    use_bias: bool = True,
    learning_rate: float = 0.03,
    momentum: float = 0.9,
    epochs: int = 200,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    meta = DATASETS_META[dataset_name]
    task = meta["task"]
    n_in = meta["n_in"]

    if task == "regression":
        resolved_out_act = output_activation or "linear"
        resolved_n_out = n_out or 1
    else:
        if loss == "bce" or output_activation == "sigmoid":
            resolved_out_act = "sigmoid"
            resolved_n_out = n_out or (1 if meta["n_classes"] == 2 else meta["n_classes"])
        else:
            resolved_out_act = output_activation or "softmax"
            resolved_n_out = n_out or meta["n_classes"]

    layers = [n_in, *hidden_layers, resolved_n_out]
    activations = [hidden_activation] * len(hidden_layers) + [resolved_out_act]

    subdir = meta["subdir"]
    prefix = meta["prefix"]
    train_file = f"data/{subdir}/{prefix}.train.{size}.csv"
    test_file = f"data/{subdir}/{prefix}.test.{size}.csv"

    config = {
        "seed": seed,
        "network": {
            "layers": layers,
            "activations": activations,
            "use_bias": use_bias,
        },
        "training": {
            "loss": loss,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "data": {
            "task": task,
            "train_file": train_file,
            "test_file": test_file,
        },
    }

    return {
        "study_name": study_name,
        "dataset": dataset_name,
        "dataset_size": size,
        "hidden_layers": hidden_layers,
        "hidden_activation": hidden_activation if hidden_layers else "none",
        "output_activation": resolved_out_act,
        "config": config,
    }


def generate_activations_study(
    datasets: list[str],
    size: int,
    seeds: list[int],
    epochs: int,
) -> list[dict]:
    """2.1.4 (1): Wpływ funkcji aktywacji wewnątrz sieci (sigmoid, tanh, relu)."""
    specs: list[dict] = []
    hidden_activations = ["sigmoid", "tanh", "relu"]
    architectures = [
        [16],
        [16, 8],
        [32, 16, 8],
    ]

    for dataset_name in datasets:
        task = DATASETS_META[dataset_name]["task"]
        default_loss = "mse" if task == "regression" else "cce"

        for hidden_layers in architectures:
            for hidden_act in hidden_activations:
                for seed in seeds:
                    specs.append(
                        build_experiment_spec(
                            study_name="activations",
                            dataset_name=dataset_name,
                            size=size,
                            hidden_layers=hidden_layers,
                            hidden_activation=hidden_act,
                            loss=default_loss,
                            epochs=epochs,
                            seed=seed,
                        )
                    )
    return specs


def generate_architecture_study(
    datasets: list[str],
    size: int,
    seeds: list[int],
    epochs: int,
) -> list[dict]:
    """2.1.4 (2): Wpływ liczby warstw ukrytych (0 do 4) oraz ich liczności."""
    specs: list[dict] = []
    architectures: list[list[int]] = [
        [],                  
        [4],                 
        [16],                
        [64],                
        [8, 4],              
        [16, 8],             
        [32, 16],            
        [16, 8, 4],          
        [32, 16, 8],         
        [16, 16, 8, 4],      
        [32, 16, 16, 8],     
    ]
    hidden_activations = ["tanh", "relu"]

    for dataset_name in datasets:
        task = DATASETS_META[dataset_name]["task"]
        default_loss = "mse" if task == "regression" else "cce"

        for hidden_layers in architectures:
            acts_to_run = ["linear"] if len(hidden_layers) == 0 else hidden_activations
            for hidden_act in acts_to_run:
                for seed in seeds:
                    specs.append(
                        build_experiment_spec(
                            study_name="architecture",
                            dataset_name=dataset_name,
                            size=size,
                            hidden_layers=hidden_layers,
                            hidden_activation=hidden_act,
                            loss=default_loss,
                            epochs=epochs,
                            seed=seed,
                        )
                    )
    return specs


def generate_losses_study(
    datasets: list[str],
    size: int,
    seeds: list[int],
    epochs: int,
) -> list[dict]:
    """2.1.4 (3): Wpływ miary błędu na wyjściu sieci (2 dla regresji, 2 dla klasyfikacji)."""
    specs: list[dict] = []
    hidden_activations = ["sigmoid", "tanh", "relu"]
    hidden_layers = [16, 8]

    for dataset_name in datasets:
        meta = DATASETS_META[dataset_name]
        task = meta["task"]

        if task == "regression":
            # Regresja: MSE vs MAE przy wyjściu liniowym
            loss_variants = [
                ("mse", "linear", 1),
                ("mae", "linear", 1),
            ]
        else:
            # Klasyfikacja wieloklasowa (lub one-hot): CCE vs MSE przy wyjściu softmax
            loss_variants = [
                ("cce", "softmax", meta["n_classes"]),
                ("mse", "softmax", meta["n_classes"]),
            ]
            # Dla klasyfikacji binarnej dodatkowo porównanie BCE vs MSE z 1 neuronem sigmoidalnym
            if meta["n_classes"] == 2:
                loss_variants.extend(
                    [
                        ("bce", "sigmoid", 1),
                        ("mse", "sigmoid", 1),
                    ]
                )

        for hidden_act in hidden_activations:
            for loss_name, out_act, n_out in loss_variants:
                for seed in seeds:
                    specs.append(
                        build_experiment_spec(
                            study_name="losses",
                            dataset_name=dataset_name,
                            size=size,
                            hidden_layers=hidden_layers,
                            hidden_activation=hidden_act,
                            loss=loss_name,
                            output_activation=out_act,
                            n_out=n_out,
                            epochs=epochs,
                            seed=seed,
                        )
                    )
    return specs


def _format_float(val: float | None) -> str:
    if val is None:
        return ""
    return f"{val:.6f}"


def run_experiments(
    specs: list[dict],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total = len(specs)
    print(f"Rozpoczynanie {total} eksperymentów. Wyniki będą zapisywane do: {output_csv}\n")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        f.flush()

        for idx, spec in enumerate(specs, start=1):
            config = spec["config"]
            net = config["network"]
            train = config["training"]
            data = config["data"]

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
                "task": data["task"],
                "dataset": spec["dataset"],
                "dataset_size": spec["dataset_size"],
                "seed": config["seed"],
                "num_hidden_layers": len(hidden_layers),
                "hidden_layers": hidden_str,
                "layers_full": layers_full_str,
                "total_params": count_parameters(net["layers"], net["use_bias"]),
                "use_bias": net["use_bias"],
                "hidden_activation": spec["hidden_activation"],
                "output_activation": spec["output_activation"],
                "loss": train["loss"],
                "learning_rate": train["learning_rate"],
                "momentum": train.get("momentum", 0.0),
                "epochs": train["epochs"],
                "batch_size": train["batch_size"],
                "train_loss_final": _format_float(metrics["train_loss_final"]),
                "val_loss_final": _format_float(metrics["val_loss_final"]),
                "val_loss_best": _format_float(metrics["val_loss_best"]),
                "best_epoch": metrics["best_epoch"],
                "test_loss": _format_float(metrics["test_loss"]),
                "train_acc": _format_float(metrics["train_acc"]),
                "val_acc": _format_float(metrics["val_acc"]),
                "test_acc": _format_float(metrics["test_acc"]),
                "test_rmse": _format_float(metrics["test_rmse"]),
                "test_mae": _format_float(metrics["test_mae"]),
                "test_rmse_inside": _format_float(metrics["test_rmse_inside"]),
                "test_rmse_outside": _format_float(metrics["test_rmse_outside"]),
                "time_sec": _format_float(metrics["time_sec"]),
            }

            writer.writerow(row)
            f.flush()

            if data["task"] == "classification":
                score_info = f"test_acc={metrics['test_acc']:.4f}, test_loss={metrics['test_loss']:.4f}"
            else:
                score_info = f"test_rmse={metrics['test_rmse']:.4f}, test_mae={metrics['test_mae']:.4f}"

            print(
                f"[{idx:3d}/{total:3d}] study={spec['study_name']:<12} | "
                f"data={spec['dataset']:<11} | layers={layers_full_str:<13} | "
                f"act={spec['hidden_activation']:<7}->{spec['output_activation']:<7} | "
                f"loss={train['loss']:<3} | {score_info} ({metrics['time_sec']:.2f}s)"
            )

    print(f"\nZakończono wszystkie eksperymenty! Zapisano {total} wierszy w: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Silnik badań eksperymentalnych sieci MLP (Sekcja 2.1.4)."
    )
    parser.add_argument(
        "--study",
        choices=["all", "activations", "architecture", "losses"],
        default="all",
        help="Wybór badania do uruchomienia (domyślnie: all).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS_META.keys()),
        default=list(DATASETS_META.keys()),
        help="Zbiory danych do uwzględnienia w badaniu.",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[100, 500, 1000, 10000],
        default=500,
        help="Rozmiar zbioru danych (domyślnie: 500).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Lista ziaren losowości (domyślnie: 42).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Liczba epok uczenia dla każdego eksperymentu (domyślnie: 200).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "experiments_results.csv",
        help="Ścieżka do wyjściowego pliku CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_specs: list[dict] = []
    if args.study in ("all", "activations"):
        all_specs.extend(
            generate_activations_study(args.datasets, args.size, args.seeds, args.epochs)
        )
    if args.study in ("all", "architecture"):
        all_specs.extend(
            generate_architecture_study(args.datasets, args.size, args.seeds, args.epochs)
        )
    if args.study in ("all", "losses"):
        all_specs.extend(
            generate_losses_study(args.datasets, args.size, args.seeds, args.epochs)
        )

    run_experiments(all_specs, args.output)

