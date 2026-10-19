import json
from pathlib import Path

from activations import AVAILABLE_ACTIVATIONS

AVAILABLE_LOSSES: tuple[str, ...] = ("mse", "mae", "cce", "bce")


def validate_config(config: dict) -> None:
    net = config.get("network", {})
    train = config.get("training", {})

    layers = net.get("layers", [])
    activations = net.get("activations", [])

    if not isinstance(layers, list) or len(layers) < 2:
        raise ValueError(
            f"Sieć musi mieć co najmniej 2 warstwy (wejściową i wyjściową), podano: {layers}"
        )

    if any(not isinstance(n, int) or n <= 0 for n in layers):
        raise ValueError(
            f"Liczba neuronów w każdej warstwie musi być dodatnią liczbą całkowitą, podano: {layers}"
        )

    # Warstwa wejściowa nie ma funkcji aktywacji, więc potrzebujemy len(layers) - 1 aktywacji
    expected_activations = len(layers) - 1
    if not isinstance(activations, list) or len(activations) != expected_activations:
        raise ValueError(
            f"Liczba funkcji aktywacji ({len(activations)}) musi odpowiadać liczbie warstw "
            f"ukrytych i wyjściowej (len(layers) - 1 = {expected_activations})."
        )

    for act in activations:
        if not isinstance(act, str) or act.strip().lower() not in AVAILABLE_ACTIVATIONS:
            raise ValueError(
                f"Nieznana funkcja aktywacji '{act}'. "
                f"Dozwolone opcje: {list(AVAILABLE_ACTIVATIONS)}"
            )

    if not isinstance(net.get("use_bias"), bool):
        raise ValueError("Parametr 'use_bias' musi być wartością logiczną (true/false).")

    loss = train.get("loss")
    if not isinstance(loss, str) or loss.strip().lower() not in AVAILABLE_LOSSES:
        raise ValueError(
            f"Nieznana funkcja straty '{loss}'. "
            f"Dozwolone opcje: {list(AVAILABLE_LOSSES)}"
        )

    if train.get("learning_rate", 0) <= 0:
        raise ValueError("Parametr 'learning_rate' musi być większy od 0.")

    if not isinstance(train.get("epochs"), int) or train["epochs"] <= 0:
        raise ValueError("Parametr 'epochs' musi być dodatnią liczbą całkowitą.")

    if not isinstance(train.get("batch_size"), int) or train["batch_size"] <= 0:
        raise ValueError("Parametr 'batch_size' musi być dodatnią liczbą całkowitą.")


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    validate_config(config)
    return config


def print_config(config: dict) -> None:
    net = config["network"]
    train = config["training"]

    print(f"Wczytano konfigurację (seed={config['seed']}):")
    print(f"  Sieć: warstwy={net['layers']}, aktywacje={net['activations']}, bias={net['use_bias']}")
    print(
        f"  Trening: loss={train['loss']}, lr={train['learning_rate']}, "
        f"epoki={train['epochs']}, batch_size={train['batch_size']}"
    )

