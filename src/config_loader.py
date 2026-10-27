import json
from pathlib import Path

from activations import AVAILABLE_ACTIVATIONS
from losses import AVAILABLE_LOSSES


def validate_config(config: dict) -> None:
    net = config.get("network", {})
    train = config.get("training", {})
    data = config.get("data", {})

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
        raise ValueError(
            "Parametr 'use_bias' musi być wartością logiczną (true/false)."
        )

    loss = train.get("loss")
    if not isinstance(loss, str) or loss.strip().lower() not in AVAILABLE_LOSSES:
        raise ValueError(
            f"Nieznana funkcja straty '{loss}'. "
            f"Dozwolone opcje: {list(AVAILABLE_LOSSES)}"
        )
    loss = loss.strip().lower()
    output_activation = activations[-1].strip().lower()
    task = data.get("task")
    if loss == "bce" and (task != "classification" or output_activation != "sigmoid"):
        raise ValueError(
            "BCE wymaga zadania klasyfikacji i sigmoidalnej funkcji aktywacji wyjścia."
        )
    if loss == "cce" and (task != "classification" or output_activation != "softmax"):
        raise ValueError(
            "CCE wymaga zadania klasyfikacji i funkcji aktywacji softmax na wyjściu."
        )
    if task == "classification" and output_activation not in ("sigmoid", "softmax"):
        raise ValueError(
            "Klasyfikacja wymaga sigmoidalnej lub softmaxowej funkcji aktywacji wyjścia."
        )

    if train.get("learning_rate", 0) <= 0:
        raise ValueError("Parametr 'learning_rate' musi być większy od 0.")

    if "momentum" in train:
        momentum = train["momentum"]
        if not isinstance(momentum, (int, float)) or not (0.0 <= momentum < 1.0):
            raise ValueError(
                f"Parametr 'momentum' musi być liczbą z zakresu [0, 1), otrzymano: {momentum}"
            )

    if not isinstance(train.get("epochs"), int) or train["epochs"] <= 0:
        raise ValueError("Parametr 'epochs' musi być dodatnią liczbą całkowitą.")

    if not isinstance(train.get("batch_size"), int) or train["batch_size"] <= 0:
        raise ValueError("Parametr 'batch_size' musi być dodatnią liczbą całkowitą.")
    validation_fraction = train.get("validation_fraction", 0.2)
    if (
        not isinstance(validation_fraction, (int, float))
        or not 0.0 < validation_fraction < 1.0
    ):
        raise ValueError(
            "Parametr 'validation_fraction' musi być liczbą z zakresu (0, 1)."
        )

    # Walidacja danych
    if not isinstance(data, dict) or not data:
        raise ValueError(
            "Brak wymaganej sekcji 'data' lub jest ona pusta w konfiguracji."
        )

    required_data_keys = ["task", "train_file", "test_file"]
    for key in required_data_keys:
        if key not in data:
            raise ValueError(f"Brak wymaganego klucza '{key}' w sekcji 'data'.")

    if data["task"] not in ["regression", "classification"]:
        raise ValueError(
            f"Nieobsługiwane zadanie w 'data.task': {data['task']}. Dozwolone: 'regression', 'classification'."
        )


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    validate_config(config)
    return config


def print_config(config: dict) -> None:
    net = config["network"]
    train = config["training"]
    data = config["data"]

    print(f"Wczytano konfigurację (seed={config['seed']}):")
    print(
        f"  Sieć: warstwy={net['layers']}, aktywacje={net['activations']}, bias={net['use_bias']}"
    )
    print(
        f"  Trening: loss={train['loss']}, lr={train['learning_rate']}, "
        f"momentum={train.get('momentum', 0.0)}, epoki={train['epochs']}, batch_size={train['batch_size']}"
    )
    print(
        f"  Dane: task={data['task']}, train={data['train_file']}, test={data['test_file']}"
    )
