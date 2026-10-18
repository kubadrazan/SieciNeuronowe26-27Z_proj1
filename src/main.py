from pathlib import Path
import numpy as np

from config_loader import load_config, print_config


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.json"
    config = load_config(config_path)

    seed = config["seed"]
    np.random.seed(seed)

    # Parametry sieci neuronowej
    layers = config["network"]["layers"]
    activations = config["network"]["activations"]
    use_bias = config["network"]["use_bias"]

    # Parametry uczenia
    loss = config["training"]["loss"]
    learning_rate = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]

    print_config(config)
