# Projekt 1: Wielowarstwowy Perceptron (MLP) – Klasyfikacja i Regresja

Implementacja sieci neuronowej MLP od podstaw (wyłącznie `numpy` i `matplotlib`) wraz z algorytmem propagacji wstecznej, optymalizatorem SGD z momentum oraz silnikiem badań eksperymentalnych.

---

## 1. Instalacja zależności

```bash
pip install -r requirements.txt
```

---

## 2. Sposoby wywołania

### A. Pojedynczy trening sieci (`src/main.py`)

Domyślnie skrypt wczytuje konfigurację z pliku `config/config.json`, przeprowadza trening i zapisuje wykresy w katalogu `plots/<nazwa-konfiguracji>/`.

```bash
# Uruchomienie z domyślnym plikiem config/config.json
python src/main.py

# Uruchomienie z własnym plikiem konfiguracyjnym
python src/main.py --config config/config.json

# Uruchomienie ze wskazaniem własnego katalogu na wykresy
python src/main.py --config config/config.json --plot-dir plots/moj_eksperyment

# Uruchomienie bez generowania wykresów
python src/main.py --no-plots
```

### B. Silnik badań eksperymentalnych (`src/experiments.py`)

Skrypt automatycznie uruchamia zestaw eksperymentów (wpływ funkcji aktywacji, liczby warstw ukrytych `0–4` oraz funkcji straty) i na bieżąco zapisuje wyniki do pliku CSV.

```bash
# Uruchomienie pełnego zestawu 150 eksperymentów (domyślnie size=10000, epochs=200)
python src/experiments.py

# Uruchomienie wszystkich badań na mniejszych zbiorach (N=500) i zapis do osobnego pliku
python src/experiments.py --size 500 --output results/experiments_results_500.csv

# Uruchomienie wyłącznie badania funkcji aktywacji
python src/experiments.py --study activations

# Uruchomienie wyłącznie badania architektury na wybranych zbiorach
python src/experiments.py --study architecture --datasets three_gauss cube --size 1000 --epochs 150

# Uruchomienie badania funkcji straty dla kilku ziaren losowości 
python src/experiments.py --study losses --seeds 42 --output results/losses_multi_seed.csv
```

**Dostępne parametry `src/experiments.py`:**
- `--study`: `all` (domyślnie), `activations`, `architecture`, `losses`
- `--datasets`: `simple`, `three_gauss`, `activation`, `cube` (domyślnie wszystkie)
- `--size`: `100`, `500`, `1000`, `10000` (domyślnie `10000`)
- `--epochs`: liczba epok (domyślnie `200`)
- `--seeds`: jedno lub więcej ziaren generatora liczb losowych (domyślnie `42`)
- `--output`: ścieżka zapisu pliku wynikowego CSV (domyślnie `results/experiments_results.csv`)

---

## 3. Konfiguracja modelu (`config/config.json`)

Przykład konfiguracji (`config/config.json`):

```json
{
  "seed": 42,
  "network": {
    "layers": [2, 16, 8, 1],
    "activations": ["sigmoid", "sigmoid", "sigmoid"],
    "use_bias": true
  },
  "training": {
    "loss": "bce",
    "learning_rate": 0.05,
    "momentum": 0.9,
    "epochs": 100,
    "batch_size": 32
  },
  "data": {
    "task": "classification",
    "train_file": "data/classification/data.simple.train.500.csv",
    "test_file": "data/classification/data.simple.test.500.csv"
  }
}
```

**Dozwolone wartości w konfiguracji:**
- `network.layers`: lista liczb neuronów w kolejnych warstwach `[wejście, ukryta_1, ..., wyjście]`. Liczba elementów w `activations` musi wynosić `len(layers) - 1`.
- `network.activations`: `"linear"`, `"sigmoid"`, `"tanh"`, `"relu"`, `"softmax"`.
- `training.loss`:
  - Klasyfikacja binarna (1 neuron wyjściowy + `"sigmoid"`): `"bce"` lub `"mse"`
  - Klasyfikacja wieloklasowa ($K$ neuronów wyjściowych + `"softmax"`): `"cce"` lub `"mse"`
  - Regresja (1 neuron wyjściowy + `"linear"`): `"mse"` lub `"mae"`
- `data.task`: `"classification"` lub `"regression"`.

### Uczenie i ocena na MNIST

Uruchamianie dla MNIST


```bash
python src/main.py --config config/mnist.json --no-plots
```


---

## 4. Struktura projektu

```text
.
├── config/
│   ├── config.json               # Domyślna konfiguracja pojedynczego uruchomienia
│   └── mnist.json                # Konfiguracja MNIST
├── data/
│   ├── classification/           # Zbiory klasyfikacyjne 
│   └── regression/               # Zbiory regresyjne 
├── plots/                        # Wygenerowane wykresy 
├── results/                      # Wyniki badań eksperymentalnych 
├── src/
│   ├── main.py                   # Punkt wejścia dla pojedynczego treningu i ewaluacji
│   ├── experiments.py            # Silnik badań eksperymentalnych
│   ├── generate_report_plots.py  # Generator wykresów porównawczych do raportu
│   ├── mlp.py                    # Implementacja warstw Dense i sieci MLP
│   ├── activations.py            # Funkcje aktywacji i ich pochodne
│   ├── losses.py                 # Funkcje straty (MSE, MAE, BCE, CCE)
│   ├── initializers.py           # Inicjalizacja wag (He, Xavier)
│   ├── optimizers.py             # Optymalizator SGD z momentum
│   ├── config_loader.py          # Walidacja i wczytywanie pliku JSON
│   ├── utils.py                  # Wczytywanie CSV, standaryzacja, podział train/val
│   └── visualization.py          # Rysowanie granic decyzyjnych, regresji, straty i wag
├── 
```
