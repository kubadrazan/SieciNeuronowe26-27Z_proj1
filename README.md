# Projekt 1: Klasyfikacja i regresja z uzyciem perceptronu

Projekt realizowany w ramach przedmiotu **Sieci Neuronowe** na Wydziale Mini / Politechnice Warszawskiej.

---

## Opis Projektu

Celem projektu jest zaimplementowanie od podstaw wielowarstwowego perceptronu (MLP) wraz z algorytmem propagacji wstecznej bledu (z uzyciem SGD i jego wariantow) oraz przeprowadzenie badan eksperymentalnych nad jego skutecznoscia.

Projekt sklada sie z dwoch czesci:

1. **Czesc I (15 pkt):**
   - Implementacja sieci MLP w jezyku Python z uzyciem wylacznie biblioteki `numpy` (brak dedykowanych bibliotek ML/DL).
   - Rozwiazywanie zadan:
     - **Regresji** dla funkcji R -> R,
     - **Klasyfikacji** wektorow w R^2.
   - Generowanie wizualizacji:
     - Wynikow uczenia i granic decyzyjnych / krzywej regresji,
     - Przebiegu bledu w kolejnych iteracjach/epokach,
     - Ewolucji wartosci wag w czasie uczenia.
   - Przeprowadzenie analizy wplywu architektury, funkcji aktywacji i miar bledu.

2. **Czesc II (5 pkt):**
   - Zastosowanie autorskiej implementacji MLP do rozpoznawania pisanych recznie cyfr z zestawu **MNIST**.
   - Optymalizacja wydajnosciowa oraz dobor hiperparametrow sieci.
   - Ocena jakosci nauczonej sieci na osobnym zbiorze testowym.

---

## Wymagania i Ograniczenia Implementacyjne

- **Jezyk i biblioteki:** Python 3.x, `numpy` (do obliczen macierzowych), `matplotlib` / `seaborn` (do wizualizacji).
- **Ograniczenie:** **Brak mozliwosci korzystania z bibliotek implementujacych sieci neuronowe** (np. PyTorch, TensorFlow, scikit-learn).
- **Powtarzalnosc:** Mozliwosc zainicjowania procesu uczenia z zadanym ziarnem generatora liczb pseudolosowych (`seed`).
- **Elastyczna konfiguracja:** Latwa zmiana liczby warstw ukrytych (0-4), liczby neuronow w warstwach, obecnosci biasu, funkcji aktywacji oraz miar bledu.

---

## Struktura Projektu

```text
.
+-- data/                  # Zbiory danych (regresja, klasyfikacja R2, MNIST)
+-- src/                   # Kod zrodlowy
|   +-- __init__.py
|   +-- mlp.py             # Implementacja wielowarstwowego perceptronu i backpropagation
|   +-- activations.py     # Funkcje aktywacji (sigmoid, tanh, ReLU, itp.) i ich pochodne
|   +-- losses.py          # Funkcje bledu (MSE, BCE, CCE) i ich pochodne
|   +-- utils.py           # Wczytywanie danych, przetwarzanie, ziarno losowosci
|   `-- visualization.py   # Modul do rysowania granic decyzyjnych, bledu i wag
+-- config/                # Pliki konfiguracyjne / skrypty z hiperparametrami
+-- plots/                 # Wygenerowane wykresy i wizualizacje do raportu
+-- main_regression.py     # Skrypt uruchamiajacy eksperymenty dla regresji
+-- main_classification.py # Skrypt uruchamiajacy eksperymenty dla klasyfikacji R2
+-- main_mnist.py          # Skrypt uruchamiajacy uczenie na zbiorze MNIST
+-- README.md              # Dokumentacja projektu
`-- requirements.txt       # Wymagane biblioteki (numpy, matplotlib)
```

---

## Zakres Badan Eksperymentalnych

Raport z projektu obejmuje analize nastepujacych zagadnien:
1. **Wplyw funkcji aktywacji:** Porownanie funkcji sigmoidalnej oraz dwoch innych funkcji (np. `tanh`, `ReLU`) w warstwach ukrytych oraz dobor wlasciwej funkcji na wyjsciu.
2. **Wplyw architektury sieci:** Badanie liczby warstw ukrytych (od 0 do 4) oraz liczby neuronow w poszczegolnych warstwach.
3. **Wplyw miary bledu:** Porownanie dwoch miar bledu dla klasyfikacji (np. MSE vs Cross-Entropy) oraz dwoch dla regresji.
4. **Powtarzalnosc i stabilnosc:** Analiza wplywu losowosci wag poczatkowych na proces uczenia.

---

## Uruchomienie i Uzycie

### 1. Instalacja zaleznosci
```bash
pip install -r requirements.txt
```

### 2. Przyklady uruchomienia

- **Eksperymenty dla regresji (R -> R):**
  ```bash
  python main_regression.py --seed 42 --layers 2 8 1 --activation sigmoid --loss mse --epochs 1000
  ```

- **Eksperymenty dla klasyfikacji (R^2):**
  ```bash
  python main_classification.py --seed 42 --layers 2 16 2 --activation sigmoid --loss bce --epochs 1000
  ```

- **Uczenie na zbiorze MNIST:**
  ```bash
  python main_mnist.py --seed 42 --epochs 50 --batch-size 64
  ```

---

## Zespol i Informacje Organizacyjne

- **Sklad zespolu:** [Imie i Nazwisko 1], [Imie i Nazwisko 2]
- **Przedmiot:** Sieci Neuronowe
- **Prowadzacy:** [Imie i Nazwisko Prowadzacego]
- **Terminy oddania:**
  - Oddanie czesci I: **27.10.2026**
  - Oddanie czesci II: **03.11.2026**
