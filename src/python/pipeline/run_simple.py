"""
Pipeline: Simple — Chronos 2 zero-shot na oryginalnych danych.

Najprostszy wariant. Brak fine-tuningu, brak dekompozycji Haar.

Flow (z przebieg-danych):
  wczytanie pliku → diff → normalizacja → Chronos 2
  → dla każdego horyzontu: rolling window → zapisz predykcje
  → obliczenie metryk → zapisanie wyników → koniec
"""

import numpy as np

from common.config import HORIZONS, VARIANT_SIMPLE
from common.data_loader import (
    load_mat_data, diff_series, normalize, denormalize,
    matlab_to_numpy, split_train_test,
)
from common.model_manager import get_pipeline
from common.rolling_window import rolling_window_predict
from common.evaluator import evaluate_all, print_metrics
from common.results_io import save_predictions, save_metrics, save_all_metrics_summary


def run(
    y_train: np.ndarray,
    y_test: np.ndarray,
    horizons: list[int] = None,
    variant_name: str = VARIANT_SIMPLE,
):
    """
    Uruchomienie pipeline Simple.

    Args:
        y_train: Dane treningowe (już po diff + normalizacji)
        y_test: Dane testowe
        horizons: Lista horyzontów [1, 16, 96] (domyślnie z config)
        variant_name: Nazwa wariantu (do zapisu wyników)

    Returns:
        Lista dict-ów z wynikami per horyzont.
    """
    if horizons is None:
        horizons = HORIZONS

    # 1. Załaduj model
    pipeline = get_pipeline()

    # 2. Dla każdego horyzontu: rolling window → ewaluacja → zapis
    all_results = []

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"[{variant_name}] Horyzont: {horizon}")
        print(f"{'='*60}")

        # Rolling window prediction
        predictions = rolling_window_predict(
            pipeline=pipeline,
            y_train=y_train,
            y_test=y_test,
            step_length=horizon,
        )

        # Ewaluacja
        metrics = evaluate_all(y_test, predictions)
        print_metrics(metrics, variant_name, horizon)

        # Zapis
        save_predictions(predictions, y_test, horizon, variant_name)
        save_metrics(metrics, horizon, variant_name)

        all_results.append({
            "horizon": horizon,
            "metrics": metrics,
            "predictions": predictions,
        })

    # Podsumowanie
    save_all_metrics_summary(all_results, variant_name)

    return all_results


# =============================================================================
# Wywołanie standalone (z linii poleceń lub MATLAB)
# =============================================================================

def run_from_matlab(y_train_raw, y_test_raw, step_length):
    """
    Punkt wejścia wywoływany z MATLAB-a.

    Zachowuje kompatybilność z istniejącym interfejsem MATLAB→Python.
    Używa jednego horyzontu (step_length) zamiast pętli po horyzontach.

    Args:
        y_train_raw: Dane treningowe z MATLABa (lista/array)
        y_test_raw: Dane testowe z MATLABa (lista/array)
        step_length: Horyzont predykcji (int)

    Returns:
        Lista floatów — predykcje.
    """
    y_train = matlab_to_numpy(y_train_raw)
    y_test = matlab_to_numpy(y_test_raw)

    pipeline = get_pipeline()
    predictions = rolling_window_predict(
        pipeline=pipeline,
        y_train=y_train,
        y_test=y_test,
        step_length=int(step_length),
    )

    return [float(x) for x in predictions]


if __name__ == "__main__":
    # Przykład użycia standalone
    from common.data_loader import load_mat_data
    from common.config import DATA_DIR

    print("=== Pipeline: Simple ===")
    # Wczytaj dane
    data = load_mat_data(DATA_DIR / "ab_diff_zestaw.mat", "ab_diff")
    y_train, y_test = split_train_test(data, test_ratio=0.2)

    # Normalizacja
    y_train_norm, mean, std = normalize(y_train)
    y_test_norm = (y_test - mean) / std

    results = run(y_train_norm, y_test_norm)
    print("\n=== KONIEC ===")
