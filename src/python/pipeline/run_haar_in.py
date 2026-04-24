"""
Pipeline: Haar-in — Dekompozycja → osobna predykcja dla każdego poziomu → złożenie.

Z kombinacje-mozliwosci:
  dane wejściowe → dekompozycja dyskretna Haar
  → wybranie poziomów do dekompozycji Haar'a
  → predykcja wybranych poziomów (każdy osobno, Chronos 2)
  → złożenie predykcji
  → różne horyzonty → ewaluacja

Kluczowa różnica vs Haar-after:
  Haar-after: najpierw składa, potem predykuje
  Haar-in:    predykuje OSOBNO per poziom, potem składa
"""

import numpy as np

from common.config import HORIZONS, VARIANT_HAAR_IN
from common.data_loader import matlab_to_numpy, split_train_test
from common.model_manager import get_pipeline
from common.rolling_window import rolling_window_predict
from common.evaluator import evaluate_all, print_metrics
from common.results_io import save_predictions, save_metrics, save_all_metrics_summary
from common.haar_utils import haar_decompose, reconstruct_single_level, get_level_names


def run(
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_levels: list[str] = None,
    max_decomposition_level: int = 5,
    horizons: list[int] = None,
    variant_name: str = VARIANT_HAAR_IN,
):
    """
    Pipeline Haar-in — predykcja PODCZAS dekompozycji.

    Dla każdego wybranego poziomu Haara:
    1. Rekonstruuje sygnał w dziedzinie czasu z tego jednego poziomu
    2. Uruchamia rolling window predykcję na tym sygnale
    3. Sumuje predykcje ze wszystkich poziomów → finalna predykcja
    """
    if horizons is None:
        horizons = HORIZONS
    if selected_levels is None:
        # Domyślnie: wszystkie poziomy
        full_data = np.concatenate([y_train, y_test])
        decomp_tmp = haar_decompose(full_data, max_level=max_decomposition_level)
        selected_levels = get_level_names(decomp_tmp)
        del decomp_tmp

    print(f"[{variant_name}] Dekompozycja Haar (level={max_decomposition_level})")
    print(f"[{variant_name}] Poziomy do predykcji: {selected_levels}")

    # 1. Dekompozycja na pełnych danych
    full_data = np.concatenate([y_train, y_test])
    decomp = haar_decompose(full_data, max_level=max_decomposition_level)

    # 2. Model
    pipeline = get_pipeline()

    # 3. Per horyzont: predykcja per poziom → suma
    all_results = []

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"[{variant_name}] Horyzont: {horizon}")
        print(f"{'='*60}")

        # Sumujemy predykcje ze wszystkich wybranych poziomów
        combined_predictions = np.zeros(len(y_test))

        for level_name in selected_levels:
            print(f"  → Predykcja poziomu: {level_name}")

            # Rekonstrukcja jednego poziomu w dziedzinie czasu
            level_signal = reconstruct_single_level(decomp, level_name)

            # Podziel na train/test
            level_train = level_signal[:len(y_train)]
            level_test = level_signal[len(y_train):]

            # Rolling window na tym poziomie
            level_predictions = rolling_window_predict(
                pipeline=pipeline,
                y_train=level_train,
                y_test=level_test,
                step_length=horizon,
            )

            combined_predictions += level_predictions

        # Ewaluacja na oryginalnych danych
        metrics = evaluate_all(y_test, combined_predictions)
        print_metrics(metrics, variant_name, horizon)

        save_predictions(combined_predictions, y_test, horizon, variant_name)
        save_metrics(metrics, horizon, variant_name)

        all_results.append({
            "horizon": horizon,
            "metrics": metrics,
            "predictions": combined_predictions,
        })

    save_all_metrics_summary(all_results, variant_name)
    return all_results
