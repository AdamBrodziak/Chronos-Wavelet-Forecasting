"""
Pipeline: Haar-after-sum — Dekompozycja Haar → złożenie wybranych poziomów → predykcja.

Z kombinacje-mozliwosci:
  dane wejściowe -> dekompozycja dyskretna Haar
  -> wybranie poziomów do dekompozycji Haar'a
  -> złożenie jednego sygnału z wybranych poziomów
    (eliminacja innych poziomów)
  -> Chronos 2 -> predykcja
  -> różne horyzonty czasowe 1, 16, 96
  -> ewaluacja predykcji

  Kluczowa różnice:
  Haar-after:     najpierw składa wybrane poziomy, potem przewiduje osobno per poziom i sumuje
  Haar-in:        przewiduje OSOBNO per poziom, potem składa predykcje
  Haar-after-sum: składa wybrane poziomy w jeden sygnał i przewiduje raz (dla całości)
"""

import numpy as np

from common.config import HORIZONS, VARIANT_HAAR_AFTER_SUM
from common.data_loader import matlab_to_numpy, split_train_test, normalize
from common.model_manager import get_pipeline
from common.rolling_window import rolling_window_predict
from common.evaluator import evaluate_all, print_metrics
from common.results_io import save_predictions, save_metrics, save_all_metrics_summary
from common.haar_utils import haar_decompose, haar_reconstruct


def run(
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_levels: list[str] = None,
    max_decomposition_level: int = 5,
    horizons: list[int] = None,
    variant_name: str = VARIANT_HAAR_AFTER_SUM,
):
    """
    Pipeline Haar-after-sum.

    Najpierw dekompozycja + rekonstrukcja z wybranych poziomów,
    potem predykcja na zrekonstruowanym sygnale.

    Args:
        y_train: Dane treningowe (oryginalne, przed dekompozycją)
        y_test: Dane testowe
        selected_levels: Poziomy do zachowania, np. ["A5", "D1"]
                         Domyślnie: ["A5"] (tylko aproksymacja)
        max_decomposition_level: Poziom dekompozycji Haar
        horizons: Lista horyzontów
        variant_name: Nazwa wariantu

    Returns:
        Lista dict-ów z wynikami per horyzont.
    """
    if horizons is None:
        horizons = HORIZONS
    if selected_levels is None:
        # Domyślnie: wszystkie poziomy
        full_data = np.concatenate([y_train, y_test])
        from common.haar_utils import get_level_names
        decomp_tmp = haar_decompose(full_data, max_level=max_decomposition_level)
        selected_levels = get_level_names(decomp_tmp)
        del decomp_tmp

    print(f"[{variant_name}] Dekompozycja Haar (level={max_decomposition_level})...")
    print(f"[{variant_name}] Wybrane poziomy: {selected_levels}")

    # 1. Dekompozycja + rekonstrukcja z wybranych poziomów
    # Robimy to na CAŁYCH danych (train+test) razem, potem dzielimy
    full_data = np.concatenate([y_train, y_test])
    decomp = haar_decompose(full_data, max_level=max_decomposition_level)
    reconstructed = haar_reconstruct(decomp, selected_levels)

    # Podziel z powrotem na train/test
    train_reconstructed = reconstructed[:len(y_train)]
    test_reconstructed = reconstructed[len(y_train):]

    print(f"[{variant_name}] Rekonstrukcja OK. "
          f"Train: {len(train_reconstructed)}, Test: {len(test_reconstructed)}")

    # 2. Model
    pipeline = get_pipeline()

    # 3. Rolling window + ewaluacja per horyzont
    all_results = []

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"[{variant_name}] Horyzont: {horizon}")
        print(f"{'='*60}")

        predictions = rolling_window_predict(
            pipeline=pipeline,
            y_train=train_reconstructed,
            y_test=test_reconstructed,
            step_length=horizon,
        )

        # Ewaluacja na oryginalnych danych testowych (nie zrekonstruowanych!)
        metrics = evaluate_all(y_test, predictions)
        print_metrics(metrics, variant_name, horizon)

        save_predictions(predictions, y_test, horizon, variant_name)
        save_metrics(metrics, horizon, variant_name)

        all_results.append({
            "horizon": horizon,
            "metrics": metrics,
            "predictions": predictions,
        })

    save_all_metrics_summary(all_results, variant_name)
    return all_results