"""
Pipeline: Haar-in — Predykcja W dziedzinie wavelet, potem IDWT.

Logika:
  1. Dekompozycja Haar sygnału treningowego na współczynniki (np. A5, D5..D1)
  2. Dla każdego WYBRANEGO poziomu:
     a) Wyciągnij wektor współczynników (krótszy niż oryginał!)
     b) Oblicz dynamiczny horyzont: wavelet_h = ceil(time_h / 2^depth)
     c) Rolling window predykcja NA WSPÓŁCZYNNIKACH (nie w dziedzinie czasu)
  3. Złóż predykowane współczynniki przez IDWT -> sygnał w dziedzinie czasu
     (współczynniki poziomów NIEpredykowanych są ZEROWANE)
  4. Ewaluacja na oryginalnych danych testowych

Kluczowe różnice vs Haar-after:
  Haar-after: rekonstruuje wybrany poziom do dziedziny czasu, potem predykcja
  Haar-in:    predykcja NA WSPÓŁCZYNNIKACH wavelet, potem IDWT do czasu

Elastyczność:
  - Dowolny podzbiór poziomów: ["A5"], ["A5", "D2"], ["A3", "D1"], itp.
  - Poziom dekompozycji ustawiany dynamicznie (max_decomposition_level)
  - Horyzont predykcji przeliczany automatycznie per poziom
"""

from __future__ import annotations

import numpy as np

from common.config import HORIZONS, VARIANT_HAAR_IN
from common.data_loader import matlab_to_numpy, split_train_test
from common.model_manager import get_pipeline
from common.rolling_window import rolling_window_predict
from common.evaluator import evaluate_all, print_metrics
from common.results_io import save_predictions, save_metrics, save_all_metrics_summary
from common.haar_utils import (
    haar_decompose,
    get_level_names,
    get_level_depth,
    compute_wavelet_horizon,
    reconstruct_from_predicted_coeffs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _required_decomp_level(selected_levels: list[str]) -> int:
    """Oblicz minimalny wymagany poziom dekompozycji na podstawie wybranych nazw.

    Np. ["A5", "D2"] -> 5,  ["A2"] -> 2,  ["A3", "D1"] -> 3
    """
    return max(get_level_depth(name) for name in selected_levels)


def _split_coeffs(
    coeffs: np.ndarray,
    n_train_samples: int,
    level_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Podziel wektor współczynników na część train / test.

    Dekompozycja Haar na całym sygnale (train+test) daje współczynniki
    o długości ~ len(signal) / 2^depth.  Próbki treningowe odpowiadają
    pierwszym ~ n_train_samples / 2^depth współczynnikom.

    Returns:
        (coeffs_train, coeffs_test)
    """
    import math
    split_idx = math.ceil(n_train_samples / (2 ** level_depth))
    # Zapewnij, że split nie wykracza poza zakres
    split_idx = min(split_idx, len(coeffs))
    return coeffs[:split_idx], coeffs[split_idx:]


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def run(
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_levels: list[str] | None = None,
    max_decomposition_level: int = 5,
    horizons: list[int] | None = None,
    variant_name: str = VARIANT_HAAR_IN,
) -> list[dict]:
    """Pipeline Haar-in — predykcja w dziedzinie współczynników wavelet.

    Args:
        y_train: Dane treningowe (oryginalne, 1-D).
        y_test:  Dane testowe (oryginalne, 1-D).
        selected_levels: Poziomy do predykcji, np. ``["A5"]``,
            ``["A5", "D2"]``, ``["A3", "D1"]``.
            ``None`` -> wszystkie poziomy z dekompozycji.
        max_decomposition_level: Maksymalny poziom dekompozycji Haar.
        horizons: Lista horyzontów czasowych (domyślnie z config).
        variant_name: Nazwa wariantu (do zapisu wyników).

    Returns:
        Lista słowników z wynikami per horyzont.
    """
    if horizons is None:
        horizons = HORIZONS

    # ----- 1. Dekompozycja ------------------------------------------------
    full_data = np.concatenate([y_train, y_test])
    decomp = haar_decompose(full_data, max_level=max_decomposition_level)
    actual_level = decomp["max_level"]

    if selected_levels is None:
        selected_levels = get_level_names(decomp)

    # Walidacja: czy wymagany poziom mieści się w faktycznym
    required = _required_decomp_level(selected_levels)
    if required > actual_level:
        raise ValueError(
            f"Wybrany poziom {required} przekracza max_level={actual_level} "
            f"(dostępne: {get_level_names(decomp)})"
        )

    print(f"[{variant_name}] Dekompozycja Haar (level={actual_level})")
    print(f"[{variant_name}] Poziomy do predykcji: {selected_levels}")
    for lvl in selected_levels:
        depth = get_level_depth(lvl)
        n_coeffs = len(decomp[lvl])
        print(f"  {lvl}: depth={depth}, "
              f"len(coeffs)={n_coeffs}")

    # ----- 2. Model -------------------------------------------------------
    pipeline = get_pipeline()

    # ----- 3. Per horyzont -------------------------------------------------
    all_results: list[dict] = []

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"[{variant_name}] Horyzont czasowy: {horizon}")
        print(f"{'='*60}")

        # Słownik predykowanych współczynników
        predicted_coeffs: dict[str, np.ndarray] = {}

        for level_name in selected_levels:
            depth = get_level_depth(level_name)
            wavelet_h = compute_wavelet_horizon(horizon, depth)

            # Pełny wektor współczynników dla tego poziomu
            full_coeffs = decomp[level_name]

            # Podziel na train/test w dziedzinie współczynników
            coeffs_train, coeffs_test = _split_coeffs(
                full_coeffs, len(y_train), depth, #TODO czy dane y_train jest w takiej samej dziedzinie co coeffs_train ? Mogą być inne długości
            )

            print(f"  -> {level_name}: depth={depth}, "
                  f"wavelet_h={wavelet_h}, "
                  f"train_coeffs={len(coeffs_train)}, "
                  f"test_coeffs={len(coeffs_test)}")

            # Rolling window NA WSPÓŁCZYNNIKACH
            coeffs_pred = rolling_window_predict(
                pipeline=pipeline,
                y_train=coeffs_train,
                y_test=coeffs_test,
                step_length=wavelet_h,
            )

            predicted_coeffs[level_name] = coeffs_pred

        # ----- 4. IDWT — złóż predykowane współczynniki ----------------
        # Potrzebujemy zrekonstruować tylko fragment odpowiadający y_test.
        # Budujemy "sztuczną" dekompozycję o kształtach identycznych jak
        # oryginalna, ale z predykowanymi współczynnikami w części testowej
        # i oryginalnymi w części treningowej.
        # Ponieważ IDWT wymaga spójnych długości, rekonstruujemy cały
        # sygnał, a potem bierzemy ostatnie len(y_test) próbek.

        recon_coeffs: dict[str, np.ndarray] = {}
        for level_name in selected_levels:
            depth = get_level_depth(level_name)
            full_coeffs = decomp[level_name]
            coeffs_train_part, _ = _split_coeffs(
                full_coeffs, len(y_train), depth,
            )
            # Sklejamy: oryginalne train + predykowane test
            recon_coeffs[level_name] = np.concatenate([
                coeffs_train_part,
                predicted_coeffs[level_name],
            ])

        # IDWT z zerami na niepredykowanych poziomach
        reconstructed = reconstruct_from_predicted_coeffs(
            decomposition=decomp,
            predicted_coeffs=recon_coeffs,
            original_length=len(full_data),
        )

        # Fragment testowy
        combined_predictions = reconstructed[len(y_train):]

        # Wyrównaj długość (zaokrąglenia w podziale współczynników)
        if len(combined_predictions) < len(y_test):
            combined_predictions = np.pad(
                combined_predictions,
                (0, len(y_test) - len(combined_predictions)),
            )
        combined_predictions = combined_predictions[:len(y_test)]

        # ----- 5. Ewaluacja -----------------------------------------------
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
