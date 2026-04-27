"""
Pipeline: Haar-in-LoRA — Predykcja W dziedzinie wavelet z LoRA fine-tuningiem per pasmo.

Logika:
  1. Dekompozycja Haar sygnału na współczynniki (np. A5, D5..D1)
  2. Dla każdego WYBRANEGO poziomu:
     a) Wyciągnij wektor współczynników (krótszy niż oryginał!)
     b) Fine-tune Chronos 2 z LoRA na tych współczynnikach (osobny model per pasmo!)
     c) Oblicz dynamiczny horyzont: wavelet_h = ceil(time_h / 2^depth)
     d) Rolling window predykcja NA WSPÓŁCZYNNIKACH
  3. Złóż predykowane współczynniki przez IDWT → sygnał w dziedzinie czasu
     (współczynniki poziomów NIEpredykowanych są ZEROWANE)
  4. Ewaluacja na oryginalnych danych testowych

Kluczowe różnice vs Haar-in (zero-shot):
  Haar-in:      jeden pretrained model Chronos 2 per pasmo
  Haar-in-LoRA: osobny LoRA fine-tuned model per pasmo (A5 ma inny model niż D2!)

Kluczowe różnice vs Haar-after-LoRA:
  Haar-after-LoRA: LoRA + predykcja w dziedzinie czasu (po rekonstrukcji)
  Haar-in-LoRA:    LoRA + predykcja NA WSPÓŁCZYNNIKACH wavelet
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from common.config import HORIZONS, VARIANT_HAAR_IN_LORA, MODELS_DIR, DEFAULT_LORA_SPLIT_RATIO
from common.data_loader import matlab_to_numpy, split_train_test
from common.model_manager import get_pipeline, load_finetuned_pipeline
from common.fine_tuner import fine_tune_lora, save_finetuned_model
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
# Helpers (identyczne jak w run_haar_in.py)
# ---------------------------------------------------------------------------

def _required_decomp_level(selected_levels: list[str]) -> int:
    """Oblicz minimalny wymagany poziom dekompozycji na podstawie wybranych nazw.

    Np. ["A5", "D2"] → 5,  ["A2"] → 2,  ["A3", "D1"] → 3
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
    variant_name: str = VARIANT_HAAR_IN_LORA,
    model_save_name: str = "chronos2_haar_in_lora",
    force_retrain: bool = False,
    lora_config: dict | None = None,
    learning_rate: float | None = None,
    num_steps: int | None = None,
) -> list[dict]:
    """Pipeline Haar-in-LoRA — predykcja w dziedzinie wavelet z LoRA per pasmo.

    Dla każdego wybranego poziomu Haara tworzy OSOBNY model LoRA,
    fine-tunowany na współczynnikach wavelet tego poziomu.

    Args:
        y_train: Dane treningowe (oryginalne, 1-D).
        y_test:  Dane testowe (oryginalne, 1-D).
        selected_levels: Poziomy do predykcji, np. ``["A5"]``,
            ``["A5", "D2"]``, ``["A3", "D1"]``.
            ``None`` → wszystkie poziomy z dekompozycji.
        max_decomposition_level: Maksymalny poziom dekompozycji Haar.
        horizons: Lista horyzontów czasowych (domyślnie z config).
        variant_name: Nazwa wariantu (do zapisu wyników).
        model_save_name: Bazowa nazwa modelu do zapisu.
        force_retrain: Wymuszenie ponownego trenowania nawet gdy model istnieje.
        lora_config: Konfiguracja LoRA (r, lora_alpha, target_modules).
        learning_rate: Learning rate dla LoRA fine-tuningu.
        num_steps: Liczba kroków treningowych LoRA.

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
        print(f"  {lvl}: depth={depth}, len(coeffs)={n_coeffs}")

    # ----- 2. LoRA fine-tuning per pasmo ----------------------------------
    # Każde pasmo dostaje swój własny model fine-tunowany na swoich
    # współczynnikach wavelet (krótszych niż oryginalny sygnał).
    level_pipelines: dict[str, object] = {}

    for level_name in selected_levels:
        depth = get_level_depth(level_name)
        full_coeffs = decomp[level_name]
        coeffs_train, _ = _split_coeffs(full_coeffs, len(y_train), depth)

        # Unikalna nazwa modelu per pasmo
        level_model_name = f"{model_save_name}_{level_name}"
        model_path = MODELS_DIR / level_model_name

        if model_path.exists() and not force_retrain:
            print(f"\n[{variant_name}] {level_name}: Wczytywanie modelu z {model_path}")
            level_pipelines[level_name] = load_finetuned_pipeline(model_path)
        else:
            print(f"\n[{variant_name}] {level_name}: LoRA fine-tuning "
                  f"(train_coeffs={len(coeffs_train)}, depth={depth})")

            base_pipeline = get_pipeline()

            # Split walidacyjny NA WSPÓŁCZYNNIKACH
            train_ft, val_ft = split_train_test(coeffs_train, test_ratio=DEFAULT_LORA_SPLIT_RATIO)

            # prediction_length = max horyzont w dziedzinie wavelet
            max_wavelet_h = compute_wavelet_horizon(max(horizons), depth)

            finetuned = fine_tune_lora(
                pipeline=base_pipeline,
                train_data=train_ft,
                prediction_length=max_wavelet_h,
                val_data=val_ft,
                lora_config=lora_config,
                learning_rate=learning_rate,
                num_steps=num_steps,
                output_dir=model_path,
            )
            save_finetuned_model(finetuned, level_model_name)
            level_pipelines[level_name] = finetuned

    # ----- 3. Per horyzont: predykcja na współczynnikach → IDWT -----------
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

            full_coeffs = decomp[level_name]
            coeffs_train, coeffs_test = _split_coeffs(
                full_coeffs, len(y_train), depth,
            )

            print(f"  -> {level_name}: depth={depth}, "
                  f"wavelet_h={wavelet_h}, "
                  f"train_coeffs={len(coeffs_train)}, "
                  f"test_coeffs={len(coeffs_test)}")

            # Rolling window NA WSPÓŁCZYNNIKACH z LoRA fine-tuned modelem
            coeffs_pred = rolling_window_predict(
                pipeline=level_pipelines[level_name],
                y_train=coeffs_train,
                y_test=coeffs_test,
                step_length=wavelet_h,
            )

            predicted_coeffs[level_name] = coeffs_pred

        # ----- 4. IDWT — złóż predykowane współczynniki ------------------
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
