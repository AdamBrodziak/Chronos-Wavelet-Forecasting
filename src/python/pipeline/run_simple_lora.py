"""
Pipeline: Simple-LoRA — Chronos 2 + LoRA fine-tuning na oryginalnych danych.

Flow (z przebieg-danych):
  wczytanie pliku → diff → normalizacja → Chronos 2
  → czy model jest już dostrojony?
    NIE → fine-tuning LoRA → załaduj dostrojony model
    TAK → załaduj dostrojony model
  → dla każdego horyzontu: rolling window → zapisz predykcje
  → obliczenie metryk → zapisanie wyników → koniec
"""

import numpy as np
from pathlib import Path

from common.config import HORIZONS, VARIANT_SIMPLE_LORA, MODELS_DIR
from common.data_loader import (
    matlab_to_numpy, split_train_test, normalize,
)
from common.model_manager import get_pipeline, load_finetuned_pipeline
from common.fine_tuner import fine_tune_lora, save_finetuned_model
from common.rolling_window import rolling_window_predict
from common.evaluator import evaluate_all, print_metrics
from common.results_io import save_predictions, save_metrics, save_all_metrics_summary


def run(
    y_train: np.ndarray,
    y_test: np.ndarray,
    horizons: list[int] = None,
    variant_name: str = VARIANT_SIMPLE_LORA,
    model_save_name: str = "chronos2_simple_lora",
    force_retrain: bool = False,
    lora_config: dict = None,
    learning_rate: float = None,
    num_steps: int = None,
):
    """
    Uruchomienie pipeline Simple-LoRA.

    Args:
        y_train: Dane treningowe (po diff + normalizacji)
        y_test: Dane testowe
        horizons: Lista horyzontów
        variant_name: Nazwa wariantu
        model_save_name: Nazwa modelu do zapisu/ładowania
        force_retrain: Czy wymusić ponowny fine-tuning
        lora_config: Konfiguracja LoRA (opcjonalnie)
        learning_rate: Learning rate (opcjonalnie)
        num_steps: Liczba kroków (opcjonalnie)

    Returns:
        Lista dict-ów z wynikami per horyzont.
    """
    if horizons is None:
        horizons = HORIZONS

    # 1. Sprawdź czy model jest już dostrojony
    model_path = MODELS_DIR / model_save_name
    
    if model_path.exists() and not force_retrain:
        print(f"[{variant_name}] Ładowanie istniejącego modelu: {model_path}")
        pipeline = load_finetuned_pipeline(model_path)
    else:
        print(f"[{variant_name}] Fine-tuning LoRA...")
        base_pipeline = get_pipeline()

        # Podział danych treningowych na train_FT i val_FT
        train_ft, val_ft = split_train_test(y_train, test_ratio=0.15)

        pipeline = fine_tune_lora(
            pipeline=base_pipeline,
            train_data=train_ft,
            prediction_length=max(horizons),
            val_data=val_ft,
            lora_config=lora_config,
            learning_rate=learning_rate,
            num_steps=num_steps,
        )

        # Zapisz model
        save_finetuned_model(pipeline, model_save_name)

    # 2. Rolling window dla każdego horyzontu
    all_results = []

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"[{variant_name}] Horyzont: {horizon}")
        print(f"{'='*60}")

        predictions = rolling_window_predict(
            pipeline=pipeline,
            y_train=y_train,
            y_test=y_test,
            step_length=horizon,
        )

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


def run_from_matlab(y_train_raw, y_test_raw, step_length):
    """Punkt wejścia z MATLAB-a."""
    y_train = matlab_to_numpy(y_train_raw)
    y_test = matlab_to_numpy(y_test_raw)

    results = run(y_train, y_test, horizons=[int(step_length)])

    # Zwróć predykcje dla jedynego horyzontu
    return [float(x) for x in results[0]["predictions"]]
