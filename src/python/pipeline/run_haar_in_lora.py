"""
Pipeline: Haar-after-sum-LoRA — Dekompozycja -> złożenie -> LoRA fine-tuning -> predykcja.

Z kombinacje-mozliwosci:
  dane wejściowe -> dekompozycja dyskretna Haar
  -> wybranie poziomów -> złożenie wybranych sygnałów
  -> lora fine-tuning Chronos 2 dla każdego poziomu/modelu
  -> Chronos 2 -> predykcja
  -> różne horyzonty -> ewaluacja
"""

#TODO LOGIKA JAK DLA HAAR-AFTER-SUM, ŹLE! takie same jak HAAR-AFTER-LORA

import numpy as np
from pathlib import Path

from common.config import HORIZONS, VARIANT_HAAR_AFTER_LORA, MODELS_DIR
from common.data_loader import matlab_to_numpy, split_train_test
from common.model_manager import get_pipeline, load_finetuned_pipeline
from common.fine_tuner import fine_tune_lora, save_finetuned_model
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
    variant_name: str = VARIANT_HAAR_AFTER_LORA,
    model_save_name: str = "chronos2_haar_after_lora",
    force_retrain: bool = False,
    lora_config: dict = None,
    learning_rate: float = None,
    num_steps: int = None,
):
    """Pipeline Haar-after-LoRA."""
    if horizons is None:
        horizons = HORIZONS
    if selected_levels is None:
        selected_levels = ["A5"]

    levels_str = "_".join(selected_levels)
    full_model_name = f"{model_save_name}_{levels_str}"

    # 1. Dekompozycja + rekonstrukcja
    full_data = np.concatenate([y_train, y_test])
    decomp = haar_decompose(full_data, max_level=max_decomposition_level)
    reconstructed = haar_reconstruct(decomp, selected_levels)

    train_recon = reconstructed[:len(y_train)]
    test_recon = reconstructed[len(y_train):]

    print(f"[{variant_name}] Rekonstrukcja z {selected_levels}: OK")

    # 2. Fine-tuning LoRA na zrekonstruowanym sygnale
    model_path = MODELS_DIR / full_model_name

    if model_path.exists() and not force_retrain:
        print(f"[{variant_name}] Ładowanie modelu: {model_path}")
        pipeline = load_finetuned_pipeline(model_path)
    else:
        print(f"[{variant_name}] Fine-tuning LoRA na danych Haar...")
        base_pipeline = get_pipeline()
        train_ft, val_ft = split_train_test(train_recon, test_ratio=0.15)

        pipeline = fine_tune_lora(
            pipeline=base_pipeline,
            train_data=train_ft,
            prediction_length=max(horizons),
            val_data=val_ft,
            lora_config=lora_config,
            learning_rate=learning_rate,
            num_steps=num_steps,
        )
        save_finetuned_model(pipeline, full_model_name)

    # 3. Rolling window + ewaluacja
    all_results = []

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"[{variant_name}] Horyzont: {horizon}")
        print(f"{'='*60}")

        predictions = rolling_window_predict(
            pipeline=pipeline,
            y_train=train_recon,
            y_test=test_recon,
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
