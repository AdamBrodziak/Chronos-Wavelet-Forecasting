"""
Pipeline: Haar-in-LoRA — Dekompozycja -> LoRA per poziom -> predykcja per poziom -> złożenie.

Z kombinacje-mozliwosci:
  dane wejściowe -> dekompozycja dyskretna Haar
  -> wybranie poziomów
  -> lora fine-tuning Chronos 2 na rozłożonych danych DLA KAŻDEGO MODELU
  -> predykcja wybranych poziomów (Chronos 2, osobny model per poziom)
  -> złożenie predykcji
  -> ewaluacja

Najbardziej złożony wariant — osobny model LoRA dla każdego poziomu Haara.
"""
#TODO POPRAWIĆ TO! OBECNIE JEST TAKIE SAME JAK HAAR-AFTER-LORA!

import numpy as np
from pathlib import Path

from common.config import HORIZONS, VARIANT_HAAR_IN_LORA, MODELS_DIR
from common.data_loader import matlab_to_numpy, split_train_test
from common.model_manager import get_pipeline, load_finetuned_pipeline
from common.fine_tuner import fine_tune_lora, save_finetuned_model
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
    variant_name: str = VARIANT_HAAR_IN_LORA,
    model_save_prefix: str = "chronos2_haar_in_lora",
    force_retrain: bool = False,
    lora_config: dict = None,
    learning_rate: float = None,
    num_steps: int = None,
):
    """
    Pipeline Haar-in-LoRA.

    Osobny model LoRA jest trenowany na sygnale z każdego poziomu Haara.
    Następnie każdy model predykuje swój poziom, a wyniki są sumowane.
    """
    if horizons is None:
        horizons = HORIZONS

    # 1. Dekompozycja
    full_data = np.concatenate([y_train, y_test])
    decomp = haar_decompose(full_data, max_level=max_decomposition_level)

    if selected_levels is None:
        selected_levels = get_level_names(decomp)

    print(f"[{variant_name}] Poziomy: {selected_levels}")

    # 2. Fine-tune osobny model LoRA per poziom
    level_pipelines = {}

    for level_name in selected_levels:
        model_name = f"{model_save_prefix}_{level_name}"
        model_path = MODELS_DIR / model_name

        level_signal = reconstruct_single_level(decomp, level_name)
        level_train = level_signal[:len(y_train)]

        if model_path.exists() and not force_retrain:
            print(f"[{variant_name}] Ładowanie modelu {level_name}: {model_path}")
            level_pipelines[level_name] = load_finetuned_pipeline(model_path)
        else:
            print(f"[{variant_name}] Fine-tuning LoRA dla poziomu {level_name}...")
            base_pipeline = get_pipeline()
            train_ft, val_ft = split_train_test(level_train, test_ratio=0.15)

            ft_pipeline = fine_tune_lora(
                pipeline=base_pipeline,
                train_data=train_ft,
                prediction_length=max(horizons),
                val_data=val_ft,
                lora_config=lora_config,
                learning_rate=learning_rate,
                num_steps=num_steps,
            )
            save_finetuned_model(ft_pipeline, model_name)
            level_pipelines[level_name] = ft_pipeline

    # 3. Per horyzont: predykcja per poziom -> suma
    all_results = []

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"[{variant_name}] Horyzont: {horizon}")
        print(f"{'='*60}")

        combined_predictions = np.zeros(len(y_test))

        for level_name in selected_levels:
            print(f"  -> Predykcja poziomu: {level_name}")

            level_signal = reconstruct_single_level(decomp, level_name)
            level_train = level_signal[:len(y_train)]
            level_test = level_signal[len(y_train):]

            level_predictions = rolling_window_predict(
                pipeline=level_pipelines[level_name],
                y_train=level_train,
                y_test=level_test,
                step_length=horizon,
            )

            combined_predictions += level_predictions

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
