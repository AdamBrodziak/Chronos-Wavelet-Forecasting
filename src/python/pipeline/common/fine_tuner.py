"""
LoRA Fine-tuning modelu Chronos 2.

Obsługuje:
- Fine-tuning z LoRA (Low-Rank Adaptation)
- Fine-tuning pełny (full)
- Grid search po hiperparametrach
- Zapis i ładowanie dostrojonych modeli
"""

import numpy as np
import torch
from pathlib import Path
from chronos import Chronos2Pipeline

from .config import (
    DEFAULT_LORA_CONFIG,
    DEFAULT_LORA_TRAINING,
    LORA_GRID,
    MODELS_DIR,
)
from .evaluator import compute_mae


# =============================================================================
# Fine-tuning
# =============================================================================

def fine_tune_lora(
    pipeline: Chronos2Pipeline,
    train_data: np.ndarray,
    prediction_length: int,
    val_data: np.ndarray = None,
    lora_config: dict = None,
    learning_rate: float = None,
    num_steps: int = None,
    batch_size: int = None,
    output_dir: str | Path = None,
) -> Chronos2Pipeline:
    """
    Fine-tuning modelu Chronos 2 za pomocą LoRA.

    Args:
        pipeline: Bazowy (pretrained) Chronos2Pipeline
        train_data: 1D numpy array z danymi treningowymi
        prediction_length: Horyzont predykcji dla fine-tuningu
        val_data: (opcjonalnie) Dane walidacyjne
        lora_config: Konfiguracja LoRA (r, lora_alpha, target_modules)
        learning_rate: Learning rate (domyślnie z config)
        num_steps: Liczba kroków treningowych (domyślnie z config)
        batch_size: Batch size (domyślnie z config)
        output_dir: Katalog na pliki tymczasowe i wagi modelu

    Returns:
        Nowy Chronos2Pipeline z dostrojonymi wagami LoRA.
    """
    # Ustawienia domyślne z config
    if lora_config is None:
        lora_config = DEFAULT_LORA_CONFIG.copy()
    if learning_rate is None:
        learning_rate = DEFAULT_LORA_TRAINING["learning_rate"]
    if num_steps is None:
        num_steps = DEFAULT_LORA_TRAINING["num_steps"]
    if batch_size is None:
        batch_size = DEFAULT_LORA_TRAINING["batch_size"]

    # Przygotuj dane treningowe w formacie Chronos 2
    train_inputs = _prepare_fit_inputs(train_data)
    val_inputs = _prepare_fit_inputs(val_data) if val_data is not None else None

    print(f"[fine_tuner] Start LoRA fine-tuning:")
    print(f"  - train_data: {len(train_data)} punktów")
    print(f"  - prediction_length: {prediction_length}")
    print(f"  - lora_config: {lora_config}")
    print(f"  - learning_rate: {learning_rate}")
    print(f"  - num_steps: {num_steps}")
    print(f"  - batch_size: {batch_size}")
    if output_dir:
        print(f"  - output_dir: {output_dir}")

    # Wywołanie fit() z Chronos 2 API
    fit_kwargs = dict(
        inputs=train_inputs,
        prediction_length=prediction_length,
        finetune_mode="lora",
        lora_config=lora_config,
        learning_rate=learning_rate,
        num_steps=num_steps,
        batch_size=batch_size,
    )
    if val_inputs is not None:
        fit_kwargs["validation_inputs"] = val_inputs
    if output_dir is not None:
        fit_kwargs["output_dir"] = str(output_dir)

    finetuned_pipeline = pipeline.fit(**fit_kwargs)

    print("[fine_tuner] LoRA fine-tuning zakończony.")
    return finetuned_pipeline


def fine_tune_full(
    pipeline: Chronos2Pipeline,
    train_data: np.ndarray,
    prediction_length: int,
    val_data: np.ndarray = None,
    learning_rate: float = 1e-6,
    num_steps: int = 1000,
    batch_size: int = 256,
    output_dir: str | Path = None,
) -> Chronos2Pipeline:
    """
    Pełny fine-tuning modelu Chronos 2 (aktualizacja wszystkich wag).

    ⚠️ Wymaga znacznie więcej VRAM niż LoRA!

    Args:
        pipeline: Bazowy Chronos2Pipeline
        train_data: Dane treningowe
        prediction_length: Horyzont predykcji
        val_data: Dane walidacyjne (opcjonalnie)
        learning_rate: Learning rate (niższy niż dla LoRA)
        num_steps: Liczba kroków
        batch_size: Batch size
        output_dir: Katalog do zapisu modelu (opcjonalnie)

    Returns:
        Fine-tuned Chronos2Pipeline.
    """
    train_inputs = _prepare_fit_inputs(train_data)
    val_inputs = _prepare_fit_inputs(val_data) if val_data is not None else None

    print(f"[fine_tuner] Start FULL fine-tuning ({len(train_data)} punktów, "
          f"lr={learning_rate}, steps={num_steps})...")

    fit_kwargs = dict(
        inputs=train_inputs,
        prediction_length=prediction_length,
        finetune_mode="full",
        learning_rate=learning_rate,
        num_steps=num_steps,
        batch_size=batch_size,
    )
    if val_inputs is not None:
        fit_kwargs["validation_inputs"] = val_inputs
    if output_dir is not None:
        fit_kwargs["output_dir"] = Path(output_dir)

    finetuned_pipeline = pipeline.fit(**fit_kwargs)

    print("[fine_tuner] Full fine-tuning zakończony.")
    return finetuned_pipeline


# =============================================================================
# Grid Search
# =============================================================================

def grid_search_lora(
    pipeline: Chronos2Pipeline,
    train_data: np.ndarray,
    val_data: np.ndarray,
    prediction_length: int,
    param_grid: dict = None,
) -> dict:
    """
    Grid search po hiperparametrach LoRA.

    Dla każdej kombinacji hiperparametrów:
    1. Fine-tune z LoRA
    2. Predykcja na val_data
    3. Obliczenie MAE
    4. Zapamiętanie najlepszego wyniku

    Args:
        pipeline: Bazowy Chronos2Pipeline (używany jako punkt startowy)
        train_data: Dane treningowe
        val_data: Dane walidacyjne
        prediction_length: Horyzont predykcji
        param_grid: Siatka hiperparametrów (domyślnie LORA_GRID z config)

    Returns:
        dict z kluczami:
            "best_params": dict — najlepsze hiperparametry
            "best_mae": float — najniższy MAE
            "best_pipeline": Chronos2Pipeline — najlepszy model
            "all_results": list[dict] — wyniki wszystkich kombinacji
    """
    if param_grid is None:
        param_grid = LORA_GRID

    from .data_loader import prepare_context_df
    from .model_manager import predict
    from itertools import product

    # Generuj wszystkie kombinacje
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    print(f"[fine_tuner] Grid search: {len(combinations)} kombinacji")

    best_mae = float("inf")
    best_params = None
    best_pipeline = None
    all_results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        print(f"\n[fine_tuner] Kombinacja {i+1}/{len(combinations)}: {params}")

        # Wyodrębnij parametry LoRA vs parametry treningowe
        lora_config = {}
        training_params = {}
        for key, value in params.items():
            if key in ("r", "lora_alpha", "target_modules"):
                lora_config[key] = value
            else:
                training_params[key] = value

        try:
            # Fine-tune
            ft_pipeline = fine_tune_lora(
                pipeline=pipeline,
                train_data=train_data,
                prediction_length=prediction_length,
                val_data=val_data,
                lora_config=lora_config if lora_config else None,
                **training_params,
            )

            # Predykcja na danych walidacyjnych
            context_df = prepare_context_df(train_data)
            y_pred = predict(ft_pipeline, context_df, prediction_length)

            # Obcięcie do długości val_data (jeśli różne)
            eval_len = min(len(y_pred), len(val_data))
            mae = compute_mae(val_data[:eval_len], y_pred[:eval_len])

            result = {"params": params, "mae": mae, "success": True}
            all_results.append(result)

            print(f"  MAE = {mae:.6f}")

            if mae < best_mae:
                best_mae = mae
                best_params = params
                best_pipeline = ft_pipeline

        except Exception as e:
            print(f"  BŁĄD: {e}")
            all_results.append({"params": params, "mae": None, "success": False, "error": str(e)})

    print(f"\n[fine_tuner] Grid search zakończony.")
    print(f"  Najlepsze parametry: {best_params}")
    print(f"  Najlepsze MAE: {best_mae:.6f}")

    return {
        "best_params": best_params,
        "best_mae": best_mae,
        "best_pipeline": best_pipeline,
        "all_results": all_results,
    }


# =============================================================================
# Zapis modelu
# =============================================================================

def save_finetuned_model(
    pipeline: Chronos2Pipeline,
    model_name: str,
    output_dir: str | Path = None,
):
    """
    Zapisuje fine-tuned model na dysk.

    Args:
        pipeline: Fine-tuned Chronos2Pipeline
        model_name: Nazwa modelu (np. "chronos2_lora_r8_lr1e-4")
        output_dir: Katalog bazowy (domyślnie MODELS_DIR z config)
    """
    if output_dir is None:
        output_dir = MODELS_DIR

    save_path = Path(output_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"[fine_tuner] Zapisuję model do: {save_path}")
    pipeline.model.save_pretrained(save_path)
    if hasattr(pipeline, "tokenizer") and pipeline.tokenizer is not None:
        pipeline.tokenizer.save_pretrained(save_path)
    print(f"[fine_tuner] Model zapisany.")


# =============================================================================
# Helpers
# =============================================================================

def _prepare_fit_inputs(data: np.ndarray) -> list:
    """
    Konwertuje numpy array na format wejściowy fit().

    Chronos 2 fit() oczekuje listy tensorów (1, T) lub dict z 'target'.

    Args:
        data: 1D numpy array

    Returns:
        Lista tensorów gotowa do fit().
    """
    if data is None:
        return None
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0) #TODO sprawdzić czy nie powinno byc torch.bfloat16
    return [tensor]
