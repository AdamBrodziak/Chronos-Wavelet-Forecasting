"""
Fine-tuning — osobny workflow dostrajania modelu Chronos 2.

Z przebieg-danych (flow Fine-tuning):
  podział danych na treningowe i walidacyjne (tren_FT + val_FT)
  -> dopasowanie (fit) na tren_FT -> predykcja na val_FT
  -> obliczenie MAE na val_FT -> zapisz MAE
  -> czy MAE jest zadowalające?
    NIE -> zmiana hiperparametrów (grid-search?) -> fit
    TAK -> zapisz ulepszony model -> koniec

Ten workflow jest NIEZALEŻNY od pipeline'ów predykcyjnych.
Służy do znalezienia najlepszych hiperparametrów LoRA,
a następnie wyprodukowania modelu gotowego do użycia.
"""

import numpy as np
from pathlib import Path

from common.config import MODELS_DIR
from common.data_loader import load_mat_data, split_train_test, normalize
from common.model_manager import get_pipeline
from common.fine_tuner import (
    fine_tune_lora,
    grid_search_lora,
    save_finetuned_model,
)
from common.evaluator import compute_mae


def run_manual(
    data: np.ndarray,
    prediction_length: int = 96,
    test_ratio: float = 0.2,
    model_name: str = "chronos2_lora_manual",
    lora_config: dict = None,
    learning_rate: float = None,
    num_steps: int = None,
    mae_threshold: float = None,
):
    """
    Ręczny fine-tuning z jednym zestawem hiperparametrów.

    Args:
        data: Pełne dane (train + test zostaną podzielone)
        prediction_length: Horyzont predykcji
        test_ratio: Stosunek danych walidacyjnych
        model_name: Nazwa modelu do zapisu
        lora_config: Konfiguracja LoRA
        learning_rate: Learning rate
        num_steps: Liczba kroków
        mae_threshold: Próg MAE — jeśli poniżej, model jest "zadowalający"

    Returns:
        dict z wynikami fine-tuningu.
    """
    # Podział
    train_data, val_data = split_train_test(data, test_ratio=test_ratio)

    print(f"[fine_tuning] Train: {len(train_data)}, Val: {len(val_data)}")

    # Załaduj bazowy model
    base_pipeline = get_pipeline()

    # Fine-tune
    ft_pipeline = fine_tune_lora(
        pipeline=base_pipeline,
        train_data=train_data,
        prediction_length=prediction_length,
        val_data=val_data,
        lora_config=lora_config,
        learning_rate=learning_rate,
        num_steps=num_steps,
    )

    # Ewaluacja na val_data
    from common.data_loader import prepare_context_df
    from common.model_manager import predict

    context_df = prepare_context_df(train_data)
    y_pred = predict(ft_pipeline, context_df, prediction_length)

    eval_len = min(len(y_pred), len(val_data))
    mae = compute_mae(val_data[:eval_len], y_pred[:eval_len])

    print(f"[fine_tuning] MAE na val: {mae:.6f}")

    # Czy zadowalające?
    is_satisfactory = True
    if mae_threshold is not None:
        is_satisfactory = mae < mae_threshold
        status = "TAK" if is_satisfactory else "NIE"
        print(f"[fine_tuning] MAE < {mae_threshold}? {status}")

    # Zapisz model
    if is_satisfactory:
        save_finetuned_model(ft_pipeline, model_name)

    return {
        "mae": mae,
        "is_satisfactory": is_satisfactory,
        "pipeline": ft_pipeline,
    }


def run_grid_search(
    data: np.ndarray,
    prediction_length: int = 96,
    test_ratio: float = 0.2,
    model_name: str = "chronos2_lora_best",
    param_grid: dict = None,
):
    """
    Grid search po hiperparametrach LoRA.

    Automatycznie przeszukuje przestrzeń hiperparametrów,
    wybiera najlepszy model i zapisuje go.

    Args:
        data: Pełne dane
        prediction_length: Horyzont predykcji
        test_ratio: Stosunek danych walidacyjnych
        model_name: Nazwa najlepszego modelu do zapisu
        param_grid: Siatka hiperparametrów (domyślnie z config)

    Returns:
        dict z wynikami grid search.
    """
    train_data, val_data = split_train_test(data, test_ratio=test_ratio)
    base_pipeline = get_pipeline()

    results = grid_search_lora(
        pipeline=base_pipeline,
        train_data=train_data,
        val_data=val_data,
        prediction_length=prediction_length,
        param_grid=param_grid,
    )

    # Zapisz najlepszy model
    if results["best_pipeline"] is not None:
        save_finetuned_model(results["best_pipeline"], model_name)

    return results


if __name__ == "__main__":
    from common.config import DATA_DIR

    print("=== Fine-tuning Workflow ===")

    # Wczytaj dane
    data = load_mat_data(DATA_DIR / "ab_diff_zestaw.mat", "ab_diff")
    data_norm, mean, std = normalize(data)

    # Opcja 1: Ręczny fine-tuning
    result = run_manual(
        data_norm,
        prediction_length=96,
        mae_threshold=0.05,
    )

    # Opcja 2: Grid search (odkomentuj)
    # result = run_grid_search(data_norm, prediction_length=96)

    print("\n=== KONIEC ===")
