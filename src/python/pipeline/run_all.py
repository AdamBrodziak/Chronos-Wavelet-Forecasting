"""
Master runner — uruchamia wszystkie lub wybrane pipeline'y z jednego miejsca.

Użycie:
    python run_all.py                          # Wszystkie warianty
    python run_all.py Simple Simple-LoRA       # Tylko wybrane
"""

import sys
import numpy as np
from datetime import datetime

from common.config import (
    DATA_DIR,
    HORIZONS,
    VARIANT_SIMPLE,
    VARIANT_SIMPLE_LORA,
    VARIANT_HAAR_AFTER,
    VARIANT_HAAR_AFTER_LORA,
    VARIANT_HAAR_IN,
    VARIANT_HAAR_IN_LORA,
    MAT_VAR_NAME_TRAIN,
    MAT_VAR_NAME_TEST,
)
from common.data_loader import load_mat_data, split_train_test

# Import pipeline'ów
import run_simple
import run_simple_lora
import run_haar_after
import run_haar_after_lora
import run_haar_in
import run_haar_in_lora

# Mapowanie nazwa → moduł
PIPELINES = {
    VARIANT_SIMPLE: run_simple,
    VARIANT_SIMPLE_LORA: run_simple_lora,
    VARIANT_HAAR_AFTER: run_haar_after,
    VARIANT_HAAR_AFTER_LORA: run_haar_after_lora,
    VARIANT_HAAR_IN: run_haar_in,
    VARIANT_HAAR_IN_LORA: run_haar_in_lora,
}


def run_all(
    variants: list[str] = None,
    data_path: str = None,
    var_name_train: str = MAT_VAR_NAME_TRAIN,
    var_name_test: str = MAT_VAR_NAME_TEST,
    test_ratio: float = 0.2,
    horizons: list[int] = None,
    haar_levels: list[str] = None,
):
    """
    Uruchom pipeline'y sekwencyjnie.

    Args:
        variants: Lista wariantów do uruchomienia (None = wszystkie)
        data_path: Ścieżka do pliku .mat
        var_name: Nazwa zmiennej w .mat
        test_ratio: Podział train/test
        horizons: Horyzonty predykcji
        haar_levels: Poziomy Haar dla wariantów Haar-*
    """
    if variants is None:
        variants = list(PIPELINES.keys())
    if horizons is None:
        horizons = HORIZONS
    if haar_levels is None:
        haar_levels = ["A5"]
    if data_path is None:
        data_path = DATA_DIR / "ab_diff_zestaw.mat"

    # 1. Wczytaj dane
    print(f"[run_all] Wczytywanie danych: {data_path}")
    y_train, y_test = load_mat_data(data_path, var_name_train, var_name_test)

    print(f"[run_all] Dane: train={len(y_train)}, test={len(y_test)}")
    print(f"[run_all] Warianty: {variants}")
    print(f"[run_all] Horyzonty: {horizons}")

    # 2. Uruchom pipeline'y
    all_results = {}
    start_time = datetime.now()

    for variant in variants:
        if variant not in PIPELINES:
            print(f"[run_all] UWAGA: Nieznany wariant '{variant}', pomijam.")
            continue

        print(f"\n{'#'*60}")
        print(f"# Pipeline: {variant}")
        print(f"# Start: {datetime.now().isoformat()}")
        print(f"{'#'*60}\n")

        module = PIPELINES[variant]

        # Warianty Haar wymagają dodatkowych parametrów
        if "Haar" in variant:
            results = module.run(
                y_train=y_train,
                y_test=y_test,
                selected_levels=haar_levels,
                horizons=horizons,
            )
        else:
            results = module.run(
                y_train=y_train,
                y_test=y_test,
                horizons=horizons,
            )

        all_results[variant] = results

    # 3. Podsumowanie
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"PODSUMOWANIE (czas: {elapsed})")
    print(f"{'='*60}")

    for variant, results in all_results.items():
        print(f"\n  {variant}:")
        for r in results:
            mae = r["metrics"]["MAE"]
            print(f"    h={r['horizon']:3d}  MAE={mae:.6f}")

    return all_results


if __name__ == "__main__":
    # CLI: python run_all.py [variant1] [variant2] ...
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
    else:
        selected = None

    run_all(variants=selected)
