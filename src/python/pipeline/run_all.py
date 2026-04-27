"""
Master runner — uruchamia wszystkie lub wybrane pipeline'y z jednego miejsca.

Uzycie:
    python run_all.py                          # Wszystkie warianty
    python run_all.py Simple Simple-LoRA       # Tylko wybrane (pozycyjnie)
    python run_all.py --variants Simple Haar-after --haar_levels A5 D2

    # Wiele kombinacji pasm na raz:
    python run_all.py --variants Haar-in Haar-after --haar_combos A5,D2  A4,D4  A3
    # ^ uruchomi Haar-in i Haar-after po 3 razy: (A5+D2), (A4+D4), (A3)
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
    VARIANT_HAAR_AFTER_SUM,
    VARIANT_HAAR_AFTER_SUM_LORA,
    MAT_VAR_NAME_TRAIN,
    MAT_VAR_NAME_TEST,
)
from common.data_loader import load_mat_data, split_train_test

# Import pipeline'ow
import run_simple
import run_simple_lora
import run_haar_after
import run_haar_after_lora
import run_haar_in
import run_haar_in_lora
import run_haar_after_sum
import run_haar_after_sum_lora

# Mapowanie nazwa -> modul
PIPELINES = {
    VARIANT_SIMPLE: run_simple,
    VARIANT_SIMPLE_LORA: run_simple_lora,
    VARIANT_HAAR_AFTER: run_haar_after,
    VARIANT_HAAR_AFTER_LORA: run_haar_after_lora,
    VARIANT_HAAR_IN: run_haar_in,
    VARIANT_HAAR_IN_LORA: run_haar_in_lora,
    VARIANT_HAAR_AFTER_SUM: run_haar_after_sum,
    VARIANT_HAAR_AFTER_SUM_LORA: run_haar_after_sum_lora,
}


def _combo_label(levels: list[str]) -> str:
    """Tworzy czytelny label z listy poziomow, np. ['A5', 'D2'] -> 'A5-D2'."""
    return "-".join(levels)


def run_all(
    variants: list[str] = None,
    data_path: str = None,
    var_name_train: str = MAT_VAR_NAME_TRAIN,
    var_name_test: str = MAT_VAR_NAME_TEST,
    test_ratio: float = 0.2,
    horizons: list[int] = None,
    haar_levels: list[str] = None,
    haar_combos: list[list[str]] = None,
):
    """
    Uruchom pipeline'y sekwencyjnie.

    Args:
        variants: Lista wariantow do uruchomienia (None = wszystkie)
        data_path: Sciezka do pliku .mat
        var_name_train: Nazwa zmiennej train w .mat
        var_name_test: Nazwa zmiennej test w .mat
        test_ratio: Podzial train/test
        horizons: Horyzonty predykcji
        haar_levels: Pojedyncza kombinacja poziomow Haar (wsteczna kompatybilnosc)
        haar_combos: Lista kombinacji poziomow Haar, np.
                     [["A5","D2"], ["A4","D4"], ["A3"]]
                     Kazda kombinacja jest uruchamiana osobno per wariant Haar.
    """
    if variants is None:
        variants = list(PIPELINES.keys())
    if horizons is None:
        horizons = HORIZONS
    if data_path is None:
        data_path = DATA_DIR / "ab_diff_zestaw.mat"

    # Rozwiaz haar_combos vs haar_levels (wsteczna kompatybilnosc)
    # haar_combos ma priorytet; jesli nie podano, uzyj haar_levels jako jednej kombinacji
    if haar_combos is None and haar_levels is not None:
        haar_combos = [haar_levels]
    # jesli oba None -> haar_combos = None (pipeline uzyje swoich domyslnych)

    # 1. Wczytaj dane
    print(f"[run_all] Wczytywanie danych: {data_path}")
    y_train, y_test = load_mat_data(data_path, var_name_train, var_name_test)

    print(f"[run_all] Dane: train={len(y_train)}, test={len(y_test)}")
    print(f"[run_all] Warianty: {variants}")
    print(f"[run_all] Horyzonty: {horizons}")
    if haar_combos:
        combos_str = [_combo_label(c) for c in haar_combos]
        print(f"[run_all] Kombinacje pasm Haar: {combos_str}")

    # 2. Uruchom pipeline'y
    all_results = {}
    start_time = datetime.now()

    for variant in variants:
        if variant not in PIPELINES:
            print(f"[run_all] UWAGA: Nieznany wariant '{variant}', pomijam.")
            continue

        module = PIPELINES[variant]

        if "Haar" in variant and haar_combos is not None:
            # --- Wiele kombinacji pasm ---
            for combo in haar_combos:
                combo_name = _combo_label(combo)
                run_label = f"{variant}({combo_name})"

                print(f"\n{'#'*60}")
                print(f"# Pipeline: {run_label}")
                print(f"# Start: {datetime.now().isoformat()}")
                print(f"{'#'*60}\n")

                results = module.run(
                    y_train=y_train,
                    y_test=y_test,
                    selected_levels=combo,
                    horizons=horizons,
                    variant_name=run_label,
                )
                all_results[run_label] = results

        elif "Haar" in variant:
            # --- Jedna kombinacja lub domyslna ---
            print(f"\n{'#'*60}")
            print(f"# Pipeline: {variant}")
            print(f"# Start: {datetime.now().isoformat()}")
            print(f"{'#'*60}\n")

            results = module.run(
                y_train=y_train,
                y_test=y_test,
                selected_levels=haar_levels,
                horizons=horizons,
            )
            all_results[variant] = results

        else:
            # --- Warianty nie-Haar (Simple, Simple-LoRA) ---
            print(f"\n{'#'*60}")
            print(f"# Pipeline: {variant}")
            print(f"# Start: {datetime.now().isoformat()}")
            print(f"{'#'*60}\n")

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

    for label, results in all_results.items():
        print(f"\n  {label}:")
        for r in results:
            mae = r["metrics"]["MAE"]
            print(f"    h={r['horizon']:3d}  MAE={mae:.6f}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Master runner -- uruchamia wszystkie lub wybrane pipeline'y.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Przyklady:
        python run_all.py Simple                                    # jeden wariant
        python run_all.py --variants Haar-in --haar_levels A5 D2    # jedna kombinacja pasm
        python run_all.py --variants Haar-in --haar_combos A5,D2 A4,D4 A3
                                                                    # trzy kombinacje pasm
        python run_all.py --variants Haar-in Haar-after --haar_combos A5,D2 A3
                                                                    # 2 warianty x 2 kombinacje
        """,
    )

    # Wsparcie dla wariantow podawanych pozycyjnie
    parser.add_argument(
        "positional_variants",
        nargs="*",
        type=str,
        help="Lista wariantow do uruchomienia podana bez flagi (wsteczna kompatybilnosc)",
    )

    parser.add_argument("--variants", nargs="+", type=str, help="Lista wariantow do uruchomienia (z flaga)")
    parser.add_argument("--data_path", type=str, help="Sciezka do pliku .mat")
    parser.add_argument("--var_name_train", type=str, default=MAT_VAR_NAME_TRAIN, help="Nazwa zmiennej train w .mat")
    parser.add_argument("--var_name_test", type=str, default=MAT_VAR_NAME_TEST, help="Nazwa zmiennej test w .mat")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Podzial train/test")
    parser.add_argument("--horizons", nargs="+", type=int, help="Horyzonty predykcji")
    parser.add_argument("--haar_levels", nargs="+", type=str, help="Jedna kombinacja poziomow Haar (wsteczna kompatybilnosc)")
    parser.add_argument("--haar_combos", nargs="+", type=str, help="Wiele kombinacji poziomow Haar, rozdzielone przecinkami, np.: A5,D2 A4,D4 A3",)

    args = parser.parse_args()

    # Złączamy logike wariantow podanych pozycyjnie i przez flage --variants
    selected_variants = args.variants if args.variants else args.positional_variants
    if not selected_variants:
        selected_variants = None

    # Parsuj --haar_combos: "A5,D2" -> ["A5","D2"]
    parsed_combos = None
    if args.haar_combos:
        parsed_combos = [combo_str.split(",") for combo_str in args.haar_combos]

    run_all(
        variants=selected_variants,
        data_path=args.data_path,
        var_name_train=args.var_name_train,
        var_name_test=args.var_name_test,
        test_ratio=args.test_ratio,
        horizons=args.horizons,
        haar_levels=args.haar_levels,
        haar_combos=parsed_combos,
    )
