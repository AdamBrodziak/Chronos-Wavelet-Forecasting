"""
Zapis wyników — predykcje, metryki, modele.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from .config import RESULTS_DIR, MODELS_DIR


def save_predictions(
    predictions: np.ndarray,
    y_true: np.ndarray,
    horizon: int,
    variant_name: str,
    output_dir: str | Path = None,
):
    """Zapisuje predykcje i wartości rzeczywiste do CSV.
    
    Args:
        predictions: Tablica predykcji model
        y_true: Tablica prawdziwych wartości
        horizon: Horyzont predykcji
        variant_name: Nazwa wariantu (opcjonalnie)
        output_dir: Katalog do zapisu (domyślnie: RESULTS_DIR)
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = f"{variant_name}_h{horizon}_{timestamp}_pred.csv"
    df = pd.DataFrame({
        "y_true": y_true[:len(predictions)],
        "y_pred": predictions,
    })
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    print(f"[results_io] Predykcje zapisane: {filepath}")


def save_metrics(
    metrics: dict[str, float],
    horizon: int,
    variant_name: str,
    output_dir: str | Path = None,
):
    """Zapisuje metryki do JSON.
    
    Args:
        metrics: Słownik z metrykami
        horizon: Horyzont predykcji
        variant_name: Nazwa wariantu (opcjonalnie)
        output_dir: Katalog do zapisu (domyślnie: RESULTS_DIR)
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    filename = f"{variant_name}_h{horizon}_{timestamp}_metrics.json"
    result = {
        "variant": variant_name,
        "horizon": horizon,
        "timestamp": datetime.now().isoformat(timespec='minutes'),
        "metrics": metrics,
    }
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[results_io] Metryki zapisane: {filepath}")


def save_all_metrics_summary(
    all_metrics: list[dict],
    variant_name: str,
    output_dir: str | Path = None,
):
    """Zapisuje podsumowanie metryk ze wszystkich horyzontów do jednego CSV.
    
    Args:
        all_metrics: Lista dict-ów z metrykami per horyzont
        variant_name: Nazwa wariantu
        output_dir: Katalog do zapisu
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in all_metrics:
        row = {"variant": variant_name, "horizon": entry["horizon"]}
        row.update(entry["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    filepath = output_dir / f"{variant_name}_{timestamp}_summary.csv"
    df.to_csv(filepath, index=False)
    print(f"[results_io] Podsumowanie zapisane: {filepath}")


def save_model(pipeline, model_name: str, output_dir: str | Path = None):
    """Zapisuje model na dysk.
    
    Args:
        pipeline: Pipeline do zapisania
        model_name: Nazwa modelu (opcjonalnie)
        output_dir: Katalog do zapisu (domyślnie: MODELS_DIR)
    """
    if output_dir is None:
        output_dir = MODELS_DIR
    save_path = Path(output_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    pipeline.model.save_pretrained(save_path)
    pipeline.tokenizer.save_pretrained(save_path)
    print(f"[results_io] Model zapisany: {save_path}")