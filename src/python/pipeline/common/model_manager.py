"""
Zarządzanie modelem Chronos 2 — ładowanie, singleton, predykcja.

Obsługuje:
- Ładowanie modelu z HuggingFace (singleton — jeden model w pamięci)
- Ładowanie fine-tuned modelu z dysku
- Ujednolicona predykcja (zwraca medianę jako np.ndarray)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from chronos import Chronos2Pipeline

from .config import MODEL_NAME, MODEL_DTYPE, MODEL_DEVICE_MAP, QUANTILE_LEVELS


# =============================================================================
# Singleton — globalny model
# =============================================================================

_PIPELINE_CACHE: dict[str, Chronos2Pipeline] = {}


def get_pipeline(
    model_name: str = MODEL_NAME,
    dtype: str = MODEL_DTYPE,
    device_map: str = MODEL_DEVICE_MAP,
) -> Chronos2Pipeline:
    """
    Ładuje model Chronos 2 (Singleton pattern).

    Model jest ładowany tylko raz na daną nazwę. Kolejne wywołania
    zwracają ten sam obiekt z cache.

    Args:
        model_name: Nazwa modelu na HuggingFace lub ścieżka lokalna
        dtype: Typ danych ("bfloat16", "float32", "float16")
        device_map: Mapowanie urządzenia ("auto", "cuda", "cpu")

    Returns:
        Załadowany Chronos2Pipeline.
    """
    cache_key = f"{model_name}_{dtype}_{device_map}"

    if cache_key not in _PIPELINE_CACHE:
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        print(f"[model_manager] Ładowanie modelu: {model_name} "
              f"(dtype={dtype}, device={device_map})...")
        
        pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=torch_dtype,
        )
        _PIPELINE_CACHE[cache_key] = pipeline
        print(f"[model_manager] Model załadowany pomyślnie.")

    return _PIPELINE_CACHE[cache_key]


def load_finetuned_pipeline(
    model_path: str | Path,
    device_map: str = MODEL_DEVICE_MAP,
) -> Chronos2Pipeline:
    """
    Ładuje fine-tuned model Chronos 2 z dysku.

    Args:
        model_path: Ścieżka do katalogu z zapisanym modelem
        device_map: Mapowanie urządzenia

    Returns:
        Fine-tuned Chronos2Pipeline.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model nie istnieje: {model_path}")

    print(f"[model_manager] Ładowanie fine-tuned modelu z: {model_path}")
    return Chronos2Pipeline.from_pretrained(
        str(model_path),
        device_map=device_map,
    )


def clear_cache():
    """Czyści cache modeli — przydatne przy zwalnianiu VRAM."""
    global _PIPELINE_CACHE
    _PIPELINE_CACHE.clear()
    torch.cuda.empty_cache()
    print("[model_manager] Cache wyczyszczony.")


# =============================================================================
# Predykcja
# =============================================================================

def predict(
    pipeline: Chronos2Pipeline,
    context_df: pd.DataFrame,
    prediction_length: int,
    quantile_levels: list[float] = None,
) -> np.ndarray:
    """
    Wykonuje predykcję modelem Chronos 2 i zwraca medianę.

    Args:
        pipeline: Załadowany Chronos2Pipeline
        context_df: DataFrame w formacie [timestamp, target, id]
        prediction_length: Horyzont predykcji (liczba kroków)
        quantile_levels: Lista kwantyli (domyślnie [0.5])

    Returns:
        1D numpy array z predykcją mediany (długość = prediction_length).
    """
    if quantile_levels is None:
        quantile_levels = QUANTILE_LEVELS

    with torch.inference_mode():
        forecast_df = pipeline.predict_df(
            context_df,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

    return _extract_median(forecast_df)


def _extract_median(forecast_df: pd.DataFrame) -> np.ndarray:
    """
    Wyciąga kolumnę mediany z forecast DataFrame.

    Chronos 2 predict_df zwraca kolumny o nazwach odpowiadających
    kwantylom (np. '0.5') — ta funkcja szuka właściwej kolumny
    z fallback w razie zmian w API.

    Args:
        forecast_df: DataFrame z wynikami predict_df

    Returns:
        1D numpy array z wartościami mediany.
    """
    # Próbujemy różne nazwy kolumn (Chronos 2 API może się różnić)
    for col_name in ['0.5', 'median', 'mean', 'forecast']:
        if col_name in forecast_df.columns:
            return forecast_df[col_name].values.astype(np.float64)

    # Fallback: ostatnia kolumna numeryczna
    numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return forecast_df[numeric_cols[-1]].values.astype(np.float64)

    raise ValueError(
        f"Nie znaleziono kolumny z medianą. "
        f"Dostępne kolumny: {list(forecast_df.columns)}"
    )
