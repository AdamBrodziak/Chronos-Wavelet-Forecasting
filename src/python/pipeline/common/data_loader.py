"""
Wczytywanie i przetwarzanie danych wejściowych.

Obsługuje:
- Konwersja z formatu MATLAB (listy, double array) do numpy/pandas
- Wczytywanie plików .mat
- Różnicowanie (diff)
- Normalizacja / denormalizacja
- Przygotowanie DataFrame w formacie wymaganym przez Chronos 2
"""

import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# Konwersja danych z MATLAB
# =============================================================================

def matlab_to_numpy(raw_data) -> np.ndarray:
    """
    Konwertuje dane wejściowe z MATLAB (lista, np.ndarray, itp.) 
    do płaskiego 1D numpy array typu float64.

    Args:
        raw_data: Dane wejściowe — mogą być listą, zagnieżdżoną listą,
                  np.ndarray lub dowolną strukturą z MATLABa.

    Returns:
        Płaski 1D numpy array (float64).
    """
    return np.asarray(raw_data, dtype=np.float64).ravel()


# =============================================================================
# Wczytywanie plików
# =============================================================================
# TODO zmienić aby odpowiadało strukturze .mat
# ab_diff_test i ab_diff_train, mają po dwie kolumny: Time i Var1
def load_mat_data(mat_path: str | Path, var_name_train: str, var_name_test: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Wczytuje zmienną z pliku .mat (MATLAB).

    Args:
        mat_path: Ścieżka do pliku .mat
        var_name_train: Nazwa zmiennej w pliku .mat
        var_name_test: Nazwa zmiennej w pliku .mat

    Returns:
        Tuple 1D numpy arrays: (y_train, y_test)
    """
    import scipy.io as sio
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Plik nie istnieje: {mat_path}")

    mat_data = sio.loadmat(str(mat_path))

    def get_var(name):
        if name not in mat_data:
            available = [k for k in mat_data.keys() if not k.startswith('__')]
            msg = f"Zmienna '{name}' nie znaleziona w {mat_path.name}. Dostępne: {available}"
            
            # Sprawdzenie czy są obiekty Opaque (tabele/timetables)
            if any(isinstance(v, sio.matlab.MatlabOpaque) for v in mat_data.values()):
                msg += (
                    "\nUWAGA: Wykryto obiekty typu Table/Timetable, których scipy.io.loadmat nie obsługuje. "
                    "W MATLAB użyj 'table2array(T)' lub zapisz jako prosty 'double' przed exportem."
                )
            raise KeyError(msg)
        
        val = mat_data[name]
        # Jeśli to tablica 2D z dwiema kolumnami, zakładamy (Timestamp, Wartość) i bierzemy drugą
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] == 2:
            return np.asarray(val[:, 1], dtype=np.float64)
        
        # W innym przypadku używamy matlab_to_numpy (ravel)
        return matlab_to_numpy(val)

    y_train = get_var(var_name_train)
    y_test = get_var(var_name_test)

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("Jedna z zmiennych jest pusta")
    
    return y_train, y_test


def load_csv_data(csv_path: str | Path, column: str = None) -> np.ndarray:
    """
    Wczytuje dane z pliku CSV.

    Args:
        csv_path: Ścieżka do pliku CSV
        column: Nazwa kolumny do wczytania (None = pierwsza kolumna numeryczna)

    Returns:
        1D numpy array z danymi.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Plik nie istnieje: {csv_path}")

    df = pd.read_csv(csv_path)
    if column is not None:
        return df[column].values.astype(np.float64)
    else:
        # Pierwsza kolumna numeryczna
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError(f"Brak kolumn numerycznych w {csv_path.name}")
        return df[numeric_cols[0]].values.astype(np.float64)


# =============================================================================
# Przetwarzanie danych
# =============================================================================

def diff_series(data: np.ndarray) -> np.ndarray:
    """
    Różnicowanie szeregu czasowego (pierwsza różnica).

    Wynik ma o 1 element mniej niż wejście.
    Stosowane gdy dane są w postaci skumulowanej (np. energia kWh).

    Args:
        data: 1D numpy array

    Returns:
        np.diff(data) — 1D array, len = len(data) - 1
    """
    return np.diff(data)


def undiff_series(diff_data: np.ndarray, first_value: float) -> np.ndarray:
    """
    Odwrócenie różnicowania — odtworzenie oryginalnego szeregu.

    Args:
        diff_data: Zróżnicowane dane (np.diff output)
        first_value: Pierwsza wartość oryginalnego szeregu

    Returns:
        Odtworzony szereg skumulowany.
    """
    return np.concatenate([[first_value], first_value + np.cumsum(diff_data)])


def normalize(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Z-score normalizacja (standaryzacja).

    Args:
        data: 1D numpy array

    Returns:
        (normalized_data, mean, std)
        Zachowaj mean i std żeby później zdenormalizować predykcje.
    """
    mean = float(np.mean(data))
    std = float(np.std(data))
    if std == 0:
        std = 1.0  # Unikamy dzielenia przez zero
    normalized = (data - mean) / std
    return normalized, mean, std


def denormalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Odwrócenie normalizacji.

    Args:
        data: Znormalizowane dane
        mean: Średnia użyta do normalizacji
        std: Odchylenie std użyte do normalizacji

    Returns:
        Dane w oryginalnej skali.
    """
    return data * std + mean


# =============================================================================
# Przygotowanie danych dla Chronos 2
# =============================================================================

#TODO zmienić timestamp, aby brało z pliku
def prepare_context_df(data: np.ndarray, series_id: str = "seria_1") -> pd.DataFrame:
    """
    Przygotowuje DataFrame w formacie wymaganym przez Chronos2Pipeline.predict_df().

    Format: kolumny [timestamp, target, id] (long format).

    Args:
        data: 1D numpy array z danymi historycznymi
        series_id: Identyfikator szeregu czasowego

    Returns:
        pd.DataFrame z kolumnami: timestamp, target, id
    """
    return pd.DataFrame({
        "timestamp": range(len(data)),
        "target": data,
        "id": series_id,
    })

# TODO zrozumieć
def prepare_train_inputs(data: np.ndarray) -> list:
    """
    Przygotowuje dane treningowe w formacie wymaganym przez pipeline.fit().

    Chronos 2 API oczekuje listy dict-ów: [{"target": tensor}, ...]
    Dla jednej serii tworzymy jeden element.

    Args:
        data: 1D numpy array z danymi

    Returns:
        Lista dict z kluczem 'target' (torch.Tensor)
    """
    import torch
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, T)
    return [tensor]


def split_train_test(data: np.ndarray, test_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Podział danych na zbiór treningowy i testowy (chronologicznie).

    Args:
        data: 1D numpy array
        test_ratio: Procent danych na test (domyślnie 20%)

    Returns:
        (train_data, test_data)
    """
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]
