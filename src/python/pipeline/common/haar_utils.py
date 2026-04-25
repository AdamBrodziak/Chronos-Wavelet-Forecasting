"""
Transformata Haar — dekompozycja i rekonstrukcja.

Obsługuje:
- Wielopoziomowa dekompozycja dyskretna (DWT) z filtrem Haar
- Rekonstrukcja wybranych poziomów
- Selekcja poziomów do dalszej predykcji

Używa PyWavelets (pywt) — standardowa biblioteka do wavelet transforms.
"""

import numpy as np

try:
    import pywt
except ImportError:
    pywt = None
    print("[haar_utils] UWAGA: pywt nie zainstalowane. "
          "Zainstaluj: pip install PyWavelets")


def haar_decompose(signal: np.ndarray, max_level: int = 5) -> dict:
    """
    Wielopoziomowa dekompozycja Haar (DWT).

    Args:
        signal: 1D numpy array — sygnał do dekompozycji
        max_level: Maksymalny poziom dekompozycji (domyślnie 5)

    Returns:
        dict z kluczami:
            "A{i}": współczynniki aproksymacji dla poziomu i
            "D{i}": współczynniki detali dla poziomu i
            "original": oryginalny sygnał
            "max_level": użyty poziom dekompozycji
    
    Przykład:
        >>> decomp = haar_decompose(signal, max_level=5)
        >>> decomp.keys()  # {"A5", "D5", "D4", "D3", "D2", "D1", "original", "max_level"}
    """
    _check_pywt()

    # Automatyczne ograniczenie poziomu (max dopuszczalny przez pywt)
    actual_max = pywt.dwt_max_level(len(signal), "haar")
    level = min(max_level, actual_max)

    # Dekompozycja: zwraca [cA_n, cD_n, cD_n-1, ..., cD_1]
    coeffs = pywt.wavedec(signal, "haar", level=level)

    result = {
        "original": signal,
        "max_level": level,
    }

    # coeffs[0] = aproksymacja najwyższego poziomu (A_n)
    result[f"A{level}"] = coeffs[0]

    # coeffs[1:] = detale od najwyższego do najniższego poziomu
    for i, detail in enumerate(coeffs[1:], start=1):
        detail_level = level - i + 1
        result[f"D{detail_level}"] = detail

    return result


def haar_reconstruct(
    decomposition: dict,
    selected_levels: list[str],
) -> np.ndarray:
    """
    Rekonstrukcja sygnału z wybranych poziomów dekompozycji Haar.

    Zeruje współczynniki NIEwybranych poziomów, a następnie
    rekonstruuje sygnał za pomocą IDWT.

    Args:
        decomposition: dict z haar_decompose()
        selected_levels: Lista nazw poziomów do zachowania,
                         np. ["A5", "D1"] lub ["A1"]

    Returns:
        1D numpy array — zrekonstruowany sygnał
    """
    _check_pywt()

    level = decomposition["max_level"]

    # Odtwórz pełną listę współczynników
    # coeffs = [cA_n, cD_n, cD_n-1, ..., cD_1]
    coeffs = []

    # Aproksymacja najwyższego poziomu
    a_key = f"A{level}"
    if a_key in selected_levels:
        coeffs.append(decomposition[a_key])
    else:
        coeffs.append(np.zeros_like(decomposition[a_key]))

    # Detale od najwyższego do najniższego
    for d_level in range(level, 0, -1):
        d_key = f"D{d_level}"
        if d_key in selected_levels:
            coeffs.append(decomposition[d_key])
        else:
            coeffs.append(np.zeros_like(decomposition[d_key]))

    # Rekonstrukcja IDWT
    reconstructed = pywt.waverec(coeffs, "haar")

    # Wyrównaj długość (waverec może dodać 1 próbkę)
    original_len = len(decomposition["original"])
    return reconstructed[:original_len]


def reconstruct_single_level(
    decomposition: dict,
    level_name: str,
) -> np.ndarray:
    """
    Rekonstrukcja sygnału z JEDNEGO poziomu.

    Przydatne dla wariantu Haar-in, gdzie każdy poziom
    jest predykowany osobno.

    Args:
        decomposition: dict z haar_decompose()
        level_name: Nazwa poziomu, np. "A5" lub "D1"

    Returns:
        Sygnał w dziedzinie czasu odpowiadający jednemu poziomowi.
    """
    return haar_reconstruct(decomposition, [level_name])


def get_level_names(decomposition: dict) -> list[str]:
    """
    Zwraca listę nazw dostępnych poziomów.

    Returns:
        np. ["A5", "D5", "D4", "D3", "D2", "D1"]
    """
    level = decomposition["max_level"]
    names = [f"A{level}"]
    for i in range(level, 0, -1):
        names.append(f"D{i}")
    return names

def get_level_depth(level_name: str) -> int:
    """
    Zwraca głębokość dekompozycji dla danego poziomu.

    Np. "A5" -> 5, "D3" -> 3, "D1" -> 1.
    Głębokość mówi, ile razy sygnał został skrócony o połowę,
    więc współczynniki na poziomie k mają długość ~ signal_len / 2^k.

    Args:
        level_name: Nazwa poziomu, np. "A5" lub "D3".

    Returns:
        Głębokość (int).
    """
    return int(level_name[1:])


def compute_wavelet_horizon(time_horizon: int, level_depth: int) -> int:
    """
    Oblicza horyzont predykcji w dziedzinie współczynników wavelet.

    Dekompozycja Haar skraca sygnał o połowę na każdym poziomie,
    więc horyzont w dziedzinie współczynników to:
        wavelet_horizon = ceil(time_horizon / 2^depth)

    Args:
        time_horizon: Horyzont predykcji w dziedzinie czasowej (np. 96).
        level_depth: Głębokość poziomu dekompozycji (np. 5 dla A5).

    Returns:
        Horyzont predykcji w dziedzinie współczynników (int, >= 1).
    """
    import math
    wh = math.ceil(time_horizon / (2 ** level_depth))
    return max(wh, 1)


def reconstruct_from_predicted_coeffs(
    decomposition: dict,
    predicted_coeffs: dict[str, np.ndarray],
    original_length: int,
) -> np.ndarray:
    """
    Rekonstrukcja sygnału w dziedzinie czasu z predykowanych współczynników.

    Składa wektor współczynników do IDWT, gdzie:
    - Poziomy obecne w ``predicted_coeffs`` używają predykowanych wartości,
    - Wszystkie pozostałe poziomy są ZEROWANE.

    Args:
        decomposition: dict z haar_decompose() — potrzebny, żeby
                       znać kształty współczynników na każdym poziomie.
        predicted_coeffs: Słownik {level_name: predicted_array},
                          np. {"A5": np.array([...]), "D2": np.array([...])}.
        original_length: Oczekiwana długość wyjściowego sygnału
                         (len(y_test) lub len(full_data)).

    Returns:
        1D numpy array — zrekonstruowany sygnał w dziedzinie czasu.
    """
    _check_pywt()

    level = decomposition["max_level"]

    # Buduj listę współczynników: [cA_n, cD_n, cD_n-1, ..., cD_1]
    coeffs: list[np.ndarray] = []

    # Aproksymacja najwyższego poziomu
    a_key = f"A{level}"
    if a_key in predicted_coeffs:
        coeffs.append(predicted_coeffs[a_key])
    else:
        coeffs.append(np.zeros_like(decomposition[a_key]))

    # Detale od najwyższego do najniższego
    for d_level in range(level, 0, -1):
        d_key = f"D{d_level}"
        if d_key in predicted_coeffs:
            coeffs.append(predicted_coeffs[d_key])
        else:
            coeffs.append(np.zeros_like(decomposition[d_key]))

    reconstructed = pywt.waverec(coeffs, "haar")
    return reconstructed[:original_length]


def _check_pywt():
    if pywt is None:
        raise ImportError(
            "PyWavelets (pywt) nie jest zainstalowane. "
            "Zainstaluj: pip install PyWavelets"
        )
