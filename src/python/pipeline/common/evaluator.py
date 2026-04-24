"""
Ewaluacja predykcji — metryki jakości.

Metryki:
- MAE, RMSE, MAPE, R2_classic, R2_alt
"""

import numpy as np


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate(y_true, y_pred)
    mask = np.abs(y_true) > 1e-10
    if not np.any(mask):
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_r2_classic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

#TODO sprawdzić poprawność wzoru względem excela
def compute_r2_alt(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate(y_true, y_pred)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(corr ** 2)


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Oblicza WSZYSTKIE metryki naraz.
    
    Args:
        y_true: Tablica prawdziwych wartości (idealna trajektoria)
        y_pred: Tablica predykcji modelu
    
    Returns: 
        Słownik z metrykami:
        - MAE: średni błąd bezwzględny
        - RMSE: pierwiastek średniego błędu kwadratowego
        - MAPE: procentowy błąd średni bezwzględny
        - R2_classic: współczynnik determinacji (klasyczny)
        - R2_alt: współczynnik determinacji (alternatywny)
    """
    return {
        "MAE": compute_mae(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAPE": compute_mape(y_true, y_pred),
        "R2_classic": compute_r2_classic(y_true, y_pred),
        "R2_alt": compute_r2_alt(y_true, y_pred),
    }


def print_metrics(metrics: dict, variant_name: str = "", horizon: int = 0):
    """Wypisuje metryki w czytelnej formie.
    
    Args:
        metrics: Słownik z metrykami (z evaluate_all)
        variant_name: Nazwa wariantu (opcjonalnie)
        horizon: Horyzont predykcji (opcjonalnie)
    """
    header = f"[evaluator] Metryki"
    if variant_name:
        header += f" — {variant_name}"
    if horizon > 0:
        header += f" (h={horizon})"
    print(header)
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.6f}")


def _validate(y_true, y_pred):
    """Waliduje dane wejściowe.
    
    Args:
        y_true: Tablica prawdziwych wartości
        y_pred: Tablica predykcji model
    
    Returns:
        (y_true, y_pred) po konwersji do numpy array i sprawdzeniu zgodności długości
    """
    # TODO zrobić to bardziej ogólnie, teraz zakłada tylko 1 cechę
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError(f"len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")
    return y_true, y_pred
