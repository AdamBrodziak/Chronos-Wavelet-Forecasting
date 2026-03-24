import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing


def forecast_holt_winters(
        y_train_raw,
        y_test_raw,
        step_length,
        seasonal_periods,
        seasonal='add',
        initialization_method='estimated'
):
    """
    Prognozuje metodą Holta-Wintersa (ExponentialSmoothing) przy użyciu
    strategii "expanding window" z ponownym trenowaniem (refitting).
    """

    step_length = int(step_length)
    seasonal_periods = int(seasonal_periods)

    # 1. Konwertuj surowe dane (np. z MATLAB double) na tablice numpy
    #    .ravel() spłaszcza dane, jeśli MATLAB przekazał je jako [[1],[2],[3]]
    np_train = np.asarray(y_train_raw).ravel()
    np_test = np.asarray(y_test_raw).ravel()

    # 2. Konwertuj tablice numpy na obiekty pd.Series
    y_train = pd.Series(np_train)
    y_test = pd.Series(np_test)
    
    # 3. Ustaw prosty indeks dla y_test, aby .index działał poprawnie
    y_test.index = range(len(y_test))
    y_train.index = range(len(y_train))

    # Lista do przechowywania fragmentów prognoz
    all_predictions_list = []

    # Teraz .copy() zadziała poprawnie
    current_training_data = y_train.copy()

    # Obliczamy liczbę pętli potrzebną do pokrycia całego zbioru testowego
    num_loops = int(np.ceil(len(y_test) / step_length))

    print(f"Rozpoczynanie prognozy (Refitting Expanding Window)...")
    print(f"Rozmiar testowy: {len(y_test)}, Krok: {step_length}, Pętle: {num_loops}")

    for i in range(num_loops):
        print(f"  Pętla {i + 1}/{num_loops} (Rozmiar danych tren.: {len(current_training_data)})...")

        # 1. Trenowanie modelu na *całych* dotychczasowych danych
        model = ExponentialSmoothing(
            current_training_data,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method=initialization_method
        )

        # Trenujemy model
        fitted_model = model.fit(optimized=True)

        # 2. Prognoza na następny `step_length`
        pred_chunk = fitted_model.forecast(steps=step_length)

        # 3. Zapisanie fragmentu prognozy
        all_predictions_list.append(pred_chunk)

        # 4. AKTUALIZACJA (Expand)
        start_idx = i * step_length
        end_idx = min((i + 1) * step_length, len(y_test))

        # Teraz .iloc[] zadziała poprawnie
        true_data_chunk = y_test.iloc[start_idx:end_idx]

        if not true_data_chunk.empty:
            current_training_data = pd.concat([current_training_data, true_data_chunk], ignore_index=True)

    # Połączenie wszystkich fragmentów prognoz w jedną serię
    final_predictions_series = pd.concat(all_predictions_list)

    final_predictions_series = final_predictions_series.iloc[:len(y_test)]
    final_predictions_series.index = y_test.index

    print("Prognoza zakończona.")
    return final_predictions_series.values