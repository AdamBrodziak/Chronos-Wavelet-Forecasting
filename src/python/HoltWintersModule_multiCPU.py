import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from joblib import Parallel, delayed
import multiprocessing

def forecast_holt_winters_parallel(
        y_train_raw,
        y_test_raw,
        step_length,
        seasonal_periods,
        seasonal='add',
        initialization_method='estimated',
        n_jobs=-1  # -1 = wszystkie dostępne rdzenie
):
    """
    Prognozuje metodą Holta-Wintersa z równoległym przetwarzaniem na CPU.
    """
    step_length = int(step_length)
    seasonal_periods = int(seasonal_periods)
    
    np_train = np.asarray(y_train_raw).ravel()
    np_test = np.asarray(y_test_raw).ravel()
    
    y_train = pd.Series(np_train)
    y_test = pd.Series(np_test)
    
    y_test.index = range(len(y_test))
    y_train.index = range(len(y_train))
    
    num_loops = int(np.ceil(len(y_test) / step_length))
    
    print(f"Rozpoczynanie prognozy (Parallel Refitting Expanding Window)...")
    print(f"Rozmiar testowy: {len(y_test)}, Krok: {step_length}, Pętle: {num_loops}")
    print(f"Używam {multiprocessing.cpu_count() if n_jobs == -1 else n_jobs} rdzeni CPU")
    
    def fit_and_forecast(i, training_data, step_length, seasonal, seasonal_periods, initialization_method):
        """Funkcja pomocnicza do równoległego przetwarzania"""
        print(f"  Pętla {i + 1}/{num_loops} (Rozmiar danych tren.: {len(training_data)})...")
        
        model = ExponentialSmoothing(
            training_data,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method=initialization_method
        )
        
        fitted_model = model.fit(optimized=True)
        pred_chunk = fitted_model.forecast(steps=step_length)
        
        return pred_chunk
    
    # Przygotowanie danych treningowych dla każdej iteracji
    training_datasets = []
    current_training_data = y_train.copy()
    
    for i in range(num_loops):
        training_datasets.append(current_training_data.copy())
        
        # Rozszerzenie danych treningowych o prawdziwe dane
        start_idx = i * step_length
        end_idx = min((i + 1) * step_length, len(y_test))
        true_data_chunk = y_test.iloc[start_idx:end_idx]
        
        if not true_data_chunk.empty:
            current_training_data = pd.concat([current_training_data, true_data_chunk], ignore_index=True)
    
    # Równoległe przetwarzanie
    all_predictions_list = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(fit_and_forecast)(
            i, training_datasets[i], step_length, seasonal, seasonal_periods, initialization_method
        )
        for i in range(num_loops)
    )
    
    final_predictions_series = pd.concat(all_predictions_list)
    final_predictions_series = final_predictions_series.iloc[:len(y_test)]
    final_predictions_series.index = y_test.index
    
    print("Prognoza zakończona.")
    return final_predictions_series.values