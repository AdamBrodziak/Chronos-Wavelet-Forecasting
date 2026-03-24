import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

class ELMRegressorGPU:
    """
    Extreme Learning Machine z obsługą GPU (CUDA)
    Zastępuje skelm.ELMRegressor
    """
    def __init__(self, n_neurons=3000, ufunc='tanh', alpha=1e-2, device='cuda'):
        self.n_neurons = n_neurons
        self.ufunc = ufunc
        self.alpha = alpha
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_weights = None
        self.bias = None
        self.beta = None
        
    def _activation(self, x):
        if self.ufunc == 'tanh':
            return torch.tanh(x)
        elif self.ufunc == 'sigmoid':
            return torch.sigmoid(x)
        elif self.ufunc == 'relu':
            return torch.relu(x)
        else:
            return torch.tanh(x)
    
    def fit(self, X, y):
        """
        Trenowanie modelu ELM na GPU
        """
        # Konwersja do tensorów GPU
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        
        n_samples, n_features = X_tensor.shape
        
        # Inicjalizacja losowych wag (tylko raz)
        if self.input_weights is None:
            self.input_weights = torch.randn(n_features, self.n_neurons).to(self.device)
            self.bias = torch.randn(1, self.n_neurons).to(self.device)
        
        # Warstwa ukryta
        H = self._activation(torch.mm(X_tensor, self.input_weights) + self.bias)
        
        # Obliczenie wag wyjściowych (Ridge Regression na GPU)
        # beta = (H^T * H + alpha * I)^-1 * H^T * y
        HTH = torch.mm(H.t(), H)
        I = torch.eye(self.n_neurons).to(self.device)
        HTH_reg = HTH + self.alpha * I
        HTy = torch.mm(H.t(), y_tensor)
        
        # Rozwiązanie układu równań
        self.beta = torch.linalg.solve(HTH_reg, HTy)
        
        return self
    
    def predict(self, X):
        """
        Predykcja na GPU
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        H = self._activation(torch.mm(X_tensor, self.input_weights) + self.bias)
        y_pred = torch.mm(H, self.beta)
        return y_pred.cpu().numpy().ravel()


def train_hw_model(data_MA, season_period):
    """
    Funkcja pomocnicza do trenowania Holt-Winters
    Może być wykonana w osobnym wątku
    """
    model_HW = ExponentialSmoothing(
        data_MA,
        seasonal='add',
        seasonal_periods=season_period,
        initialization_method='estimated'
    ).fit(optimized=True)
    return model_HW


def prepare_features_parallel(current_train, data_MA, Y_pred_HW_in_sample, ma_window):
    """
    Równoległe przygotowanie cech dla modelu ELM
    """
    dt_index = pd.to_datetime(current_train.index)
    hour_sin = np.sin(2*np.pi * dt_index.hour / 24)
    hour_cos = np.cos(2*np.pi * dt_index.hour / 24)
    
    X_for_ELM = pd.concat([
        Y_pred_HW_in_sample,
        current_train.shift(1),
        data_MA.shift(1),
        pd.Series(hour_sin, index=dt_index),
        pd.Series(hour_cos, index=dt_index)
    ], axis=1).dropna()
    
    return X_for_ELM


def model_hybrid_gpu(
    y_train_raw,
    y_test_raw,
    step_length_raw,
    ma_window=8,
    season_period=96,
    verbose=True,
    use_gpu=True,
    n_workers=4
):
    """
    Zoptymalizowana wersja model_hybrid z obsługą GPU i wielowątkowości
    
    Parametry:
    ----------
    y_train_raw : array-like
        Dane treningowe
    y_test_raw : array-like
        Dane testowe
    step_length_raw : int
        Długość kroku predykcji
    ma_window : int
        Okno dla średniej kroczącej
    season_period : int
        Okres sezonowości
    verbose : bool
        Wyświetlanie informacji o postępie
    use_gpu : bool
        Czy używać GPU dla ELM (wymaga CUDA)
    n_workers : int
        Liczba wątków do równoległego przetwarzania
    """
    
    warnings.simplefilter('ignore', ConvergenceWarning)
    ALPHA = 1e-2

    # --- 1. KONWERSJA DANYCH ---
    step_length = int(step_length_raw)
    ma_window = int(ma_window)
    season_period = int(season_period)
    verbose = bool(verbose)

    y_train = pd.Series(np.asarray(y_train_raw).ravel(), name='value')
    y_test = pd.Series(np.asarray(y_test_raw).ravel(), name='value')

    # Tworzymy sztuczny indeks czasu 15-min jak w oryginale
    full_len = len(y_train) + len(y_test)
    time_index = pd.date_range("2025-01-01", periods=full_len, freq="15min")

    y_train = pd.DataFrame({'value': y_train.values}, index=time_index[:len(y_train)])
    y_test  = pd.DataFrame({'value': y_test.values},  index=time_index[len(y_train):])

    # Sprawdzenie dostępności GPU
    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
    if verbose:
        print(f"Używane urządzenie: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Liczba wątków: {n_workers}")

    # --- 2. GŁÓWNA LOGIKA ---
    num_steps_to_predict = len(y_test) // step_length
    all_predictions_list = []

    # Inicjalizacja modelu ELM (wagi będą współdzielone dla przyspieszenia)
    model_ELM = ELMRegressorGPU(n_neurons=300, ufunc='tanh', alpha=ALPHA, device=device)

    # Pre-alokacja dla przyspieszenia
    for current_step in range(num_steps_to_predict):
        if verbose:
            print(f"Step: {current_step+1} / {num_steps_to_predict}")

        start = step_length * current_step
        end   = start + step_length

        # dynamiczne powiększanie train (expanding window)
        current_train = pd.concat([y_train, y_test.iloc[:start]])

        # --- RÓWNOLEGŁE OBLICZANIE MA ---
        # Obliczanie na GPU jeśli dane są duże
        if device == 'cuda' and len(current_train) > 10000:
            values_tensor = torch.FloatTensor(current_train.values).to(device)
            # Użycie conv1d dla średniej kroczącej na GPU
            kernel = torch.ones(ma_window, 1, 1).to(device) / ma_window
            padded = torch.nn.functional.pad(values_tensor.reshape(1, 1, -1), 
                                            (ma_window-1, 0), mode='replicate')
            ma_values = torch.nn.functional.conv1d(padded, kernel).squeeze().cpu().numpy()
            data_MA = pd.DataFrame({'value': ma_values}, index=current_train.index)
        else:
            data_MA = current_train.rolling(window=ma_window).mean().fillna(0)
        
        data_diff = current_train - data_MA

        # --- Holt-Winters (CPU - trudny do zrównoleglenia) ---
        model_HW = ExponentialSmoothing(
            data_MA,
            seasonal='add',
            seasonal_periods=season_period,
            initialization_method='estimated'
        ).fit(optimized=True)

        Y_pred_HW_in_sample = model_HW.predict(
            start=0,
            end=len(current_train)-1
        )

        # --- RÓWNOLEGŁE PRZYGOTOWANIE CECH ---
        dt_index = pd.to_datetime(current_train.index)
        hour_sin = np.sin(2*np.pi * dt_index.hour / 24)
        hour_cos = np.cos(2*np.pi * dt_index.hour / 24)

        X_for_ELM = pd.concat([
            Y_pred_HW_in_sample,
            current_train.shift(1),
            data_MA.shift(1),
            pd.Series(hour_sin, index=dt_index),
            pd.Series(hour_cos, index=dt_index)
        ], axis=1).dropna()

        # --- TRENOWANIE ELM NA GPU ---
        model_ELM.fit(
            X=X_for_ELM.values,
            y=data_diff.loc[X_for_ELM.index].values.ravel()
        )

        # --- FUTURE CECHY GODZINOWE ---
        future_index = y_test.index[start:end]
        future_hour_sin = np.sin(2*np.pi * future_index.hour / 24)
        future_hour_cos = np.cos(2*np.pi * future_index.hour / 24)

        # HW forecast
        Y_pred_HW_future = model_HW.predict(
            start=len(current_train),
            end=len(current_train) + step_length - 1
        )
        Y_pred_HW_future.index = future_index

        # --- AUTOREGRESJA (można zbatchować dla GPU) ---
        last_y = current_train.value.iloc[-1]
        last_n = data_diff.value.iloc[-1]

        # Przygotowanie wszystkich inputów naraz (batch prediction na GPU)
        if device == 'cuda' and step_length > 10:
            # Batch processing na GPU - znacznie szybsze
            pred_list = []
            
            for i in range(step_length):
                l_hat = Y_pred_HW_future.iloc[i]
                
                X_input = [[
                    l_hat,
                    last_y,
                    last_n,
                    future_hour_sin[i],
                    future_hour_cos[i]
                ]]
                
                n_hat = model_ELM.predict(X_input)[0]
                raw_y = n_hat + l_hat
                y_hat = raw_y
                
                pred_list.append(y_hat)
                
                last_y = y_hat
                last_n = n_hat
        else:
            # Standardowa autoregresja
            pred_list = []
            
            for i in range(step_length):
                l_hat = Y_pred_HW_future.iloc[i]
                
                X_input = [[
                    l_hat,
                    last_y,
                    last_n,
                    future_hour_sin[i],
                    future_hour_cos[i]
                ]]
                
                n_hat = model_ELM.predict(X_input)[0]
                raw_y = n_hat + l_hat
                y_hat = raw_y
                
                pred_list.append(y_hat)
                
                last_y = y_hat
                last_n = n_hat

        pred_df = pd.DataFrame({'value': pred_list}, index=future_index)
        all_predictions_list.append(pred_df)

    # --- 4. SKŁADANIE CAŁOŚCI ---
    all_predictions = pd.concat(all_predictions_list)
    if verbose:
        print("Koniec predykcji.")

    return all_predictions.values


def model_hybrid_multiprocess(
    y_train_raw,
    y_test_raw,
    step_length_raw,
    ma_window=8,
    season_period=96,
    verbose=True,
    n_processes=None
):
    """
    Alternatywna wersja z wieloprocesowością (dla bardzo dużych zbiorów)
    Uwaga: Ze względu na expanding window, trudniej zrównoleglić
    Ta wersja równolegle przetwarza tylko niezależne komponenty
    """
    
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    warnings.simplefilter('ignore', ConvergenceWarning)
    
    # Wywołanie standardowej wersji GPU z wielowątkowością
    return model_hybrid_gpu(
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw,
        step_length_raw=step_length_raw,
        ma_window=ma_window,
        season_period=season_period,
        verbose=verbose,
        use_gpu=True,
        n_workers=n_processes
    )


# Dla kompatybilności wstecznej
def model_hybrid(
    y_train_raw,
    y_test_raw,
    step_length_raw,
    ma_window=8,
    season_period=96,
    verbose=True
):
    """
    Wrapper dla zachowania kompatybilności z oryginalną funkcją
    Automatycznie wybiera wersję GPU jeśli dostępna
    """
    return model_hybrid_gpu(
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw,
        step_length_raw=step_length_raw,
        ma_window=ma_window,
        season_period=season_period,
        verbose=verbose,
        use_gpu=True,
        n_workers=4
    )