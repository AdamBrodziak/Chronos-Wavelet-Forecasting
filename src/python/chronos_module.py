import torch
from chronos import ChronosPipeline
from chronos import Chronos2Pipeline
import numpy as np
import pandas as pd

# Zmienna globalna, aby nie ładować modelu przy każdym wywołaniu funkcji
PIPELINE = None

def get_pipeline():
    """
    Ładuje model tylko raz (Singleton pattern).
    """
    global PIPELINE
    if PIPELINE is None:
        print("Ładowanie modelu Chronos po raz pierwszy...")
        PIPELINE = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="auto")
    return PIPELINE

def run_forecast(signal_data, prediction_length):
    """
    Funkcja wywoływana z MATLABa.
    """
    pipeline = get_pipeline()
    
    # 1. KONWERSJA DANYCH (MATLAB -> PyTorch)
    if isinstance(signal_data, list):
        #context_tensor = torch.tensor(signal_data)
        np_train = np.asarray(signal_data).ravel()
        
        # 2. Konwertuj tablice numpy na obiekty pd.Series
        y_train = pd.Series(np_train)

        context_df = y_train.to_frame(name="target").reset_index()
        context_df = context_df.rename(columns={"index": "timestamp"})
        context_df["id"] = "seria_1"
    else:
        context_df = torch.tensor(np.array(signal_data))
    
    # 2. PREDYKCJA
    forecast_df = pipeline.predict_df(
            context_df, 
            prediction_length=int(prediction_length),
            quantile_levels=[0.5],
            id_column="id",  # Column identifying different time series
            timestamp_column="timestamp",  # Column with datetime information
            target="target",  # Column(s) with time series values to predict
        )
        
    # Zazwyczaj kolumna nazywa się 'q_0.5' dla kwantyla 0.5
    if '0.5' in forecast_df.columns:
        result_values = forecast_df['0.5'].values
    elif 'median' in forecast_df.columns:
        result_values = forecast_df['median'].values
    elif 'mean' in forecast_df.columns:
         result_values = forecast_df['mean'].values
    else:
        # Fallback: bierzemy ostatnią kolumnę, jeśli nazewnictwo jest inne
        result_values = forecast_df.iloc[:, -1].values

    # Zwracamy listę floatów
    return result_values.tolist()