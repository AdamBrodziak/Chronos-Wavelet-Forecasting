import torch
from chronos import Chronos2Pipeline
import numpy as np
import pandas as pd

# --- Zmienna globalna dla modelu (Singleton) ---
PIPELINE = None

def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        print("Ładowanie modelu Chronos-2 (Singleton)...")
        PIPELINE = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2", 
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    return PIPELINE

def run_forecast_expanding(y_train_raw, y_test_raw, step_length):
    """
    Symuluje strategię Expanding Window (jak w Holt-Winters), ale dla Chronosa.
    
    Args:
        y_train_raw: Lista/tablica z danymi historycznymi.
        y_test_raw: Lista/tablica z danymi, które chcemy przewidzieć (służy też do aktualizacji wiedzy modelu).
        step_length: Co ile próbek aktualizujemy kontekst modelu o prawdziwe dane.
    """
    print('CUDA dostepne:', torch.cuda.is_available())
    print('Wersja:', torch.__version__)

    step_length = int(step_length)
    pipeline = get_pipeline()
    
    # 1. Przygotowanie danych (Spłaszczenie do 1D numpy array)
    np_train = np.array(y_train_raw).flatten()
    np_test = np.array(y_test_raw).flatten()
    
    # Obiekt do przechowywania narastającej historii (Train + fragmenty Test)
    # Używamy listy dla szybkości, potem będziemy konwertować do DataFrame w pętli
    current_history = np_train.tolist()
    
    # Lista na wyniki
    all_predictions = []
    
    # Liczba pętli
    num_loops = int(np.ceil(len(np_test) / step_length))
    
    print(f"[Python Chronos] Start Expanding Window. Kroki: {num_loops}, Step: {step_length}")
    
    for i in range(num_loops):
        if i%10 == 0:
            print(f"[Python Chronos] Krok {i+1}/{num_loops}")
        # --- A. PRZYGOTOWANIE KONTEKSTU ---
        # Tworzymy DataFrame z aktualnej historii
        context_df = pd.DataFrame({
            "target": current_history,
            "timestamp": range(len(current_history)), # Sztuczny czas 0, 1, 2...
            "id": "seria_1"
        })
        
        # --- B. PREDYKCJA ---
        # Określamy ile przewidzieć. Zawsze przewidujemy 'step_length', 
        # nawet w ostatniej pętli (potem przytniemy nadmiar).
        with torch.inference_mode():
            forecast_df = pipeline.predict_df(
                context_df,
                prediction_length=step_length,
                quantile_levels=[0.5], # Tylko mediana
                id_column="id",
                timestamp_column="timestamp",
                target="target"
            )
        
        # --- C. WYCIĄGNIĘCIE WYNIKU ---
        # Szukamy kolumny z medianą (może być 'forecast_0.5', 'q_0.5' lub 'median')
        # Chronos-2 zwraca często "fcst" lub nazwy kwantyli.
        # W predict_df dla amazon/chronos-2 kolumny to zazwyczaj quantile level float.
        result_values = forecast_df['0.5'].values
 
        # Dodajemy predykcję do listy wyników
        all_predictions.extend(result_values)
        
        # --- D. AKTUALIZACJA (EXPAND) ---
        # Dodajemy PRAWDZIWE dane z y_test do historii, żeby w kolejnym kroku model o nich wiedział
        start_idx = i * step_length
        end_idx = min((i + 1) * step_length, len(np_test))
        
        true_chunk = np_test[start_idx:end_idx]
        current_history.extend(true_chunk)
        
        # Logowanie postępu (opcjonalne, bo spowalnia w MATLABie)
        # print(f"  Pętla {i+1}/{num_loops} zakończona.")

    # --- KONIEC ---
    # Przycinamy wynik, jeśli w ostatniej pętli przewidzieliśmy więcej niż zostało danych testowych
    final_prediction = all_predictions[:len(np_test)]
    
    # Zwracamy listę (MATLAB sam przekonwertuje to na double vector)
    return [float(x) for x in final_prediction]