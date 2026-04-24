"""
Expanding Window (Rolling Window) predictor.

Implementuje strategię Expanding Window — model otrzymuje coraz więcej danych
historycznych z każdym krokiem. Po każdej predykcji prawdziwe wartości
są dodawane do kontekstu.

Używany przez WSZYSTKIE pipeline'y do generowania predykcji.
"""

import numpy as np
from chronos import Chronos2Pipeline

from .data_loader import prepare_context_df
from .model_manager import predict

# TODO w przyszłości train i test mogą mieć więcej niż 1 ceche (future_covariates -> covariates)
# TODO może zrobić to w sposób bardziej ogólny?
def rolling_window_predict(
    pipeline: Chronos2Pipeline,
    y_train: np.ndarray,
    y_test: np.ndarray,
    step_length: int,
) -> np.ndarray:
    """
    Expanding Window predictor — model „widzi" coraz więcej danych.

    W każdym kroku:
    1. Tworzy kontekst z aktualnej historii (train + dotychczasowe fragmenty test)
    2. Przewiduje step_length kroków naprzód
    3. Dodaje PRAWDZIWE wartości z y_test do historii (expand)
    4. Powtarza aż do wyczerpania y_test

    Args:
        pipeline: Załadowany Chronos2Pipeline (pretrained lub fine-tuned)
        y_train: 1D numpy array — dane historyczne (kontekst początkowy)
        y_test: 1D numpy array — dane testowe (do rozszerzania kontekstu)
        step_length: Co ile kroków aktualizujemy kontekst

    Returns:
        1D numpy array — predykcje dla y_test (długość = len(y_test)).
    """
    step_length = int(step_length)
    num_loops = int(np.ceil(len(y_test) / step_length))

    # Narastająca historia — zaczynamy od y_train
    current_history = y_train.tolist()

    # Kolekcja wyników
    all_predictions = []

    print(f"[rolling_window] Start Expanding Window: "
          f"{num_loops} kroków, step_length={step_length}, "
          f"train={len(y_train)}, test={len(y_test)}")

    for i in range(num_loops):
        if i % 10 == 0:
            print(f"[rolling_window] Krok {i+1}/{num_loops}")

        # A. Przygotuj kontekst z aktualnej historii
        context_df = prepare_context_df(np.array(current_history))

        # B. Predykcja
        step_predictions = predict(
            pipeline,
            context_df,
            prediction_length=step_length,
        )

        # C. Dodaj predykcję do wyników
        all_predictions.extend(step_predictions.tolist())

        # D. Expand — dodaj PRAWDZIWE dane z y_test
        start_idx = i * step_length
        end_idx = min((i + 1) * step_length, len(y_test))
        true_chunk = y_test[start_idx:end_idx]
        current_history.extend(true_chunk.tolist())

    # Przytnij do dokładnej długości y_test
    # (ostatnia pętla mogła przewidzieć więcej niż zostało danych)
    final_predictions = np.array(all_predictions[:len(y_test)], dtype=np.float64) #TODO float64?

    print(f"[rolling_window] Zakończono. Predykcji: {len(final_predictions)}")
    return final_predictions
