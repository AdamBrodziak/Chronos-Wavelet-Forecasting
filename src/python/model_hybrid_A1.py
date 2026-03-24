import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from skelm import ELMRegressor
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def model_hybrid(
    y_train_raw,
    y_test_raw,
    step_length_raw,
    ma_window=8,
    season_period=96,
    verbose=True
):

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

    # --- 2. GŁÓWNA LOGIKA ---
    num_steps_to_predict = len(y_test) // step_length
    all_predictions_list = []

    for current_step in range(num_steps_to_predict):
        print(f"Step: {current_step+1} / {num_steps_to_predict}")

        start = step_length * current_step
        end   = start + step_length

        # dynamiczne powiększanie train
        current_train = pd.concat([y_train, y_test.iloc[:start]])

        # cechy MA + reszta
        data_MA   = current_train.rolling(window=ma_window).mean().fillna(0)
        data_diff = current_train - data_MA

        # --- Holt-Winters ---
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

        # --- CECHY GODZINOWE
        dt_index = pd.to_datetime(current_train.index)
        hour_sin = np.sin(2*np.pi * dt_index.hour / 24)
        hour_cos = np.cos(2*np.pi * dt_index.hour / 24)

        # --- CECHY DLA ELM
        X_for_ELM = pd.concat([
            Y_pred_HW_in_sample,
            current_train.shift(1),
            #data_MA.shift(1),
            pd.Series(hour_sin, index=dt_index),
            pd.Series(hour_cos, index=dt_index)
        ], axis=1).dropna()

        model_ELM = ELMRegressor(n_neurons=3000, ufunc='tanh',alpha=ALPHA)
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

        # autoregresja
        last_y = current_train.value.iloc[-1]
        last_n = data_diff.value.iloc[-1]

        pred_list = []

        for i in range(step_length):
            l_hat = Y_pred_HW_future.iloc[i]

            X_input = [[ #zmiana dla A1
                l_hat,
                last_y,
                #last_n,
                future_hour_sin[i],
                future_hour_cos[i]
            ]]

            n_hat = model_ELM.predict(X_input)[0]
            raw_y = n_hat + l_hat
            #y_hat = np.clip(raw_y, -4.0, 4.0)
            y_hat = raw_y

            pred_list.append(y_hat)

            last_y = y_hat
            last_n = n_hat

        pred_df = pd.DataFrame({'value': pred_list}, index=future_index)
        all_predictions_list.append(pred_df)

    # --- 4. SKŁADANIE CAŁOŚCI ---
    all_predictions = pd.concat(all_predictions_list)
    print("Koniec predykcji.")

    return all_predictions.values
