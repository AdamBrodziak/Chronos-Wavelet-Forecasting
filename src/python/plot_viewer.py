# %%
import os
import json
import re
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Ścieżka do folderu z wynikami
results_dir = Path(r"c:\Users\Asus\Studia\Indywidualny Program Studiów\IPS - Projekt\dane - semestr 5\MATLAB_project\results\pipeline_results")

def find_matching_json(base_name, directory):
    """
    Znajduje odpowiedni plik .json z metrykami na podstawie nazwy pliku .csv z predykcjami.
    Uwzględnia różne formatowania daty w nazwach plików ('-' vs 'T').
    """
    # Próba dopasowania do typowego formatu: Wariant_hXX_YYYY-MM-DD-HH-MM
    m = re.match(r"(.*_h\d+)_(\d{4}-\d{2}-\d{2})[-T](\d{2}-\d{2})", base_name)
    if m:
        prefix = m.group(1)
        date_part = m.group(2)
        time_part = m.group(3)
        
        json_name_1 = f"{prefix}_{date_part}-{time_part}_metrics.json"
        json_name_2 = f"{prefix}_{date_part}T{time_part}_metrics.json"
        
        if (directory / json_name_1).exists():
            return directory / json_name_1
        if (directory / json_name_2).exists():
            return directory / json_name_2
            
    # Metoda zapasowa: szukanie pliku z metrykami rozpoczynającego się od tego samego wariantu i horyzontu
    prefix = base_name.split('_2026')[0] 
    for j_path in directory.glob(f"{prefix}*_metrics.json"):
        return j_path
        
    return None

# %%
# Zbieranie wszystkich plików z predykcjami i ich metryk
csv_files = sorted(results_dir.glob("*_pred.csv"))
data_pairs = []

for csv_path in csv_files:
    base_name = csv_path.name.replace("_pred.csv", "")
    json_path = find_matching_json(base_name, results_dir)
    
    data_pairs.append({
        "name": base_name,
        "csv": csv_path,
        "json": json_path
    })

print(f"Znaleziono {len(data_pairs)} plików z predykcjami w folderze {results_dir.name}.")

# %%
def plot_interactive_predictions(pair):
    """
    Generuje interaktywny wykres Plotly dla danej pary plików (predykcje + metryki).
    """
    name = pair["name"]
    csv_path = pair["csv"]
    json_path = pair["json"]
    
    # 1. Wczytanie predykcji i prawdziwych wartości
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Błąd podczas wczytywania {csv_path.name}: {e}")
        return
        
    # 2. Wczytanie i formatowanie metryk jakości
    metrics_text = ""
    if json_path and json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                metrics_data = json.load(f)
                metrics = metrics_data.get("metrics", {})
                
                # Formatowanie metryk pogrubieniem dla czytelności
                metrics_str = " | ".join([f"<b>{k}</b>: {v:.4f}" for k, v in metrics.items()])
                metrics_text = f"<br><span style='font-size: 13px;'>{metrics_str}</span>"
            except Exception as e:
                metrics_text = f"<br><span style='font-size: 13px; color: red;'>Błąd wczytywania metryk: {e}</span>"
    else:
        metrics_text = "<br><span style='font-size: 13px; color: orange;'>Brak pliku z metrykami w folderze</span>"
        
    # 3. Utworzenie wykresu Plotly
    fig = go.Figure()
    
    # Wartości prawdziwe
    fig.add_trace(go.Scatter(
        y=df['y_true'],
        mode='lines',
        name='Wartości Prawdziwe',
        line=dict(color='#1f77b4', width=2), # Ciemniejszy niebieski
        opacity=0.8
    ))
    
    # Predykcje
    fig.add_trace(go.Scatter(
        y=df['y_pred'],
        mode='lines',
        name='Predykcje',
        line=dict(color='#ff7f0e', width=1.5), # Pomarańczowy
        opacity=0.9
    ))
    
    # Dodatkowa seria: Błąd bezwzględny (domyślnie ukryta, można aktywować w legendzie)
    fig.add_trace(go.Scatter(
        y=abs(df['y_true'] - df['y_pred']),
        mode='lines',
        name='Błąd Bezwzględny',
        line=dict(color='#d62728', width=1, dash='dot'),
        visible='legendonly' 
    ))
    
    # Ekstrakcja nazwy wariantu i horyzontu do tytułu
    m = re.match(r"(.*)_h(\d+)_", name)
    title_variant = f"Wariant: <b>{m.group(1)}</b>, Horyzont: <b>{m.group(2)}</b>" if m else f"Plik: {name}"
    
    # 4. Stylizacja layoutu wykresu
    fig.update_layout(
        title=f"Porównanie Wartości Prawdziwych i Predykcji<br>{title_variant}{metrics_text}",
        xaxis_title="Krok czasowy",
        yaxis_title="Znormalizowana Wartość",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        margin=dict(t=110) # Zwiększony margines u góry na wielolinijkowy tytuł
    )
    
    # Dodanie suwaka czasu na osi X (rangeslider) dla łatwego przybliżania fragmentów sygnału
    fig.update_xaxes(rangeslider_visible=True)
    
    fig.show()

# %%
# Kategoryzacja dostępnych wyników według wariantów i wyświetlenie podsumowania
from collections import defaultdict
variants = defaultdict(list)

for pair in data_pairs:
    m = re.match(r"(.*)_h(\d+)_", pair["name"])
    if m:
        variant_name = m.group(1)
        variants[variant_name].append(pair)
    else:
        variants["Inne"].append(pair)

print("\nDostępne warianty predykcji:")
for v, files in variants.items():
    print(f" - {v} (Liczba plików: {len(files)})")

# %%
# === PRZYKŁADY WYŚWIETLANIA ===

# 1. Wyświetlenie wykresów dla konkretnego wariantu (np. 'Haar-after-LoRA')
target_variant = 'Simple'
if target_variant in variants:
    print(f"\nGenerowanie wykresów dla wariantu: {target_variant}...")
    # Wyświetlamy np. pierwsze 3 z danego wariantu by nie przeciążyć środowiska
    for pair in variants[target_variant][:3]:
        plot_interactive_predictions(pair)
else:
    # Jeżeli wariantu nie ma, rysujemy ostatni z listy wszystkich
    if data_pairs:
        plot_interactive_predictions(data_pairs[-1])

# %%
# 2. Wyświetlenie wyników z konkretnym horyzontem czasowym, np. h=16
# h16_pairs = [p for p in data_pairs if "_h16_" in p["name"]]
# for pair in h16_pairs[:3]:
#     plot_interactive_predictions(pair)
