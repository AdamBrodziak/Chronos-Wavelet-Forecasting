"""
Konfiguracja projektu — stałe, ścieżki, domyślne hiperparametry.

Centralne miejsce dla wszystkich parametrów eksperymentalnych,
żeby każdy pipeline korzystał z tych samych ustawień.
"""

from pathlib import Path

# =============================================================================
# Ścieżki
# =============================================================================
# Bazowy katalog projektu (MATLAB_project/)
PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "pipeline_results"
SUMMARY_DIR = PROJECT_ROOT / "results" / "pipeline_results" / "summary"
MODELS_DIR = PROJECT_ROOT / "models" / "finetuned"


# Pliki .mat
# MAT_VAR_NAME_TRAIN = "ab_diff_train_tab_raw"
# MAT_VAR_NAME_TEST = "ab_diff_test_tab_raw"
MAT_VAR_NAME_TRAIN = "ab_diff_train_norm"
MAT_VAR_NAME_TEST = "ab_diff_test_norm"

# =============================================================================
# Model Chronos 2
# =============================================================================
MODEL_NAME = "amazon/chronos-2"
MODEL_DTYPE = "bfloat16"       # torch.bfloat16
MODEL_DEVICE_MAP = "auto"      # "auto", "cuda", "cpu"

# =============================================================================
# Horyzonty predykcji (w krokach czasowych)
# =============================================================================
# HORIZONS = [1, 16, 96]
HORIZONS = [1,16,96]

# =============================================================================
# Kwantyle do predykcji
# =============================================================================
QUANTILE_LEVELS = [0.5]  # Tylko mediana

# =============================================================================
# Fine-tuning LoRA — domyślne hiperparametry
# =============================================================================
DEFAULT_LORA_CONFIG = {
    "r": 8,              # rank
    "lora_alpha": 16,
    # Można tu dodać: "target_modules": ["self_attention.q", "self_attention.v"]
}

DEFAULT_LORA_TRAINING = {
    "learning_rate": 1e-4,
    "num_steps": 1000,
    "batch_size": 32,
}

# Grid search — przestrzeń hiperparametrów
LORA_GRID = {
    "r": [4, 8, 16],
    "lora_alpha": [8, 16, 32],
    "learning_rate": [1e-4, 5e-5, 1e-5],
    "num_steps": [500, 1000, 2000],
}

# =============================================================================
# Metryki ewaluacyjne
# =============================================================================
METRIC_NAMES = ["MAE", "RMSE", "MAPE", "R2_classic", "R2_alt"]

# =============================================================================
# Nazwy wariantów pipeline
# =============================================================================
VARIANT_SIMPLE = "Simple"
VARIANT_SIMPLE_LORA = "Simple-LoRA"
VARIANT_HAAR_AFTER = "Haar-after"
VARIANT_HAAR_AFTER_LORA = "Haar-after-LoRA"
VARIANT_HAAR_IN = "Haar-in"
VARIANT_HAAR_IN_LORA = "Haar-in-LoRA"
