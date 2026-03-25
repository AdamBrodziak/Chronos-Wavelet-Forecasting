# Comparative Analysis of Foundation Models and Wavelet-Enhanced Methods in Load Forecasting

## Abstract

This repository contains the official codebase for the study "Comparative Analysis of Foundation Models and Wavelet-Enhanced Methods in Load Forecasting". 

This study rigorously compares the zero-shot Chronos 2 Foundation Model against classical forecasting methods enhanced by Haar Wavelet decomposition for "stair-case" load forecasting. Our findings confirm the raw Foundation Model’s superiority (achieving an MAE of 0.0441), whereas explicit domain-specific decomposition degraded accuracy (best MAE: 0.0703). Unlike classical baselines that deteriorated rapidly over time, Chronos 2 maintained stability across various forecasting horizons. We conclude that manual feature engineering (such as wavelet decomposition) disrupts the latent representation learning of transformers, rendering traditional domain-specific preprocessing redundant in the era of foundation models.

## Repository Structure

The project is built on a hybrid architecture, utilizing MATLAB for workflow orchestration and Live Scripts, and Python for the underlying predictive models and foundation model pipelines.

* `data/`: Contains raw (`.csv`) and processed (`.mat`) datasets.

* `src/matlab/`: MATLAB Live Scripts (`.mlx`) representing various method combinations.

* `src/python/`: Python modules invoked by MATLAB, including the foundation model integrations.

## Experimental Protocol

* **Dataset:** The primary working dataset (`ab`) contains incremental energy load data, transformed into differences (`ab\_diff`). 

* **Models Tested:** Chronos 2 (Zero-shot Foundation Model), SARIMA, ELM, and various Hybrid / Baseline methods.

* **Feature Engineering:** Haar Wavelet filters (e.g., Level 1 and Level 5 decompositions) applied to approximation coefficients.

* **Forecasting Horizons:** The experiments span multiple prediction horizons, specifically 1, 16, 32, and 96 steps/months.


## System Requirements

To successfully reproduce this project, you need both MATLAB and Python environments configured to communicate with each other.


* **MATLAB:** R202X or newer (Ensure you have a version compatible with your Python release).

* **Python:** Python 3.9+ (Tested with PyTorch and Chronos pipelines).

* **Hardware:** A CUDA-enabled GPU is highly recommended for running the Chronos 2 model efficiently.


## Getting Started \& How to Reproduce

### 1. Python Environment Setup

First, prepare the Python environment with the required dependencies.

```bash

cd src/python

python -m venv .venv

source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate

pip install -r requirements.txt

```


### 2. MATLAB Configuration

Open MATLAB and configure the Python interpreter. Run the following command in the MATLAB Command Window, replacing the path with the absolute path to your virtual environment's executable:


```bash

pe = pyenv('Version', 'C:\\path\\to\\your\\project\\src\\python\\.venv\\Scripts\\python.exe');

```

### 3. Running the Experiments

Since there is no single monolithic execution script, experiments are modularized.

1. Open MATLAB and navigate to the src/matlab/ directory.

2. Ensure the data/processed/ directory is in your MATLAB Path.


3. Open and run the specific .mlx file corresponding to the experiment you wish to reproduce. For example, to test the Haar Wavelet Level 1 decomposition paired with Chronos, run:
   - Haar\_A1\_CHRONOS.mlx
4. The scripts will automatically invoke the Python modules located in src/python/python\_modules/ and save the prediction results as .xlsx files in the results/raw\_forecasts/ directory.

## License
This project is licensed under the **MIT License**. 
You are free to use, modify, and distribute this software for both academic and commercial purposes. However, if you use this code, data, or models in your research, **you are required to cite our paper**.
