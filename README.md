# TER 2025 — Anomaly Detection, Forecasting & Interpolation

This repository contains experiments on meteorological time-series data (Météo France). The work covers three tracks:
- Detecting anomalies with deep generative models (VAE + GAN).
- Forecasting temperature with recurrent networks.
- Interpolating missing values with classical methods and sequence models.

## Repository Contents
- `meteofrance.csv` — Source dataset (semicolon separated) used across notebooks.
- `anomaly_vae.ipynb` — Variational Autoencoder for anomaly detection with clustering of flagged points.
- `anomaly_gan.ipynb` — GAN-based anomaly scoring using discriminator outputs and optional clustering.
- `forecast.ipynb` — LSTM/Bidirectional LSTM temperature forecast; exports `forecast_model.h5`.
- `interpolation.ipynb` — Single-station gap filling (linear, spline, Bi-LSTM variants) with RMSE comparisons.
- `interpolation_allstation.ipynb` — Multi-station interpolation prototype; notes on hardware constraints.
- `forecast_model.h5` — Trained forecast model produced by `forecast.ipynb`.

## Data Notes
- File uses separator `;` and includes at least `AAAAMMJJHH`, `U`, `T`, `PSTAT` among the columns.
- Clean/normalize missing values before training; some notebooks inject artificial gaps for evaluation.

## Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
pip install skimpy  # used for quick dataframe summaries
```
If you plan to run notebooks, also install Jupyter:
```bash
pip install notebook
```

## Running the Notebooks
1. Activate the virtual environment (`source .venv/bin/activate`).
2. Launch Jupyter: `jupyter notebook`.
3. Open the notebook of interest and run all cells.
   - For anomaly notebooks, adjust thresholds/percentiles to tune sensitivity.
   - `forecast.ipynb` saves the trained network to `forecast_model.h5`.
   - Interpolation notebooks compare several methods; hardware limits may require downsampling for `interpolation_allstation.ipynb`.

## Suggested Next Steps
- Log metrics (loss curves, RMSE, anomaly counts) to a lightweight tracker for easier comparisons.
- Parameterize common preprocessing so the different notebooks share the same scaling and splitting logic.
