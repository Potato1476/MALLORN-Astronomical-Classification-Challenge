# MALLORN-Astronomical-Classification-Challenge

Astronomical time-series classification for the MALLORN challenge. This repo contains EDA and modeling notebooks that build lightcurve features, extract tsfresh statistics, and train an XGBoost + LightGBM ensemble.

## Project structure

- `eda-step.ipynb` - EDA notebook: data overview, missing values, filter distribution, and basic plotting.
- `gia_bao_xinh_gai_score_06120.ipynb` - Main training notebook (feature engineering + tsfresh + XGB/LGB ensemble).
- `gia-bao-xinh-gai_score_05706.ipynb` - Earlier training notebook version.
- `README.md` - Project overview and usage.

## Dataset layout

Expected dataset files (Kaggle-style input directory):

- `train_log.csv`
- `test_log.csv`
- `split_01/` ... `split_20/`
  - `train_full_lightcurves.csv`
  - `test_full_lightcurves.csv`

The notebooks auto-detect `DATA_DIR` by searching common Kaggle input paths. If running locally, set `DATA_DIR` to the folder that contains the files above.

## Approach (high level)

- **Feature engineering (per filter + global)**: statistics over flux, clipped flux, SNR, cadence, and peak/shape features (rise/decay slopes, asymmetry, AUC, smoothness, local peaks).
- **tsfresh multi-channel features**: build long-form time series across filters and extract EfficientFCParameters.
- **Modeling**: Stratified K-Fold training with XGBoost (GPU histogram) and LightGBM; ensemble via weighted blending and threshold optimization for F1.

## Environment

Typical Python dependencies:

- `numpy`, `pandas`, `scikit-learn`
- `xgboost`, `lightgbm`
- `tsfresh`
- `matplotlib`, `seaborn`

If running on Kaggle, the base environment usually includes most of these. Install missing packages as needed:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm tsfresh matplotlib seaborn
```

## How to run

1. Open `eda-step.ipynb` to explore the data and confirm dataset paths.
2. Open `gia_bao_xinh_gai_score_06120.ipynb` to train models and generate predictions.
3. Adjust `DATA_DIR` if your dataset is stored outside the default Kaggle input path.

## Notes

- The notebooks are designed for the MALLORN dataset structure and may need minor path changes if you reorganize data locally.
- Score values included in notebook filenames reflect experiments at the time of submission.
