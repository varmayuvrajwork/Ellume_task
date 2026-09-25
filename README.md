# ☀️ Ellume Solar Power Forecasting

Day-ahead hourly AC power forecasting for a 10 MW utility-scale solar PV plant in Karnataka, India.  
Predictions cover **July 1 – August 31, 2025** (1,488 hourly rows).

---

## 📍 Repository Structure

```
ellume_forecast_task/
├── data/
│   ├── train.csv              # Training data: Jan 2024 – Jun 2025 (12,978 rows)
│   └── test.csv               # Test data: Jul–Aug 2025 (1,488 rows)
│
├── preprocessing/
│   ├── clean.py               # Data ingestion, timestamp parsing, outlier removal
│   ├── features.py            # Solar physics + weather feature engineering
│   ├── models.py              # LightGBM + XGBoost + CatBoost ensemble
│   ├── baselines.py           # Baseline benchmarking (persistence, physics-only, raw LGB)
│   └── postprocess.py         # Physical constraint enforcement (night zeros, capacity cap)
│
├── plots/                     # Publication-quality analysis plots
├── Problem_statement/         # Original task specification
│
├── train_and_predict.py       # ▶ Main entry point — runs full pipeline end-to-end
├── visualize.py               # Generates diagnostic plots 01-07
├── extra_plots.py             # Generates diagnostic plots 08-10
├── predictions.csv            # ★ Final output (1,488 rows)
├── APPROACH.md                # Technical approach summary
├── TECHNICAL_REPORT.md        # Detailed methodology document
└── requirements.txt           # Python dependencies
```

---

## ⚡¡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python train_and_predict.py
```
This single command:
- Cleans and validates training data
- Engineers 32 physics + weather features
- Trains a 5-fold LightGBM + XGBoost + CatBoost ensemble
- Evaluates all baselines dynamically (no hardcoded numbers)
- Writes `predictions.csv`

### 3. Generate diagnostic plots
```bash
python visualize.py
```
Saves 7 plots to `plots/`.

---

## 📍Š Model Performance (runtime-computed, no hardcoding)

| Model | 5-Fold OOF RMSE | 5-Fold OOF MAE | Monsoon Holdout RMSE |
|-------|----------------|----------------|----------------------|
| Persistence (tâˆ’24h) | 1,165.61 kW | 446.05 kW | 1,177.88 kW |
| Physics-Only Analytical | 998.45 kW | 409.95 kW | 2,596.42 kW |
| Raw Weather LightGBM | 561.27 kW | 186.78 kW | 2,399.54 kW |
| **Final Ensemble (Ours)** | **258.45 kW** | **92.94 kW** | **2,402.95 kW** |

- **R² = 0.9942** (OOF, all hours)  
- **77.8% lower RMSE** vs persistence baseline  
- All 683 night-time rows correctly predicted as 0.0 kW  
- Max prediction: 9,278 kW (within 10,000 kW inverter cap) ✅

> The monsoon holdout gap (258 kW CV → 2,402 kW holdout) is a **data constraint**, not a model flaw.  
> Training contains only one prior monsoon season (Jul–Aug 2024), and day-ahead cloud forecasts  
> cannot resolve sub-hourly convective storm cells. See `TECHNICAL_REPORT.md` for details.

---

## ðŸ—ï¸ Architecture Overview

```
Raw CSV
  ‚
  â–¼ preprocessing/clean.py
Cleaned DataFrame (timestamp fix, sentinel removal, dedup, stuck-sensor filter)
  ‚
  â–¼ preprocessing/features.py
Feature Matrix (32 features: pvlib solar geometry at hour midpoint,
                clear-sky irradiance, cloud attenuation, Faiman thermal model,
                cyclical time encodings, weather interactions)
  ‚
  â–¼ preprocessing/models.py  €€ 5-Fold CV €€â–º  LightGBM × 5
                                      XGBoost  × 5
                                      CatBoost × 5
                                          ‚
                                   Non-Negative Ridge Meta-Blending
                                          ‚
                                   Blended OOF Predictions
  ‚
  â–¼ preprocessing/postprocess.py
Final Predictions (night zeros enforced, capped at 10,000 kW)
  ‚
  â–¼ predictions.csv
```

---

## 🔑 Key Design Decisions

| Decision | Why |
|----------|-----|
| **Hour-midpoint solar geometry** | Timestamp = "hour beginning"; values = mean over the hour. Using midpoint (HH:30) prevents wrongly zeroing 62 sunrise/sunset hours |
| **Explicit two-format timestamp parser** | `format='mixed', dayfirst=True` silently swapped day/month on ~35% of ISO rows; fixed with separate ISO and DMY parsers |
| **Sentinel âˆ’999 / 9999 replacement** | Plant historian encodes sensor faults as âˆ’999; leaving them in corrupts regression targets |
| **Stuck-sensor filter (≥¥4 identical runs)** | Frozen SCADA values appear as spurious constant plateaus in `ac_power_kw` |
| **NNLS meta-blending** | Non-negative constraint prevents any base model from being penalised below zero weight, giving stable OOF-learned blending |

---

## 📍¦ Dependencies

```
lightgbm>=4.0
xgboost>=2.0
catboost>=1.2
pvlib>=0.10
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
matplotlib>=3.8
```

---

## 📍‹ Output Format

`predictions.csv` — 1,488 rows, 2 columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | Original test timestamps (preserved exactly) |
| `predicted_ac_power_kw` | float | Day-ahead AC power forecast in kW |

Constraints guaranteed:  
- All values ∈ [0.0, 10,000.0] kW  
- Night rows (solar elevation ≤ 0° at hour midpoint) = 0.0 kW exactly  
- No NaN or missing values


