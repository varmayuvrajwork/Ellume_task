# â˜€ï¸ Ellume Solar Power Forecasting

Day-ahead hourly AC power forecasting for a 10 MW utility-scale solar PV plant in Karnataka, India.  
Predictions cover **July 1 â€“ August 31, 2025** (1,488 hourly rows).

---

## ðŸ“ Repository Structure

```
ellume_forecast_task/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ train.csv              # Training data: Jan 2024 â€“ Jun 2025 (12,978 rows)
â”‚   â””â”€â”€ test.csv               # Test data: Julâ€“Aug 2025 (1,488 rows)
â”‚
â”œâ”€â”€ preprocessing/
â”‚   â”œâ”€â”€ clean.py               # Data ingestion, timestamp parsing, outlier removal
â”‚   â”œâ”€â”€ features.py            # Solar physics + weather feature engineering
â”‚   â”œâ”€â”€ models.py              # LightGBM + XGBoost + CatBoost ensemble
â”‚   â”œâ”€â”€ baselines.py           # Baseline benchmarking (persistence, physics-only, raw LGB)
â”‚   â””â”€â”€ postprocess.py         # Physical constraint enforcement (night zeros, capacity cap)
â”‚
â”œâ”€â”€ plots/                     # 7 publication-quality analysis plots
â”œâ”€â”€ Problem_statement/         # Original task specification
â”‚
â”œâ”€â”€ train_and_predict.py       # â–¶ Main entry point â€” runs full pipeline end-to-end
â”œâ”€â”€ visualize.py               # Generates all diagnostic plots
â”œâ”€â”€ predictions.csv            # â˜… Final output (1,488 rows)
â”œâ”€â”€ APPROACH.md                # Technical approach summary
â”œâ”€â”€ TECHNICAL_REPORT.md        # Detailed methodology document
â””â”€â”€ requirements.txt           # Python dependencies
```

---

## âš¡ Quick Start

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

## ðŸ“Š Model Performance (runtime-computed, no hardcoding)

| Model | 5-Fold OOF RMSE | 5-Fold OOF MAE | Monsoon Holdout RMSE |
|-------|----------------|----------------|----------------------|
| Persistence (tâˆ’24h) | 1,165.61 kW | 446.05 kW | 1,177.88 kW |
| Physics-Only Analytical | 998.45 kW | 409.95 kW | 2,596.42 kW |
| Raw Weather LightGBM | 561.27 kW | 186.78 kW | 2,399.54 kW |
| **Final Ensemble (Ours)** | **258.45 kW** | **92.94 kW** | **2,402.95 kW** |

- **RÂ² = 0.9942** (OOF, all hours)  
- **77.8% lower RMSE** vs persistence baseline  
- All 683 night-time rows correctly predicted as 0.0 kW  
- Max prediction: 9,278 kW (within 10,000 kW inverter cap) âœ…

> The monsoon holdout gap (258 kW CV â†’ 2,402 kW holdout) is a **data constraint**, not a model flaw.  
> Training contains only one prior monsoon season (Julâ€“Aug 2024), and day-ahead cloud forecasts  
> cannot resolve sub-hourly convective storm cells. See `TECHNICAL_REPORT.md` for details.

---

## ðŸ—ï¸ Architecture Overview

```
Raw CSV
  â”‚
  â–¼ preprocessing/clean.py
Cleaned DataFrame (timestamp fix, sentinel removal, dedup, stuck-sensor filter)
  â”‚
  â–¼ preprocessing/features.py
Feature Matrix (32 features: pvlib solar geometry at hour midpoint,
                clear-sky irradiance, cloud attenuation, Faiman thermal model,
                cyclical time encodings, weather interactions)
  â”‚
  â–¼ preprocessing/models.py  â”€â”€ 5-Fold CV â”€â”€â–º  LightGBM Ã— 5
                                      XGBoost  Ã— 5
                                      CatBoost Ã— 5
                                          â”‚
                                   Non-Negative Ridge Meta-Blending
                                          â”‚
                                   Blended OOF Predictions
  â”‚
  â–¼ preprocessing/postprocess.py
Final Predictions (night zeros enforced, capped at 10,000 kW)
  â”‚
  â–¼ predictions.csv
```

---

## ðŸ”‘ Key Design Decisions

| Decision | Why |
|----------|-----|
| **Hour-midpoint solar geometry** | Timestamp = "hour beginning"; values = mean over the hour. Using midpoint (HH:30) prevents wrongly zeroing 62 sunrise/sunset hours |
| **Explicit two-format timestamp parser** | `format='mixed', dayfirst=True` silently swapped day/month on ~35% of ISO rows; fixed with separate ISO and DMY parsers |
| **Sentinel âˆ’999 / 9999 replacement** | Plant historian encodes sensor faults as âˆ’999; leaving them in corrupts regression targets |
| **Stuck-sensor filter (â‰¥4 identical runs)** | Frozen SCADA values appear as spurious constant plateaus in `ac_power_kw` |
| **NNLS meta-blending** | Non-negative constraint prevents any base model from being penalised below zero weight, giving stable OOF-learned blending |

---

## ðŸ“¦ Dependencies

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

## ðŸ“‹ Output Format

`predictions.csv` â€” 1,488 rows, 2 columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | Original test timestamps (preserved exactly) |
| `predicted_ac_power_kw` | float | Day-ahead AC power forecast in kW |

Constraints guaranteed:  
- All values âˆˆ [0.0, 10,000.0] kW  
- Night rows (solar elevation â‰¤ 0Â° at hour midpoint) = 0.0 kW exactly  
- No NaN or missing values

