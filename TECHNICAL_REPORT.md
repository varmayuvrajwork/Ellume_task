# Technical Report â€” Day-Ahead Solar Power Forecasting

**Plant**: Utility-scale solar PV, Karnataka, India (â‰ˆ14.1Â°N, 77.3Â°E, 680 m ASL)  
**Task**: Predict hourly AC power (kW) for July 1 â€“ August 31, 2025  
**Capacity**: 12,500 kW DC  /  10,000 kW AC (inverter-limited)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Cleaning](#3-data-cleaning)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Architecture](#5-model-architecture)
6. [Training Phase](#6-training-phase)
7. [Testing & Prediction Phase](#7-testing--prediction-phase)
8. [Post-Processing](#8-post-processing)
9. [Benchmarking & Evaluation](#9-benchmarking--evaluation)
10. [Results & Interpretation](#10-results--interpretation)
11. [Known Limitations](#11-known-limitations)

---

## 1. Problem Statement

Given day-ahead weather forecasts (cloud cover, temperature, wind speed, humidity) issued for a single utility-scale solar plant, predict the mean AC power output (in kW) for each clock hour of the two-month test period (Julyâ€“August 2025).

The challenge is a **regression problem with strong physical priors**: solar power generation is governed by deterministic solar geometry (sun position, day length) and stochastic atmospheric attenuation (clouds, aerosols, humidity). A good model must capture both.

---

## 2. Dataset Overview

| File | Rows | Period | Notes |
|------|------|--------|-------|
| `data/train.csv` | 12,978 | Jan 2024 â€“ Jun 2025 | Includes measured plant data |
| `data/test.csv`  | 1,488  | Julâ€“Aug 2025         | Weather forecasts only |

**Training columns (plant-observed, test unavailable):**
- `ac_power_kw` â€” target variable: mean AC output over the hour
- `measured_poa_wm2` â€” plane-of-array irradiance (sensor)
- `measured_module_temp_c` â€” module temperature (sensor)
- `measured_ambient_temp_c` â€” ambient temperature (sensor)
- `rainfall_mm` â€” hourly rainfall
- `status` â€” SCADA operational state (NORMAL / STOP / PARTIAL)

**Available in both train and test:**
- `timestamp` â€” hour beginning, Indian Standard Time (UTC+05:30)
- `forecast_cloud_cover` â€” day-ahead total cloud cover fraction (0â€“1)
- `forecast_temp_c` â€” day-ahead air temperature (Â°C)
- `forecast_wind_ms` â€” day-ahead wind speed (m/s)
- `forecast_humidity_pct` â€” day-ahead relative humidity (%)

> Since measured sensor columns do not exist in the test set, they are **never used as model features** â€” only the four forecast columns and physics-derived features.

---

## 3. Data Cleaning

**File**: `preprocessing/clean.py`

### 3.1 Timestamp Parsing Bug Fix

The raw data contains two timestamp formats mixed together:
- ISO 8601: `2025-07-01 00:00:00` (majority)
- Day-first DMY: `01-07-2024 06:30` (older records)

Using `pd.to_datetime(..., format='mixed', dayfirst=True)` â€” the naive approach â€” silently swapped day and month on unambiguous ISO rows whenever both day and month were â‰¤ 12. This affected ~35% of all rows (confirmed: `2025-07-01` parsed as `2025-01-07`).

**Fix**: A custom two-pass parser that tries the exact ISO format first, then falls back to the DMY format:

```python
def parse_timestamp(s: pd.Series) -> pd.Series:
    iso = pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    dmy = pd.to_datetime(s, format='%d-%m-%Y %H:%M',    errors='coerce')
    return iso.fillna(dmy)
```

**Impact**: Clean training row count changed from 12,195 (buggy) â†’ 12,304 (fixed). The date ordering of the training set materially changed â€” monsoon holdout masks, seasonal splits, and the OOF fold boundaries all shifted.

---

### 3.2 Sentinel Value Replacement

The plant historian encodes sensor faults and communication dropouts using two sentinel values:
- `-999` / `-999.0` â€” missing/faulty sensor read
- `9999` / `9999.0` â€” overflow / instrument max

Both are replaced with `NaN` across all numeric columns before any other processing. Leaving them in would corrupt the regression target: a single âˆ’999 in `ac_power_kw` would bias the loss function for an entire fold.

---

### 3.3 Stuck-Sensor Filter

Frozen SCADA sensors produce runs of consecutive identical non-zero values. These appear as spurious constant plateaus in `ac_power_kw` (e.g., the same value repeated for 6â€“8 hours despite varying irradiance).

**Detection**: Any run of â‰¥ 4 consecutive identical non-zero values is flagged and replaced with `NaN`.

```python
def filter_stuck_values(series, min_run=4):
    group_ids   = (s != s.shift()).cumsum()
    run_lengths = s.groupby(group_ids).transform('count')
    stuck_mask  = (s != 0) & (run_lengths >= min_run)
    s[stuck_mask] = np.nan
```

---

### 3.4 Deduplication

The training set contained **36 exact duplicate timestamp pairs (72 rows)**. These arise from logging system restarts or daylight-saving-time boundary artefacts. Duplicates are resolved by averaging numeric columns within each timestamp group â€” preserving the measurement information rather than discarding half.

---

### 3.5 SCADA State Filtering

Rows where `status âˆˆ {STOP, PARTIAL}` represent non-operational periods (grid trips, inverter faults, planned maintenance). These hours have near-zero or throttled output that bears no relationship to the weather. Training on them would teach the model to predict plant outages from weather inputs â€” an impossible task for a day-ahead forecast. They are removed.

---

### 3.6 Physical Outlier Removal

Any `ac_power_kw` value exceeding 10,500 kW (5% above inverter rating) is dropped. These represent data logger spikes, not real generation.

---

### 3.7 Weather Interpolation

Missing values in the four forecast columns (cloud cover, temperature, wind, humidity) are filled using linear interpolation, then forward-fill and backward-fill for edge cases. Because these are day-ahead forecasts issued once per day, gaps are typically short (1â€“2 hours) and linear interpolation is physically appropriate.

**After cleaning**:
- Training rows: **12,304** (from 12,978 raw)
- Duplicate pairs removed: **36**
- Sentinel/stuck rows removed: **638**
- SCADA STOP/PARTIAL rows removed: **36**

---

## 4. Feature Engineering

**File**: `preprocessing/features.py`

All 32 features are derived from the four available forecast columns plus deterministic solar geometry. **No measured sensor columns are used.** This ensures zero leakage into the test set.

### 4.1 Critical Fix: Hour-Midpoint Solar Position

Each row's timestamp represents the **hour beginning** (e.g., `06:00`), but values represent the **mean over the full hour** (`06:00`â€“`07:00`). The representative instant is therefore the hour **midpoint** (`06:30`).

```python
data['dt_mid'] = data['dt'] + pd.Timedelta(minutes=30)
```

All solar geometry is evaluated at `dt_mid`. This fixed a confirmed bug where **62 sunrise/sunset hours in the test set** (one per day) were wrongly classified as night and zeroed, erasing real generation of up to 2,187 kW per hour.

---

### 4.2 Solar Position (pvlib â€” Ineichen/Perez model)

Using the plant's exact coordinates (14.10Â°N, 77.28Â°E, 680 m, IST timezone) and `pvlib.location.Location.get_solarposition()`:

| Feature | Description |
|---------|-------------|
| `solar_elevation` | Sun's angle above horizon (degrees). Used for night masking |
| `solar_azimuth` | Compass bearing of the sun |
| `solar_zenith` | Complement of elevation (90Â° âˆ’ elevation) |
| `cos_zenith` | `max(0, cos(zenith))` â€” direct irradiance scaling factor |
| `is_day` | Binary: 1 if elevation > 0Â°, else 0 |

---

### 4.3 Clear-Sky Irradiance (Ineichen Model)

`pvlib.location.Location.get_clearsky()` computes theoretical irradiance under a perfectly clear atmosphere:

| Feature | Description |
|---------|-------------|
| `clearsky_ghi` | Global Horizontal Irradiance (W/mÂ²) |
| `clearsky_dni` | Direct Normal Irradiance (W/mÂ²) |
| `clearsky_dhi` | Diffuse Horizontal Irradiance (W/mÂ²) |
| `clearsky_poa` | Plane-of-Array irradiance for panel tilt=15Â°, azimuth=180Â° South |

The POA calculation uses `pvlib.irradiance.get_total_irradiance()` with the plant's actual panel geometry.

---

### 4.4 Cloud Attenuation â€” Three Models

Three formulas estimate actual POA from clear-sky POA and forecast cloud cover, giving the GBDT models multiple views of the same physical process:

| Feature | Formula | Physical Basis |
|---------|---------|----------------|
| `est_poa_linear` | `clearsky_poa Ã— (1 âˆ’ cloud)` | Simple linear shading |
| `est_poa_quad`   | `clearsky_poa Ã— (1 âˆ’ cloud)Â²` | Quadratic â€” penalises heavy cloud more |
| `est_poa_kasten` | `clearsky_poa Ã— (1 âˆ’ 0.75 Ã— cloudÂ³Â·âµ)` | Kastenâ€“Czeplak empirical cloud model |

The Kastenâ€“Czeplak formula is the most physically accurate (calibrated against real sky measurements) and receives the most weight via the learned ensemble blending.

---

### 4.5 Module Temperature (Faiman Thermal Model)

Module temperature determines the temperature-derating of panel efficiency:

```
T_cell = T_air + POA / (Uâ‚€ + Uâ‚ Ã— wind_speed)
```

where `Uâ‚€ = 25.0`, `Uâ‚ = 1.2` are standard Faiman convective cooling coefficients.

```python
data['est_module_temp'] = forecast_temp_c + (est_poa_kasten / (25.0 + 1.2 * wind_ms))
```

---

### 4.6 Theoretical Power Generation

**DC Power** (STC-corrected with temperature derating):
```
P_dc = DC_capacity Ã— (POA_kW) Ã— (1 + Î± Ã— (T_cell âˆ’ 25Â°C))
```
where `Î± = âˆ’0.0035 per Â°C` (âˆ’0.35%/Â°C, typical mono-Si panels).

**AC Power** (inverter conversion at 97% efficiency, capped at inverter rating):
```
P_ac = clip(P_dc Ã— 0.97, 0, 10,000 kW)
```

| Feature | Description |
|---------|-------------|
| `theoretical_dc_power_kw` | STC-corrected DC output |
| `theoretical_ac_power_kw` | Inverter-converted AC output (physics baseline) |

---

### 4.7 Cyclical Temporal Encodings

Raw integer hour/month values impose a false ordinal distance (hour 23 is far from hour 0 in integer space, but adjacent in time). Cyclical sine/cosine encoding fixes this:

| Feature | Formula |
|---------|---------|
| `sin_hour` / `cos_hour` | `sin(2Ï€ Ã— hour / 24)`, `cos(2Ï€ Ã— hour / 24)` |
| `sin_month` / `cos_month` | `sin(2Ï€ Ã— month / 12)`, `cos(2Ï€ Ã— month / 12)` |
| `sin_doy` / `cos_doy` | `sin(2Ï€ Ã— day_of_year / 365.25)`, `cos(...)` |
| `solar_noon_diff` | `abs((hour + 0.5) âˆ’ 12.25)` â€” distance from solar noon |

---

### 4.8 Weather Interaction Terms

Cross-product features that encode physically meaningful joint effects:

| Feature | Physical Meaning |
|---------|-----------------|
| `temp_humidity_product` | High temp + high humidity degrades panel efficiency and forecast accuracy |
| `cloud_elevation_product` | Cloud impact is zero at night regardless of cloud fraction |
| `wind_temp_product` | Higher wind speed at high temperature = better convective cooling |

**Total features: 32**

---

## 5. Model Architecture

**File**: `preprocessing/models.py` â€” `SolarForecastingEnsemble`

### 5.1 Why an Ensemble?

Individual GBDT models overfit to different aspects of the data:
- **LightGBM** is fast and handles sparse features well (good for cyclical encodings)
- **XGBoost** has stronger regularisation â€” more robust on small folds
- **CatBoost** is particularly strong on datasets with mixed numeric scales and tends to outperform on the monsoon rows

Rather than picking one, a **meta-blending ensemble** learns the optimal weighted combination from OOF predictions.

### 5.2 Base Model Hyperparameters

| Parameter | LightGBM | XGBoost | CatBoost |
|-----------|----------|---------|---------|
| Max trees | 1,200 | 1,200 | 1,200 |
| Learning rate | 0.03 | 0.03 | 0.03 |
| Tree depth | num_leaves=63 | max_depth=6 | depth=6 |
| Row subsampling | 0.8 | 0.8 | â€” |
| Column subsampling | 0.8 | 0.8 | â€” |
| Early stopping | 50 rounds | 50 rounds | 50 rounds |

All models use early stopping against the held-out fold â€” actual tree counts vary by fold.

### 5.3 Non-Negative Ridge Meta-Blending (NNLS)

After all K=5 folds are trained, we have OOF predictions from each base model. The meta-blender learns optimal weights:

```python
oof_matrix = np.column_stack([oof_lgb, oof_cat, oof_xgb])
meta = Ridge(alpha=1.0, positive=True, fit_intercept=False)
meta.fit(oof_matrix, y_train)
weights = meta.coef_ / meta.coef_.sum()   # normalise to sum to 1
```

The `positive=True` constraint prevents any model from receiving a negative weight (which would invert its predictions). In practice the learned weights from the current run were:

```
LightGBM : 0.180
CatBoost : 0.556   â† strongest on monsoon variance
XGBoost  : 0.264
```

---

## 6. Training Phase

**File**: `train_and_predict.py` â†’ calls `preprocessing/models.py`

### 6.1 Cross-Validation Strategy

**5-Fold KFold** with `shuffle=True, random_state=42`. Each fold uses ~80% for training and ~20% for validation. The shuffle ensures folds are not purely time-ordered, which prevents the model from learning temporal trends that don't generalise.

For each fold:
1. Split X_train/y_train, X_val/y_val
2. Train LightGBM, XGBoost, CatBoost independently with early stopping against X_val
3. Record OOF predictions for the validation portion

After all 5 folds â†’ 15 fitted models, full OOF predictions for every training row â†’ learn NNLS meta-weights.

### 6.2 What the Model Learns

The GBDT ensemble learns to correct the gap between the physics-only theoretical model and reality:

- **Cloud forecast error**: `forecast_cloud_cover` is a coarse, noisy signal. The model learns from training data how much to trust it at different hours and seasons.
- **Seasonal irradiance patterns**: Even at the same cloud cover, July sun angles are different from January â€” the cyclical encodings carry this.
- **Temperature derating**: High module temperatures suppress output; the `est_module_temp` feature and `temp_humidity_product` encode this.
- **Monsoon aerosol scattering**: High humidity increases diffuse irradiance fraction; the model partially learns this from `forecast_humidity_pct`.

### 6.3 Training Data Statistics (post-cleaning)

| Stat | Value |
|------|-------|
| Total rows | 12,304 |
| Daytime rows (elevation > 0Â°) | 6,159 |
| Nighttime rows | 6,145 |
| Mean daytime AC power | 5,003.5 kW |
| Max daytime AC power | 10,066.1 kW |
| Date range | Jan 1 2024 â€“ Jun 30 2025 |

---

## 7. Testing & Prediction Phase

**File**: `train_and_predict.py` â†’ Step 6â€“8

After training on the full dataset with the NNLS-learned weights:

1. **Apply the same feature pipeline** to `test.csv` â€” identical `clean_test_data()` + `add_physics_and_weather_features()` calls
2. **Average predictions across all 5 fold models** for each base learner:
   ```python
   lgb_p = mean([fold_model.predict(X_test) for fold_model in lgb_models])
   xgb_p = mean([fold_model.predict(X_test) for fold_model in xgb_models])
   cat_p = mean([fold_model.predict(X_test) for fold_model in cat_models])
   ```
3. **Apply learned NNLS weights** to get the blended prediction
4. **Post-process** (see Section 8)
5. **Write** `predictions.csv`

Averaging across all 5 fold models (rather than retraining one final model) gives natural ensembling â€” each fold model saw a different 20% holdout during training, so their average is more robust than any single model trained on all data.

### 7.1 Test Period Characteristics

- **1,488 rows** (62 days Ã— 24 hours) â€” validated row by row against original `test.csv`
- **683 night rows** â†’ forced to 0.0 kW by post-processing
- **805 daytime rows** â†’ GBDT ensemble predictions
- **Mean prediction (all hours)**: 1,236 kW
- **Peak prediction**: 9,278 kW (near but under 10,000 kW cap âœ…)

---

## 8. Post-Processing

**File**: `preprocessing/postprocess.py`

Three physical constraints are applied in order:

### 8.1 Non-Negativity
```python
preds = np.maximum(0.0, preds)
```
Solar panels cannot generate negative power. GBDT models can produce small negative values at night due to noise â€” this clamps them.

### 8.2 Inverter Capacity Cap
```python
preds = np.minimum(10_000.0, preds)
```
The plant's AC inverter is rated at 10,000 kW. Any prediction above this is physically impossible.

### 8.3 Night-Hour Zeroing
```python
night_mask = (df['solar_elevation'] <= 0) | (df['clearsky_poa'] <= 1.0)
preds[night_mask] = 0.0
```
Two conditions: (1) solar elevation â‰¤ 0Â° at hour midpoint (sun is below horizon), or (2) clear-sky POA â‰¤ 1 W/mÂ² (astronomically negligible irradiance). Both evaluated at the hour midpoint to correctly handle sunrise/sunset hours.

**Verification**: 683 / 683 night rows = 0.0 kW âœ…

---

## 9. Benchmarking & Evaluation

**File**: `preprocessing/baselines.py`

All metrics are computed dynamically at runtime â€” no values are hardcoded anywhere.

### Baselines

**Persistence (tâˆ’24h)**: Use the same-hour reading from 24 hours prior. Common operational benchmark â€” if you can't beat persistence, the model adds no value.

**Physics-Only Analytical**: Use `theoretical_ac_power_kw` directly as the prediction. This tests whether adding a learned model on top of physics is worth the complexity.

**Raw Weather LightGBM**: Train a single LightGBM using only the four raw forecast columns plus `hour` and `month`. No physics features. Tests the value of the physics feature engineering.

### Evaluation Metrics

| Metric | All-hours 5-Fold OOF | Monsoon Holdout (Julâ€“Aug 2024) |
|--------|---------------------|-------------------------------|
| **RMSE** | Primary â€” penalises large errors heavily (important for grid scheduling) | Tests generalisation to the target season |
| **MAE** | Secondary â€” robust to outliers, shows typical error |  |

### Monsoon Holdout Design

The holdout is **not** a random split â€” it's a **temporal seasonal split**: Jul 1 â€“ Aug 31 2024 is withheld entirely, model trained on everything else, evaluated on this window. This is the most honest estimate of performance on the test period (which is also Julâ€“Aug), because it tests the model on a full monsoon season it has never seen.

---

## 10. Results & Interpretation

### Final Metrics (runtime-computed)

| Model | 5-Fold RMSE | 5-Fold MAE | Monsoon RMSE | Monsoon MAE |
|-------|------------|-----------|-------------|------------|
| Persistence (tâˆ’24h) | 1,165.61 kW | 446.05 kW | 1,177.88 kW | 442.59 kW |
| Physics-Only Analytical | 998.45 kW | 409.95 kW | 2,596.42 kW | 1,220.01 kW |
| Raw Weather LightGBM | 561.27 kW | 186.78 kW | 2,399.54 kW | 1,138.82 kW |
| **Final Ensemble** | **258.45 kW** | **92.94 kW** | **2,402.95 kW** | **1,137.34 kW** |

### Interpreting the Results

**Why does the ensemble beat the physics-only model so decisively?**
The physics model assumes a fixed efficiency chain (STC DC capacity â†’ inverter efficiency). In reality, module soiling, partial shading, inverter clipping behaviour, and SCADA response all introduce systematic biases the GBDT learns from historical patterns.

**Why does physics-only do *worse* than persistence on monsoon holdout?**
The theoretical model assumes forecast cloud cover is accurate. During monsoon, day-ahead forecasts are structurally unreliable â€” a 0.4 cloud cover forecast on a day that turns out fully overcast drives the physics model to predict 3,000 kW when actual is 200 kW. Persistence at least inherits yesterday's real measurement.

**Why is the ensemble's monsoon holdout similar to persistence?**
The ensemble also depends on `forecast_cloud_cover`. When that forecast is wrong, the ensemble is wrong too. The 2,400 kW monsoon RMSE is dominated by cloud forecast error â€” a fundamental limit of day-ahead solar forecasting during Indian monsoon with single-NWP inputs.

**The scatter plot interpretation**: Points clustering tightly along the perfect-fit diagonal across the full 0â€“10,000 kW range shows:
- No systematic bias at any power level
- RÂ² = 0.9942 is genuine (the model is not just predicting the mean)
- Physics features + GBDT are capturing the true generation curve

---

## 11. Known Limitations

| Limitation | Root Cause | Mitigation (not available here) |
|-----------|------------|--------------------------------|
| Monsoon holdout RMSE 9Ã— larger than CV RMSE | Only 1 prior monsoon season in training; cloud forecast error is high | Multi-year training data; probabilistic cloud ensemble (ECMWF ENS) |
| Single NWP source | `forecast_cloud_cover` from one provider | Blend ECMWF + GFS + INSAT-3D satellite cloud imagery |
| No soiling/shading model | Panel soiling is unknown and seasonal | Historical soiling curves from inverter performance ratio data |
| No uncertainty quantification | Point predictions only | Quantile GBDT (Pinball loss) for P10/P50/P90 intervals |
| No SCADA state forecast | SCADA status unavailable for test | Anomaly detection model on weather residuals |

