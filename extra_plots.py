"""
extra_plots.py
==============
Generates three additional diagnostic plots:
  08 — Feature Importance (top 20, averaged across 5-fold LightGBM models)
  09 — Temperature vs Power coloured by Humidity
  10 — Predicted vs Actual error map (5-fold OOF LightGBM)

Run:
    python extra_plots.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import lightgbm as lgb

from preprocessing.clean import clean_train_data
from preprocessing.features import add_physics_and_weather_features

os.makedirs('plots', exist_ok=True)

PALETTE = {
    'bg':      '#0f0f1a',
    'card':    '#1c1c2e',
    'grid':    '#2a2a3e',
    'text':    '#e0e0e0',
    'subtext': '#9090a0',
    'orange':  '#FF7043',
    'green':   '#66BB6A',
    'amber':   '#FFA726',
    'blue':    '#4FC3F7',
}

plt.rcParams.update({
    'figure.facecolor':  PALETTE['bg'],
    'axes.facecolor':    PALETTE['card'],
    'axes.edgecolor':    PALETTE['grid'],
    'axes.labelcolor':   PALETTE['text'],
    'axes.titlecolor':   PALETTE['text'],
    'xtick.color':       PALETTE['subtext'],
    'ytick.color':       PALETTE['subtext'],
    'grid.color':        PALETTE['grid'],
    'grid.alpha':        0.4,
    'text.color':        PALETTE['text'],
    'legend.facecolor':  PALETTE['card'],
    'legend.edgecolor':  PALETTE['grid'],
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

kw_fmt = FuncFormatter(lambda x, _: f'{x/1000:.1f}K' if abs(x) >= 1000 else f'{x:.0f}')

# ── Load & prepare training data ──────────────────────────────────────────
print("Loading and preparing data...")
train_raw   = pd.read_csv('data/train.csv')
train_clean = clean_train_data(train_raw, drop_outliers=True)
train_df    = add_physics_and_weather_features(train_clean)

exclude = {'timestamp', 'dt', 'dt_mid', 'status', 'status_clean',
           'measured_poa_wm2', 'measured_module_temp_c',
           'measured_ambient_temp_c', 'rainfall_mm', 'ac_power_kw'}
feature_cols = [c for c in train_df.columns if c not in exclude]

X = train_df[feature_cols]
y = train_df['ac_power_kw']

# ── 5-Fold LightGBM — collect OOF predictions + feature importances ───────
print("Running 5-fold LightGBM for OOF predictions and feature importance...")
kf       = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
imp_sum   = np.zeros(len(feature_cols))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

    m = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42 + fold, verbose=-1,
    )
    m.fit(X_tr, y_tr,
          eval_X=X_va, eval_y=y_va,
          callbacks=[lgb.early_stopping(40, verbose=False)])
    oof_preds[va_idx] = m.predict(X_va)
    imp_sum += m.feature_importances_

# Night rows: force OOF to zero (same physical rule as postprocess)
night_mask = train_df['solar_elevation'] <= 0
oof_preds  = np.maximum(0.0, oof_preds)
oof_preds  = np.minimum(10000.0, oof_preds)
oof_preds[night_mask] = 0.0

rmse = root_mean_squared_error(y, oof_preds)
mae  = mean_absolute_error(y, oof_preds)
print(f"  OOF RMSE: {rmse:.2f} kW  |  MAE: {mae:.2f} kW")

# ── Plot 08 — Feature Importance ─────────────────────────────────────────
print("Plotting Fig 08: Feature Importance...")

avg_imp   = imp_sum / 5
imp_df    = pd.DataFrame({'feature': feature_cols, 'importance': avg_imp})
imp_df    = imp_df.sort_values('importance', ascending=True).tail(20)

pct       = imp_df['importance'] / imp_df['importance'].sum() * 100
colors    = [PALETTE['orange'] if p > pct.median() else PALETTE['blue'] for p in pct]

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor(PALETTE['bg'])

bars = ax.barh(imp_df['feature'], pct, color=colors, edgecolor='none', height=0.65)
for bar, val in zip(bars, pct):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', fontsize=9, color=PALETTE['text'])

ax.set_xlabel('Relative Importance (%)', fontsize=12)
ax.set_title('Feature Importance — Top 20 Features\n'
             '(averaged across 5 LightGBM folds)',
             fontsize=14, fontweight='bold')
ax.set_xlim(0, pct.max() * 1.18)
ax.grid(True, axis='x')

# Legend
import matplotlib.patches as mpatches
leg = [mpatches.Patch(color=PALETTE['orange'], label='Above-median importance'),
       mpatches.Patch(color=PALETTE['blue'],   label='Below-median importance')]
ax.legend(handles=leg, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('plots/08_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved plots/08_feature_importance.png")

# ── Plot 09 — Temperature vs Power, coloured by Humidity ──────────────────
print("Plotting Fig 09: Temperature vs Power coloured by Humidity...")

day = train_df[train_df['solar_elevation'] > 0].copy()

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle('Temperature vs AC Power — coloured by Humidity\n'
             '(daytime hours only)',
             fontsize=14, fontweight='bold')

for ax, col, label, cmap, note in [
    (axes[0], 'forecast_humidity_pct', 'Relative Humidity (%)', 'RdYlGn_r',
     'High humidity → haze → power loss'),
    (axes[1], 'forecast_cloud_cover',  'Cloud Cover (fraction)',  'RdYlGn_r',
     'High cloud → irradiance loss → power loss'),
]:
    sc = ax.scatter(
        day['forecast_temp_c'],
        day['ac_power_kw'],
        c=day[col],
        cmap=cmap,
        alpha=0.35,
        s=8,
        vmin=day[col].quantile(0.02),
        vmax=day[col].quantile(0.98),
        rasterized=True,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label, color=PALETTE['text'])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE['subtext'])
    cbar.ax.yaxis.set_tick_params(color=PALETTE['subtext'])

    ax.set_xlabel('Forecast Temperature (°C)', fontsize=11)
    ax.set_ylabel('Actual AC Power (kW)',       fontsize=11)
    ax.yaxis.set_major_formatter(kw_fmt)
    ax.set_title(f'Colour = {label}', fontsize=11)
    ax.grid(True)

    ax.text(0.04, 0.97, note, transform=ax.transAxes,
            fontsize=9, va='top', color=PALETTE['amber'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor=PALETTE['bg'], alpha=0.7))

plt.tight_layout()
plt.savefig('plots/09_temperature_vs_power_humidity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved plots/09_temperature_vs_power_humidity.png")

# ── Plot 10 — Predicted vs Actual Error Map ───────────────────────────────
print("Plotting Fig 10: Predicted vs Actual error map...")

residuals = oof_preds - y.values

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle('Predicted vs Actual Power — Error Map\n'
             '(5-fold LightGBM out-of-fold predictions, daytime only)',
             fontsize=14, fontweight='bold')

day_idx = train_df['solar_elevation'] > 0
y_day   = y.values[day_idx]
p_day   = oof_preds[day_idx]
res_day = residuals[day_idx]

# Left: scatter coloured by residual magnitude
ax = axes[0]
sc = ax.scatter(y_day, p_day,
                c=res_day,
                cmap='RdYlGn',
                alpha=0.3, s=6,
                vmin=-2500, vmax=2500,
                rasterized=True)
cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Error (Predicted − Actual) kW', color=PALETTE['text'])
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE['subtext'])
cbar.ax.yaxis.set_tick_params(color=PALETTE['subtext'])

lim = [0, 10200]
ax.plot(lim, lim, '--', color=PALETTE['amber'], lw=1.8,
        label='Perfect prediction line (y = x)')
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel('Actual AC Power (kW)',    fontsize=11)
ax.set_ylabel('Predicted AC Power (kW)', fontsize=11)
ax.xaxis.set_major_formatter(kw_fmt)
ax.yaxis.set_major_formatter(kw_fmt)
ax.set_title('Points on the line = perfect\n'
             'Above = over-predicted  |  Below = under-predicted',
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(True)

ann = (f'RMSE : {rmse:.0f} kW\n'
       f'MAE  : {mae:.0f} kW\n'
       f'R²   : {1 - np.var(residuals)/np.var(y.values):.4f}')
ax.text(0.04, 0.96, ann, transform=ax.transAxes, fontsize=10, va='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=PALETTE['bg'], alpha=0.85))

# Right: residual distribution histogram
ax2 = axes[1]
bins = np.linspace(-4000, 4000, 80)
n, _, patches = ax2.hist(res_day, bins=bins, color=PALETTE['blue'],
                          edgecolor='none', alpha=0.8)
for patch in patches:
    if patch.get_x() < 0:
        patch.set_facecolor(PALETTE['orange'])   # over-predicted = orange
    else:
        patch.set_facecolor(PALETTE['green'])    # under-predicted = green

ax2.axvline(0,         color=PALETTE['amber'], lw=2,   ls='--', label='Zero error')
ax2.axvline(res_day.mean(), color='white',     lw=1.5, ls=':',
            label=f'Mean error: {res_day.mean():.0f} kW')
ax2.set_xlabel('Prediction Error (Predicted − Actual) kW', fontsize=11)
ax2.set_ylabel('Number of Hours',                           fontsize=11)
ax2.set_title('Error Distribution\n'
              'Orange = over-predicted  |  Green = under-predicted',
              fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, axis='y')

pct_within_500  = (np.abs(res_day) < 500).mean()  * 100
pct_within_1000 = (np.abs(res_day) < 1000).mean() * 100
note = (f'{pct_within_500:.1f}% of daytime hours\nwithin ±500 kW\n'
        f'{pct_within_1000:.1f}% within ±1,000 kW')
ax2.text(0.97, 0.96, note, transform=ax2.transAxes, fontsize=10, va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor=PALETTE['bg'], alpha=0.85))

plt.tight_layout()
plt.savefig('plots/10_predicted_vs_actual_error_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved plots/10_predicted_vs_actual_error_map.png")

print("\n" + "=" * 55)
print(" Extra plots complete.")
print("=" * 55)
print("  08_feature_importance.png")
print("  09_temperature_vs_power_humidity.png")
print("  10_predicted_vs_actual_error_map.png")
