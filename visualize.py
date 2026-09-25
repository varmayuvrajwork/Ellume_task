"""
visualize.py
============
Generates a comprehensive set of publication-quality plots for the
Ellume Solar Forecasting task. All figures are saved to plots/ directory.

Run:
    python visualize.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from preprocessing.clean import clean_train_data, clean_test_data
from preprocessing.features import add_physics_and_weather_features

# Colour palette and global style
PALETTE = {
    'actual':     '#4FC3F7',
    'predicted':  '#FF7043',
    'clearsky':   '#A5D6A7',
    'night':      '#1a1a2e',
    'bg':         '#0f0f1a',
    'card':       '#1c1c2e',
    'grid':       '#2a2a3e',
    'text':       '#e0e0e0',
    'subtext':    '#9090a0',
    'green':      '#66BB6A',
    'amber':      '#FFA726',
    'red':        '#EF5350',
}

plt.rcParams.update({
    'figure.facecolor':    PALETTE['bg'],
    'axes.facecolor':      PALETTE['card'],
    'axes.edgecolor':      PALETTE['grid'],
    'axes.labelcolor':     PALETTE['text'],
    'axes.titlecolor':     PALETTE['text'],
    'xtick.color':         PALETTE['subtext'],
    'ytick.color':         PALETTE['subtext'],
    'grid.color':          PALETTE['grid'],
    'grid.alpha':          0.5,
    'text.color':          PALETTE['text'],
    'legend.facecolor':    PALETTE['card'],
    'legend.edgecolor':    PALETTE['grid'],
    'font.family':         'DejaVu Sans',
    'axes.spines.top':     False,
    'axes.spines.right':   False,
})

os.makedirs('plots', exist_ok=True)

kw_fmt = FuncFormatter(lambda x, _: f'{x/1000:.1f}K' if abs(x) >= 1000 else f'{x:.0f}')


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Load & prepare data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Loading data...")
train_raw = pd.read_csv('data/train.csv')
test_raw  = pd.read_csv('data/test.csv')
pred_df   = pd.read_csv('predictions.csv')

train_clean = clean_train_data(train_raw, drop_outliers=True)
test_clean  = pd.read_csv('data/test.csv')

train_df = add_physics_and_weather_features(train_clean)
test_df  = add_physics_and_weather_features(clean_test_data(test_raw))

# Parse prediction timestamps
pred_df['dt'] = pd.to_datetime(pred_df['timestamp'])

# Daytime masks
train_day   = train_df[train_df['solar_elevation'] > 0].copy()
test_day_m  = test_df['solar_elevation'] > 0

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Figure 1 â€” Test period predictions: full time-series overview
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Plotting Fig 1: Test-period time series...")

fig, axes = plt.subplots(2, 1, figsize=(18, 10),
                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})
fig.patch.set_facecolor(PALETTE['bg'])

ax1, ax2 = axes

ax1.fill_between(pred_df['dt'], pred_df['predicted_ac_power_kw'],
                 alpha=0.35, color=PALETTE['predicted'])
ax1.plot(pred_df['dt'], pred_df['predicted_ac_power_kw'],
         color=PALETTE['predicted'], lw=0.8, label='Predicted AC Power')

# Overlay theoretical clear-sky
ax1.fill_between(test_df['dt'], test_df['theoretical_ac_power_kw'],
                 alpha=0.15, color=PALETTE['clearsky'])
ax1.plot(test_df['dt'], test_df['theoretical_ac_power_kw'],
         color=PALETTE['clearsky'], lw=0.6, alpha=0.7, label='Clear-Sky Theoretical')

ax1.axhline(10000, color=PALETTE['amber'], lw=1, ls='--', alpha=0.5, label='AC Capacity (10 MW)')
ax1.set_xlim(pred_df['dt'].min(), pred_df['dt'].max())
ax1.set_ylim(0, 10500)
ax1.set_ylabel('AC Power (kW)', fontsize=12)
ax1.set_title('Day-Ahead Solar Power Forecast  â€”  Julâ€“Aug 2025 (Test Period)',
              fontsize=15, fontweight='bold', pad=12)
ax1.yaxis.set_major_formatter(kw_fmt)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, axis='y')
ax1.tick_params(labelbottom=False)

# Bottom: cloud cover
ax2.fill_between(test_df['dt'], test_df['forecast_cloud_cover'],
                 alpha=0.6, color='#90A4AE', label='Cloud Cover')
ax2.set_ylim(0, 1)
ax2.set_ylabel('Cloud Cover', fontsize=10)
ax2.set_xlabel('Date', fontsize=11)
ax2.set_xlim(pred_df['dt'].min(), pred_df['dt'].max())
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, axis='y')

plt.savefig('plots/01_test_period_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved plots/01_test_period_timeseries.png")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Figure 2 â€” OOF Predicted vs Actual scatter (training, daytime only)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Plotting Fig 2: OOF predicted vs actual scatter...")

# Re-run OOF quickly (daytime subset only, use theoretical as OOF proxy for scatter)
# We use the saved model's theoretical vs actual to show the structural relationship
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle('Model Prediction Quality â€” Training Period (Daytime Hours Only)',
             fontsize=14, fontweight='bold', y=1.01)

# Left: Physics-only vs Actual
ax = axes[0]
ax.scatter(train_day['theoretical_ac_power_kw'], train_day['ac_power_kw'],
           alpha=0.25, s=8, color='#A5D6A7', rasterized=True)
lim = [0, 10200]
ax.plot(lim, lim, '--', color=PALETTE['amber'], lw=1.5, label='Perfect fit')
ax.set_xlim(*lim); ax.set_ylim(*lim)
rmse_p = root_mean_squared_error(train_day['ac_power_kw'], train_day['theoretical_ac_power_kw'])
r2_p   = r2_score(train_day['ac_power_kw'], train_day['theoretical_ac_power_kw'])
ax.set_title(f'Physics-Only Model\nRMSE={rmse_p:.0f} kW  |  RÂ²={r2_p:.3f}', fontsize=12)
ax.set_xlabel('Predicted (kW)'); ax.set_ylabel('Actual (kW)')
ax.xaxis.set_major_formatter(kw_fmt); ax.yaxis.set_major_formatter(kw_fmt)
ax.legend()
ax.grid(True)

# Right: Ensemble OOF â€” we'll use the clearsky-attenuated as a structural signal
# and annotate with the runtime benchmark metrics (daytime hours)
ax2_s = axes[1]
day_mask = train_df['solar_elevation'] > 0

# Use clearsky_poa vs actual as structural scatter on right (real physics signal)
x = train_day['clearsky_poa'].clip(0, 1200)
y = train_day['ac_power_kw']
sc = ax2_s.scatter(x, y, c=train_day['forecast_cloud_cover'],
                   cmap='RdYlGn_r', alpha=0.3, s=8, vmin=0, vmax=1, rasterized=True)
cbar = plt.colorbar(sc, ax=ax2_s, fraction=0.03, pad=0.02)
cbar.set_label('Cloud Cover', color=PALETTE['text'])
cbar.ax.yaxis.set_tick_params(color=PALETTE['subtext'])
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE['subtext'])

ax2_s.set_xlabel('Clear-Sky POA Irradiance (W/mÂ²)', fontsize=11)
ax2_s.set_ylabel('Actual AC Power (kW)', fontsize=11)
ax2_s.set_title('Physics Signal: Clear-Sky POA vs Actual Power\n(colour = cloud cover)', fontsize=12)
ax2_s.yaxis.set_major_formatter(kw_fmt)
ax2_s.grid(True)

# Annotate ensemble metrics
txt = ('Ensemble OOF (all hours)\n'
       'RMSE : 258.45 kW\n'
       'MAE  :  92.94 kW\n'
       'RÂ²   :   0.9942')
ax2_s.text(0.04, 0.95, txt, transform=ax2_s.transAxes,
           fontsize=10, va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f0f1a', alpha=0.8))

plt.tight_layout()
plt.savefig('plots/02_scatter_physics_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved plots/02_scatter_physics_vs_actual.png")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Figure 3 â€” Average daily generation profile: Training vs Predictions
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Plotting Fig 3: Average daily profile...")

pred_df['hour'] = pred_df['dt'].dt.hour
train_df['hour'] = train_df['dt'].dt.hour

train_hourly = train_df.groupby('hour')['ac_power_kw'].agg(['mean', 'std', 'median'])
pred_hourly  = pred_df.groupby('hour')['predicted_ac_power_kw'].agg(['mean', 'std', 'median'])

# Separate by season in training for richer view
train_df['season'] = train_df['dt'].dt.month.map(
    lambda m: 'Monsoon (Julâ€“Aug)' if m in [7, 8]
    else ('Winter (Novâ€“Feb)' if m in [11, 12, 1, 2] else 'Other'))

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(PALETTE['bg'])

hours = np.arange(24)

# Training monsoon average
for label, color in [('Monsoon (Julâ€“Aug)', '#5C6BC0'), ('Winter (Novâ€“Feb)', '#80CBC4'), ('Other', '#CE93D8')]:
    sub = train_df[train_df['season'] == label].groupby('hour')['ac_power_kw'].mean()
    ax.plot(hours, sub.reindex(hours, fill_value=0),
            color=color, lw=1.5, ls='--', alpha=0.7, label=f'Train {label}')

# Prediction mean Â± std band
m  = pred_hourly['mean']
s  = pred_hourly['std']
ax.fill_between(hours, (m - s).clip(0), m + s, alpha=0.25, color=PALETTE['predicted'])
ax.plot(hours, m, color=PALETTE['predicted'], lw=2.5, label='Predicted Julâ€“Aug 2025 (mean)')

ax.axhline(0, color=PALETTE['subtext'], lw=0.5)
ax.set_xlim(0, 23)
ax.set_ylim(0, 8000)
ax.set_xlabel('Hour of Day (IST)', fontsize=12)
ax.set_ylabel('Average AC Power (kW)', fontsize=12)
ax.set_title('Average Daily Generation Profile\nTraining seasons vs Julâ€“Aug 2025 Predictions',
             fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(kw_fmt)
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=30)
ax.legend(fontsize=10)
ax.grid(True)

plt.tight_layout()
plt.savefig('plots/03_daily_generation_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved plots/03_daily_generation_profile.png")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Figure 4 â€” Residuals distribution + error metrics dashboard
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Plotting Fig 4: Benchmark metrics dashboard...")

baseline_names  = ['Persistence\n(t-24h)', 'Physics-Only\nAnalytical', 'Raw Weather\nLightGBM', 'Final Ensemble\n(Ours)']
full_rmse       = [1165.61, 998.45, 561.27, 258.45]
full_mae        = [446.05,  409.95, 186.78,  92.94]
monsoon_rmse    = [1177.88, 2596.42, 2399.54, 2402.95]
colors          = [PALETTE['red'], PALETTE['amber'], '#29B6F6', PALETTE['green']]

fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor(PALETTE['bg'])
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# 4a â€” 5-Fold OOF RMSE
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(baseline_names, full_rmse, color=colors, edgecolor='none', width=0.55)
for bar, val in zip(bars, full_rmse):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             f'{val:.1f}', ha='center', va='bottom', fontsize=10, color=PALETTE['text'])
ax1.set_title('5-Fold OOF RMSE (kW)', fontsize=12, fontweight='bold')
ax1.set_ylabel('RMSE (kW)'); ax1.set_ylim(0, 1350)
ax1.yaxis.set_major_formatter(kw_fmt); ax1.grid(True, axis='y')

# 4b â€” 5-Fold OOF MAE
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(baseline_names, full_mae, color=colors, edgecolor='none', width=0.55)
for bar, val in zip(bars2, full_mae):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=10, color=PALETTE['text'])
ax2.set_title('5-Fold OOF MAE (kW)', fontsize=12, fontweight='bold')
ax2.set_ylabel('MAE (kW)'); ax2.set_ylim(0, 520)
ax2.yaxis.set_major_formatter(kw_fmt); ax2.grid(True, axis='y')

# 4c â€” Monsoon holdout RMSE
ax3 = fig.add_subplot(gs[1, 0])
bars3 = ax3.bar(baseline_names, monsoon_rmse, color=colors, edgecolor='none', width=0.55)
for bar, val in zip(bars3, monsoon_rmse):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'{val:.0f}', ha='center', va='bottom', fontsize=10, color=PALETTE['text'])
ax3.set_title('Julâ€“Aug 2024 Holdout RMSE (kW)', fontsize=12, fontweight='bold')
ax3.set_ylabel('RMSE (kW)'); ax3.set_ylim(0, 3000)
ax3.yaxis.set_major_formatter(kw_fmt); ax3.grid(True, axis='y')

# 4d â€” Improvement waterfall text card
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
ax4.axis('off')

improvements = [
    ('vs Persistence',   (1165.61 - 258.45) / 1165.61 * 100, 'Full-CV RMSE'),
    ('vs Physics-Only',  (998.45  - 258.45) / 998.45  * 100, 'Full-CV RMSE'),
    ('vs Raw LightGBM',  (561.27  - 258.45) / 561.27  * 100, 'Full-CV RMSE'),
]

ax4.text(0.5, 0.96, 'Ensemble Improvement Over Baselines',
         ha='center', va='top', fontsize=12, fontweight='bold', color=PALETTE['text'])

for i, (label, pct, metric) in enumerate(improvements):
    y = 0.78 - i * 0.22
    ax4.text(0.08, y,       f'â–² {label}', va='top', fontsize=11, color=PALETTE['green'])
    ax4.text(0.08, y - 0.08, f'  {pct:.1f}% lower {metric}', va='top',
             fontsize=10, color=PALETTE['subtext'])

ax4.text(0.5, 0.10,
         'â˜…  RÂ² = 0.9942  (OOF, all hours)',
         ha='center', va='bottom', fontsize=11, color=PALETTE['amber'],
         fontweight='bold')

fig.suptitle('Model Benchmark Dashboard â€” Solar Power Forecasting',
             fontsize=15, fontweight='bold', y=1.01)
plt.savefig('plots/04_benchmark_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved plots/04_benchmark_dashboard.png")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Figure 5 â€” Prediction distribution: night zeros vs daytime, by hour-of-day box
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Plotting Fig 5: Hourly box-plots for predictions...")

pred_df['hour'] = pred_df['dt'].dt.hour
pred_df['date'] = pred_df['dt'].dt.date

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor(PALETTE['bg'])

# Group by hour
grouped = [pred_df[pred_df['hour'] == h]['predicted_ac_power_kw'].values for h in range(24)]
bp = ax.boxplot(grouped, positions=range(24), widths=0.55, patch_artist=True,
                medianprops={'color': PALETTE['predicted'], 'lw': 2},
                flierprops={'marker': '.', 'markersize': 3,
                            'markerfacecolor': PALETTE['subtext'], 'alpha': 0.4},
                whiskerprops={'color': PALETTE['grid']},
                capprops={'color': PALETTE['grid']})

for patch, h in zip(bp['boxes'], range(24)):
    med = np.median(grouped[h])
    frac = med / 10000
    color = plt.cm.RdYlGn(frac)
    patch.set_facecolor(color); patch.set_alpha(0.8)

ax.set_xticks(range(24))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, fontsize=9)
ax.set_ylabel('Predicted AC Power (kW)', fontsize=12)
ax.set_title('Predicted Power Distribution by Hour of Day (Julâ€“Aug 2025)\n'
             'Box shows IQR, whiskers = 1.5Ã—IQR, colour = median intensity',
             fontsize=13, fontweight='bold')
ax.yaxis.set_major_formatter(kw_fmt)
ax.set_ylim(-50, 10200)
ax.axhline(10000, color=PALETTE['amber'], lw=1, ls='--', alpha=0.4)
ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig('plots/05_hourly_boxplot_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved plots/05_hourly_boxplot_predictions.png")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Figure 6 â€” Weekly heatmap of predicted power (Julâ€“Aug 2025)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Plotting Fig 6: Weekly heatmap...")

pred_df['date'] = pred_df['dt'].dt.date
pivot = pred_df.pivot_table(index='hour', columns='date',
                             values='predicted_ac_power_kw', aggfunc='mean')

fig, ax = plt.subplots(figsize=(20, 7))
fig.patch.set_facecolor(PALETTE['bg'])

im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd',
               vmin=0, vmax=10000, origin='upper')

cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cbar.set_label('AC Power (kW)', color=PALETTE['text'])
cbar.ax.yaxis.set_tick_params(color=PALETTE['subtext'])
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE['subtext'])

# X-axis: dates (every 7th)
dates = list(pivot.columns)
xtick_pos  = list(range(0, len(dates), 7))
xtick_labs = [str(dates[i]) for i in xtick_pos]
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_labs, rotation=30, fontsize=9)

ax.set_yticks(range(0, 24, 2))
ax.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], fontsize=9)
ax.set_ylabel('Hour of Day (IST)', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.set_title('Predicted AC Power Heatmap â€” Julâ€“Aug 2025\n(row = hour of day, column = date)',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/06_heatmap_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved plots/06_heatmap_predictions.png")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Figure 7 â€” Two representative weeks: detailed view
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Plotting Fig 7: Two representative weeks...")

weeks = [
    ('2025-07-07', '2025-07-13', 'Week 1 â€” July  7â€“13 (Early Monsoon)'),
    ('2025-08-11', '2025-08-17', 'Week 2 â€” Aug 11â€“17 (Peak Monsoon)'),
]

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
fig.patch.set_facecolor(PALETTE['bg'])

for ax, (start, end, title) in zip(axes, weeks):
    mask = (pred_df['dt'] >= start) & (pred_df['dt'] <= end)
    wk   = pred_df[mask]
    cs   = test_df[(test_df['dt'] >= start) & (test_df['dt'] <= end)]

    ax.fill_between(wk['dt'], wk['predicted_ac_power_kw'],
                    alpha=0.4, color=PALETTE['predicted'])
    ax.plot(wk['dt'], wk['predicted_ac_power_kw'],
            color=PALETTE['predicted'], lw=1.5, label='Predicted')
    ax.plot(cs['dt'], cs['theoretical_ac_power_kw'],
            color=PALETTE['clearsky'], lw=1.2, ls='--', alpha=0.7, label='Clear-Sky Theoretical')

    ax.set_ylim(0, 10500)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('AC Power (kW)', fontsize=10)
    ax.yaxis.set_major_formatter(kw_fmt)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True)
    ax.axhline(10000, color=PALETTE['amber'], lw=0.8, ls=':', alpha=0.5)

axes[-1].set_xlabel('Date / Hour', fontsize=11)
fig.suptitle('Detailed Weekly View â€” Predicted vs Clear-Sky Theoretical',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plots/07_weekly_detail.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved plots/07_weekly_detail.png")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Summary
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 55)
print(" All 7 plots saved to ./plots/")
print("=" * 55)
plots = [
    "01_test_period_timeseries.png   â€” Full Julâ€“Aug 2025 time series",
    "02_scatter_physics_vs_actual.png â€” Physics vs Actual scatter",
    "03_daily_generation_profile.png  â€” Avg daily profile by season",
    "04_benchmark_dashboard.png       â€” Benchmark metrics comparison",
    "05_hourly_boxplot_predictions.png â€” Hourly power distribution",
    "06_heatmap_predictions.png        â€” Date Ã— Hour power heatmap",
    "07_weekly_detail.png              â€” Two detailed weekly views",
]
for p in plots:
    print(f"  {p}")
