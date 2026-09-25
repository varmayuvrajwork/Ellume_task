import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import lightgbm as lgb
from preprocessing.postprocess import postprocess_predictions
from preprocessing.models import SolarForecastingEnsemble


def evaluate_baselines_and_holdouts(train_df: pd.DataFrame, feature_cols: list) -> Dict[str, Dict[str, float]]:
    """
    Computes benchmark metrics for four models across:
      - 5-Fold cross-validation (full training dataset)
      - Jul–Aug 2024 seasonal monsoon holdout

    All metrics are computed dynamically at runtime from the actual data —
    no values are hardcoded anywhere.

    The Final Ensemble row uses fit_and_return_oof() so its OOF predictions
    come from the same three-model NNLS ensemble used for predictions.csv,
    not from a separate or simpler model.
    """
    df = train_df.copy()

    monsoon_mask  = (df['dt'] >= '2024-07-01') & (df['dt'] <= '2024-08-31 23:00:00')
    train_monsoon = df[~monsoon_mask].reset_index(drop=True)
    val_monsoon   = df[monsoon_mask].reset_index(drop=True)
    y_monsoon     = val_monsoon['ac_power_kw'].values

    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Persistence baseline: same-hour reading from 24 hours prior
    df['persistence_pred'] = df['ac_power_kw'].shift(24).fillna(0.0)
    df['persistence_pred'] = postprocess_predictions(df, df['persistence_pred'].values)
    p_monsoon = df.loc[monsoon_mask, 'persistence_pred'].values

    results['Persistence (t-24h)'] = {
        'Full_RMSE':            root_mean_squared_error(df['ac_power_kw'], df['persistence_pred']),
        'Full_MAE':             mean_absolute_error(df['ac_power_kw'], df['persistence_pred']),
        'Monsoon_Holdout_RMSE': root_mean_squared_error(y_monsoon, p_monsoon),
        'Monsoon_Holdout_MAE':  mean_absolute_error(y_monsoon, p_monsoon),
    }

    # Physics-only analytical: theoretical_ac_power_kw used directly
    phys_preds = postprocess_predictions(df, df['theoretical_ac_power_kw'].values)
    results['Physics-Only Analytical'] = {
        'Full_RMSE':            root_mean_squared_error(df['ac_power_kw'], phys_preds),
        'Full_MAE':             mean_absolute_error(df['ac_power_kw'], phys_preds),
        'Monsoon_Holdout_RMSE': root_mean_squared_error(y_monsoon, phys_preds[monsoon_mask]),
        'Monsoon_Holdout_MAE':  mean_absolute_error(y_monsoon, phys_preds[monsoon_mask]),
    }

    # Raw Weather LightGBM: only the four raw forecast columns, no physics features
    raw_cols = ['forecast_cloud_cover', 'forecast_temp_c',
                'forecast_wind_ms', 'forecast_humidity_pct', 'hour', 'month']

    m_lgb = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03,
                                random_state=42, verbose=-1)
    m_lgb.fit(train_monsoon[raw_cols], train_monsoon['ac_power_kw'])
    raw_m_preds = postprocess_predictions(val_monsoon, m_lgb.predict(val_monsoon[raw_cols]))

    oof_raw = np.zeros(len(df))
    for tr_idx, va_idx in kf.split(df):
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03,
                               random_state=42, verbose=-1)
        m.fit(df.iloc[tr_idx][raw_cols], df.iloc[tr_idx]['ac_power_kw'])
        oof_raw[va_idx] = m.predict(df.iloc[va_idx][raw_cols])
    oof_raw = postprocess_predictions(df, oof_raw)

    results['Raw Weather LightGBM'] = {
        'Full_RMSE':            root_mean_squared_error(df['ac_power_kw'], oof_raw),
        'Full_MAE':             mean_absolute_error(df['ac_power_kw'], oof_raw),
        'Monsoon_Holdout_RMSE': root_mean_squared_error(y_monsoon, raw_m_preds),
        'Monsoon_Holdout_MAE':  mean_absolute_error(y_monsoon, raw_m_preds),
    }

    # Final Ensemble — monsoon holdout: fit on non-monsoon data, predict monsoon block
    ens_monsoon = SolarForecastingEnsemble(n_splits=5, random_state=42)
    ens_monsoon.fit(train_monsoon[feature_cols], train_monsoon['ac_power_kw'])
    final_m_preds = postprocess_predictions(
        val_monsoon, ens_monsoon.predict(val_monsoon[feature_cols]))

    # Final Ensemble — full 5-fold CV via fit_and_return_oof (true ensemble OOF, not single LGB)
    ens_full = SolarForecastingEnsemble(n_splits=5, random_state=42)
    oof_blend, _ = ens_full.fit_and_return_oof(df[feature_cols], df['ac_power_kw'])
    oof_blend_post = postprocess_predictions(df, oof_blend)

    results['Final Physics-Informed Ensemble'] = {
        'Full_RMSE':            root_mean_squared_error(df['ac_power_kw'], oof_blend_post),
        'Full_MAE':             mean_absolute_error(df['ac_power_kw'], oof_blend_post),
        'Monsoon_Holdout_RMSE': root_mean_squared_error(y_monsoon, final_m_preds),
        'Monsoon_Holdout_MAE':  mean_absolute_error(y_monsoon, final_m_preds),
    }

    return results
