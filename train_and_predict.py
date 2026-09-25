import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from preprocessing.clean import clean_train_data, clean_test_data
from preprocessing.features import add_physics_and_weather_features
from preprocessing.models import SolarForecastingEnsemble
from preprocessing.postprocess import postprocess_predictions
from preprocessing.baselines import evaluate_baselines_and_holdouts


def main():
    print("=======================================================")
    print(" Day-Ahead Solar Power Generation Forecasting Pipeline")
    print("=======================================================")

    train_path, test_path = 'data/train.csv', 'data/test.csv'
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("Error: data files not found."); sys.exit(1)

    print("\n[Step 1] Loading raw datasets...")
    train_raw = pd.read_csv(train_path)
    test_raw  = pd.read_csv(test_path)
    print(f"  Raw train rows : {len(train_raw)}")
    print(f"  Raw test rows  : {len(test_raw)}")

    print("\n[Step 2] Cleaning data...")
    train_clean = clean_train_data(train_raw, drop_outliers=True)
    test_clean  = clean_test_data(test_raw)
    print(f"  Clean train rows : {len(train_clean)}")
    print(f"  Clean test rows  : {len(test_clean)}")

    print("\n[Step 3] Extracting physics & weather features...")
    train_df = add_physics_and_weather_features(train_clean)
    test_df  = add_physics_and_weather_features(test_clean)

    exclude = {'timestamp', 'dt', 'dt_mid', 'status', 'status_clean',
               'measured_poa_wm2', 'measured_module_temp_c',
               'measured_ambient_temp_c', 'rainfall_mm', 'ac_power_kw'}
    feature_cols = [c for c in test_df.columns if c not in exclude]
    print(f"  Features : {len(feature_cols)}")

    X_train = train_df[feature_cols]
    y_train = train_df['ac_power_kw']

    print("\n[Step 4] Evaluating baselines and monsoon holdout...")
    baseline_results = evaluate_baselines_and_holdouts(train_df, feature_cols)

    print("\n" + "=" * 85)
    print(" BENCHMARK TABLE  (all metrics in kW, computed at runtime)")
    print("=" * 85)
    print(f"{'Model':<36} {'5-Fold RMSE':>11} {'5-Fold MAE':>10} {'Monsoon RMSE':>13} {'Monsoon MAE':>12}")
    print("-" * 85)
    for name, v in baseline_results.items():
        print(f"{name:<36} {v['Full_RMSE']:>11.2f} {v['Full_MAE']:>10.2f} "
              f"{v['Monsoon_Holdout_RMSE']:>13.2f} {v['Monsoon_Holdout_MAE']:>12.2f}")
    print("=" * 85)

    print("\n[Step 5] Training full-dataset ensemble for predictions.csv...")
    ensemble = SolarForecastingEnsemble(n_splits=5, random_state=42)
    oof_blend, cv_metrics = ensemble.fit_and_return_oof(X_train, y_train)

    oof_post = postprocess_predictions(train_df, oof_blend)
    day_mask = train_df['solar_elevation'] > 0
    print("\n=== Post-Processed OOF Performance ===")
    print(f"  Full Dataset RMSE  : {root_mean_squared_error(y_train, oof_post):.2f} kW")
    print(f"  Full Dataset MAE   : {mean_absolute_error(y_train, oof_post):.2f} kW")
    print(f"  Daytime Only RMSE  : {root_mean_squared_error(y_train[day_mask], oof_post[day_mask]):.2f} kW")
    print(f"  Daytime Only MAE   : {mean_absolute_error(y_train[day_mask], oof_post[day_mask]):.2f} kW")

    print("\n[Step 6] Predicting test period (2025-07-01 to 2025-08-31)...")
    raw_preds = ensemble.predict(test_df[feature_cols])

    print("\n[Step 7] Applying physical constraints...")
    final_preds = postprocess_predictions(test_df, raw_preds)

    output_df = pd.DataFrame({
        'timestamp':             test_raw['timestamp'],
        'predicted_ac_power_kw': final_preds,
    })

    print("\n[Step 8] Validating and writing predictions.csv...")
    assert len(output_df) == 1488
    assert list(output_df.columns) == ['timestamp', 'predicted_ac_power_kw']
    assert output_df['predicted_ac_power_kw'].isnull().sum() == 0
    assert (output_df['timestamp'].values == test_raw['timestamp'].values).all()
    assert (output_df['predicted_ac_power_kw'] >= 0.0).all()
    assert (output_df['predicted_ac_power_kw'] <= 10000.0).all()

    output_df.to_csv('predictions.csv', index=False)
    print("  SUCCESS: predictions.csv written.")

    print("\nPrediction summary:")
    print(output_df['predicted_ac_power_kw'].describe().to_string())

    night_count = (test_df['solar_elevation'] <= 0).sum()
    night_zeros = (output_df.loc[test_df['solar_elevation'] <= 0,
                                  'predicted_ac_power_kw'] == 0.0).sum()
    print(f"\nNight-zero check   : {night_zeros}/{night_count} rows == 0.0 kW")

    sr_mask = (test_df['dt'].dt.hour == 6)
    print(f"Sunrise (06:00 IST): avg {output_df.loc[sr_mask, 'predicted_ac_power_kw'].mean():.2f} kW, "
          f"{(output_df.loc[sr_mask, 'predicted_ac_power_kw'] > 0).sum()}/{sr_mask.sum()} non-zero")


if __name__ == '__main__':
    main()
