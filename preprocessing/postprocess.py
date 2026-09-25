import numpy as np
import pandas as pd

AC_CAPACITY_KW = 10000.0


def postprocess_predictions(df: pd.DataFrame, raw_preds: np.ndarray) -> np.ndarray:
    """
    Enforces three physical constraints on raw model output:
    1. Non-negativity  — clamp to 0 kW minimum
    2. Inverter cap    — clamp to 10,000 kW maximum
    3. Night zeroing   — zero any row where solar elevation <= 0° at hour midpoint
                         or clear-sky POA <= 1.0 W/m² (astronomically negligible)
    """
    preds = np.copy(raw_preds)

    preds = np.maximum(0.0, preds)
    preds = np.minimum(AC_CAPACITY_KW, preds)

    if 'solar_elevation' in df.columns:
        night_mask = df['solar_elevation'] <= 0
        if 'clearsky_poa' in df.columns:
            night_mask = night_mask | (df['clearsky_poa'] <= 1.0)
        preds[night_mask] = 0.0

    return preds
