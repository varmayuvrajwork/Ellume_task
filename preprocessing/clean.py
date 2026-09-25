import pandas as pd
import numpy as np


def parse_timestamp(s: pd.Series) -> pd.Series:
    """
    Two-format parser for plant historian timestamps.
    Handles ISO (%Y-%m-%d %H:%M:%S) and DMY (%d-%m-%Y %H:%M) formats
    without the day/month swap that format='mixed', dayfirst=True causes
    on unambiguous ISO rows.
    """
    iso = pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    dmy = pd.to_datetime(s, format='%d-%m-%Y %H:%M', errors='coerce')
    return iso.fillna(dmy)


def filter_stuck_values(series: pd.Series, min_run: int = 4) -> pd.Series:
    """
    Replaces runs of min_run or more consecutive identical non-zero values
    with NaN to remove frozen SCADA sensor stretches.
    """
    s = series.copy()
    non_zero = (s != 0) & (~s.isna())
    group_ids = (s != s.shift()).cumsum()
    run_lengths = s.groupby(group_ids).transform('count')
    stuck_mask = non_zero & (run_lengths >= min_run)
    s[stuck_mask] = np.nan
    return s


def clean_train_data(df: pd.DataFrame, drop_outliers: bool = True) -> pd.DataFrame:
    """
    Prepares the training dataset:
    - Replaces -999 and 9999 sentinels with NaN
    - Filters frozen SCADA sensor stretches (>= 4 identical non-zero values)
    - Parses timestamps with explicit ISO / DMY format resolution
    - Deduplicates timestamps by averaging numeric columns (36 pairs)
    - Filters STOP / PARTIAL non-operational SCADA states
    - Interpolates missing forecast weather variables
    - Drops rows with NaN target and physical outliers (> 10,500 kW)
    """
    data = df.copy()

    sentinels = [-999, -999.0, 9999, 9999.0]
    for col in data.columns:
        if data[col].dtype in [np.float64, np.int64, float, int]:
            data[col] = data[col].replace(sentinels, np.nan)

    data['dt'] = parse_timestamp(data['timestamp'])
    data = data.sort_values('dt').reset_index(drop=True)

    if 'status' in data.columns:
        data['status_clean'] = data['status'].astype(str).str.strip().str.upper()
        data['status_clean'] = data['status_clean'].replace({'NAN': np.nan, 'NONE': np.nan})

    if 'ac_power_kw' in data.columns:
        data['ac_power_kw'] = filter_stuck_values(data['ac_power_kw'], min_run=4)

    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    agg_dict = {col: 'mean' for col in num_cols}
    if 'timestamp' in data.columns:
        agg_dict['timestamp'] = 'first'
    if 'status_clean' in data.columns:
        agg_dict['status_clean'] = 'first'

    data = data.groupby('dt', as_index=False).agg(agg_dict)

    fcst_cols = ['forecast_cloud_cover', 'forecast_temp_c', 'forecast_wind_ms', 'forecast_humidity_pct']
    for col in fcst_cols:
        if col in data.columns:
            data[col] = data[col].interpolate(method='linear').ffill().bfill()

    data = data.dropna(subset=['ac_power_kw']).reset_index(drop=True)

    if drop_outliers:
        if 'status_clean' in data.columns:
            data = data[~data['status_clean'].isin(['STOP', 'PARTIAL'])].reset_index(drop=True)
        data = data[data['ac_power_kw'] <= 10500].reset_index(drop=True)

    return data


def clean_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the test dataset:
    - Preserves original timestamp strings
    - Replaces -999 and 9999 sentinels with NaN
    - Interpolates missing weather forecast values
    - Parses timestamps with explicit two-format parser
    """
    data = df.copy()

    sentinels = [-999, -999.0, 9999, 9999.0]
    for col in data.columns:
        if data[col].dtype in [np.float64, np.int64, float, int]:
            data[col] = data[col].replace(sentinels, np.nan)

    data['dt'] = parse_timestamp(data['timestamp'])
    data = data.sort_values('dt').set_index('dt')

    fcst_cols = ['forecast_cloud_cover', 'forecast_temp_c', 'forecast_wind_ms', 'forecast_humidity_pct']
    for col in fcst_cols:
        if col in data.columns:
            data[col] = data[col].interpolate(method='time').ffill().bfill()

    data = data.reset_index()
    data = data.sort_values('dt').reset_index(drop=True)

    return data
