import pandas as pd
import numpy as np
import pvlib

from preprocessing.clean import parse_timestamp

PLANT_LAT      = 14.10
PLANT_LON      = 77.28
PLANT_ALT      = 680.0
PLANT_TILT     = 15.0
PLANT_AZIMUTH  = 180.0
DC_CAPACITY_KW = 12500.0
AC_CAPACITY_KW = 10000.0
TEMP_COEFF     = -0.0035   # -0.35% per °C relative to STC (25°C)


def add_physics_and_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives 32 features from solar physics geometry, clear-sky irradiance,
    cloud attenuation, module temperature, theoretical power, and cyclical
    temporal encodings.

    Solar position is evaluated at the hour MIDPOINT (dt + 30 min) because
    each row's timestamp is the hour beginning while values are the mean over
    the full hour. Evaluating at HH:00 instead of HH:30 incorrectly zeroes
    62 sunrise/sunset hours in the test set.
    """
    data = df.copy()

    if 'dt' not in data.columns:
        data['dt'] = parse_timestamp(data['timestamp'])

    data['dt_mid'] = data['dt'] + pd.Timedelta(minutes=30)

    location = pvlib.location.Location(
        latitude=PLANT_LAT,
        longitude=PLANT_LON,
        altitude=PLANT_ALT,
        tz='Asia/Kolkata'
    )

    times_mid = pd.DatetimeIndex(data['dt_mid']).tz_localize('Asia/Kolkata')
    sol_pos   = location.get_solarposition(times_mid)

    data['solar_elevation'] = sol_pos['elevation'].values
    data['solar_azimuth']   = sol_pos['azimuth'].values
    data['solar_zenith']    = sol_pos['zenith'].values
    data['cos_zenith']      = np.maximum(0.0, np.cos(np.radians(sol_pos['zenith'].values)))
    data['is_day']          = (data['solar_elevation'] > 0).astype(int)

    clearsky = location.get_clearsky(times_mid)
    data['clearsky_ghi'] = clearsky['ghi'].values
    data['clearsky_dni'] = clearsky['dni'].values
    data['clearsky_dhi'] = clearsky['dhi'].values

    poa_clearsky = pvlib.irradiance.get_total_irradiance(
        surface_tilt=PLANT_TILT,
        surface_azimuth=PLANT_AZIMUTH,
        solar_zenith=sol_pos['zenith'].values,
        solar_azimuth=sol_pos['azimuth'].values,
        dni=clearsky['dni'].values,
        ghi=clearsky['ghi'].values,
        dhi=clearsky['dhi'].values,
    )
    data['clearsky_poa'] = np.maximum(0.0, poa_clearsky['poa_global'])

    # Three cloud attenuation formulas — linear, quadratic, and Kasten-Czeplak
    cloud = np.clip(data['forecast_cloud_cover'].values, 0.0, 1.0)
    data['est_poa_linear'] = data['clearsky_poa'] * (1.0 - cloud)
    data['est_poa_quad']   = data['clearsky_poa'] * ((1.0 - cloud) ** 2)
    data['est_poa_kasten'] = data['clearsky_poa'] * (1.0 - 0.75 * (cloud ** 3.5))

    # Faiman thermal model: T_cell = T_air + POA / (U0 + U1 * wind)
    u0, u1 = 25.0, 1.2
    wind    = np.maximum(0.0, data['forecast_wind_ms'].values)
    data['est_module_temp'] = data['forecast_temp_c'].values + (
        data['est_poa_kasten'] / (u0 + u1 * wind)
    )

    # STC-corrected DC power with temperature derating, then inverter conversion
    poa_kw      = data['est_poa_kasten'] / 1000.0
    temp_factor = 1.0 + TEMP_COEFF * (data['est_module_temp'] - 25.0)
    data['theoretical_dc_power_kw'] = np.maximum(0.0, DC_CAPACITY_KW * poa_kw * temp_factor)
    data['theoretical_ac_power_kw'] = np.clip(
        data['theoretical_dc_power_kw'] * 0.97, 0.0, AC_CAPACITY_KW
    )

    # Zero all irradiance and power estimates for night rows
    night_cols = ['est_poa_linear', 'est_poa_quad', 'est_poa_kasten',
                  'theoretical_dc_power_kw', 'theoretical_ac_power_kw']
    data.loc[data['solar_elevation'] <= 0, night_cols] = 0.0

    # Cyclical temporal encodings
    hours  = data['dt'].dt.hour.values
    months = data['dt'].dt.month.values
    doys   = data['dt'].dt.dayofyear.values

    data['hour']        = hours
    data['month']       = months
    data['day_of_year'] = doys

    data['sin_hour']  = np.sin(2 * np.pi * hours  / 24.0)
    data['cos_hour']  = np.cos(2 * np.pi * hours  / 24.0)
    data['sin_month'] = np.sin(2 * np.pi * months / 12.0)
    data['cos_month'] = np.cos(2 * np.pi * months / 12.0)
    data['sin_doy']   = np.sin(2 * np.pi * doys   / 365.25)
    data['cos_doy']   = np.cos(2 * np.pi * doys   / 365.25)

    data['solar_noon_diff'] = np.abs((hours + 0.5) - 12.25)

    # Weather interaction terms
    data['temp_humidity_product']   = data['forecast_temp_c']      * data['forecast_humidity_pct']
    data['cloud_elevation_product'] = data['forecast_cloud_cover']  * data['solar_elevation']
    data['wind_temp_product']       = data['forecast_wind_ms']      * data['forecast_temp_c']

    return data
