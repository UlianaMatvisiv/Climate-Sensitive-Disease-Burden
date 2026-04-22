import pandas as pd
from pathlib import Path
import numpy as np
SCRIPT_DIR = Path(__file__).resolve().parent
WEATHER_PATH = (SCRIPT_DIR.parent / "DATA" / "weather.csv").resolve()
HEALTH_PATH  = (SCRIPT_DIR.parent / "DATA" /"IHME.csv").resolve()
weather = pd.read_csv(WEATHER_PATH, index_col=0)
weather_df = weather[weather['year'] <= 2023]
health_df = pd.read_csv(HEALTH_PATH, index_col=0)

health  = health_df.drop(columns = ['Deaths', 'Incidence'])
health_wide = health.pivot_table(
    index=["country","year","income_level_mode", 'income_level'], columns="cause",
    values=["DALYs"], aggfunc="first").reset_index()
health_wide.columns = [
    f"{c[0]}_{c[1].replace(' ','_').replace('&','and').replace('/','_')}" if c[1] else c[0]
    for c in health_wide.columns]
df_wide = weather_df.merge(health_wide, on=["country","year"], how="inner")

var_directions = {
    # Heat stress
    "mean_temp":                  "BOTH",
    "max_temp":                   "HIGH",
    "anomaly_heat_days":          "HIGH",
    "heat_episodes":              "HIGH",
    "extreme_heat_episodes":      "HIGH",
    "extreme_area":               "HIGH",
    "caution_area":               "HIGH",
    "mean_caution_area":          "HIGH",
    "mean_extreme_area":          "HIGH",
    "max_heat_intensity":         "HIGH",
    "max_rolling_heat_stress":    "HIGH",
    "total_tropical_nights":      "HIGH",
    # Cold stress
    "anomaly_cold_days":          "HIGH",
    "annual_freeze_burden":       "HIGH",
    "min_temp":                   "LOW",
    "winter_mean_snow_density":   "HIGH",
    "mean_cold_area":             "HIGH",
    # Precipitation / Drought
    "heavy_rain_days":            "HIGH",
    "extreme_rain_days":          "HIGH",
    "max_year_precip":            "HIGH",
    "annual_dry_area":            "HIGH",
    "drought_episodes":           "HIGH",
    "mean_dry_area":              "HIGH",
    "evap_deficit":               "HIGH",
    # Runoff / Flood
    "runoff_days":                "HIGH",
    "max_runoff":                 "HIGH",
    # Air quality
    "pm2.5_mean":                 "HIGH",
    "ozone_mean":                 "HIGH",
    "no2_mean":                   "HIGH",
    # Wind / Pressure
    "max_monthly_wind_speed":     "HIGH",
    "mean_annual_wind_speed":     "HIGH",
    "pressure_variability":       "HIGH",
    # Moisture / Evaporation
    "mean_soil_moisture":         "LOW",
    "min_monthly_soil_moisture":  "LOW",
    "annual_total_evap":          "BOTH",
    "annual_potential_evap":      "HIGH",
    "mean_annual_rsn":            "BOTH",
    # Energy fluxes
    "mean_annual_net_solar":      "BOTH",
    "mean_annual_net_thermal":    "BOTH",
    "mean_latent_heat":           "BOTH",
    "mean_sensible_heat":         "BOTH"
}

CLIMATE_VARS = {
    "Temperature / Heat": [
        "mean_temp", "max_temp", "anomaly_heat_days",
        "heat_episodes", "extreme_heat_episodes",
        "extreme_area", "caution_area",
        "mean_caution_area", "mean_extreme_area",
        "max_heat_intensity", "max_rolling_heat_stress",
        "total_tropical_nights"
    ],

    "Cold / Freeze": [
        "anomaly_cold_days", "annual_freeze_burden",
        "min_temp", "winter_mean_snow_density",
        "mean_cold_area"
    ],

    "Air Pollution": [
        "pm25_mean", "ozone_mean", "no2_mean"
    ],

    "Precipitation / Drought": [
        "heavy_rain_days", "extreme_rain_days",
        "max_year_precip", "annual_dry_area",
        "drought_episodes", "mean_dry_area",
        "evap_deficit"
    ],

    "Runoff / Flood": [
        "runoff_days", "max_runoff"
    ],

    "Wind / Pressure": [
        "max_monthly_wind_speed", "mean_annual_wind_speed",
        "pressure_variability"
    ],

    "Soil / Moisture": [
        "mean_soil_moisture", "min_monthly_soil_moisture",
        "annual_total_evap", "annual_potential_evap"
    ],

    "Energy Fluxes": [
        "mean_annual_net_solar", "mean_annual_net_thermal",
        "mean_latent_heat", "mean_sensible_heat"
    ],
}
metrics = ["Deaths", "DALYs", "Incidence"]
pct_high = 95
pct_low  = 5
records = []
for country, grp in weather_df.groupby("country"):
    for var, direction in var_directions.items():
        if var not in grp.columns:
            continue
        series = grp[var].dropna()
        if len(series) < 5:
            continue
        records.append({
            "country":      country,
            "variable":     var,
            "direction":    direction,
            "thresh_high":  np.percentile(series, pct_high),
            "thresh_low":   np.percentile(series, pct_low),
            "mean":         series.mean(),
            "std":          series.std(),
            "n_obs":        len(series),
        })
if not records:
    percentiles = pd.DataFrame()
else:
    percentiles = pd.DataFrame(records).set_index(["country", "variable"])

if percentiles.empty:
    extreme_df = pd.DataFrame()
else:
    rows = []
    for (country, var), thresholds in percentiles.iterrows():

        subset = (weather_df[weather_df["country"] == country][["year", var]]
                    .dropna()
                    .copy())
        if subset.empty:
            continue

        for _, row in subset.iterrows():
            val  = row[var]
            year = int(row["year"])

            is_extreme_high = (val >= thresholds["thresh_high"]) and (thresholds["direction"] in ("HIGH", "BOTH"))
            is_extreme_low  = (val <= thresholds["thresh_low"])  and (thresholds["direction"] in ("LOW",  "BOTH"))
            is_extreme = is_extreme_high or is_extreme_low

            if is_extreme_high:
                direction = "HIGH"
            elif is_extreme_low:
                direction = "LOW"
            else:
                direction = "normal"

            rows.append({
                "country":              country,
                "year":                 year,
                "variable":             var,
                "all_time_mean":        thresholds["mean"],
                "all_time_std":         thresholds["std"],
                "thresh_high":          thresholds["thresh_high"],
                "thresh_low":           thresholds["thresh_low"],
                "value":                val,
                "semantic_direction":   thresholds["direction"],
                "deviation_from_mean":  val - thresholds["mean"],
                "z_score":              (val - thresholds["mean"]) / thresholds["std"] if thresholds["std"] > 0 else 0.0,
                "direction":            direction,
                "is_extreme":           is_extreme,
            })
    extreme_df = pd.DataFrame(rows) if rows else pd.DataFrame()

def smooth_income_strict(series, min_run=5):
    s = series.copy().reset_index(drop=True)
    out = s.copy()

    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1

        run_length = j - i
        if run_length < min_run:
            left_val  = s[i-1] if i > 0 else None
            right_val = s[j] if j < len(s) else None

            fill_val = left_val if left_val == right_val else left_val or right_val
            out[i:j] = fill_val

        i = j

    return out.values

NUM_COLS = df_wide.select_dtypes(include="number").columns.tolist()
H_KEYS = [c for c in df_wide.columns if c.startswith('DALYs_')]
C_KEYS = [c for c in NUM_COLS if c not in H_KEYS and c != 'year']
ECON_KEYS = ['gdp', 'health_exp', 'city', 'rural']
urban_df = pd.read_csv('../Data/urban.csv', index_col=0)
urban_df = urban_df[urban_df['country'] != 'WORLD']

ISO_LOOKUP = {
    "Ukraine": "UKR",
    "Bosnia and Herz.": "BIH", 
    "Germany": "DEU",
    "Austria": "AUT",
    "Switzerland": "CHE",
    "Netherlands": "NLD",
    "Denmark": "DNK",
    "Sweden": "SWE",
    "Finland": "FIN",
    "Pakistan": "PAK",
    "Sri Lanka": "LKA",
    "Thailand": "THA",
    "Philippines": "PHL",
    "Japan": "JPN",
    "Canada": "CAN",
    "Australia": "AUS",
    "Brazil": "BRA",
    "Bolivia": "BOL",
    "Colombia": "COL",
    "Cuba": "CUB",
    "Honduras": "HND",
    "Nicaragua": "NIC",
    "Zimbabwe": "ZWE",
    "Malawi": "MWI",
    "Niger": "NER",
    "Burundi": "BDI",
    "Chad": "TCD",
    "Central African Rep.": "CAF",
    "Mozambique": "MOZ",
    "Sierra Leone": "SLE",
    "Madagascar": "MDG",
    "Uganda": "UGA",
    "Gabon": "GAB",
    "South Africa": "ZAF",
    "Botswana": "BWA",
    "Algeria": "DZA",
    "Morocco": "MAR",
    "Tunisia": "TUN",
    "Egypt": "EGY",
    "United States of America": "USA",
    "Kazakhstan": "KAZ",
}
health_df["iso3"] = health_df["country"].map(ISO_LOOKUP)