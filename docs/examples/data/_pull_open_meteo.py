"""Pull real climate data from the Open-Meteo API into checked-in CSV snapshots.

This script is a *reproducibility record*, not part of the test suite or the
notebook execution path. It is run once (manually) to regenerate the CSV files
in this directory; the notebooks then read those local CSVs so that
``uv run poe docs`` never touches the network.

Usage
-----
    uv run python docs/examples/data/_pull_open_meteo.py

Uses ``OPEN_METEO_API_KEY`` (Open-Meteo Pro) if available. The script reads it
from the environment or from a local ``.env`` file at the repository root. Without a key
it falls back to the free public endpoints, which return identical ERA5 / CMIP6
data at lower rate limits.

Data sources
------------
- Historical reanalysis: ERA5 / ERA5-Land (Copernicus / ECMWF) via Open-Meteo.
- Climate projections: CMIP6 HighResMIP (downscaled) via Open-Meteo.
- Marine: ERA5 wave reanalysis via Open-Meteo.
Attribution: data by Open-Meteo.com (CC-BY 4.0) and the underlying providers.
"""

from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import requests

HERE = Path(__file__).parent


def _load_api_key() -> str | None:
    """Return the Open-Meteo API key from the env or the repo-root .env."""
    import os

    key = os.environ.get("OPEN_METEO_API_KEY")
    if key:
        return key
    env_path = HERE.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPEN_METEO_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return None


API_KEY = _load_api_key()
_PRO = API_KEY is not None
_PREFIX = "customer-" if _PRO else ""
print(f"Using {'Pro (customer-*)' if _PRO else 'free public'} endpoints.\n")


def _get(service: str, params: dict) -> dict:
    """GET one Open-Meteo endpoint with the API key and basic retries."""
    url = f"https://{_PREFIX}{service}.open-meteo.com/v1/{service.split('-')[0]}"
    params = dict(params)
    if API_KEY:
        params["apikey"] = API_KEY
    last_err = None
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as err:
            last_err = err
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def _save(df: pd.DataFrame, name: str) -> None:
    path = HERE / name
    df.to_csv(path, index=False)
    print(f"  wrote {name}: {len(df)} rows, {list(df.columns)}")


# --------------------------------------------------------------------------- #
# A1 — regression: daily mean temperature, Reykjavik (oceanic, damped season). #
# --------------------------------------------------------------------------- #
def pull_regression() -> None:
    print("A1  regression: Reykjavik daily mean temperature 2020-2023")
    js = _get(
        "archive-api",
        {
            "latitude": 64.1466,
            "longitude": -21.9426,
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "daily": "temperature_2m_mean",
            "timezone": "UTC",
        },
    )
    df = pd.DataFrame(
        {
            "date": js["daily"]["time"],
            "temperature_2m_mean": js["daily"]["temperature_2m_mean"],
        }
    ).dropna()
    _save(df, "reykjavik_daily_temperature.csv")


# --------------------------------------------------------------------------- #
# A3 — heteroscedastic: hourly solar radiation vs hour of day, Barcelona.      #
#      Midday variance is large (cloud-driven); night is deterministic zero.   #
# --------------------------------------------------------------------------- #
def pull_solar() -> None:
    print("A3  heteroscedastic: Barcelona hourly solar radiation, 2023 Apr-Sep")
    js = _get(
        "archive-api",
        {
            "latitude": 41.3874,
            "longitude": 2.1686,
            "start_date": "2023-04-01",
            "end_date": "2023-09-30",
            "hourly": "shortwave_radiation,cloud_cover",
            "timezone": "Europe/Madrid",
        },
    )
    df = pd.DataFrame(
        {
            "time": js["hourly"]["time"],
            "shortwave_radiation": js["hourly"]["shortwave_radiation"],
            "cloud_cover": js["hourly"]["cloud_cover"],
        }
    ).dropna()
    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    _save(df, "barcelona_solar_radiation.csv")


# --------------------------------------------------------------------------- #
# A4 — Poisson counts: annual hot-day / frost-day counts, Madrid 1960-2023.    #
# --------------------------------------------------------------------------- #
def pull_counts() -> None:
    print("A4  Poisson: Madrid annual hot-day / frost-day counts 1960-2023")
    js = _get(
        "archive-api",
        {
            "latitude": 40.4168,
            "longitude": -3.7038,
            "start_date": "1960-01-01",
            "end_date": "2023-12-31",
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "Europe/Madrid",
        },
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(js["daily"]["time"]),
            "tmax": js["daily"]["temperature_2m_max"],
            "tmin": js["daily"]["temperature_2m_min"],
        }
    ).dropna()
    daily["year"] = daily["date"].dt.year
    annual = (
        daily.groupby("year")
        .agg(
            hot_days_30=("tmax", lambda s: int((s >= 30.0).sum())),
            hot_days_35=("tmax", lambda s: int((s >= 35.0).sum())),
            frost_days=("tmin", lambda s: int((s <= 0.0).sum())),
            n_days=("tmax", "size"),
        )
        .reset_index()
    )
    # keep only complete years
    annual = annual[annual["n_days"] >= 365].drop(columns="n_days")
    _save(annual, "madrid_annual_extreme_days.csv")


# --------------------------------------------------------------------------- #
# A5 — multi-output: correlated marine wave components, Atlantic W of Ireland. #
# --------------------------------------------------------------------------- #
def pull_marine() -> None:
    print("A5  multi-output: Atlantic wave components, Dec 2023 (storm season)")
    js = _get(
        "marine-api",
        {
            "latitude": 53.5,
            "longitude": -11.0,
            "start_date": "2023-12-01",
            "end_date": "2023-12-31",
            "hourly": "wave_height,swell_wave_height,wind_wave_height",
            "timezone": "UTC",
        },
    )
    df = pd.DataFrame(
        {
            "time": js["hourly"]["time"],
            "wave_height": js["hourly"]["wave_height"],
            "swell_wave_height": js["hourly"]["swell_wave_height"],
            "wind_wave_height": js["hourly"]["wind_wave_height"],
        }
    ).dropna()
    _save(df, "atlantic_wave_components.csv")


# --------------------------------------------------------------------------- #
# B1 — climate extrapolation: CMIP6 annual mean temperature, Paris 1950-2050. #
# --------------------------------------------------------------------------- #
def pull_climate() -> None:
    print("B1  climate: Paris CMIP6 annual mean temperature 1950-2050 (7 models)")
    models = [
        "CMCC_CM2_VHR4",
        "FGOALS_f3_H",
        "HiRAM_SIT_HR",
        "MRI_AGCM3_2_S",
        "EC_Earth3P_HR",
        "MPI_ESM1_2_XR",
        "NICAM16_8S",
    ]
    frames = []
    for model in models:
        js = _get(
            "climate-api",
            {
                "latitude": 48.8566,
                "longitude": 2.3522,
                "start_date": "1950-01-01",
                "end_date": "2050-12-31",
                "models": model,
                "daily": "temperature_2m_mean",
            },
        )
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(js["daily"]["time"]),
                "temp": js["daily"]["temperature_2m_mean"],
            }
        ).dropna()
        daily["year"] = daily["date"].dt.year
        annual = daily.groupby("year")["temp"].mean().reset_index()
        annual["model"] = model
        frames.append(annual)
        print(f"    {model}: {len(annual)} years")
    out = pd.concat(frames, ignore_index=True).rename(
        columns={"temp": "annual_mean_temp"}
    )
    _save(out, "paris_climate_projection.csv")


if __name__ == "__main__":
    pull_regression()
    pull_solar()
    pull_counts()
    pull_marine()
    pull_climate()
    print("\nDone.")
