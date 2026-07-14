"""Pull real climate data from the Open-Meteo API into checked-in CSV snapshots.

This script is a *reproducibility record*, not part of the test suite or the
notebook execution path. It is run once (manually) to regenerate the CSV files
in this directory; the notebooks then read those local CSVs so that
``uv run poe docs-build`` never touches the network.

Usage
-----
    uv run python examples/data/_pull_open_meteo.py

Uses ``OPEN_METEO_API_KEY`` (Open-Meteo Pro) if available. The script reads it
from the environment or from a local ``.env`` file at the repository root.
Without a key it falls back to the free public endpoints, which return
identical ERA5 data at lower rate limits.

Data sources
------------
- Historical reanalysis: ERA5 / ERA5-Land (Copernicus / ECMWF) via Open-Meteo.
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
# Poisson counts: annual hot-day / frost-day counts, Madrid 1960-2023.        #
# --------------------------------------------------------------------------- #
def pull_counts() -> None:
    print("Poisson: Madrid annual hot-day / frost-day counts 1960-2023")
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
# Multi-output: correlated marine wave components, Atlantic W of Ireland.     #
# --------------------------------------------------------------------------- #
def pull_marine() -> None:
    print("multi-output: Atlantic wave components, Dec 2023 (storm season)")
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



if __name__ == "__main__":
    pull_counts()
    pull_marine()
    print("\nDone.")
