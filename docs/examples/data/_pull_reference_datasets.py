"""Pull the public reference datasets used by the notebooks into checked-in files.

This script is a *reproducibility record*, not part of the test suite or the
notebook execution path. It is run once (manually) to regenerate the files in
this directory; the notebooks then read those local copies so that
``uv run poe docs`` never touches the network.

Usage
-----
    uv run --extra docs python docs/examples/data/_pull_reference_datasets.py

The ``--extra docs`` is only needed for the UCI Auto MPG pull, which uses
``ucimlrepo``; the other three pulls need nothing beyond ``pandas`` and
``requests``.

Data sources
------------
- Mauna Loa CO2 (``mauna_loa_co2.csv``): monthly mean atmospheric CO2 measured
  at the Mauna Loa Observatory, Hawaii. Source:
  https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv
  Provider: NOAA Global Monitoring Laboratory (Lan, Tans & Thoning). Work of the
  US Government, so public domain / not subject to copyright; NOAA request
  citation of the dataset. Vendored verbatim, including the ``#`` comment header
  that carries NOAA's own citation and contact details.
- Gulf of Mexico velocities (``gulfdata_train.csv``, ``gulfdata_test.csv``):
  drifter observations (train) and a gridded ocean-current field (test) over the
  Gulf of Mexico. Source:
  https://raw.githubusercontent.com/JaxGaussianProcesses/static/main/data/gulfdata_{train,test}.csv
  These live in the JaxGaussianProcesses organisation's own ``static`` repository
  and are redistributed here under that repository's terms. Vendored verbatim.
- UCI Auto MPG (``auto_mpg.csv``): fuel consumption for 398 cars. Source:
  https://archive.ics.uci.edu/dataset/9/auto-mpg (fetched via ``ucimlrepo``).
  Licence: CC BY 4.0. The features and the target are concatenated into a single
  frame so the notebook can split them back out without ``ucimlrepo``.
"""

from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import requests

HERE = Path(__file__).parent


def _download(url: str, name: str) -> None:
    """Save one URL verbatim, with basic retries."""
    last_err = None
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            path = HERE / name
            path.write_bytes(resp.content)
            print(f"  wrote {name}: {len(resp.content)} bytes")
            return
        except requests.RequestException as err:
            last_err = err
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def _save(df: pd.DataFrame, name: str) -> None:
    path = HERE / name
    df.to_csv(path, index=False)
    print(f"  wrote {name}: {len(df)} rows, {list(df.columns)}")


# --------------------------------------------------------------------------- #
# intro_to_kernels, state_space_gps — Mauna Loa monthly mean CO2 record.       #
# --------------------------------------------------------------------------- #
def pull_mauna_loa_co2() -> None:
    print("intro_to_kernels / state_space_gps: Mauna Loa monthly mean CO2")
    _download(
        "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
        "mauna_loa_co2.csv",
    )


# --------------------------------------------------------------------------- #
# oceanmodelling — Gulf of Mexico drifters (train) and current field (test).   #
# --------------------------------------------------------------------------- #
def pull_gulf_velocities() -> None:
    print("oceanmodelling: Gulf of Mexico drifter and ocean-current velocities")
    base = "https://raw.githubusercontent.com/JaxGaussianProcesses/static/main/data/"
    for name in ("gulfdata_train.csv", "gulfdata_test.csv"):
        _download(base + name, name)


# --------------------------------------------------------------------------- #
# oak — UCI Auto MPG, features and target concatenated into one frame.        #
# --------------------------------------------------------------------------- #
def pull_auto_mpg() -> None:
    print("oak: UCI Auto MPG (id=9)")
    from ucimlrepo import fetch_ucirepo

    auto_mpg = fetch_ucirepo(id=9)
    features = auto_mpg.data.features
    targets = auto_mpg.data.targets
    # Column order is load-bearing: the notebook reconstructs ``features`` by
    # selecting every column except the target, and reports them in this order.
    _save(pd.concat([features, targets], axis=1), "auto_mpg.csv")


if __name__ == "__main__":
    pull_mauna_loa_co2()
    pull_gulf_velocities()
    pull_auto_mpg()
    print("\nDone.")
