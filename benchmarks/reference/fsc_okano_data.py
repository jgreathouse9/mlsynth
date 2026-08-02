"""Rebuild the Okano-Kurisu (2026) FSC panels in ``basedata/`` from source.

The authors ship their three datasets as R lists in ``asfr.RData``,
``aad.RData`` and ``service.RData`` (https://github.com/RyoOkano21/FSC). Those
are nested lists rather than data frames, so ``pyreadr`` returns nothing for
them; the ``rdata`` package parses them directly and no R installation is
needed. Run this only to regenerate the CSVs -- the benchmark case reads the
committed files.

    pip install rdata
    git clone https://github.com/RyoOkano21/FSC
    python -m benchmarks.reference.fsc_okano_data FSC

Unit order is the order of the authors' lists, which their Tables 1-3 follow and
which the FSC weights confirm (the three donors Table 3 gives nonzero weight land
on France, Greece and the United States under this labelling). The treated unit
is first in every list.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_OUT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "basedata"))

FERTILITY_UNITS = [
    "East Germany", "Austria", "Belgium", "Bulgaria", "Canada", "Switzerland",
    "Czechia", "Denmark", "Spain", "Finland", "France", "Hungary", "Ireland",
    "Italy", "Japan", "Netherlands", "Portugal", "Slovakia", "Sweden",
    "United Kingdom", "United States",
]
MORTALITY_UNITS = [
    "Russia", "Austria", "Belgium", "Denmark", "Finland", "France", "Germany",
    "Iceland", "Ireland", "Italy", "Luxembourg", "Netherlands", "Norway",
    "Portugal", "Spain", "Sweden", "Switzerland", "United Kingdom",
]
SERVICE_UNITS = [
    "United Kingdom", "Australia", "Austria", "Canada", "Czechia", "Estonia",
    "Finland", "France", "Greece", "Hungary", "Ireland", "Iceland", "Italy",
    "Japan", "Korea", "Luxembourg", "Latvia", "New Zealand", "Portugal",
    "Slovenia", "Sweden", "Turkiye", "United States",
]
SERVICE_CATEGORIES = ["SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL"]


def _cube(repo: str, filename: str, key: str) -> np.ndarray:
    import rdata

    obj = rdata.read_rda(os.path.join(repo, filename))[key]
    return np.array([[np.asarray(c, dtype=float) for c in unit]
                     for unit in obj])


def _emit(cube: np.ndarray, units, times, arguments, argument_name: str,
          value_name: str, first_treated_index: int, path: str) -> None:
    n_units, n_times, n_args = cube.shape
    assert (len(units), len(times), len(arguments)) == cube.shape, cube.shape
    ui, ti, ai = np.meshgrid(np.arange(n_units), np.arange(n_times),
                             np.arange(n_args), indexing="ij")
    frame = pd.DataFrame({
        "unit": np.asarray(units)[ui.ravel()],
        "time": np.asarray(times)[ti.ravel()],
        argument_name: np.asarray(arguments)[ai.ravel()],
        value_name: cube.ravel(),
        "treat": ((ui.ravel() == 0)
                  & (ti.ravel() >= first_treated_index)).astype(int),
    })
    frame.to_csv(os.path.join(_OUT, path), index=False)
    print(f"{path:30s} {len(frame):>7,} rows")


def main(repo: str) -> None:
    # 6.1 fertility -- ASFR curves over age, T0 = 16 so 1972 is first treated
    _emit(_cube(repo, "asfr.RData", "asfr_list"), FERTILITY_UNITS,
          np.arange(1956, 1976), np.arange(12, 56), "age", "asfr", 16,
          "okano_fsc_fertility.csv")

    # 6.2 mortality -- age-at-death quantile functions, T0 = 21 -> 1991
    _emit(_cube(repo, "aad.RData", "AaD_Quant_list"), MORTALITY_UNITS,
          np.arange(1970, 2000), np.round(np.linspace(0.01, 0.99, 100), 10),
          "prob", "quantile", 21, "okano_fsc_mortality.csv")

    # 6.3 service trade -- vech of the 9x9 covariance, T0 = 29 -> 2016Q2.
    # covvec_list is the row-major lower triangle of covmat_list, with no
    # sqrt(2) on the off-diagonals (so it is not the Frobenius isometry).
    rows, cols = np.tril_indices(9)
    pairs = [f"{SERVICE_CATEGORIES[r]}:{SERVICE_CATEGORIES[c]}"
             for r, c in zip(rows, cols)]
    quarters = [f"{y}Q{q}" for y in range(2009, 2019) for q in (1, 2, 3, 4)][:38]
    _emit(_cube(repo, "service.RData", "covvec_list"), SERVICE_UNITS,
          quarters, pairs, "pair", "cov", 29, "okano_fsc_service.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "FSC")
