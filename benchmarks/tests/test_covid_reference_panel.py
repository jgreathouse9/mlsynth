"""The two sides of the Sweden benchmark must read the same panel.

``scmo_covid_sweden`` compares mlsynth against a captured run of the authors'
own COVID_analysis.R. The Python side reads
``basedata/tlp_covid_sweden.parquet``; the R side reads
``benchmarks/reference/scmo_covid_sweden/covid_panel.csv``, written from that
Parquet so the reference is reproducible offline without a Parquet reader in R.

Two files holding one dataset is a drift the comparison cannot see: if either
were edited alone, the case would keep passing or failing for reasons that have
nothing to do with the estimator. So the equality is a test, not a convention.

Levels: smoke, unit invariants, failure.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "basedata" / "tlp_covid_sweden.parquet"
CSV = ROOT / "benchmarks" / "reference" / "scmo_covid_sweden" / "covid_panel.csv"

KEYS = ["code", "location", "date"]


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(KEYS).reset_index(drop=True)


def _mismatches(left: pd.DataFrame, right: pd.DataFrame) -> list:
    """Columns whose values differ, treating NaN as equal to NaN."""
    bad = []
    for col in left.columns:
        a, b = left[col], right[col]
        if a.dtype.kind in "fi" and b.dtype.kind in "fi":
            if not np.allclose(a.to_numpy(float), b.to_numpy(float),
                               rtol=0, atol=1e-9, equal_nan=True):
                bad.append(col)
        elif not a.equals(b):
            bad.append(col)
    return bad


# --------------------------------------------------------------------------- smoke
def test_both_files_are_present():
    assert PARQUET.exists() and CSV.exists()


# ------------------------------------------------------------------ unit invariants
def test_the_two_files_hold_the_same_panel():
    left, right = _load(PARQUET), _load(CSV)
    assert list(left.columns) == list(right.columns)
    assert left.shape == right.shape
    assert _mismatches(left, right) == []


def test_the_panel_is_the_one_the_case_expects():
    df = _load(PARQUET)
    assert df["code"].nunique() == 27
    assert "SWE" in set(df["code"])
    assert df["date"].min() == pd.Timestamp("2019-01-01")
    assert df["date"].max() == pd.Timestamp("2020-09-30")
    for outcome in ("covid_cases", "deaths_TOTAL", "labour_hours", "gdp_a",
                    "retail_both", "CPI"):
        assert outcome in df.columns


def test_missingness_is_all_or_nothing_within_a_date():
    """Each outcome is observed for every country on a date or for none of them,
    once the country each domain holds out is dropped. The matching matrix keeps
    complete columns only, so partial coverage would silently drop periods."""
    df = _load(PARQUET)
    held_out = {"deaths_TOTAL": "IRL", "labour_absence": "DEU", "labour_hours": "DEU"}
    for outcome in df.columns.drop(KEYS):
        sub = df[df["code"] != held_out.get(outcome, "")]
        counts = sub.groupby("date")[outcome].agg(["count", "size"])
        partial = counts[(counts["count"] > 0) & (counts["count"] < counts["size"])]
        assert partial.empty, f"{outcome}: {len(partial)} dates observed for some units only"


# ----------------------------------------------------------------------- failure
def test_the_comparison_catches_a_difference():
    """The guard above is only worth having if it can fail."""
    left = _load(PARQUET)
    right = left.copy()
    right.loc[right.index[0], "covid_cases"] = 1.0
    assert _mismatches(left, right) == ["covid_cases"]


def test_the_comparison_catches_a_changed_label():
    left = _load(PARQUET)
    right = left.copy()
    right.loc[right.index[0], "location"] = "Atlantis"
    assert _mismatches(left, right) == ["location"]
