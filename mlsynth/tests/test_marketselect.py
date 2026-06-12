import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from mlsynth.utils.datautils import geoex_dataprep
from mlsynth.utils.geolift_helpers.marketselect.helpers.similarity import (
    correlation_matrix,
    rank_markets_by_correlation,
)
from mlsynth.exceptions import MlsynthDataError


def planted_panel():
    """Ywide whose correlation ordering is known by construction.

    A and B move together (B is an affine image of A -> corr = +1); C is the
    exact negative of A (corr = -1); D is a low-correlation sawtooth.
    """
    t = np.arange(1, 9, dtype=float)
    Ywide = pd.DataFrame(
        {
            "A": t,
            "B": 2.0 * t + 1.0,                                  # corr(A, B) = +1
            "C": -t,                                             # corr(A, C) = -1
            "D": np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float),
        },
        index=pd.Index(range(1, 9), name="time"),
    )
    Ywide.columns.name = "unit"
    return Ywide


# === correlation_matrix ===

def test_correlation_matrix_smoke_and_labels():
    Ywide = planted_panel()
    C = correlation_matrix(Ywide)
    assert isinstance(C, pd.DataFrame)
    assert C.shape == (4, 4)
    assert C.index.equals(Ywide.columns)      # labeled by the unit Index, both axes
    assert C.columns.equals(Ywide.columns)
    assert_allclose(np.diag(C.to_numpy()), 1.0)
    assert C.loc["A", "B"] == pytest.approx(1.0)
    assert C.loc["A", "C"] == pytest.approx(-1.0)


def test_correlation_matrix_symmetric_bounded_unit_diagonal():
    """Invariants: symmetric, unit diagonal, off-diagonals within [-1, 1]."""
    M = correlation_matrix(planted_panel()).to_numpy()
    assert_allclose(M, M.T)                       # symmetric
    assert_allclose(np.diag(M), 1.0)              # unit diagonal
    off = M[~np.eye(M.shape[0], dtype=bool)]
    assert np.all((off >= -1 - 1e-12) & (off <= 1 + 1e-12))


def test_correlation_matrix_identical_units_are_one():
    """Affine-identical and exactly-duplicated series correlate at +1."""
    Ywide = planted_panel()
    assert correlation_matrix(Ywide).loc["A", "B"] == pytest.approx(1.0)  # B = 2A+1
    Ywide = Ywide.assign(A2=Ywide["A"])           # exact duplicate column
    assert correlation_matrix(Ywide).loc["A", "A2"] == pytest.approx(1.0)


def test_correlation_matrix_constant_unit_is_nan():
    """A zero-variance unit has undefined correlation -> NaN row/column."""
    Ywide = planted_panel().assign(E=7.0)         # constant series
    C = correlation_matrix(Ywide)
    assert np.isnan(C.loc["E"].drop("E")).all()   # E vs every other unit
    assert np.isnan(C.loc["A", "E"])


def test_correlation_matrix_single_unit_returns_one_by_one():
    """The matrix maker does not itself require >=2 units (the ranker does);
    a single column yields a 1x1 frame with a unit diagonal."""
    C = correlation_matrix(planted_panel()[["A"]])
    assert C.shape == (1, 1)
    assert C.loc["A", "A"] == pytest.approx(1.0)


# === rank_markets_by_correlation ===

def test_rank_markets_shape_index_and_self_excluded():
    Ywide = planted_panel()
    R = rank_markets_by_correlation(Ywide)
    assert isinstance(R, pd.DataFrame)
    assert R.shape == (4, 3)                   # L anchors x (L-1) neighbor ranks
    assert R.index.equals(Ywide.columns)       # anchors are the unit Index
    for anchor in R.index:                     # no anchor in its own neighbor row
        assert anchor not in set(R.loc[anchor].tolist())


def test_rank_markets_descending_by_correlation():
    Ywide = planted_panel()
    R = rank_markets_by_correlation(Ywide)
    assert R.loc["A"].iloc[0] == "B"           # closest = corr +1
    assert R.loc["A"].iloc[-1] == "C"          # farthest = corr -1


def test_rank_markets_two_units():
    Ywide = planted_panel()[["A", "B"]]
    R = rank_markets_by_correlation(Ywide)
    assert R.shape == (2, 1)
    assert R.loc["A"].iloc[0] == "B"
    assert R.loc["B"].iloc[0] == "A"


def test_rank_markets_consumes_geoex_output():
    """Composes directly with geoex_dataprep's Ywide."""
    df = pd.DataFrame(
        {
            "unit": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "time": list(range(5)) * 3,
            "outcome": list(np.arange(5.0)) + list(2.0 * np.arange(5))
            + list(-np.arange(5.0)),
        }
    )
    out = geoex_dataprep(df, "unit", "time", "outcome")
    R = rank_markets_by_correlation(out["Ywide"])
    assert R.loc["A"].iloc[0] == "B"           # A~B positive, A~C negative


def test_rank_markets_zero_variance_unit_ranks_last():
    """A constant series has undefined correlation; it sorts to the bottom of
    every other unit's ranking rather than crashing."""
    Ywide = planted_panel()
    Ywide["E"] = 5.0                           # constant -> NaN correlations
    R = rank_markets_by_correlation(Ywide)
    assert R.loc["A"].iloc[-1] == "E"


def test_rank_markets_too_few_units_raises():
    Ywide = planted_panel()[["A"]]
    with pytest.raises(MlsynthDataError, match="at least 2"):
        rank_markets_by_correlation(Ywide)
