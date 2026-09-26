"""Synthetic stand-ins for Hsiao and Zhou's two smoking panels.

Section 7 runs on three panels. One of them, the Election Day Registration
turnout panel, is already in this repository as
``basedata/xu_edr_turnout.parquet`` -- it is the paper's ``turnout.csv`` row for
row. The other two are not, so without them both empirical arms refused to run.
These generators replace them with panels drawn to behave the same way, which
makes the paper's data an option instead of a prerequisite.

Nothing in them is measured. Units are integers, periods are integers counted
from zero, every constant below is a design parameter with a round value, and
no observation, label, year or magnitude from any real panel appears here or is
recoverable from it. A run on a stand-in says so in its output and prints no
comparison against the published tables, because there is nothing to compare.

What the consumption panel reproduces, and why each part is there:

=================  =====  ===================================================
constant           value  what it sets
=================  =====  ===================================================
N_UNITS            39     one treated unit and 38 controls, as Section 7 has
N_PERIODS          31     with treatment at period 19
START / END        130/48 the outcome's decline over the window
FACTOR_AR          0.75   persistence of the two common factors
LNINCOME_LEVEL     9.5    the near-constant regressor's level
LNINCOME_SD        0.25   and its spread -- deliberately small
BETA               (...)  the coefficients, with a large one on that regressor
=================  =====  ===================================================

The near-constant regressor is the part that matters most. ``lnincome`` in the
real panel is a log, nearly flat across states, carrying a coefficient of about
60, which ``BETA[0]`` matches. That combination is what made Bai's PCA2
iteration stall in #647, and a stand-in without it would let this study's
regression test pass for a reason that has nothing to do with the defect.

The size of ``BETA[0]`` sets how far PCA2 stalls, so it sets how much power the
regression test has. Measured over six seeds, PCA2 from a zero start lands above
the reached minimum by this much:

===========  =================  ===================
``BETA[0]``  worst-case excess  two-factor share
===========  =================  ===================
6            0.003 percent      0.93
20           2.7 percent        0.91
60           31 percent         0.74
===========  =================  ===================

At 6 the gap sits eleven orders of magnitude above floating-point noise but
only four above nothing, which is too thin to assert across BLAS
implementations. At 60 it matches the 18 percent the real panel shows and the
factor structure survives. ``benchmarks/tests/test_hsiao_zhou_standin.py``
holds the generator to a margin, not to a strict inequality.

The expenditure panel's defining feature is its shape: 38 controls against a
nine-period pre-period, a pool more than four times as wide as the sample it is
fitted on. That is the rank-deficient regime its arm exists to exercise, and it
is why CCE behaves so differently between the two tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- the consumption panel -------------------------------------------------
N_UNITS = 39
N_PERIODS = 31
T0 = 19
START, END = 130.0, 48.0
UNIT_SPREAD = 22.0
FACTOR_AR = 0.75
N_FACTORS = 2
LNINCOME_LEVEL, LNINCOME_SD = 9.5, 0.25
EDU_LEVEL, EDU_SD = 14.0, 2.5
POVERTY_LEVEL, POVERTY_SD = 12.0, 3.5
BETA = (60.0, -0.9, -0.6)         # large coefficient on the flat regressor
IDIO_SD = 6.0

# --- the expenditure panel -------------------------------------------------
EXP_PERIODS = 21
EXP_T0 = 9
EXP_START, EXP_END = 8.95, 9.80
EXP_UNIT_SPREAD = 0.18
EXP_IDIO_SD = 0.035
EXP_BETA = 0.35


def _factors(n_periods, n_factors, rng, ar=FACTOR_AR):
    """Persistent common factors, standardised so their scale is a choice."""
    f = np.zeros((n_periods, n_factors))
    for t in range(1, n_periods):
        f[t] = ar * f[t - 1] + rng.standard_normal(n_factors)
    return (f - f.mean(axis=0)) / f.std(axis=0)


def consumption_panel(seed: int = 0, n_units: int = N_UNITS,
                      n_periods: int = N_PERIODS) -> pd.DataFrame:
    """A stand-in for ``smoking.csv``: units, periods, outcome, three covariates.

    Long form with ``state`` / ``year`` / ``cigsale`` / ``lnincome`` /
    ``EduAttain`` / ``Poverty``, the column names the arm reads. Unit 1 is the
    treated one. ``year`` counts from zero, so nothing here can be mistaken for
    a calendar.
    """
    if n_units < 3 or n_periods < 5:
        raise ValueError(
            f"a panel needs at least 3 units and 5 periods; got "
            f"{n_units} and {n_periods}")
    rng = np.random.default_rng(seed)

    trend = np.linspace(START, END, n_periods)
    level = rng.normal(0.0, UNIT_SPREAD, n_units)
    f = _factors(n_periods, N_FACTORS, rng)
    loadings = rng.standard_normal((n_units, N_FACTORS)) * 4.0

    lnincome = LNINCOME_LEVEL + rng.normal(0.0, LNINCOME_SD,
                                           (n_periods, n_units))
    edu = EDU_LEVEL + rng.normal(0.0, EDU_SD, (n_periods, n_units)) \
        + np.linspace(0.0, 3.0, n_periods)[:, None]
    poverty = POVERTY_LEVEL + rng.normal(0.0, POVERTY_SD,
                                         (n_periods, n_units))

    y = (trend[:, None] + level[None, :] + f @ loadings.T
         + BETA[0] * (lnincome - LNINCOME_LEVEL)
         + BETA[1] * (edu - EDU_LEVEL)
         + BETA[2] * (poverty - POVERTY_LEVEL)
         + rng.normal(0.0, IDIO_SD, (n_periods, n_units)))

    return pd.DataFrame({
        "state": np.repeat(np.arange(1, n_units + 1), n_periods),
        "year": np.tile(np.arange(n_periods), n_units),
        "cigsale": y.T.ravel(),
        "lnincome": lnincome.T.ravel(),
        "EduAttain": edu.T.ravel(),
        "Poverty": poverty.T.ravel(),
    })


def expenditure_panel(seed: int = 0, n_units: int = N_UNITS,
                      n_periods: int = EXP_PERIODS) -> pd.DataFrame:
    """A stand-in for ``smoking-health-expenditure.csv``.

    Long form with ``state`` / ``year`` / ``lnhexpense`` / ``lnincome``. The
    outcome is on a log scale and rises; the donor pool is deliberately far
    wider than the nine-period pre-period, which is the regime this arm is for.
    """
    if n_units < 3 or n_periods < 5:
        raise ValueError(
            f"a panel needs at least 3 units and 5 periods; got "
            f"{n_units} and {n_periods}")
    rng = np.random.default_rng(seed)

    trend = np.linspace(EXP_START, EXP_END, n_periods)
    level = rng.normal(0.0, EXP_UNIT_SPREAD, n_units)
    f = _factors(n_periods, N_FACTORS, rng)
    loadings = rng.standard_normal((n_units, N_FACTORS)) * 0.05
    lnincome = LNINCOME_LEVEL + rng.normal(0.0, LNINCOME_SD,
                                           (n_periods, n_units))

    y = (trend[:, None] + level[None, :] + f @ loadings.T
         + EXP_BETA * (lnincome - LNINCOME_LEVEL)
         + rng.normal(0.0, EXP_IDIO_SD, (n_periods, n_units)))

    return pd.DataFrame({
        "state": np.repeat(np.arange(1, n_units + 1), n_periods),
        "year": np.tile(np.arange(n_periods), n_units),
        "lnhexpense": y.T.ravel(),
        "lnincome": lnincome.T.ravel(),
    })


# ---------------------------------------------------------------------------
# Where each arm's panel comes from
# ---------------------------------------------------------------------------

def resolve_panels(data_dir=None, seed: int = 0):
    """The three Section 7 panels, real where they exist and drawn where not.

    ``data_dir`` is the paper's replication directory, normally supplied
    through ``MLSYNTH_HZ_DATA``. The two smoking panels are read from it when
    it holds them and drawn otherwise. The turnout panel is different: it needs
    no stand-in, because ``basedata/xu_edr_turnout.parquet`` is the paper's own
    ``turnout.csv``, so it is read from the repository whichever way the other
    two resolve.

    Returns
    -------
    dict
        ``consumption``, ``expenditure``, ``turnout`` and ``real``, the last a
        bool saying whether the two smoking panels came from the paper. Callers
        print the published comparison only when it is true.
    """
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[3]
    turnout = pd.read_parquet(repo / "basedata" / "xu_edr_turnout.parquet")

    d = pathlib.Path(data_dir) if data_dir else None
    real = bool(d and (d / "smoking.csv").is_file()
                and (d / "smoking-health-expenditure.csv").is_file())
    if real:
        return {"consumption": pd.read_csv(d / "smoking.csv"),
                "expenditure": pd.read_csv(d / "smoking-health-expenditure.csv"),
                "turnout": turnout, "real": True}
    return {"consumption": consumption_panel(seed=seed),
            "expenditure": expenditure_panel(seed=seed),
            "turnout": turnout, "real": False}
