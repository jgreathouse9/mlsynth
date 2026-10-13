"""The sunny-donor screen's verdict as typed fields on the result.

Test-first (per ``agents/agents_tests.md``): written before the fields exist, so
every test here is RED until they land.

``solve_mscmt`` reports the screen's findings in a free-form ``metadata`` dict.
CLAUDE.md invariant 7 asks that a diagnostic a caller might act on become a typed
field, and there are three distinct things a caller needs to tell apart:

* how many donors the screen kept and how many it pruned;
* whether the screen could have pruned anything at all on a design of this shape.
  When ``rank(Xt) = J`` the centred columns are independent, every donor is sunny
  by algebra, and an all-sunny answer says nothing about the donor pool. A pool
  that is irreducible and a test that is inapplicable produce the same counts;
* which branch of Becker and Klossner's Figure 2 cascade ran, since an exact
  predictor fit selects weights by their Eq (10) while the third branch returns
  what a differential-evolution search landed on.

The screen does not run on every path. Two early exits precede it (the
unconstrained outcome optimum being predictor-feasible, and a single predictor
fixing ``V`` up to scale), and ``prune_shady=False`` disables it. On those paths
the vacuity flag is absent, because the question was never asked -- which is not
the same answer as ``False``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth.config_models import MethodDetailsResults
from mlsynth.utils.bilevel import BilevelProblem, solve_bilevel
from mlsynth.utils.solvers.sunny import sunny_donors, sunny_screen_is_vacuous

from mlsynth.tests.test_mscmt_sunny_cascade import (
    _exact_fit_design,
    _single_sunny_design,
)

TREATED, AGG = "Basque Country (Pais Vasco)", "Spain (Espana)"
_BASQUE = Path(__file__).resolve().parents[2] / "basedata" / "basque_mscmt.csv"


# --------------------------------------------------------------------------- #
# designs
# --------------------------------------------------------------------------- #
def _problem(seed: int = 0, K: int = 5, J: int = 9, T: int = 14) -> BilevelProblem:
    """Predictor matching with ``K < J``, so the screen runs and is not vacuous."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(K, 3))
    X0 = base @ rng.normal(size=(3, J)) + 0.1 * rng.normal(size=(K, J))
    X1 = rng.normal(size=K)
    Y0 = np.cumsum(rng.normal(size=(T, J)), axis=0) + 5.0
    y1 = Y0 @ rng.dirichlet(np.ones(J)) + 0.05 * rng.normal(size=T)
    return BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)


def _vacuous_problem(seed: int = 4, K: int = 6, J: int = 3, T: int = 16) -> BilevelProblem:
    """``K > J`` with independent centred columns, so ``rank(Xt) = J`` and the
    gate fires: every donor is sunny by algebra and no linear program runs."""
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=K)
    X0 = X1[:, None] + rng.normal(size=(K, J))
    Y0 = np.cumsum(rng.normal(size=(T, J)), axis=0) + 5.0
    y1 = Y0 @ rng.dirichlet(np.ones(J)) + 0.05 * rng.normal(size=T)
    return BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)


def _single_predictor_problem(seed: int = 7, J: int = 5, T: int = 12) -> BilevelProblem:
    """One predictor, so ``V`` is fixed up to scale and the run exits before the
    screen is ever reached."""
    rng = np.random.default_rng(seed)
    X1 = np.array([1.0])
    X0 = rng.normal(size=(1, J)) + 2.0
    Y0 = np.cumsum(rng.normal(size=(T, J)), axis=0) + 5.0
    y1 = Y0 @ rng.dirichlet(np.ones(J)) + 0.05 * rng.normal(size=T)
    return BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)


def _basque_predictor_design():
    """Abadie and Gardeazabal's thirteen-predictor specification, the design MSCMT
    runs its screen on and reports "16 out of 16" sunny donors for."""
    d = pd.read_csv(_BASQUE)
    w16, w19 = (1964, 1969), (1961, 1969)
    covs = ["school.illit", "school.prim", "school.med", "school.higher", "invest",
            "gdpcap", "sec.agriculture", "sec.energy", "sec.industry",
            "sec.construction", "sec.services.venta", "sec.services.nonventa",
            "popdens"]
    win = {**{c: w16 for c in covs[:5]}, "gdpcap": (1960, 1969),
           **{c: w19 for c in covs[6:12]}, "popdens": (1969, 1969)}
    units = [u for u in d.regionname.unique() if u != AGG]
    donors = [u for u in units if u != TREATED]
    X = pd.DataFrame({c: d[d.year.between(*win[c])].groupby("regionname")[c].mean()
                      for c in covs}).loc[units]
    return (X.loc[donors].values.astype(float).T, X.loc[TREATED].values.astype(float),
            donors)


def _basque_outcome_only():
    """Twenty pre-treatment periods of ``gdpcap`` against sixteen donors, where
    ``rank(Xt) = 16 = J`` and the gate fires."""
    d = pd.read_csv(_BASQUE)
    piv = d.pivot(index="year", columns="regionname", values="gdpcap")
    donors = [r for r in piv.columns if r not in (TREATED, AGG)]
    yrs = [y for y in piv.index if y < 1975]
    A = piv.loc[yrs, TREATED].values.astype(float)
    B = np.column_stack([piv.loc[yrs, c].values.astype(float) for c in donors])
    return B, A


def _mscmt(prob, **kw):
    kw.setdefault("maxiter", 6)
    kw.setdefault("popsize", 4)
    kw.setdefault("seed", 0)
    return solve_bilevel(prob, method="mscmt", **kw)


# --------------------------------------------------------------------------- #
# the fields are declared, not carried by extra='allow'
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field", ["n_sunny", "n_shady_pruned", "sunny_screen_vacuous", "mscmt_branch"]
)
def test_the_field_is_declared_on_method_details(field):
    """``MethodDetailsResults`` sets ``extra='allow'``, so an undeclared name would
    round-trip silently and no contract test would cover it. Declaring it is what
    makes the name part of the supported surface."""
    assert field in MethodDetailsResults.model_fields


def test_the_declared_fields_default_to_none():
    """Purely additive: an estimator that never screens reports nothing, not zero.
    A default of ``0`` would read as "the screen ran and pruned nothing"."""
    md = MethodDetailsResults(method_name="whatever")
    assert md.n_sunny is None
    assert md.n_shady_pruned is None
    assert md.sunny_screen_vacuous is None
    assert md.mscmt_branch is None


# --------------------------------------------------------------------------- #
# the solver reports the vacuity flag alongside the counts it already reported
# --------------------------------------------------------------------------- #
def test_outer_search_reports_the_vacuity_flag():
    prob = _problem()
    sol = _mscmt(prob)
    assert sol.metadata["mscmt_branch"] == "outer-search"
    assert sol.metadata["sunny_screen_vacuous"] is False


def test_the_flag_agrees_with_the_screen_called_directly():
    """The reported flag cannot drift from what the screen would say on the same
    design, which is the failure a hard-coded constant would pass."""
    for prob in (_problem(), _problem(seed=3), _vacuous_problem()):
        sol = _mscmt(prob)
        assert sol.metadata["sunny_screen_vacuous"] == sunny_screen_is_vacuous(
            prob.X0, prob.X1
        )


def test_the_counts_agree_with_the_screen_called_directly():
    prob = _problem()
    sol = _mscmt(prob)
    sunny = sunny_donors(prob.X0, prob.X1)
    assert sol.metadata["n_sunny"] == int(sunny.sum())
    assert sol.metadata["n_shady_pruned"] == int((~sunny).sum())


def test_a_vacuous_design_reports_true_and_keeps_every_donor():
    prob = _vacuous_problem()
    assert sunny_screen_is_vacuous(prob.X0, prob.X1)          # the design, not the report
    sol = _mscmt(prob)
    assert sol.metadata["sunny_screen_vacuous"] is True
    assert sol.metadata["n_sunny"] == prob.n_donors
    assert sol.metadata["n_shady_pruned"] == 0


def test_exact_fit_branch_reports_the_flag():
    sol = _mscmt(_exact_fit_design())
    assert sol.metadata["mscmt_branch"] == "exact-fit"
    assert sol.metadata["n_sunny"] == 0
    assert sol.metadata["sunny_screen_vacuous"] is False


def test_single_sunny_branch_reports_the_flag():
    sol = _mscmt(_single_sunny_design())
    assert sol.metadata["mscmt_branch"] == "single-sunny"
    assert sol.metadata["n_sunny"] == 1
    assert sol.metadata["sunny_screen_vacuous"] is False


# --------------------------------------------------------------------------- #
# paths where the screen never ran: absent, which is not False
# --------------------------------------------------------------------------- #
def test_the_flag_is_absent_when_a_single_predictor_exits_early():
    prob = _single_predictor_problem()
    sol = _mscmt(prob)
    assert "sunny_screen_vacuous" not in sol.metadata
    assert "n_sunny" not in sol.metadata


def test_the_flag_is_absent_when_the_screen_is_disabled():
    """``prune_shady=False`` skips the screen, so reporting a verdict for it would
    describe a computation that did not happen."""
    sol = _mscmt(_problem(), prune_shady=False)
    assert "sunny_screen_vacuous" not in sol.metadata


# --------------------------------------------------------------------------- #
# the designs the issue names
# --------------------------------------------------------------------------- #
def test_basque_predictor_spec_reports_sixteen_of_sixteen_sunny():
    """Matches MSCMT's own console line, ``Number of 'sunny' donors: 16 out of 16``.
    The counts are settled by the screen before the outer search starts, so the
    search budget here does not affect them."""
    B, A, donors = _basque_predictor_design()
    rng = np.random.default_rng(0)
    T = 20
    Y0 = np.cumsum(rng.normal(size=(T, len(donors))), axis=0) + 5.0
    y1 = Y0 @ rng.dirichlet(np.ones(len(donors))) + 0.05 * rng.normal(size=T)
    sol = _mscmt(BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=A, X0=B))
    assert sol.metadata["n_sunny"] == 16
    assert sol.metadata["n_shady_pruned"] == 0
    # 13 predictors against 16 donors cannot reach rank J, so the gate cannot fire:
    # all-sunny here is a fact about the pool, not about the design's shape.
    assert sol.metadata["sunny_screen_vacuous"] is False


def test_basque_outcome_only_design_is_vacuous():
    """Twenty pre-periods against sixteen donors gives ``rank(Xt) = 16 = J``, so
    every donor is sunny by algebra and the all-sunny verdict carries no
    information about the donor pool."""
    B, A = _basque_outcome_only()
    assert sunny_screen_is_vacuous(B, A) is True
    assert bool(sunny_donors(B, A).all())


# --------------------------------------------------------------------------- #
# end to end, through the estimator's result
# --------------------------------------------------------------------------- #
def _panel():
    rng = np.random.default_rng(3)
    rows = []
    for unit in range(6):
        level = 1.0 + unit
        for t in range(12):
            y = level + 0.3 * t + rng.normal(scale=0.05)
            if unit == 0 and t >= 8:
                y += 2.0
            rows.append({"unit": unit, "time": t, "y": y,
                         "x": level + rng.normal(scale=0.01),
                         "x2": level ** 2 + rng.normal(scale=0.01),
                         "treat": int(unit == 0 and t >= 8)})
    return pd.DataFrame(rows)


def _fit_mscmt():
    from mlsynth import VanillaSC
    from mlsynth.utils.vanillasc_helpers.config import VanillaSCConfig

    return VanillaSC(VanillaSCConfig(
        df=_panel(), outcome="y", treat="treat", unitid="unit", time="time",
        backend="mscmt", covariates=["x", "x2"],
    )).fit()


def test_vanillasc_surfaces_the_typed_fields():
    md = _fit_mscmt().method_details
    assert isinstance(md.n_sunny, int)
    assert isinstance(md.n_shady_pruned, int)
    assert isinstance(md.sunny_screen_vacuous, bool)
    assert md.mscmt_branch in {"exact-fit", "single-sunny", "outer-search"}


def test_the_counts_partition_the_donor_pool():
    """Sunny and shady are complementary over the whole pool, so the two counts sum
    to the donor count -- not to the number of weight-bearing donors, which is
    smaller again. On this panel one donor carries weight out of five, while the
    screen splits the five into two sunny and three shady: being usable, being
    used, and being irreducible are three different things."""
    res = _fit_mscmt()
    md = res.method_details
    n_donors = _panel().unit.nunique() - 1
    assert md.n_sunny + md.n_shady_pruned == n_donors
    assert len(res.weights.donor_weights) <= md.n_sunny


def _vacuous_panel():
    """Three donors against four covariates, so the predictor design has
    ``rank(Xt) = 3 = J``: every donor is sunny by algebra and none can be pruned."""
    rng = np.random.default_rng(17)
    rows = []
    for unit in range(4):
        level = 1.0 + 2.0 * unit
        for t in range(12):
            y = level + 0.3 * t + rng.normal(scale=0.05)
            if unit == 0 and t >= 8:
                y += 2.0
            rows.append({"unit": unit, "time": t, "y": y,
                         "treat": int(unit == 0 and t >= 8),
                         **{f"c{k}": level * (k + 1) + rng.normal() for k in range(4)}})
    return pd.DataFrame(rows)


def test_a_vacuous_predictor_design_reports_every_donor_sunny():
    """The counts are not interchangeable, and this design says which is which: a
    vacuous design keeps the whole pool, so ``n_sunny`` is the donor count and
    ``n_shady_pruned`` is zero. Reporting the two the other way round would pass a
    test that only checked they sum to the pool."""
    from mlsynth import VanillaSC
    from mlsynth.utils.vanillasc_helpers.config import VanillaSCConfig

    panel = _vacuous_panel()
    res = VanillaSC(VanillaSCConfig(
        df=panel, outcome="y", treat="treat", unitid="unit", time="time",
        backend="mscmt", covariates=[f"c{k}" for k in range(4)],
    )).fit()
    md = res.method_details
    n_donors = panel.unit.nunique() - 1
    assert md.sunny_screen_vacuous is True
    assert md.n_sunny == n_donors
    assert md.n_shady_pruned == 0


def test_a_backend_that_never_screens_reports_none():
    from mlsynth import VanillaSC
    from mlsynth.utils.vanillasc_helpers.config import VanillaSCConfig

    res = VanillaSC(VanillaSCConfig(
        df=_panel(), outcome="y", treat="treat", unitid="unit", time="time",
    )).fit()
    md = res.method_details
    assert md.n_sunny is None
    assert md.sunny_screen_vacuous is None
    assert md.mscmt_branch is None
