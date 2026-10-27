"""The per-unit synthetic control behind SPILLSYNTH's iscm, iterative and grossi.

``build_unit_sc`` fits one unit's synthetic control. With no predictor block it
is simplex least squares on the pre-period outcomes, or, with
``intercept=True``, the same program on outcomes centred by their own
pre-period means, with the level shift added back. Both are solved exactly by
the active set behind :func:`mlsynth.utils.solvers.ridge_augment.simplex_qp`.

They were solved by the FISTA primitive
:func:`mlsynth.utils.solvers.simplex.simplex_lstsq`. On German reunification it
ran to its 2000-iteration cap on 31 of the 34 leave-one-out fits (17 units, both
modes) and stopped short of the optimum on 32 of them: weights up to 0.03 away
and the pre-period loss up to 0.18% above the minimum, which left the inclusive
ATT 3.3 USD from the reference solved to the simplex. Each fit cost about 37 ms;
the active set takes under 1 ms. On panels drawn like the ones below it failed
the optimality certificate on 67 of 300 draws.

* Optimality: the weights pass a KKT certificate computed from the design and
  the target, which holds whatever solver produced them -- on every German unit
  and over a generated domain.
* Invariance: the intercept fit's pre-period gap has mean zero and does not move
  when each unit's series is shifted by its own constant; the plain fit's does
  not move under a common shift.
* Degenerate pools: one donor, a duplicated donor, a target inside the hull,
  flat series.
* Failure: non-finite outcomes raise MlsynthDataError at ingestion and in the
  helper, since the active set returns weights for a design carrying a NaN; an
  empty donor pool raises it in the helper.
* Reference: iscm and iterative reproduce the reference's algorithm solved to
  the simplex (``benchmarks/reference/clone_iscm``) to 1e-3 USD.
* Work: no first-order iterations, and each fit is certified in at most ``2J``
  active-set pivots -- machine-independent proxies, as in
  ``test_simplex_active_set_perf``.

The mutants are catalogued under ``spillsynth-unit-sc`` in
``tools/mutation/targets.toml``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlsynth import SPILLSYNTH
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.solvers.minnorm import simplex_point_is_optimal
from mlsynth.utils.spillsynth_helpers import build_unit_sc, run_iscm, run_iterative
from mlsynth.utils.spillsynth_helpers.setup import prepare_spillsynth_inputs

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "repgermany.dta"


@pytest.fixture(scope="module")
def german_panel():
    if not _DATA.exists():
        pytest.skip("repgermany.dta not available")
    d = pd.read_stata(_DATA)[["country", "year", "gdp"]].copy()
    d["gdp"] = d["gdp"].astype(float)
    d["treat"] = ((d.country == "West Germany") & (d.year >= 1990)).astype(int)
    return d


@pytest.fixture(scope="module")
def german_inputs(german_panel):
    return prepare_spillsynth_inputs(
        df=german_panel, outcome="gdp", treat="treat", unitid="country",
        time="year", affected_units=["Austria"])


def _program(Y, target, donors, T0, intercept):
    """The design and target the fit solves, scaled to unit magnitude.

    Dividing both by one constant leaves the minimisers unchanged, and the KKT
    certificate needs it: its tolerance has an absolute floor, which on outcomes
    near 1e4 asks an exact fit for a residual of 1e-12 -- the rounding of the
    data itself.
    """
    y1, Y0 = Y[target, :T0], Y[donors, :T0].T
    if intercept:
        Y0, y1 = Y0 - Y0.mean(axis=0), y1 - y1.mean()
    s = max(float(np.abs(Y0).max()), float(np.abs(y1).max()), 1e-300)
    return Y0 / s, y1 / s


def _panel(seed, J, T0, scale, T1=3):
    """A trending factor panel in levels: row 0 the target, rows ``1..J`` donors."""
    rng = np.random.default_rng(seed)
    T = T0 + T1
    F = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    L = rng.normal(size=(J + 1, 2))
    return scale * (5.0 + L @ F.T + 0.3 * rng.normal(size=(J + 1, T)))


def _fit(Y, T0, intercept):
    return build_unit_sc(0, np.arange(1, Y.shape[0]), Y, T0, intercept=intercept)


# --------------------------------------------------------------------------- #
# Optimality
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("intercept", [False, True])
def test_every_german_unit_fit_is_the_exact_optimum(german_inputs, intercept):
    """Each unit's fit against all the others passes the KKT certificate."""
    Y, T0, N = german_inputs.Y, german_inputs.T0, german_inputs.N
    not_optimal = []
    for i in range(N):
        donors = np.array([j for j in range(N) if j != i])
        w, cf, gap, pre_rmspe, sol = build_unit_sc(i, donors, Y, T0,
                                                    intercept=intercept)
        assert sol is None
        assert w.shape == (N - 1,)
        assert np.all(np.isfinite(cf)) and np.isfinite(pre_rmspe)
        np.testing.assert_allclose(cf + gap, Y[i], rtol=1e-12)
        if not simplex_point_is_optimal(*_program(Y, i, donors, T0, intercept), w):
            not_optimal.append(i)
    assert not not_optimal, (
        f"units {not_optimal} were fit to weights that do not solve their "
        f"program (intercept={intercept})")


_SEEDS = st.integers(min_value=0, max_value=2**32 - 1)


@given(seed=_SEEDS, J=st.integers(1, 8), T0=st.integers(2, 12),
       scale=st.sampled_from([1.0, 1e3, 1e4]), intercept=st.booleans())
def test_fit_is_the_exact_optimum_over_generated_panels(seed, J, T0, scale,
                                                        intercept):
    """Feasible and KKT-optimal, with the gap and RMSPE consistent with ``cf``.

    The domain covers more donors than pre-periods, a single donor, the
    two-period minimum the setup admits, and outcomes in levels at scales up to
    1e4, where the outcome Gram's entries pass 1e10.
    """
    Y = _panel(seed, J, T0, scale)
    w, cf, gap, pre_rmspe, _ = _fit(Y, T0, intercept)
    assert w.shape == (J,)
    assert w.min() >= 0.0 and abs(w.sum() - 1.0) <= 1e-9
    assert simplex_point_is_optimal(
        *_program(Y, 0, np.arange(1, J + 1), T0, intercept), w)
    np.testing.assert_allclose(cf + gap, Y[0], rtol=1e-12)
    assert pre_rmspe == pytest.approx(np.sqrt(np.mean(gap[:T0] ** 2)),
                                      rel=1e-12, abs=1e-12)


# --------------------------------------------------------------------------- #
# Invariance
# --------------------------------------------------------------------------- #
@given(seed=_SEEDS, J=st.integers(1, 8), T0=st.integers(2, 12),
       scale=st.sampled_from([1.0, 1e3, 1e4]))
def test_intercept_fit_absorbs_each_units_level(seed, J, T0, scale):
    """The level shift takes up the pre-period mean, so a per-unit shift is inert.

    Centring removes a constant added to any one series from the program, and
    the level shift ``mean(y1_pre) - mean(Y0_pre) w`` takes it back out of the
    gap. The pre-period gap is unique even where the weights are not -- it is
    the residual of a least-squares fit -- so that is what is compared.
    """
    Y = _panel(seed, J, T0, scale)
    _, _, gap, _, _ = _fit(Y, T0, intercept=True)
    span = float(np.abs(Y).max())
    assert abs(gap[:T0].mean()) <= 1e-12 * span

    shifts = np.random.default_rng(seed ^ 0x5EED).normal(size=(J + 1, 1)) * 10 * scale
    _, _, gap_shifted, _, _ = _fit(Y + shifts, T0, intercept=True)
    np.testing.assert_allclose(gap_shifted[:T0], gap[:T0], rtol=0,
                               atol=1e-6 * span)


@given(seed=_SEEDS, J=st.integers(1, 8), T0=st.integers(2, 12),
       scale=st.sampled_from([1.0, 1e3, 1e4]))
def test_plain_fit_ignores_a_common_shift(seed, J, T0, scale):
    """Weights summing to one carry a constant added to every series through."""
    Y = _panel(seed, J, T0, scale)
    _, _, gap, _, _ = _fit(Y, T0, intercept=False)
    _, _, gap_shifted, _, _ = _fit(Y + 7.5 * scale, T0, intercept=False)
    span = float(np.abs(Y).max())
    np.testing.assert_allclose(gap_shifted[:T0], gap[:T0], rtol=0,
                               atol=1e-6 * span)


# --------------------------------------------------------------------------- #
# Degenerate pools
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("intercept", [False, True])
def test_a_single_donor_takes_all_the_weight(intercept):
    Y = _panel(3, J=1, T0=6, scale=100.0)
    w, cf, _, _, _ = _fit(Y, 6, intercept)
    np.testing.assert_array_equal(w, [1.0])
    level = (Y[0, :6].mean() - Y[1, :6].mean()) if intercept else 0.0
    np.testing.assert_allclose(cf, Y[1] + level, rtol=1e-12)


@pytest.mark.parametrize("intercept", [False, True])
def test_a_duplicated_donor_leaves_the_fit_unchanged(intercept):
    """Two identical columns make the optimum a face; the fit on it is unique."""
    Y = _panel(5, J=4, T0=8, scale=1e3)
    w, _, gap, _, _ = _fit(Y, 8, intercept)
    Yd = np.vstack([Y, Y[2]])
    wd, _, gap_d, _, _ = _fit(Yd, 8, intercept)
    assert simplex_point_is_optimal(*_program(Yd, 0, np.arange(1, 6), 8, intercept), wd)
    assert wd[1] + wd[4] == pytest.approx(w[1], abs=1e-9)
    np.testing.assert_allclose(gap_d, gap, rtol=0, atol=1e-6)


@pytest.mark.parametrize("intercept", [False, True])
def test_a_target_inside_the_hull_is_fit_exactly(intercept):
    """A convex combination of the donors comes back with zero pre-period gap.

    In levels the first-order solver stopped 0.002 from the combination, at a
    pre-period RMSPE of 0.80 on series near 1e4.
    """
    rng = np.random.default_rng(11)
    donors = 1e4 + np.cumsum(rng.normal(scale=300, size=(5, 30)), axis=1)
    w_true = np.array([0.6, 0.0, 0.3, 0.1, 0.0])
    Y = np.vstack([w_true @ donors, donors])
    w, _, _, pre_rmspe, _ = _fit(Y, 24, intercept)
    np.testing.assert_allclose(w, w_true, atol=1e-8)
    assert pre_rmspe <= 1e-8 * np.abs(Y).max()


def test_flat_series_have_nothing_to_fit_in_intercept_mode():
    """Centred flat series are all zero: every simplex point is optimal, and the
    level shift alone reproduces the target, so the gap is zero."""
    Y = np.vstack([np.full(10, 3.0), np.full(10, 1.0), np.full(10, 5.0)])
    w, cf, gap, pre_rmspe, _ = _fit(Y, 6, intercept=True)
    assert w.min() >= 0.0 and w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(gap, 0.0, atol=1e-12)
    assert pre_rmspe == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("intercept", [False, True])
def test_an_empty_donor_pool_is_refused(intercept):
    Y = _panel(0, J=3, T0=5, scale=1.0)
    with pytest.raises(MlsynthDataError, match="empty donor pool"):
        build_unit_sc(0, np.array([], dtype=int), Y, 5, intercept=intercept)


@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("row, col", [(0, 2), (2, 1), (2, 7)])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_outcomes_are_refused_not_fit(intercept, row, col, bad):
    """In the target or a donor, pre-period or post: the active set returns
    finite-looking weights for a design carrying a NaN, so the helper refuses it
    before solving."""
    Y = _panel(0, J=3, T0=5, scale=1.0)
    Y[row, col] = bad
    with pytest.raises(MlsynthDataError, match="non-finite"):
        build_unit_sc(0, np.arange(1, 4), Y, 5, intercept=intercept)


def test_a_non_finite_unit_outside_the_fit_is_not_its_concern():
    """The check covers the target and its donors, not the rest of the panel."""
    Y = _panel(0, J=4, T0=5, scale=1.0)
    Y[4, 1] = np.inf
    w, cf, _, _, _ = build_unit_sc(0, np.arange(1, 4), Y, 5)
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(cf))


@pytest.mark.parametrize("method", ["cd", "iscm", "iterative", "grossi"])
@pytest.mark.parametrize("year", [1970, 1995])
def test_an_infinite_outcome_is_refused_at_ingestion(german_panel, method, year):
    """Every SPILLSYNTH method sees the same panel; ``inf`` passed the
    missing-cell check and came back as an ATT of nan or -inf, or failed inside
    the solver with an IndexError."""
    d = german_panel.copy()
    d.loc[(d.country == "Norway") & (d.year == year), "gdp"] = np.inf
    with pytest.raises(MlsynthDataError, match="non-finite") as info:
        SPILLSYNTH(dict(df=d, outcome="gdp", treat="treat", unitid="country",
                        time="year", method=method, affected_units=["Austria"],
                        display_graphs=False)).fit()
    assert "Norway" in str(info.value)


# --------------------------------------------------------------------------- #
# Reference: the reference's algorithm with its program solved to the simplex
# --------------------------------------------------------------------------- #
# ``converged_reference_german`` (OSQP at 1e-9) for the inclusive cells. The
# iterative cells are the same algorithm, ``_converged_iterative``, run on the
# panel as floats: the reference's ``repgermany.dta`` stores gdp as int32, and
# the cleaned synthetic written into that array is truncated to whole dollars,
# which moves its post-only cell to -1295.3477.
_CONVERGED = {
    "w_A": 0.4542477064847917,
    "l_WG": 0.3550418242972873,
    "inclusive_att": -1275.5304041033035,
    "naive_att": -1474.4511163878094,
    "iterative_att": -1295.5483482714747,
    "iterative_att_replace_pre": -1615.787265257831,
}


def test_inclusive_scm_reproduces_the_converged_reference(german_inputs):
    fit = run_iscm(german_inputs, intercept=True)
    assert fit.cross_weights["Austria in West Germany"] == pytest.approx(
        _CONVERGED["w_A"], abs=1e-6)
    assert fit.cross_weights["West Germany in Austria"] == pytest.approx(
        _CONVERGED["l_WG"], abs=1e-6)
    assert fit.att == pytest.approx(_CONVERGED["inclusive_att"], abs=1e-3)
    assert fit.att_scm == pytest.approx(_CONVERGED["naive_att"], abs=1e-3)


@pytest.mark.parametrize("replace_pre, key", [
    (False, "iterative_att"), (True, "iterative_att_replace_pre")])
def test_iterative_scm_reproduces_the_converged_reference(german_inputs,
                                                          replace_pre, key):
    fit = run_iterative(german_inputs, intercept=True, replace_pre=replace_pre)
    assert fit.att == pytest.approx(_CONVERGED[key], abs=1e-3)
    if not replace_pre:
        # the refit sees the observed pre-period, so its weights are the naive
        # fit's and so is the contaminated counterfactual built from them
        assert fit.att_scm == pytest.approx(_CONVERGED["naive_att"], abs=1e-3)


# --------------------------------------------------------------------------- #
# Work
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("intercept", [False, True])
def test_outcome_fits_cost_pivots_not_an_iteration_budget(german_inputs,
                                                          monkeypatch, intercept):
    """No first-order projections, and each active set certifies in ``2J`` pivots.

    The first-order solver projected onto the simplex once per iteration, 2000
    times a fit; the inclusive method on this panel made 6000 projections.
    """
    from mlsynth.utils.solvers import active_set, simplex

    projections = {"n": 0}
    solves = []
    real_project, real_qp = simplex.project_simplex, active_set.solve_simplex_qp

    def project(*a, **kw):
        projections["n"] += 1
        return real_project(*a, **kw)

    def qp(B, A, **kw):
        out = real_qp(B, A, **kw)
        if kw.get("return_info"):
            solves.append((np.shape(B)[1], out[1]))
        return out

    monkeypatch.setattr(simplex, "project_simplex", project)
    monkeypatch.setattr(active_set, "solve_simplex_qp", qp)
    run_iscm(german_inputs, intercept=intercept)
    run_iterative(german_inputs, intercept=intercept)

    assert projections["n"] == 0
    # iscm: West Germany, Austria, and the restricted pool; iterative: Austria's
    # clean and West Germany's refit
    assert len(solves) == 5
    for J, info in solves:
        assert info["converged"]
        assert info["pivots"] <= 2 * J
