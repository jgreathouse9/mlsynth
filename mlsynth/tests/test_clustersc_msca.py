"""MSCa weights on the CLUSTERSC RPCA arm: free intercept, donors on the simplex.

``weight_objective="msca"`` fits

.. math::

   \\min_{a,\; w \\geq 0,\; \\mathbf{1}^\\top w = 1}
       \\| y^- - L^- w - a \\mathbf{1} \\|_2^2,

the Li-Shankar (2023) MSC(a) program against the denoised donors. It keeps
the Abadie-Diamond-Hainmueller convex-hull restriction on the donor weights
and frees the level, so the treated unit no longer has to lie inside the
donors' hull in levels -- only parallel to a point in it.

The substitution is an identification change, not a fit improvement: it
replaces "the treated unit is in the donors' convex hull" with "the level
gap between the treated unit and its synthetic control is constant". The
tests below pin what the program is, that the intercept is applied to the
post-period counterfactual and reported to the caller, and that the
degenerate inputs behave.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.clustersc_helpers.rpca.pipeline import run_rpca
from mlsynth.utils.clustersc_helpers.rpca.weights import solve_msca, solve_nnls
from mlsynth.utils.clustersc_helpers.pcr.convex import solve_simplex

_T0, _T = 24, 32
_OFFSET = 7.5


@pytest.fixture(scope="module")
def offset_panel():
    """Treated = convex combination of donors, lifted by a constant.

    The lift is what gives these tests power. Without it the intercept is
    zero at the optimum, MSCa and simplex coincide, and every assertion
    below passes for a reason that has nothing to do with the intercept.
    """
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 1.0, _T)
    basis = np.column_stack([np.ones(_T), t, np.sin(3.0 * t), t ** 2])
    donors = basis @ rng.normal(size=(4, 6)) * 3.0 + 40.0
    donors += rng.normal(scale=0.05, size=donors.shape)
    truth = np.zeros(6)
    truth[[0, 2, 4]] = (0.5, 0.3, 0.2)
    treated = donors @ truth + _OFFSET + rng.normal(scale=0.02, size=_T)
    return treated, donors, [f"d{i}" for i in range(6)], truth


@pytest.fixture(scope="module")
def centered_panel():
    """Treated is an exact convex combination -- no level gap to absorb."""
    rng = np.random.default_rng(5)
    t = np.linspace(0.0, 1.0, _T)
    donors = np.column_stack([np.ones(_T), t, np.cos(2.0 * t), t ** 1.5]) @ \
        rng.normal(size=(4, 5)) * 2.0 + 30.0
    truth = np.array([0.4, 0.0, 0.35, 0.0, 0.25])
    return donors @ truth, donors, [f"d{i}" for i in range(5)], truth


# --------------------------------------------------------------------------
# Smoke
# --------------------------------------------------------------------------
def test_solve_msca_returns_finite_weights_and_intercept(offset_panel):
    treated, donors, _, _ = offset_panel
    beta, a = solve_msca(donors[:_T0], treated[:_T0])
    assert beta.shape == (donors.shape[1],)
    assert np.all(np.isfinite(beta)) and np.isfinite(a)


def test_pipeline_accepts_msca_end_to_end(offset_panel):
    treated, donors, names, _ = offset_panel
    fit = run_rpca(treated_outcome=treated, donor_outcomes=donors,
                   donor_names=names, T0=_T0, weight_objective="msca",
                   k_clusters=1)
    assert np.all(np.isfinite(fit.counterfactual))
    assert np.isfinite(fit.att) and np.isfinite(fit.pre_rmse)
    assert fit.metadata["weight_objective"] == "msca"


# --------------------------------------------------------------------------
# Unit: what the program is
# --------------------------------------------------------------------------
def test_donor_weights_stay_on_the_simplex(offset_panel):
    """The hull restriction survives; only the level is freed."""
    treated, donors, _, _ = offset_panel
    beta, _ = solve_msca(donors[:_T0], treated[:_T0])
    assert beta.min() >= -1e-8
    assert beta.sum() == pytest.approx(1.0, abs=1e-6)


def test_the_intercept_is_free_to_be_negative(offset_panel):
    """A treated unit below its donors needs a negative intercept.

    Constraining the intercept to be non-negative alongside the donor
    weights is the natural mis-implementation; it would clamp to zero here
    and the fit would collapse onto the simplex solution.
    """
    treated, donors, _, _ = offset_panel
    lowered = treated - 2.0 * _OFFSET
    _, a = solve_msca(donors[:_T0], lowered[:_T0])
    assert a < -1.0


def test_the_intercept_recovers_a_known_level_shift(offset_panel):
    """The intercept is identified here even though the weights are not.

    The donor block is ill-conditioned (cond ~ 3e3), so many weight
    vectors fit equally well and asserting on them would pin solver noise.
    The level shift is the mean gap between the treated unit and its
    synthetic control, which every one of those vectors agrees on.
    """
    treated, donors, _, truth = offset_panel
    beta, a = solve_msca(donors[:_T0], treated[:_T0])
    assert a == pytest.approx(_OFFSET, abs=0.15)
    on_support = float(beta[np.flatnonzero(truth)].sum())
    assert on_support > 0.95
    fitted = donors[:_T0] @ beta + a
    assert fitted == pytest.approx(treated[:_T0], abs=0.2)


def test_shifting_the_treated_unit_moves_only_the_intercept(offset_panel):
    """Shift equivariance: y -> y + c sends a -> a + c with w unchanged.

    This is the defining property of a free intercept and it fails for
    every objective that does not have one.
    """
    treated, donors, _, _ = offset_panel
    b0, a0 = solve_msca(donors[:_T0], treated[:_T0])
    b1, a1 = solve_msca(donors[:_T0], treated[:_T0] + 13.0)
    assert a1 - a0 == pytest.approx(13.0, abs=1e-5)
    assert b1 == pytest.approx(b0, abs=1e-5)


def test_msca_is_never_worse_than_simplex_in_pre_period_fit(offset_panel):
    """Simplex is MSCa with the intercept pinned to zero, so it is nested."""
    treated, donors, _, _ = offset_panel
    D, y = donors[:_T0], treated[:_T0]
    b_m, a_m = solve_msca(D, y)
    b_s = solve_simplex(D, y)
    r_m = np.sqrt(np.mean((y - D @ b_m - a_m) ** 2))
    r_s = np.sqrt(np.mean((y - D @ b_s) ** 2))
    assert r_m <= r_s + 1e-8
    assert r_m < 0.5 * r_s          # and on an offset panel, strictly better


def test_intercept_is_near_zero_when_the_treated_unit_is_in_the_hull(centered_panel):
    treated, donors, _, _ = centered_panel
    _, a = solve_msca(donors[:_T0], treated[:_T0])
    assert abs(a) < 1e-3


# --------------------------------------------------------------------------
# Unit: the intercept reaches the counterfactual and the caller
# --------------------------------------------------------------------------
def test_the_free_intercept_zeroes_the_mean_pre_period_residual(offset_panel):
    """The first-order condition for an unconstrained ``a``.

    Differentiating the objective in ``a`` gives ``sum(y - Lw - a) = 0``,
    so the mean pre-period gap is exactly zero whatever the donor weights
    do. No objective without a free intercept satisfies this, which makes
    it the sharpest available check that the intercept was fitted, not
    defaulted.
    """
    treated, donors, names, _ = offset_panel
    fit = run_rpca(treated_outcome=treated, donor_outcomes=donors,
                   donor_names=names, T0=_T0, weight_objective="msca",
                   k_clusters=1)
    assert float(np.mean(fit.gap[:_T0])) == pytest.approx(0.0, abs=1e-7)


def test_simplex_does_not_zero_the_mean_pre_period_residual(offset_panel):
    """The control for the test above: without the intercept the treated
    unit's level gap has nowhere to go, so the residual carries it."""
    treated, donors, names, _ = offset_panel
    fit = run_rpca(treated_outcome=treated, donor_outcomes=donors,
                   donor_names=names, T0=_T0, weight_objective="simplex",
                   k_clusters=1)
    assert abs(float(np.mean(fit.gap[:_T0]))) > 1.0


def test_the_intercept_is_applied_in_the_post_period_too(offset_panel):
    """Lifting the treated unit lifts the whole counterfactual, ATT intact.

    With ``k_clusters=1`` the donor set and the denoised donor matrix do
    not depend on the treated unit, so shifting it by ``c`` moves only the
    weight step: the donor weights are unchanged and the intercept rises
    by ``c``. Applying the intercept pre-period and dropping it afterwards
    would leave the post-period counterfactual flat and pull the ATT down
    by ``c`` -- the bias this objective is most exposed to.
    """
    treated, donors, names, _ = offset_panel
    shift = 6.0
    base = run_rpca(treated_outcome=treated, donor_outcomes=donors,
                    donor_names=names, T0=_T0, weight_objective="msca",
                    k_clusters=1)
    lifted = run_rpca(treated_outcome=treated + shift, donor_outcomes=donors,
                      donor_names=names, T0=_T0, weight_objective="msca",
                      k_clusters=1)
    # 1e-4 is CLARABEL's reach on a panel at this scale, not a weakened
    # check: dropping the intercept post-period would miss by ~10 here.
    assert lifted.intercept - base.intercept == pytest.approx(shift, abs=1e-4)
    assert lifted.counterfactual == pytest.approx(base.counterfactual + shift,
                                                  abs=1e-4)
    assert lifted.att == pytest.approx(base.att, abs=1e-4)


def test_the_intercept_is_reported_not_swallowed(offset_panel):
    treated, donors, names, _ = offset_panel
    fit = run_rpca(treated_outcome=treated, donor_outcomes=donors,
                   donor_names=names, T0=_T0, weight_objective="msca",
                   k_clusters=1)
    assert fit.intercept == pytest.approx(fit.metadata["weight_intercept"], abs=0)
    assert abs(fit.intercept) > 1.0


@pytest.mark.parametrize("objective", ["nnls", "simplex"])
def test_non_intercept_objectives_report_a_zero_intercept(offset_panel, objective):
    treated, donors, names, _ = offset_panel
    fit = run_rpca(treated_outcome=treated, donor_outcomes=donors,
                   donor_names=names, T0=_T0, weight_objective=objective,
                   k_clusters=1)
    assert fit.intercept == 0.0
    assert fit.metadata["weight_intercept"] == 0.0


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------
def test_single_donor_puts_all_weight_on_it_and_the_rest_in_the_intercept():
    rng = np.random.default_rng(3)
    d = np.linspace(10.0, 20.0, _T)[:, None] + rng.normal(scale=0.01, size=(_T, 1))
    y = d[:, 0] + 4.0
    beta, a = solve_msca(d[:_T0], y[:_T0])
    assert beta == pytest.approx([1.0], abs=1e-6)
    assert a == pytest.approx(4.0, abs=0.05)


def test_duplicate_donors_split_mass_without_moving_the_fit(offset_panel):
    treated, donors, _, _ = offset_panel
    D = donors[:_T0]
    y = treated[:_T0]
    b0, a0 = solve_msca(D, y)
    Ddup = np.column_stack([D, D[:, 0]])
    b1, a1 = solve_msca(Ddup, y)
    assert a1 == pytest.approx(a0, abs=1e-4)
    assert b1.sum() == pytest.approx(1.0, abs=1e-6)
    assert (Ddup @ b1) == pytest.approx(D @ b0, abs=1e-3)


def test_two_pre_periods_is_solvable(offset_panel):
    treated, donors, _, _ = offset_panel
    beta, a = solve_msca(donors[:2], treated[:2])
    assert beta.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(a)


def test_a_constant_donor_block_leaves_the_intercept_carrying_the_level():
    """Degenerate: every donor is the same flat line. The donor term is then
    a constant whatever the weights, so the intercept is identified only up
    to that constant -- the fitted counterfactual must still be right."""
    y = np.full(_T, 9.0)
    D = np.full((_T, 3), 2.0)
    beta, a = solve_msca(D[:_T0], y[:_T0])
    assert beta.sum() == pytest.approx(1.0, abs=1e-6)
    assert (D[:_T0] @ beta + a) == pytest.approx(y[:_T0], abs=1e-5)


# --------------------------------------------------------------------------
# Failure: bad input is reported, not swallowed
# --------------------------------------------------------------------------
def test_msca_is_an_accepted_objective_on_the_config():
    from mlsynth.config_models import CLUSTERSCConfig
    import inspect
    field = CLUSTERSCConfig.model_fields["weight_objective"]
    assert "msca" in str(field.annotation)


def test_non_2d_donor_matrix_raises_estimation_error(offset_panel):
    treated, donors, _, _ = offset_panel
    with pytest.raises(MlsynthEstimationError, match="2D"):
        solve_msca(donors[:_T0, 0], treated[:_T0])


def test_pre_period_length_mismatch_raises_estimation_error(offset_panel):
    treated, donors, _, _ = offset_panel
    with pytest.raises(MlsynthEstimationError, match="mismatch"):
        solve_msca(donors[:_T0], treated[: _T0 - 3])


def test_a_solver_failure_is_reported_not_returned_as_weights(offset_panel, monkeypatch):
    """CLARABEL has returned ``infeasible`` on a simplex at real panel
    magnitudes, so this branch is defensive but not unreachable. A silent
    ``None`` here would surface as a crash deep in the projection step
    instead of as the estimation error that names the cause."""
    import cvxpy as cp
    treated, donors, _, _ = offset_panel
    monkeypatch.setattr(cp.Problem, "solve", lambda self, **kw: None)
    with pytest.raises(MlsynthEstimationError, match="did not converge"):
        solve_msca(donors[:_T0], treated[:_T0])


def test_unknown_weight_objective_still_raises_config_error(offset_panel):
    treated, donors, names, _ = offset_panel
    with pytest.raises(MlsynthConfigError):
        run_rpca(treated_outcome=treated, donor_outcomes=donors,
                 donor_names=names, T0=_T0, weight_objective="mscz")


# --------------------------------------------------------------------------
# The public estimator surface
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def germany_df():
    import pandas as pd
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "basedata" / "german_reunification.csv")
    df["treat"] = df["Reunification"].astype(int)
    return df


def _cfg(df, **kw):
    return dict(df=df, outcome="gdp", treat="treat", unitid="country",
                time="year", method="rpca", display_graphs=False, **kw)


def test_clustersc_runs_msca_end_to_end(germany_df):
    from mlsynth import CLUSTERSC
    r = CLUSTERSC(_cfg(germany_df, weight_objective="msca")).fit()
    assert np.isfinite(r.effects.att)
    assert np.isfinite(r.fit_diagnostics.rmse_pre)
    w = r.weights.donor_weights
    assert min(w.values()) >= -1e-8
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_clustersc_reports_the_objective_and_its_intercept(germany_df):
    """A caller reading the weights alone is wrong by the intercept at every
    period, so both have to reach the result."""
    from mlsynth import CLUSTERSC
    r = CLUSTERSC(_cfg(germany_df, weight_objective="msca")).fit()
    params = r.method_details.parameters_used
    assert params["weight_objective"] == "msca"
    assert np.isfinite(params["weight_intercept"])
    assert abs(params["weight_intercept"]) > 0.0


def test_clustersc_msca_beats_simplex_on_pre_fit(germany_df):
    """Nested programs: simplex is msca with the intercept pinned to zero."""
    from mlsynth import CLUSTERSC
    m = CLUSTERSC(_cfg(germany_df, weight_objective="msca")).fit()
    s = CLUSTERSC(_cfg(germany_df, weight_objective="simplex")).fit()
    assert m.fit_diagnostics.rmse_pre <= s.fit_diagnostics.rmse_pre + 1e-6


def test_clustersc_default_still_reports_a_zero_intercept(germany_df):
    """The default path is unchanged: Bayani's nnls, fitted through the
    origin, and the Bayani weights to the pinned tolerance."""
    from mlsynth import CLUSTERSC
    r = CLUSTERSC(_cfg(germany_df)).fit()
    params = r.method_details.parameters_used
    assert params["weight_objective"] == "nnls"
    assert params["weight_intercept"] == 0.0
    w = r.weights.donor_weights
    assert w["Norway"] == pytest.approx(0.485, abs=5e-3)
    assert w["France"] == pytest.approx(0.354, abs=5e-3)


# --------------------------------------------------------------------------
# scpi prediction intervals must be centred on the fit that was reported
# --------------------------------------------------------------------------
def _spy_scpi(monkeypatch):
    """Capture the arguments the pipeline hands scpi."""
    import mlsynth.utils.clustersc_helpers.scpi_pi as scpi_pi
    seen = {}
    real = scpi_pi.scpi_pi_inference

    def spy(treated, donor_full, T0, weights, **kw):
        seen["weights"] = np.asarray(weights, float).copy()
        seen["constant"] = kw.get("constant", False)
        seen["n_donor_cols"] = np.asarray(donor_full).shape[1]
        return real(treated, donor_full, T0, weights, **kw)

    monkeypatch.setattr(scpi_pi, "scpi_pi_inference", spy)
    return seen


def _germany_arrays(df):
    from mlsynth.utils.datautils import dataprep
    prep = dataprep(df, "country", "year", "gdp", "treat")
    return dict(treated_outcome=np.asarray(prep["y"], float).ravel(),
                donor_outcomes=np.asarray(prep["donor_matrix"], float),
                donor_names=list(prep["donor_names"]),
                T0=int(prep["pre_periods"]))


def test_scpi_receives_the_msca_intercept_as_its_own_column(germany_df, monkeypatch):
    """scpi refits under its own constraint family, so it has to be told
    which family that is. An MSC(a) fit is simplex donors plus a free
    intercept, which is scpi's ``constant=True``; handing it the donor
    weights alone would have it re-estimate a plain simplex fit and report
    intervals for a model the caller was never given.
    """
    seen = _spy_scpi(monkeypatch)
    fit = run_rpca(**_germany_arrays(germany_df), weight_objective="msca",
                   compute_scpi_pi=True, scpi_constraint="simplex", scpi_sims=50)
    assert seen["constant"] is True
    assert seen["weights"].size == seen["n_donor_cols"] + 1
    assert seen["weights"][-1] == pytest.approx(fit.intercept, abs=0)
    assert abs(fit.intercept) > 1.0


@pytest.mark.parametrize("objective", ["nnls", "simplex"])
def test_scpi_gets_no_constant_for_the_origin_objectives(
        germany_df, monkeypatch, objective):
    seen = _spy_scpi(monkeypatch)
    run_rpca(**_germany_arrays(germany_df), weight_objective=objective,
             compute_scpi_pi=True, scpi_constraint="simplex", scpi_sims=50)
    assert seen["constant"] is False
    assert seen["weights"].size == seen["n_donor_cols"]


# --------------------------------------------------------------------------
# A better pre-period fit is not a better estimate
# --------------------------------------------------------------------------
_GERMANY_REFERENCE_ATT = -1543.5


def test_msca_matches_the_reference_on_a_well_spanned_cluster(germany_df):
    """fGRC clustering plus HSVT keeps the donors West Germany needs, and
    there every objective agrees to within 60 of the pinned prototype."""
    from mlsynth import CLUSTERSC
    good = dict(rpca_method="HSVT", cluster_method="fgrc", fgrc_k=2)
    for objective in ("nnls", "simplex", "msca"):
        r = CLUSTERSC(_cfg(germany_df, weight_objective=objective, **good)).fit()
        assert r.effects.att == pytest.approx(_GERMANY_REFERENCE_ATT, abs=60.0), (
            f"{objective} drifted from the reference on a well-spanned cluster")


def test_msca_improves_the_fit_without_repairing_a_bad_donor_set(germany_df):
    """The failure this objective is most likely to produce.

    FPCA clustering on West Germany drops donors carrying 0.55 of the
    pool-optimal convex mass, which the spannability check reports. Against
    that cluster the intercept cuts the pre-period error well below the
    simplex's and still leaves the effect far short, because a level shift
    cannot stand in for donors that are missing. A reader treating the
    improved fit as a sign the problem went away gets an ATT roughly half
    the reference.
    """
    from mlsynth import CLUSTERSC
    fits = {o: CLUSTERSC(_cfg(germany_df, weight_objective=o)).fit()
            for o in ("nnls", "simplex", "msca")}
    rmse = {o: f.fit_diagnostics.rmse_pre for o, f in fits.items()}
    err = {o: abs(f.effects.att - _GERMANY_REFERENCE_ATT) for o, f in fits.items()}

    # Better fit than the simplex ...
    assert rmse["msca"] < 0.6 * rmse["simplex"]
    # ... and further from the reference than the objective that extrapolates.
    assert err["msca"] > 3.0 * err["nnls"]
    assert err["msca"] > 500.0
    # The level shift is doing the work a missing donor should do.
    params = fits["msca"].method_details.parameters_used
    assert params["weight_intercept"] > 300.0
