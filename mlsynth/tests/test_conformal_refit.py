"""The conformal refit rule: which control the test-inversion procedure refits.

Chernozhukov, Wuthrich & Zhu's conformal test subtracts a candidate effect from
the treated post-treatment outcomes, refits the control on the adjusted series,
and compares the post-block residual against the pre-block ones. The refit is
the only place the procedure touches the estimator, so the rule it uses is what
ties the inference to the estimate: a p-value computed from a refit the point
estimate did not use is a p-value about a different control.

Two rules are supported, and they are not interchangeable:

* ``"sc"`` -- the simplex synthetic control (non-negative weights summing to
  one). This is ``estimation_method = "sc"`` in the authors' ``scinference`` R
  package and ``progfunc = "None"`` in ``augsynth``. Convexity bounds the fit,
  so an injected post-period effect cannot be re-levelled away.
* ``"ridge"`` -- the ridge-augmented SCM, ``augsynth``'s ``progfunc = "Ridge"``
  and the refit its conformal mode is defined on. The augmentation leaves the
  simplex by construction, which is what makes it a bias correction and also
  what lets it absorb a post-period level shift.

The reference values below are the authors' own package: ``scinference`` at
``567c688`` on the Swedish carbon tax panel, moving-block scheme, reproduced
here to six decimals.
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue
from mlsynth.utils.conformal import CONFORMAL_REFIT_RULES, conformal_refit_gaps

CARBONTAX = "basedata/carbontax_data.dta"

# scinference @ 567c688, estimation_method="sc", permutation_method="mb",
# alpha=0.1, lsei_type=1, on the panel built below.
SCINFERENCE_CARBONTAX_MB = 0.391304
# Its p-value under any injected post-period effect from +1 upward: the floor of
# the moving-block reference set, 1 / T with T = 46.
SCINFERENCE_CARBONTAX_MB_FLOOR = 0.021739


@pytest.fixture(scope="module")
def carbontax():
    """Sweden vs 14 OECD donors, CO2 transport per capita, treated from 1990."""
    ct = pd.read_stata(CARBONTAX)
    wide = ct.pivot(index="year", columns="country", values="CO2_transport_capita")
    y = wide["Sweden"].to_numpy(dtype=float)
    Y0 = wide.drop(columns=["Sweden"]).to_numpy(dtype=float)
    pre = int((wide.index.to_numpy() < 1990).sum())
    return y, Y0, pre


def _panel(seed=0, pre=18, post=6, n_donors=5):
    rng = np.random.default_rng(seed)
    total = pre + post
    Y0 = rng.normal(size=(total, n_donors)) + rng.normal(size=(1, n_donors)) * 3.0
    y = Y0 @ rng.dirichlet(np.ones(n_donors)) + 0.2 * rng.normal(size=total)
    return y, Y0, pre


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rule", CONFORMAL_REFIT_RULES)
def test_refit_returns_one_finite_gap_per_period(rule):
    y, Y0, _ = _panel()
    gaps = conformal_refit_gaps(y, Y0, rule=rule)
    assert gaps.shape == y.shape
    assert np.all(np.isfinite(gaps))


@pytest.mark.parametrize("rule", CONFORMAL_REFIT_RULES)
def test_refit_can_return_its_base_weights(rule):
    """``return_base`` hands back the weights the next refit can warm-start from."""
    y, Y0, _ = _panel()
    gaps, w_base = conformal_refit_gaps(y, Y0, rule=rule, return_base=True)
    assert gaps.shape == y.shape
    assert w_base.shape == (Y0.shape[1],)
    assert np.all(w_base >= -1e-9)
    assert float(w_base.sum()) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# unit: the rule is what reproduces the authors' package
# --------------------------------------------------------------------------

def test_sc_rule_reproduces_scinference_under_the_null(carbontax):
    """The p-value with no injected effect matches ``scinference`` exactly."""
    y, Y0, pre = carbontax
    p = conformal_pvalue(y, Y0, pre, conformal_type="block", refit="sc")
    assert p == pytest.approx(SCINFERENCE_CARBONTAX_MB, abs=5e-6)


@pytest.mark.parametrize("effect", [1.0, 5.0, 20.0, 100.0])
def test_sc_rule_reproduces_scinference_under_an_injected_effect(carbontax, effect):
    """And matches it at the floor, for every injected effect the authors' run used.

    This is the assertion the suite was missing. Nothing previously checked that
    the conformal test had any power at all: the existing tests pin the shape of
    the interval, the range of the p-value and its determinism, all of which a
    procedure with no power satisfies.
    """
    y, Y0, pre = carbontax
    shifted = y.copy()
    shifted[pre:] += effect
    p = conformal_pvalue(shifted, Y0, pre, conformal_type="block", refit="sc")
    assert p == pytest.approx(SCINFERENCE_CARBONTAX_MB_FLOOR, abs=5e-6)


def test_ridge_rule_is_the_augsynth_refit_and_is_kept(carbontax):
    """The augmented rule stays available and stays the default.

    ``conformal_pvalue`` is augsynth's conformal p-value and the GeoX augsynth
    engine depends on reproducing it, so changing the default would be a
    different defect. Pinned here so the default cannot drift silently.
    """
    y, Y0, pre = carbontax
    explicit = conformal_pvalue(y, Y0, pre, conformal_type="block", refit="ridge")
    default = conformal_pvalue(y, Y0, pre, conformal_type="block")
    assert explicit == default
    assert explicit != pytest.approx(SCINFERENCE_CARBONTAX_MB, abs=5e-6)


def test_reselecting_the_penalty_on_the_full_window_costs_the_test_its_power(carbontax):
    """The second fault the same incident had, isolated.

    ``conformal_pvalue`` called without ``lambda_`` cross-validates the ridge
    penalty on the matching window -- which, for a conformal refit, is the whole
    path including the post block. That window picks a penalty orders of
    magnitude below the one the pre-period fit picks, and the augmentation it
    then permits absorbs the effect: the p-value stops responding to the effect
    size entirely, reading 0.348 whether the injected effect is 5 or 100.

    Pinning the penalty at the fitted value -- what augsynth does, and what the
    GeoX engine and VanillaSC both pass -- leaves the test its power. The
    docstring used to call re-selection "slightly anti-conservative"; the
    numbers below are why it no longer does.
    """
    y, Y0, pre = carbontax
    from mlsynth.utils.bilevel.ridge_augment import ridge_augment_weights

    fitted_lambda = float(ridge_augment_weights(y[:pre], Y0[:pre]).lambda_)
    stalled, powered = [], []
    for effect in (5.0, 100.0):
        shifted = y.copy()
        shifted[pre:] += effect
        stalled.append(conformal_pvalue(shifted, Y0, pre, conformal_type="block",
                                        refit="ridge"))
        powered.append(conformal_pvalue(shifted, Y0, pre, conformal_type="block",
                                        refit="ridge", lambda_=fitted_lambda))
    # A twentyfold larger effect leaves the re-selected penalty's p-value
    # unmoved -- that is the ceiling, stated as an assertion.
    assert stalled[0] == pytest.approx(stalled[1], abs=1e-9)
    assert stalled[0] > 0.3
    # The pinned penalty reaches the floor of the reference set for both.
    assert powered == [pytest.approx(SCINFERENCE_CARBONTAX_MB_FLOOR, abs=5e-6)] * 2


def test_the_two_rules_disagree_on_the_same_panel(carbontax):
    """If they agreed, the argument would be decoration rather than a choice."""
    y, Y0, pre = carbontax
    sc = conformal_refit_gaps(y, Y0, rule="sc")
    ridge = conformal_refit_gaps(y, Y0, rule="ridge")
    assert not np.allclose(sc, ridge)
    # The augmented fit is the better in-sample fit -- that is what it is for.
    assert np.mean(np.abs(ridge)) < np.mean(np.abs(sc))


# --------------------------------------------------------------------------
# edge
# --------------------------------------------------------------------------

def test_single_donor_leaves_the_only_weight_at_one():
    """With one donor the simplex is a point, so the fit is that donor."""
    y, Y0, _ = _panel(n_donors=1)
    gaps, w = conformal_refit_gaps(y, Y0, rule="sc", return_base=True)
    assert w == pytest.approx(np.array([1.0]))
    assert gaps == pytest.approx(y - Y0[:, 0])


def test_collinear_donors_still_produce_finite_gaps():
    """Duplicated donors make the weights non-unique; the fit is still unique."""
    y, Y0, _ = _panel(n_donors=3)
    duplicated = np.hstack([Y0, Y0])
    gaps = conformal_refit_gaps(y, duplicated, rule="sc")
    assert np.all(np.isfinite(gaps))
    assert gaps == pytest.approx(conformal_refit_gaps(y, Y0, rule="sc"), abs=1e-6)


def test_a_treated_series_outside_the_hull_is_not_forced_inside():
    """The gaps report the miss rather than the refit pretending to close it."""
    y, Y0, _ = _panel()
    y = y + 50.0
    gaps = conformal_refit_gaps(y, Y0, rule="sc")
    assert np.all(gaps > 0.0)


def test_single_post_period_is_a_valid_block(carbontax):
    """A one-period post block is the per-period test ``conformal_intervals`` runs."""
    y, Y0, _ = carbontax
    p = conformal_pvalue(y, Y0, y.size - 1, conformal_type="block", refit="sc")
    assert 0.0 < p <= 1.0


# --------------------------------------------------------------------------
# failure: the contracts the rule argument now enforces
# --------------------------------------------------------------------------

def test_unknown_rule_is_rejected():
    y, Y0, _ = _panel()
    with pytest.raises(MlsynthConfigError, match="refit rule"):
        conformal_refit_gaps(y, Y0, rule="lasso")


def test_covariates_are_rejected_by_the_sc_rule():
    """The simplex rule is outcome-only, so silently dropping covariates is wrong.

    ``scinference``'s ``estimation_method = "sc"`` matches on outcomes alone.
    Accepting ``Z0`` / ``z1`` and ignoring them would produce a fit that is not
    the one the caller asked for, and no output would show it.
    """
    y, Y0, _ = _panel()
    Z0 = np.ones((Y0.shape[1], 2))
    with pytest.raises(MlsynthConfigError, match="covariates"):
        conformal_refit_gaps(y, Y0, rule="sc", Z0=Z0, z1=np.ones(2))


def test_mismatched_donor_length_is_reported():
    y, Y0, _ = _panel()
    with pytest.raises((MlsynthDataError, ValueError)):
        conformal_refit_gaps(y, Y0[:-1], rule="sc")


def test_empty_donor_pool_is_reported():
    y, _, _ = _panel()
    with pytest.raises((MlsynthConfigError, MlsynthDataError, ValueError)):
        conformal_refit_gaps(y, np.empty((y.size, 0)), rule="sc")


def test_conformal_pvalue_rejects_an_unknown_rule(carbontax):
    y, Y0, pre = carbontax
    with pytest.raises(MlsynthConfigError, match="refit rule"):
        conformal_pvalue(y, Y0, pre, refit="nonesuch")
