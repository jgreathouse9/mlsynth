"""Metamorphic invariants of the conformal refit rule.

A test-inversion conformal procedure works by refitting the control on outcomes
that have had a candidate effect subtracted, then asking whether the residuals
on the post block look like the residuals on the pre block. Everything the test
can see is the residual path, so the refit rule decides how much the test can
see.

The invariant these properties pin is *saturation*. Inject a constant shift
``tau`` into the post block of the treated series and refit. The counterfactual
the refit produces on the *pre* block must stop moving once ``tau`` is large:
the pre-period outcomes were not touched, so a refit that keeps chasing a larger
and larger post block is redistributing the treatment effect into the very
residuals that form the test's reference distribution. Convexity gives
saturation for free -- a convex combination of donors is bounded by the donor
pool whatever the treated series does -- so the ``"sc"`` rule satisfies it on
every panel, exactly. An unconstrained rule does not: its displacement is linear
in ``tau``, without bound.

The consequence, asserted here on the p-value itself, is power. If the pre-block
residuals grow with ``tau`` alongside the post-block ones, the observed
statistic and its reference distribution grow together and the p-value stalls at
a ceiling no effect size can push it past. Saturation is what stops that.

The properties:

* saturation -- ``d(10 tau) ~ d(tau)`` for large ``tau``, where ``d`` is the
  largest pre-block displacement of the counterfactual;
* boundedness -- that displacement never exceeds the donor pool's own range,
  the bound convexity supplies;
* power -- a large enough injected effect drives the block-scheme p-value to
  ``1 / T``, the floor of its reference set;
* equivariance -- the refit commutes with a common rescaling of the treated and
  donor series, and with a relabelling of the donors.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.conformal import conformal_refit_gaps
from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue

_SETTINGS = settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)

# Large enough that the simplex weights have reached a vertex, so the two scales
# below differ only in whether the rule keeps responding after that.
_TAU = 1.0e3


@st.composite
def panels(draw):
    """A donor panel with a treated unit inside the donor pool's convex hull."""
    n_donors = draw(st.integers(3, 12))
    pre = draw(st.integers(14, 32))
    post = draw(st.integers(3, 14))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    total = pre + post
    # Heterogeneous donor levels and trends: the structure an unconstrained
    # refit exploits to manufacture a post-only level shift.
    Y0 = (rng.normal(size=(total, n_donors))
          + rng.normal(size=(1, n_donors)) * 3.0
          + np.arange(total)[:, None] * rng.normal(scale=0.1, size=(1, n_donors)))
    y = Y0 @ rng.dirichlet(np.ones(n_donors)) + 0.2 * rng.normal(size=total)
    return y, Y0, pre


def _counterfactual(y, Y0, pre, tau, rule):
    """The refit's fitted counterfactual with ``tau`` injected into the post block."""
    shifted = np.asarray(y, dtype=float).copy()
    shifted[pre:] += tau
    return shifted - conformal_refit_gaps(shifted, Y0, rule=rule)


def _pre_displacement(y, Y0, pre, tau, rule):
    """How far the pre-block counterfactual moves when ``tau`` is injected."""
    base = _counterfactual(y, Y0, pre, 0.0, rule)
    shifted = _counterfactual(y, Y0, pre, tau, rule)
    return float(np.max(np.abs(shifted[:pre] - base[:pre])))


@given(panels())
@_SETTINGS
def test_sc_refit_saturates_in_the_injected_shift(panel):
    """Ten times the shift must not move the pre block ten times as far.

    The pre-period outcomes are identical in both refits; only the post block
    differs. A rule whose pre-block fit keeps tracking the post block is
    contaminating the test's reference distribution with the effect it is meant
    to detect.
    """
    y, Y0, pre = panel
    near = _pre_displacement(y, Y0, pre, _TAU, "sc")
    far = _pre_displacement(y, Y0, pre, 10.0 * _TAU, "sc")
    assert far <= near + 1e-6 * max(near, 1.0)


@given(panels())
@_SETTINGS
def test_sc_refit_stays_inside_the_donor_pool(panel):
    """The refit counterfactual is a convex combination, so the pool bounds it.

    This is the reason saturation holds, asserted directly: whatever is done to
    the treated series, the fitted values cannot leave the per-period range of
    the donors, so their displacement is bounded by the pool's own spread.
    """
    y, Y0, pre = panel
    lo, hi = Y0.min(axis=1), Y0.max(axis=1)
    for tau in (0.0, _TAU, 10.0 * _TAU):
        cf = _counterfactual(y, Y0, pre, tau, "sc")
        tol = 1e-6 * max(1.0, float(np.max(np.abs(Y0))))
        assert np.all(cf >= lo - tol) and np.all(cf <= hi + tol)


@given(panels())
@_SETTINGS
def test_sc_refit_has_power_against_a_large_effect(panel):
    """A large enough effect must reach the floor of the reference set.

    Under the block scheme the reference set is the ``T`` cyclic shifts of the
    residual path, one of which is the observed path, so ``1 / T`` is the
    smallest p-value attainable. Any procedure worth inverting must get there
    for an effect large enough, and a saturating refit does.
    """
    y, Y0, pre = panel
    T = y.size
    shifted = y.copy()
    shifted[pre:] += 1.0e4 * max(1.0, float(np.std(y)))
    p = conformal_pvalue(shifted, Y0, pre, conformal_type="block", refit="sc")
    assert p == pytest.approx(1.0 / T)


@given(panels(), st.floats(0.25, 4.0))
@_SETTINGS
def test_refit_gaps_are_scale_equivariant(panel, scale):
    """Rescaling treated and donors together rescales the gaps by the same factor."""
    y, Y0, _ = panel
    base = conformal_refit_gaps(y, Y0, rule="sc")
    scaled = conformal_refit_gaps(y * scale, Y0 * scale, rule="sc")
    assert np.allclose(scaled, base * scale, atol=1e-6 * max(1.0, abs(scale)))


@given(panels(), st.integers(0, 2**31 - 1))
@_SETTINGS
def test_refit_gaps_ignore_donor_order(panel, seed):
    """Donors are exchangeable: permuting the columns cannot change the gaps."""
    y, Y0, _ = panel
    order = np.random.default_rng(seed).permutation(Y0.shape[1])
    base = conformal_refit_gaps(y, Y0, rule="sc")
    permuted = conformal_refit_gaps(y, Y0[:, order], rule="sc")
    assert np.allclose(permuted, base, atol=1e-6)


@given(panels())
@_SETTINGS
def test_ridge_refit_does_not_saturate(panel):
    """The augmented rule keeps responding, which is why it must not be the default.

    Not a defect in the ridge augmentation itself -- relaxing the simplex is the
    whole point of it, and augsynth's conformal mode is defined on the augmented
    fit. It is a fact about the rule that the caller has to choose knowingly,
    and pinning it here is what makes the choice visible.
    """
    y, Y0, pre = panel
    near = _pre_displacement(y, Y0, pre, _TAU, "ridge")
    far = _pre_displacement(y, Y0, pre, 10.0 * _TAU, "ridge")
    assert far > 2.0 * near
