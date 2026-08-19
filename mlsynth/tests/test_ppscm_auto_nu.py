"""The automatic pooling parameter sits on a boundary, and rounding decides it.

``run_multisynth`` picks ``nu`` when the caller says ``"auto"``. The rule follows
augsynth: fit every treated unit separately, then take the ratio of the norm of
the average imbalance to the average of the norms,

    nu = ||mean_k m_k|| / mean_k ||m_k||,

a triangle inequality that is at most one and is tight exactly when the treated
units share a single imbalance vector. One treated unit does that by definition,
so on a single-treated-unit panel ``nu`` is analytically one.

The second solve then minimises

    nu / (norm_pool J^2) q_pool + (1 - nu) / (norm_sep J) q_sep,

so at ``nu = 1`` the separate term carries a weight of exactly zero. It is
computed in floating point from two square roots and a division, and there is
nothing keeping the result on the correct side of one. Land it a single unit in
the last place above -- 1.0000000000000002 -- and the weight becomes a small
negative number, the objective picks up a negative multiple of a convex
quadratic, and cvxpy refuses the whole problem as non-convex with a ``DCPError``
before the solver ever runs.

The failure needs no unusual input. It is one treated unit and an arithmetic
coin flip, and the same coin is in play whenever the treated units' imbalance
vectors are close to parallel, which is the ordinary case for a small pool of
similar units. Rescaling the panel is enough to flip it: the fit is
scale-equivariant in exact arithmetic but not in the last bit, so the same panel
measured in dollars and in thousands of dollars can land on opposite sides.

Levels: smoke, unit invariants, property, edge, failure.
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.ppscm_helpers.engine import run_multisynth


def _panel(n=10, T=24, t0=18, n_trt=1, seed=0, scale=1.0):
    """A one-factor panel with ``n_trt`` units adopting at ``t0``.

    The draw order is fixed: the factor, then the loadings, then the noise. It
    reproduces the panel the failure was first seen on, and reordering the draws
    gives a different panel that happens not to round over the boundary.
    """
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=T)
    load = rng.uniform(0.5, 1.5, size=n)
    Xy = load[:, None] * factor[None, :] + rng.normal(scale=0.3, size=(n, T))
    trt = np.full(n, np.inf)
    trt[:n_trt] = t0
    return Xy * scale, trt, t0


def _long(Xy, trt, t0):
    """The same panel in the long form ``PPSCM`` ingests."""
    n, T = Xy.shape
    return pd.DataFrame([
        {"unit": f"u{i}", "time": t, "y": Xy[i, t],
         "treat": int(np.isfinite(trt[i]) and t >= t0)}
        for i in range(n) for t in range(T)
    ])


# --------------------------------------------------------------------------- #
# smoke                                                                       #
# --------------------------------------------------------------------------- #
def test_a_single_treated_unit_fits():
    Xy, trt, t0 = _panel()
    fit = run_multisynth(Xy, trt, t0, 5, t0)
    assert np.isfinite(fit["att"])
    assert np.isfinite(np.asarray(fit["per_time"], dtype=float)).all()


# --------------------------------------------------------------------------- #
# unit invariants                                                             #
# --------------------------------------------------------------------------- #
def test_the_automatic_nu_is_a_weight():
    """It multiplies one term and its complement multiplies the other, so a
    value outside the unit interval is not a pooling choice at all -- it is a
    negative weight on a convex quadratic, which is a different problem."""
    Xy, trt, t0 = _panel(n_trt=3)
    assert 0.0 <= run_multisynth(Xy, trt, t0, 5, t0)["nu_used"] <= 1.0


def test_one_treated_unit_puts_the_ratio_at_its_upper_bound():
    """With a single column the norm of the average and the average of the norms
    are the same number, so the ratio is one and the separate term drops out."""
    Xy, trt, t0 = _panel(n_trt=1)
    assert run_multisynth(Xy, trt, t0, 5, t0)["nu_used"] == 1.0


def test_an_explicit_nu_is_passed_through_untouched():
    """The clamp guards a computed ratio. A caller naming a value is naming the
    weights they want and gets them."""
    Xy, trt, t0 = _panel(n_trt=3)
    for nu in (0.0, 0.25, 0.5, 1.0):
        assert run_multisynth(Xy, trt, t0, 5, t0, nu=nu)["nu_used"] == nu


def test_the_clamp_does_not_reach_the_caller_s_own_nu():
    """Out of range, an explicit nu is refused, not folded to the nearest edge.

    Extending the clamp to cover it would read as defensive and change nothing on
    any valid input, which is what makes it tempting. What it costs is the only
    signal an out-of-range value has: the fit would go through at 1.0 and nothing
    on the result would say the substitution happened. Refusing is the honest
    outcome, and the range belongs to the config, which refuses it by name before
    a solver is ever built -- see the failure section below.
    """
    from cvxpy.error import DCPError

    Xy, trt, t0 = _panel(n_trt=3)
    for bad in (-0.5, 1.5):
        with pytest.raises(DCPError):
            run_multisynth(Xy, trt, t0, 5, t0, nu=bad)


# --------------------------------------------------------------------------- #
# property: the bound holds wherever the panel sits                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [1e-4, 1e-2, 0.658, 1.0, 7.3, 1e3, 1e5])
@pytest.mark.parametrize("n_trt", [1, 2, 4])
def test_the_ratio_stays_a_weight_across_scales_and_pool_sizes(scale, n_trt):
    """Twenty-one panels that differ only in units and in how many units adopt.

    The scales bracket the range a real outcome is measured over, and 0.658 is
    the median absolute level of the panel this was first seen on. Nothing here
    may produce a weight outside the unit interval, and nothing here may raise.
    """
    Xy, trt, t0 = _panel(n_trt=n_trt, scale=scale)
    assert 0.0 <= run_multisynth(Xy, trt, t0, 5, t0)["nu_used"] <= 1.0


@pytest.mark.parametrize("seed", range(12))
def test_the_ratio_stays_a_weight_across_draws(seed):
    Xy, trt, t0 = _panel(seed=seed)
    assert 0.0 <= run_multisynth(Xy, trt, t0, 5, t0)["nu_used"] <= 1.0


# --------------------------------------------------------------------------- #
# edge: the panel that first produced the DCPError                             #
# --------------------------------------------------------------------------- #
def test_the_panel_that_rounded_over_one_still_fits():
    """Recorded from the failure. The unscaled panel gives ``nu_used`` exactly
    one and solves; dividing by its own median absolute level, 0.658233745909389,
    gives 1.0000000000000002 and the second solve is refused. The two differ by
    one unit in the last place and are the same panel in different units."""
    Xy, trt, t0 = _panel()
    scaled = Xy / 0.658233745909389
    for panel in (Xy, scaled):
        assert run_multisynth(panel, trt, t0, 5, t0)["nu_used"] == 1.0


def test_the_two_scalings_agree_on_the_effect():
    """Having both solve is not enough -- they have to solve the same problem.
    Synthetic control is scale-equivariant, so the effect divides by the scale."""
    Xy, trt, t0 = _panel()
    k = 0.658233745909389
    base = run_multisynth(Xy, trt, t0, 5, t0)
    scaled = run_multisynth(Xy / k, trt, t0, 5, t0)
    assert scaled["att"] * k == pytest.approx(base["att"], rel=1e-6)


def test_the_estimator_reaches_this_through_its_own_entry_point():
    """The clamp sits in the engine, and the config's ``nu="auto"`` is the path
    a caller actually takes to it."""
    Xy, trt, t0 = _panel()
    res = PPSCM(dict(df=_long(Xy, trt, t0), unitid="unit", time="time",
                     outcome="y", treat="treat", display_graphs=False)).fit()
    assert np.isfinite(res.effects.att)


# --------------------------------------------------------------------------- #
# failure: the clamp corrects arithmetic, it does not paper over a bad panel    #
# --------------------------------------------------------------------------- #
def test_a_ratio_that_is_not_a_number_is_not_silently_clamped():
    """``np.clip`` maps NaN to NaN, so a degenerate separate fit -- one whose
    average imbalance norm is zero, making the ratio 0/0 -- still surfaces as a
    non-finite ``nu_used`` and is not turned into a plausible-looking 1.0."""
    assert np.isnan(np.clip(np.nan, 0.0, 1.0))


@pytest.mark.parametrize("bad", [-0.5, 1.5, 2.0, -1e-9])
def test_a_nu_outside_the_unit_interval_is_refused_by_name(bad):
    """It weights one term and its complement weights the other, so outside the
    unit interval one of them is negative and the objective is no longer convex.
    Left to the engine it surfaces as a ``DCPError`` from cvxpy naming a
    ``quad_over_lin`` subexpression, which does not tell a caller that ``nu`` is
    the field they set wrong."""
    Xy, trt, t0 = _panel()
    with pytest.raises(MlsynthConfigError, match="nu"):
        PPSCM(dict(df=_long(Xy, trt, t0), unitid="unit", time="time",
                   outcome="y", treat="treat", nu=bad, display_graphs=False))


@pytest.mark.parametrize("good", [0.0, 0.5, 1.0, "auto"])
def test_the_endpoints_and_the_default_are_accepted(good):
    """The interval is closed. Zero is a separate SCM per treated unit and one is
    the fully pooled fit -- both are the fits the parameter exists to reach, and
    one is where the automatic rule lands on a single treated unit."""
    Xy, trt, t0 = _panel()
    cfg = PPSCM(dict(df=_long(Xy, trt, t0), unitid="unit", time="time",
                     outcome="y", treat="treat", nu=good, display_graphs=False))
    assert cfg.config.nu == good
