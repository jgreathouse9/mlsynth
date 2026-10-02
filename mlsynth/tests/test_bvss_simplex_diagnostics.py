"""What BVSS reports about the simplex constraint it softens.

The question Xu and Zhou set out to answer is whether the simplex constraint
should be imposed at all, and their answer is a parameter, not an argument.
Equation (2) gives the actual weights

    w_gamma | gamma, mu_gamma, tau, phi  ~  N(mu_gamma, (tau / phi) I),

so the weights are centred on a point of the simplex and scattered around it
with standard deviation sqrt(tau / phi) per coordinate. Their Section 2.1 reads
that scatter directly: a posterior for tau concentrating near zero says the data
support the constraint, and one staying away from zero says the data reject it.

That makes sqrt(tau / phi) the quantity a user needs in order to decide whether a
hard-simplex method was appropriate for their panel, and until this module it
was reachable only by pulling raw posterior samples off the result. CLAUDE.md
invariant 7 asks for a typed field instead.

The scale is analytic in the draws of tau, phi and |gamma|, so nothing here
depends on how the counterfactual is built. That matters: the counterfactual's
construction is an open question recorded in docs/bvss.rst, and the simplex
verdict should not inherit it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import BVSS
from mlsynth.config_models import BVSSConfig

N_ITER, BURN = 300, 150


def _panel(weights, n_donors=6, T=40, T0=30, lift=0.0, noise=0.05,
           off_simplex=0.0, seed=0):
    """A panel whose treated unit is a known combination of its donors.

    ``weights`` are applied to the donors; ``off_simplex`` adds a constant to
    every weight, so the sum leaves one by ``off_simplex * n_donors`` while the
    fit stays good. That is the lever the diagnostic has to detect.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(5.0, 15.0, size=(T, n_donors))
    w = np.asarray(weights, dtype=float) + off_simplex
    y = X @ w + rng.normal(scale=noise, size=T)
    y[T0:] += lift

    rows = []
    for j in range(n_donors):
        for t in range(T):
            rows.append(dict(unit=f"d{j}", time=t, y=float(X[t, j]), treat=0))
    for t in range(T):
        rows.append(dict(unit="treated", time=t, y=float(y[t]),
                         treat=int(t >= T0)))
    return pd.DataFrame(rows)


def _fit(df, seed=0):
    return BVSS(BVSSConfig(df=df, unitid="unit", time="time", outcome="y",
                           treat="treat", display_graphs=False, seed=seed,
                           n_iter=N_ITER, burn_in=BURN)).fit()


@pytest.fixture(scope="module")
def on_simplex():
    """Treated unit is an exact convex combination of its donors."""
    return _fit(_panel([0.5, 0.3, 0.2, 0.0, 0.0, 0.0]))


@pytest.fixture(scope="module")
def off_simplex():
    """The same fit quality, with the weights summing well away from one."""
    return _fit(_panel([0.5, 0.3, 0.2, 0.0, 0.0, 0.0], off_simplex=0.25))


# --------------------------------------------------------------------------- #
# the field exists and is typed
# --------------------------------------------------------------------------- #
def test_the_result_carries_a_simplex_diagnostic(on_simplex):
    assert on_simplex.simplex is not None


@pytest.mark.parametrize("field", [
    "tau_mean", "tau_median", "tau_q025", "tau_q975",
    "deviation_scale", "weight_sum_sd", "relative_deviation",
    "model_size_mean",
])
def test_every_documented_field_is_present_and_finite(on_simplex, field):
    value = getattr(on_simplex.simplex, field)
    assert isinstance(value, float)
    assert np.isfinite(value)


def test_the_diagnostic_is_reachable_without_touching_raw_samples(on_simplex):
    """A caller should not have to reach into ``posterior`` to get the verdict.

    That is the whole point of the field: the raw draws stay available for
    anyone who wants them, and the summary is on the result.
    """
    assert on_simplex.simplex.tau_mean == pytest.approx(
        float(np.asarray(on_simplex.posterior.tau).mean()), rel=1e-12)


# --------------------------------------------------------------------------- #
# the quantities are the ones equation (2) defines
# --------------------------------------------------------------------------- #
def test_deviation_scale_is_the_root_of_tau_over_phi(on_simplex):
    """Equation (2)'s per-coordinate standard deviation, averaged over draws."""
    tau = np.asarray(on_simplex.posterior.tau, dtype=float)
    phi = np.asarray(on_simplex.posterior.phi, dtype=float)
    assert on_simplex.simplex.deviation_scale == pytest.approx(
        float(np.sqrt(tau / phi).mean()), rel=1e-12)


def test_weight_sum_sd_scales_with_the_square_root_of_the_model_size(on_simplex):
    """``sum(w)`` has variance ``|gamma| tau / phi`` under equation (2).

    The coordinates are independent given the parameters, so their variances
    add and the sum's standard deviation grows as the root of the model size.
    """
    tau = np.asarray(on_simplex.posterior.tau, dtype=float)
    phi = np.asarray(on_simplex.posterior.phi, dtype=float)
    size = (np.asarray(on_simplex.posterior.mu) != 0).sum(axis=0)
    assert on_simplex.simplex.weight_sum_sd == pytest.approx(
        float(np.sqrt(size * tau / phi).mean()), rel=1e-12)


def test_relative_deviation_compares_the_scatter_to_a_typical_weight(on_simplex):
    """A deviation of 0.05 means one thing across three donors and another
    across thirty, so the field reports it against ``1 / |gamma|``."""
    s = on_simplex.simplex
    assert s.relative_deviation == pytest.approx(
        s.deviation_scale * s.model_size_mean, rel=1e-9)


def test_the_quantiles_bracket_the_median(on_simplex):
    s = on_simplex.simplex
    assert s.tau_q025 <= s.tau_median <= s.tau_q975


# --------------------------------------------------------------------------- #
# it has to discriminate, which is the only reason to report it
# --------------------------------------------------------------------------- #
def test_leaving_the_simplex_raises_the_deviation_scale(on_simplex, off_simplex):
    """The claim the paper makes, as a test.

    Two panels with the same donors, the same noise and the same fit quality;
    one is an exact convex combination and the other sums to 2.5. A diagnostic
    that cannot separate them reports nothing a user could act on.
    """
    assert off_simplex.simplex.deviation_scale > on_simplex.simplex.deviation_scale


def test_an_exact_convex_combination_reads_as_supporting_the_simplex(on_simplex):
    """The scatter stays small next to a typical weight when the data comply."""
    assert on_simplex.simplex.relative_deviation < 0.5


# --------------------------------------------------------------------------- #
# failure and edge behaviour
# --------------------------------------------------------------------------- #
def test_the_scale_is_non_negative(on_simplex, off_simplex):
    for res in (on_simplex, off_simplex):
        assert res.simplex.deviation_scale >= 0.0
        assert res.simplex.weight_sum_sd >= 0.0


def test_the_verdict_does_not_move_when_the_treated_level_moves():
    """The intercept absorbs a level shift, so the simplex verdict is unchanged.

    BVSS demeans both sides by their pre-treatment means, which is a free
    intercept: fitting ``(Y - Ybar) = (X - Xbar) w`` is ``Y = (Ybar - Xbar w) +
    X w``. Adding a constant to the treated series therefore moves the
    intercept and nothing else, and the data's reading of the simplex has to
    come back identical.

    Measured on the paper's watches panel, dropping the intercept instead
    multiplies the deviation scale by 2.8 and takes the model size from 1.8
    donors to 4.1 -- without one, the fit recruits extra donors to manufacture
    the level. That is the comparison this test guards the near end of.
    """
    base = _panel([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
    shifted = base.copy()
    mask = shifted["unit"] == "treated"
    shifted.loc[mask, "y"] = shifted.loc[mask, "y"] + 3.7

    a = _fit(base).simplex
    b = _fit(shifted).simplex
    assert b.deviation_scale == pytest.approx(a.deviation_scale, rel=1e-10)
    assert b.tau_mean == pytest.approx(a.tau_mean, rel=1e-10)
    assert b.model_size_mean == pytest.approx(a.model_size_mean, rel=1e-10)


def test_the_diagnostic_survives_a_single_donor():
    """One donor forces every weight to one, so the simplex cannot be tested.

    The field still has to be well formed; a degenerate panel is not licence to
    return ``None`` where a float is documented.
    """
    res = _fit(_panel([1.0], n_donors=1))
    assert np.isfinite(res.simplex.deviation_scale)
    assert res.simplex.model_size_mean <= 1.0
