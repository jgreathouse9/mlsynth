"""Where ATEL's diversified weights come from, and whether they earn their keep.

Fan and Liao (2022) recommend four constructions for the weight matrix and only
the first needs covariates:

* 4.1 loading characteristics -- a sieve basis in observed covariates;
* 4.2 moving windows -- trimmed PCA loadings from an earlier sample split;
* 4.3 initial transformation -- a sieve basis in the first observation,
  ``w_ik = phi_k(x_i0)``, correlated with the loadings through
  ``x_0 = B f_0 + u_0``;
* 4.4 Hadamard projection -- deterministic plus/minus one columns.

``weight_source`` selects among 4.1, 4.3 and 4.4. Section 4.2 is left out: it
needs a sample split and serial independence of the errors, which is a different
assumption burden and belongs in its own change.

Two conditions from their Assumption 2.1 are checkable on the constructed
weights, and are checked here: (i) the entries stay bounded, and (ii)
``lambda_min(W'W / N) > c``, so the weights are not degenerate among
themselves. The rank condition that involves the unobserved loadings,
``rank(W'B / N) = r``, remains undiagnosable and these tests do not pretend
otherwise.

Reference pins were produced by the author's MATLAB toolbox under Octave 8.4.
The toolbox has no initial-transformation mode, so it was fed a covariate cube
whose single covariate is the initial outcome tiled across periods -- which
makes its ``construct_weights`` compute exactly ``phi_k(x_i0)`` in every block,
the same object this code builds directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import ATEL
from mlsynth.config_models import ATELConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.atel_helpers.sieve import (
    basis_values,
    hadamard_weights,
    tile_unit_weights,
)

HCW = "basedata/HongKong.csv"


def _synthetic_panel():
    """The panel from ``test_atel.py``, built the same way and without an RNG."""
    n_units, T, T0 = 9, 14, 9
    i = np.arange(n_units)[:, None]
    t = np.arange(T)[None, :]
    x1 = 0.5 + 0.3 * np.sin(0.7 * t + 0.4 * i)
    x2 = 0.2 + 0.4 * np.cos(0.5 * t + 0.9 * i)
    Y = (
        10.0
        + 3.0 * x1 * np.sin(0.9 * t)
        + 2.0 * x2 * np.cos(0.6 * t)
        + 0.10 * np.cos(2.1 * t + 1.3 * i)
    )
    Y[0, T0:] += 4.0
    return Y, np.stack([x1, x2], axis=2), T0


def _long(Y, X, T0):
    rows = []
    for u in range(Y.shape[0]):
        for p in range(Y.shape[1]):
            rows.append(
                {
                    "unit": f"u{u}", "period": p, "y": Y[u, p],
                    "x1": X[u, p, 0], "x2": X[u, p, 1],
                    "treat": int(u == 0 and p >= T0),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synth_long():
    Y, X, T0 = _synthetic_panel()
    return _long(Y, X, T0), T0


@pytest.fixture
def hcw():
    return pd.read_csv(HCW)


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="period",
                n_factors=2, display_graphs=False)
    base.update(kw)
    return ATELConfig(**base)


def _hcw_cfg(df, **kw):
    base = dict(df=df, outcome="GDP", treat="Integration", unitid="Country",
                time="Time", n_factors=2, display_graphs=False)
    base.update(kw)
    return ATELConfig(**base)


def _lambda_min(W):
    W = np.asarray(W, dtype=float)
    return float(np.linalg.eigvalsh(W.T @ W / W.shape[0]).min())


# ------------------------------------------------------------ configuration
def test_the_default_source_is_covariates(synth_long):
    """Back-compatibility: an existing config keeps its meaning."""
    df, _T0 = synth_long
    assert _cfg(df, covariates=["x1", "x2"]).weight_source == "covariates"


@pytest.mark.parametrize("source", ["initial", "hadamard"])
def test_a_covariate_free_source_needs_no_covariates(synth_long, source):
    df, _T0 = synth_long
    assert _cfg(df, weight_source=source).covariates == []


@pytest.mark.parametrize("source", ["initial", "hadamard"])
def test_a_covariate_free_source_refuses_covariates(synth_long, source):
    """Supplying columns that cannot be used is a mistake, not a preference."""
    df, _T0 = synth_long
    with pytest.raises(MlsynthConfigError, match="covariates"):
        _cfg(df, weight_source=source, covariates=["x1"])


def test_the_covariate_source_still_requires_covariates(synth_long):
    df, _T0 = synth_long
    with pytest.raises(MlsynthConfigError, match="covariates"):
        _cfg(df, weight_source="covariates", covariates=[])


def test_the_multiple_of_P_rule_does_not_bind_without_covariates(synth_long):
    """The rule exists because of the (basis, covariate) block slice.

    With one weight series per basis function there is no covariate ordering to
    permute, so an odd factor count is well defined.
    """
    df, _T0 = synth_long
    assert _cfg(df, weight_source="initial", n_factors=3).n_factors == 3
    with pytest.raises(MlsynthConfigError, match="multiple"):
        _cfg(df, weight_source="covariates", covariates=["x1", "x2"], n_factors=3)


# ------------------------------------------------------- Hadamard weights
@pytest.mark.parametrize("n_units,R", [(8, 2), (25, 4), (9, 3), (16, 6)])
def test_hadamard_weights_are_signs_with_a_leading_column_of_ones(n_units, R):
    W = hadamard_weights(n_units, R)
    assert W.shape == (n_units, R)
    assert set(np.unique(W)) <= {-1.0, 1.0}
    assert np.allclose(W[:, 0], 1.0)


@pytest.mark.parametrize("n_units,R", [(25, 2), (25, 4), (16, 3)])
def test_hadamard_weights_satisfy_assumption_2_1(n_units, R):
    """(i) bounded entries, (ii) a non-degenerate second moment."""
    W = hadamard_weights(n_units, R)
    assert np.abs(W).max() == 1.0
    assert _lambda_min(W) > 1e-8


def test_hadamard_weights_need_at_least_two_columns():
    with pytest.raises(MlsynthConfigError):
        hadamard_weights(10, 1)


# ------------------------------------------------------------ tiling blocks
def test_tiling_repeats_each_weight_across_every_period():
    W_unit = np.array([[1.0, -2.0], [3.0, 4.0], [0.5, 0.25]])
    blocks = tile_unit_weights(W_unit, 4)
    assert blocks.shape == (3, 8)
    assert np.allclose(blocks[:, 0:4], W_unit[:, [0]])
    assert np.allclose(blocks[:, 4:8], W_unit[:, [1]])


# ------------------------------------- which basis keeps Assumption 2.1(ii)
def test_the_bspline_basis_keeps_the_weights_well_conditioned(hcw):
    """Partition of unity bounds the entries at one, and the moment stays away
    from zero, on the outcome in its natural units."""
    z = hcw[hcw.Time == 0].set_index("Country").GDP.to_numpy(float)
    for J in (2, 3, 4, 5, 6):
        W = basis_values(z, J, "bspline")
        assert np.abs(W).max() <= 1.0 + 1e-12
        assert _lambda_min(W) > 1e-4


def test_a_polynomial_basis_on_small_outcomes_degenerates(hcw):
    """Why bspline is the default for the initial transformation.

    Fan and Liao's simulations use ``phi_k(z) = z^k``, which is fine for a
    covariate of order one. On a growth rate of order 0.01 the columns collapse
    onto each other and Assumption 2.1(ii) fails numerically.
    """
    z = hcw[hcw.Time == 0].set_index("Country").GDP.to_numpy(float)
    assert _lambda_min(basis_values(z, 2, "polynomial")) < 1e-4
    assert _lambda_min(basis_values(z, 6, "polynomial")) < 1e-12


# -------------------------------------------- holding out the first period
def test_the_initial_source_holds_out_the_period_it_builds_weights_from(hcw):
    """``(f_0, u_0)`` has to be independent of the errors that remain."""
    full = ATEL(_hcw_cfg(hcw, weight_source="covariates",
                         covariates=["GDP"])).fit()  # same panel, nothing dropped
    held = ATEL(_hcw_cfg(hcw, weight_source="initial")).fit()
    assert held.inputs.n_periods == full.inputs.n_periods - 1
    assert held.inputs.n_pre == full.inputs.n_pre - 1
    first_period = hcw[hcw.Time == hcw.Time.min()]
    expected = first_period.set_index("Country").GDP
    got = pd.Series(held.inputs.initial_outcome, index=held.inputs.unit_labels)
    assert np.allclose(got.to_numpy(), expected.reindex(got.index).to_numpy())


def test_the_initial_weights_do_not_move_over_time(hcw):
    res = ATEL(_hcw_cfg(hcw, weight_source="initial", n_factors=3)).fit()
    T = res.inputs.n_periods
    W = res.weights_matrix
    for j in range(3):
        block = W[:, j * T : (j + 1) * T]
        assert np.allclose(block, block[:, [0]])


def _poke_one_donor_after_the_first_period(df, unit, amount=7.5, before=None):
    """Shift one donor's outcomes after the first period.

    The perturbation has to fall on a single unit. A shift applied to every unit
    alike moves the covariate's location without changing its cross-sectional
    pattern, and a B-spline basis takes its knots from the pooled range, so such
    a shift leaves the basis values almost unchanged -- a test built on one
    would pass whatever the weights were built from.
    """
    poked = df.copy()
    rows = (poked.Country == unit) & (poked.Time > poked.Time.min())
    if before is not None:
        rows &= poked.Time < before
    poked.loc[rows, "GDP"] = poked.loc[rows, "GDP"] + amount
    return poked


def test_the_initial_weights_ignore_every_later_outcome(hcw):
    """The exogeneity Assumption 2.1(iii) asks for, stated as a test.

    Moving one donor's outcomes after the first period must leave the weight
    matrix untouched, because the weights see only the first period.
    """
    donor = [u for u in hcw.Country.unique() if u != "Hong Kong"][0]
    base = ATEL(_hcw_cfg(hcw, weight_source="initial", n_factors=3)).fit()
    poked = _poke_one_donor_after_the_first_period(hcw, donor)
    moved = ATEL(_hcw_cfg(poked, weight_source="initial", n_factors=3)).fit()
    assert np.allclose(base.weights_matrix, moved.weights_matrix, atol=1e-12)


def test_a_covariate_built_from_later_outcomes_does_move(hcw):
    """The contrast, and why the pre-period mean is not a legitimate weight.

    The same single-donor perturbation moves weights built from a pre-period
    mean, because that mean is a function of the errors the projection is
    supposed to diversify away.
    """
    donor = [u for u in hcw.Country.unique() if u != "Hong Kong"][0]

    def with_pre_mean(df):
        means = df[df.Time < 44].groupby("Country").GDP.mean().rename("gdp_pre_mean")
        return df.merge(means, on="Country")

    wa = ATEL(_hcw_cfg(with_pre_mean(hcw), weight_source="covariates",
                       covariates=["gdp_pre_mean"])).fit().weights_matrix
    poked = _poke_one_donor_after_the_first_period(hcw, donor, before=44)
    wb = ATEL(_hcw_cfg(with_pre_mean(poked), weight_source="covariates",
                       covariates=["gdp_pre_mean"])).fit().weights_matrix
    assert not np.allclose(wa, wb, atol=1e-8)


# ----------------------------------------------------------- the diagnostic
@pytest.mark.parametrize("source", ["initial", "hadamard"])
def test_the_weight_conditioning_is_reported(hcw, source):
    res = ATEL(_hcw_cfg(hcw, weight_source=source, n_factors=3)).fit()
    lam = res.diagnostics["weight_lambda_min"]
    assert np.isfinite(lam) and lam > 0.0


def test_the_conditioning_diagnostic_also_covers_the_covariate_source(synth_long):
    df, _T0 = synth_long
    res = ATEL(_cfg(df, covariates=["x1", "x2"], n_factors=2)).fit()
    assert res.diagnostics["weight_lambda_min"] > 0.0


# --------------------------------------------------------- reference pins
@pytest.mark.parametrize(
    "J,atel,se,h",
    [
        (2, 0.017059613337, 0.007489244307, 0.50),
        (3, 0.032379413503, 0.004549043775, 0.95),
        (4, 0.031679151600, 0.004530813550, 0.95),
    ],
)
def test_hcw_initial_matches_the_matlab_toolbox(hcw, J, atel, se, h):
    res = ATEL(_hcw_cfg(hcw, weight_source="initial", n_factors=J)).fit()
    assert res.atel == pytest.approx(atel, rel=1e-8)
    assert res.inference.standard_error == pytest.approx(se, rel=1e-8)
    assert res.bandwidth == pytest.approx(h)


@pytest.mark.parametrize(
    "J,atel,se", [(2, 3.795441035243, 0.240832317743), (3, 3.693235652631, 0.024817321011)]
)
def test_synthetic_initial_matches_the_matlab_toolbox(synth_long, J, atel, se):
    df, _T0 = synth_long
    res = ATEL(_cfg(df, weight_source="initial", n_factors=J)).fit()
    assert res.atel == pytest.approx(atel, rel=1e-8)
    assert res.inference.standard_error == pytest.approx(se, rel=1e-8)


# --------------------------------- an exactly determined design is refused
def test_a_pre_period_exactly_as_long_as_the_design_is_refused(synth_long):
    """``T0 == 2J`` leaves no residual degree of freedom.

    The local linear design carries a level and a slope per factor, so at
    ``T0 = 2J`` the pre-period fit is exact, every residual is zero, and the
    variance estimate collapses. The reference implementation returns a standard
    error of 6.8e-09 and a p-value of 0 on exactly this configuration; that is a
    confident-looking answer with nothing behind it.
    """
    df, T0 = synth_long
    with pytest.raises(MlsynthDataError, match="degree of freedom"):
        ATEL(_cfg(df, weight_source="initial", n_factors=4)).fit()  # T0 becomes 8 = 2J


def test_one_spare_pre_period_is_enough(synth_long):
    df, _T0 = synth_long
    res = ATEL(_cfg(df, covariates=["x1", "x2"], n_factors=4)).fit()  # T0 = 9 > 8
    assert np.isfinite(res.atel)
    assert res.inference.standard_error > 0.0


# ------------------------------------------------------------- end to end
@pytest.mark.parametrize("source", ["initial", "hadamard"])
def test_a_covariate_free_source_fills_the_standard_contract(hcw, source):
    res = ATEL(_hcw_cfg(hcw, weight_source=source, n_factors=3)).fit()
    assert np.isfinite(res.atel)
    assert res.weights is not None and not res.weights.is_empty
    assert res.effects.att is not None
    assert res.time_series.counterfactual_outcome is not None
    assert res.diagnostics["weight_source"] == source
    assert set(res.donor_weights) == set(res.inputs.unit_labels[1:])


def test_hadamard_weights_reproduce_the_counterfactual_identity(hcw):
    """The implied-weight identity holds whatever the weights are built from."""
    res = ATEL(_hcw_cfg(hcw, weight_source="hadamard", n_factors=3)).fit()
    T0 = res.inputs.n_pre
    donors = res.inputs.outcomes[1:]
    rebuilt = (donors[:, T0:] * res.implied_donor_weights).sum(axis=0)
    assert np.allclose(rebuilt, res.time_series.counterfactual_outcome[T0:], atol=1e-9)


# ---------------------------------------------------- helper-level guards
def test_building_weights_without_the_held_out_outcome_is_refused(synth_long):
    """``build_weights`` is reachable on its own, so it states its own needs."""
    from mlsynth.utils.atel_helpers.pipeline import build_weights
    from mlsynth.utils.atel_helpers.setup import prepare_atel_inputs

    df, _T0 = synth_long
    inputs = prepare_atel_inputs(df, "y", "treat", "unit", "period", ["x1"], 2)
    assert inputs.initial_outcome is None
    with pytest.raises(MlsynthDataError, match="held-out first outcome"):
        build_weights(inputs, 2, "bspline", "initial")


def test_an_unknown_weight_source_is_refused_where_it_is_read(synth_long):
    from mlsynth.utils.atel_helpers.pipeline import build_weights
    from mlsynth.utils.atel_helpers.setup import prepare_atel_inputs

    df, _T0 = synth_long
    inputs = prepare_atel_inputs(df, "y", "treat", "unit", "period", ["x1"], 2)
    with pytest.raises(MlsynthConfigError, match="Unknown weight_source"):
        build_weights(inputs, 2, "bspline", "wavelet")


def test_holding_out_the_only_pre_period_is_refused():
    """A single pre-period leaves nothing once the weights have taken it."""
    rows = []
    for u in range(4):
        for t in range(3):
            rows.append({"unit": f"u{u}", "period": t, "y": 1.0 + u + 0.5 * t,
                         "treat": int(u == 0 and t >= 1)})
    df = pd.DataFrame(rows)
    with pytest.raises(MlsynthDataError, match="no pre-treatment period"):
        ATEL(_cfg(df, weight_source="initial", n_factors=2)).fit()


def test_hadamard_weights_need_at_least_one_unit():
    with pytest.raises(MlsynthDataError, match="at least one unit"):
        hadamard_weights(0, 2)


def test_tiling_refuses_malformed_input():
    with pytest.raises(MlsynthDataError, match="2-D"):
        tile_unit_weights(np.ones(5), 3)
    with pytest.raises(MlsynthDataError, match="at least one period"):
        tile_unit_weights(np.ones((5, 2)), 0)


# ------------------------------- what the conditioning diagnostic returns
def test_the_conditioning_diagnostic_returns_the_smallest_eigenvalue():
    """Not merely a positive number: the weakest direction, by identity.

    Two weight series that nearly coincide give a spectrum with a large and a
    small eigenvalue, and the diagnostic has to report the small one -- reading
    the other way round would call a degenerate weight matrix healthy.
    """
    from mlsynth.utils.atel_helpers.sieve import tile_unit_weights, weight_conditioning

    n_donors, T = 12, 5
    rng = np.random.default_rng(0)
    first = rng.normal(size=n_donors)
    W_unit = np.column_stack([first, first + 1e-3 * rng.normal(size=n_donors)])
    spectrum = np.linalg.eigvalsh(W_unit.T @ W_unit / n_donors)

    got = weight_conditioning(tile_unit_weights(W_unit, T), 2, T)
    assert got == pytest.approx(float(spectrum.min()), rel=1e-12)
    assert float(spectrum.min()) < 1e-3 * float(spectrum.max())   # the test has power


def test_the_conditioning_diagnostic_reports_the_worst_period():
    """With time-varying weights it is a minimum over periods, not an average."""
    from mlsynth.utils.atel_helpers.sieve import weight_conditioning

    n_donors, T = 10, 4
    rng = np.random.default_rng(1)
    healthy = rng.normal(size=(n_donors, 2))
    blocks = np.zeros((n_donors, 2 * T))
    per_period = []
    for t in range(T):
        W_t = healthy.copy()
        if t == 2:
            W_t[:, 1] = W_t[:, 0]          # one period where the series coincide
        blocks[:, 0 * T + t] = W_t[:, 0]
        blocks[:, 1 * T + t] = W_t[:, 1]
        per_period.append(float(np.linalg.eigvalsh(W_t.T @ W_t / n_donors).min()))

    got = weight_conditioning(blocks, 2, T)
    assert got == pytest.approx(min(per_period), rel=1e-12)
    assert got == pytest.approx(per_period[2], rel=1e-12)
    assert per_period[2] < 1e-8 < min(p for i, p in enumerate(per_period) if i != 2)
