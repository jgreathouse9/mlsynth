"""Generative property tests for the SDIDGEO scoring engines.

The seam exists so the estimator scoring candidates can be swapped. What makes
that safe is not that each engine passes its own tests -- it is that both
satisfy the same contract, so the pipeline above them cannot tell which is
running except through the numbers. So the properties here are parametrized
over the engine registry: every engine must satisfy all of them, and a third
engine added later inherits the suite by construction.

Why this layer. Property tests generate inside the specification, which is where
a fault of omission lives (see the instrument contract in
``agents/agents_tests.md``). An engine adapter is close to the Layer 1 target
that contract prefers -- a pure function of two arrays -- with the caveat that a
solver runs inside it, so tolerances come from measured solver spread and the
example counts stay modest.

The relations asserted are the metamorphic ones that carry the estimator's
meaning:

* scale -- rescaling the panel scales the ATT by the same factor and leaves the
  donor weights alone, because a synthetic control is a weighted average and
  averaging commutes with scaling;
* location -- shifting every outcome by a constant moves nothing, since both
  engines match shapes and absorb levels (SDID differences them out, augsynth
  carries an intercept under fixed effects);
* permutation -- relabelling donors permutes the weight vector identically and
  leaves the counterfactual untouched, so the answer does not depend on column
  order;
* the analytic shortcut -- injecting an effect and refitting must agree with
  shifting the ATT arithmetically, which is the equality that licenses the
  shortcut the scoring loop relies on.

Named degenerate panels are kept as ``@example`` instead of left to the
generator: a single donor, a constant treated series, and a donor pool that
already reproduces the treated path exactly are the corners a real geo panel
produces.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.sdidgeo_helpers.engines import ENGINE_NAMES, resolve_engine

ENGINES = sorted(ENGINE_NAMES)

# Solver tolerance. Both engines run a numerical program, so equalities that are
# exact in arithmetic hold to solver precision and not to machine epsilon.
_ATOL = 1e-6
_RTOL = 1e-6

_SETTINGS = settings(
    max_examples=25, deadline=None, derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def _panel(draw, n_periods: int, n_donors: int) -> tuple:
    """A geo-shaped panel: donor levels over a range, a shared seasonal, noise."""
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    t = np.arange(n_periods)
    levels = rng.uniform(50.0, 500.0, size=n_donors)
    season = 10.0 * np.sin(2 * np.pi * t / 7.0)
    Y0 = (levels[None, :] + season[:, None]
          + rng.normal(0.0, 5.0, size=(n_periods, n_donors)))
    w = rng.dirichlet(np.ones(n_donors))
    y = Y0 @ w + rng.normal(0.0, 3.0, size=n_periods)
    return y, Y0


@st.composite
def panels(draw, min_donors: int = 3, max_donors: int = 8):
    n_periods = draw(st.integers(min_value=14, max_value=26))
    n_donors = draw(st.integers(min_value=min_donors, max_value=max_donors))
    y, Y0 = _panel(draw, n_periods, n_donors)
    n_pre = n_periods - draw(st.integers(min_value=3, max_value=6))
    assume(n_pre >= 8)
    return y, Y0, n_pre, n_periods - 1


@pytest.mark.parametrize("engine", ENGINES)
class TestMetamorphicRelations:
    """Relations every engine must satisfy, over the panel domain."""

    @_SETTINGS
    @given(case=panels())
    def test_scale_equivariance(self, engine, case):
        # Y -> cY scales the ATT by c and leaves the donor weights unchanged.
        y, Y0, n_pre, end = case
        eng = resolve_engine(engine)
        c = 7.5
        base = eng.fit_once(y, Y0, n_pre, n_pre, end, 1)
        scaled = eng.fit_once(y * c, Y0 * c, n_pre, n_pre, end, 1)
        np.testing.assert_allclose(scaled.donor_weights, base.donor_weights,
                                   atol=_ATOL)
        np.testing.assert_allclose(eng.att(scaled, y * c, n_pre, end),
                                   c * eng.att(base, y, n_pre, end),
                                   rtol=_RTOL, atol=_ATOL)

    @_SETTINGS
    @given(case=panels())
    def test_location_invariance(self, engine, case):
        # Both engines match shapes and absorb levels, so a common shift moves
        # neither the weights nor the estimated effect.
        y, Y0, n_pre, end = case
        eng = resolve_engine(engine)
        a = 123.0
        base = eng.fit_once(y, Y0, n_pre, n_pre, end, 1)
        shifted = eng.fit_once(y + a, Y0 + a, n_pre, n_pre, end, 1)
        np.testing.assert_allclose(shifted.donor_weights, base.donor_weights,
                                   atol=_ATOL)
        np.testing.assert_allclose(eng.att(shifted, y + a, n_pre, end),
                                   eng.att(base, y, n_pre, end),
                                   rtol=_RTOL, atol=_ATOL)

    @_SETTINGS
    @given(case=panels(), key=st.integers(min_value=0, max_value=2**31 - 1))
    def test_donor_permutation_equivariance(self, engine, case, key):
        # Relabelling donors permutes the weights identically and leaves the
        # counterfactual alone: column order is not information.
        y, Y0, n_pre, end = case
        eng = resolve_engine(engine)
        perm = np.random.default_rng(key).permutation(Y0.shape[1])
        base = eng.fit_once(y, Y0, n_pre, n_pre, end, 1)
        shuffled = eng.fit_once(y, Y0[:, perm], n_pre, n_pre, end, 1)
        np.testing.assert_allclose(shuffled.donor_weights,
                                   base.donor_weights[perm], atol=_ATOL)
        np.testing.assert_allclose(shuffled.counterfactual, base.counterfactual,
                                   atol=1e-4, rtol=1e-6)

    @_SETTINGS
    @given(case=panels())
    def test_counterfactual_spans_the_panel(self, engine, case):
        y, Y0, n_pre, end = case
        fit = resolve_engine(engine).fit_once(y, Y0, n_pre, n_pre, end, 1)
        assert fit.counterfactual.shape == y.shape
        assert np.all(np.isfinite(fit.counterfactual))
        assert fit.donor_weights.shape == (Y0.shape[1],)
        assert np.isfinite(fit.pre_rmspe)

    @_SETTINGS
    @given(case=panels(),
           effects=st.lists(st.floats(min_value=-0.5, max_value=0.5,
                                      allow_nan=False, allow_infinity=False),
                            min_size=1, max_size=4))
    def test_analytic_shortcut_equals_the_refit(self, engine, case, effects):
        # The scoring loop shifts the ATT by effect x mean(treated_post) instead
        # of refitting. That is exact only because the fit uses pre-period data
        # alone, so it is a claim about the engine, asserted over the domain.
        y, Y0, n_pre, end = case
        eng = resolve_engine(engine)
        fit = eng.fit_once(y, Y0, n_pre, n_pre, end, 1)
        kw = dict(n_draws=4, ns=20, n_tr=1, seed=0, inference="placebo")
        fast = eng.sweep_p_values(fit, y, Y0, n_pre, n_pre, end, effects,
                                  analytic=True, **kw)
        slow = eng.sweep_p_values(fit, y, Y0, n_pre, n_pre, end, effects,
                                  analytic=False, **kw)
        np.testing.assert_allclose(slow["tau"], fast["tau"], rtol=1e-8,
                                   atol=1e-8)


@pytest.mark.parametrize("engine", ENGINES)
class TestDegenerateCorners:
    """Named panels a generator is unlikely to produce, kept as examples."""

    def _run(self, engine, y, Y0, n_pre):
        return resolve_engine(engine).fit_once(y, Y0, n_pre, n_pre,
                                               len(y) - 1, 1)

    def test_single_donor(self, engine):
        rng = np.random.default_rng(0)
        Y0 = rng.normal(100.0, 5.0, size=(20, 1))
        y = Y0[:, 0] + rng.normal(0.0, 1.0, size=20)
        fit = self._run(engine, y, Y0, 15)
        assert fit.donor_weights.shape == (1,)
        assert np.all(np.isfinite(fit.counterfactual))

    def test_donor_reproduces_the_treated_path(self, engine):
        # Perfect pre-fit: the imbalance denominators can go to zero, so the
        # engine must still return finite weights and a finite counterfactual.
        rng = np.random.default_rng(1)
        Y0 = rng.normal(100.0, 5.0, size=(20, 3))
        y = Y0[:, 0].copy()
        fit = self._run(engine, y, Y0, 15)
        assert np.all(np.isfinite(fit.counterfactual))
        assert np.all(np.isfinite(fit.donor_weights))

    def test_constant_treated_series(self, engine):
        rng = np.random.default_rng(2)
        Y0 = rng.normal(100.0, 5.0, size=(20, 3))
        y = np.full(20, 50.0)
        fit = self._run(engine, y, Y0, 15)
        assert np.all(np.isfinite(fit.counterfactual))
