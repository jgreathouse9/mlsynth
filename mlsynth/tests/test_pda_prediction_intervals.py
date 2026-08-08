"""Jiang et al. (2025) bootstrap prediction intervals for PDA: engine tests.

``inferutils.pda_prediction_intervals`` is a general, estimator-agnostic engine:
given the treated series, control matrix, a fitted counterfactual + support, and
a ``refit`` callback (fit the point estimator on a bootstrap pre-period), it
returns equal-tailed (EQ) and symmetric (SY) prediction intervals for the
post-period counterfactual ``Y_t`` and the treatment effect ``Delta_t``, via the
dependent-wild / residual bootstrap of Algorithm 2.1.

These tests pin the contract TDD-first:

1. shape / structure of the returned intervals (EQ + SY, effect + counterfactual),
2. interval ordering (lower <= point <= upper),
3. the studentization label: post-selection OLS HAC sandwich when the support is
   low-rank-feasible, sigma^2-only fallback otherwise,
4. reproducibility under a fixed seed,
5. nominal coverage on a known DGP (the method's central claim),
6. input-validation failures.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.inferutils import pda_prediction_intervals


# ---------------------------------------------------------------------------
# A simple OLS PDA estimator to drive the engine (refit callback)
# ---------------------------------------------------------------------------

def _ols_fit(y, X, T0, support):
    """Post-selection OLS on ``support`` (with intercept); full-T counterfactual."""
    cols = np.asarray(support, dtype=int)
    Z = np.column_stack([np.ones(T0), X[:T0][:, cols]])
    coef, *_ = np.linalg.lstsq(Z, y[:T0], rcond=None)
    Zf = np.column_stack([np.ones(X.shape[0]), X[:, cols]])
    return Zf @ coef


def _make_refit(X, T0, support):
    def refit(y_boot):
        cf = _ols_fit(y_boot, X, T0, support)
        return cf, np.asarray(support, dtype=int)
    return refit


def _dgp(seed, T0=40, T1=6, p=4, effect=0.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T0 + T1, p))
    beta = np.array([1.0, -0.5, 0.3, 0.0])[:p]
    y0 = X @ beta + rng.standard_normal(T0 + T1) * 0.5
    y = y0.copy()
    y[T0:] += effect
    return y, X


def _fit(y, X, T0, support):
    return _ols_fit(y, X, T0, support)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

class TestStructure:
    def test_keys_and_shapes(self):
        y, X = _dgp(0)
        T0, T1 = 40, 6
        support = [0, 1, 2, 3]
        cf = _fit(y, X, T0, support)
        out = pda_prediction_intervals(
            y, X, T0, counterfactual=cf, support=support,
            refit=_make_refit(X, T0, support), alpha=0.05, n_boot=199, seed=1,
        )
        assert out["studentization"] in ("sandwich", "sigma2")
        for block in ("effect", "counterfactual"):
            b = out[block]
            for key in ("point", "eq_lower", "eq_upper", "sy_lower", "sy_upper"):
                assert b[key].shape == (T1,)
        assert out["se"].shape == (T1,)
        # effect point == y_post - cf_post
        np.testing.assert_allclose(out["effect"]["point"], y[T0:] - cf[T0:], atol=1e-12)

    def test_interval_ordering(self):
        y, X = _dgp(2)
        T0, support = 40, [0, 1, 2, 3]
        cf = _fit(y, X, T0, support)
        out = pda_prediction_intervals(
            y, X, T0, counterfactual=cf, support=support,
            refit=_make_refit(X, T0, support), n_boot=199, seed=3,
        )
        for block in ("effect", "counterfactual"):
            b = out[block]
            assert (b["eq_lower"] <= b["eq_upper"] + 1e-9).all()
            assert (b["sy_lower"] <= b["sy_upper"] + 1e-9).all()
            # symmetric interval is centered on the point estimate
            mid = 0.5 * (b["sy_lower"] + b["sy_upper"])
            np.testing.assert_allclose(mid, b["point"], atol=1e-9)


# ---------------------------------------------------------------------------
# Studentization label
# ---------------------------------------------------------------------------

class TestStudentization:
    def test_sandwich_when_support_feasible(self):
        y, X = _dgp(0)
        support = [0, 1, 2, 3]
        cf = _fit(y, X, 40, support)
        out = pda_prediction_intervals(
            y, X, 40, counterfactual=cf, support=support,
            refit=_make_refit(X, 40, support), n_boot=99, seed=1,
        )
        assert out["studentization"] == "sandwich"

    def test_sigma2_fallback_when_support_empty(self):
        y, X = _dgp(0)
        cf = _fit(y, X, 40, [0, 1, 2, 3])
        # empty support -> no sandwich -> sigma^2-only studentization
        out = pda_prediction_intervals(
            y, X, 40, counterfactual=cf, support=[],
            refit=lambda yb: (cf, np.array([], dtype=int)), n_boot=99, seed=1,
        )
        assert out["studentization"] == "sigma2"
        # se reduces to a single sigma across periods (no V_t term)
        assert np.allclose(out["se"], out["se"][0])

    def test_sigma2_fallback_when_support_exceeds_T0(self):
        # |support| >= T0 -> sandwich Gram is singular -> fallback.
        rng = np.random.default_rng(5)
        T0, T1, p = 8, 4, 10
        X = rng.standard_normal((T0 + T1, p))
        y = rng.standard_normal(T0 + T1)
        support = list(range(p))
        cf = np.zeros(T0 + T1)
        out = pda_prediction_intervals(
            y, X, T0, counterfactual=cf, support=support,
            refit=lambda yb: (cf, np.asarray(support)), n_boot=49, seed=1,
        )
        assert out["studentization"] == "sigma2"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_intervals(self):
        y, X = _dgp(7)
        support = [0, 1, 2, 3]
        cf = _fit(y, X, 40, support)
        kw = dict(counterfactual=cf, support=support,
                  refit=_make_refit(X, 40, support), n_boot=199)
        a = pda_prediction_intervals(y, X, 40, seed=42, **kw)
        b = pda_prediction_intervals(y, X, 40, seed=42, **kw)
        np.testing.assert_allclose(a["effect"]["eq_lower"], b["effect"]["eq_lower"])
        np.testing.assert_allclose(a["effect"]["sy_upper"], b["effect"]["sy_upper"])


# ---------------------------------------------------------------------------
# Coverage (the method's central claim)
# ---------------------------------------------------------------------------

class TestCoverage:
    @pytest.mark.slow
    def test_nominal_coverage_of_zero_effect(self):
        # 95% EQ prediction interval for Delta_{T0+1} should cover the true
        # zero effect about 95% of the time across DGP draws.
        T0, T1, p = 45, 1, 4
        support = [0, 1, 2, 3]
        covered = 0
        reps = 200
        for s in range(reps):
            y, X = _dgp(1000 + s, T0=T0, T1=T1, p=p, effect=0.0)
            cf = _fit(y, X, T0, support)
            out = pda_prediction_intervals(
                y, X, T0, counterfactual=cf, support=support,
                refit=_make_refit(X, T0, support), alpha=0.05, n_boot=199, seed=s,
            )
            lo, hi = out["effect"]["eq_lower"][0], out["effect"]["eq_upper"][0]
            covered += int(lo <= 0.0 <= hi)
        cov = covered / reps
        # nominal 0.95; MC SE ~ sqrt(.95*.05/200) ~ 1.5pp -> allow [0.90, 0.99]
        assert 0.90 <= cov <= 0.995, f"coverage {cov:.3f} off nominal 0.95"


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

class TestFailures:
    def test_length_mismatch(self):
        y, X = _dgp(0)
        cf = _fit(y, X, 40, [0, 1, 2, 3])
        with pytest.raises(MlsynthDataError):
            pda_prediction_intervals(y[:-1], X, 40, counterfactual=cf,
                                     support=[0], refit=lambda yb: (cf, [0]))

    def test_no_post_periods(self):
        y, X = _dgp(0, T0=46, T1=0)
        cf = _fit(y, X, 46, [0, 1, 2, 3])
        with pytest.raises(MlsynthConfigError):
            pda_prediction_intervals(y, X, 46, counterfactual=cf, support=[0],
                                     refit=lambda yb: (cf, [0]))

    def test_bad_n_boot(self):
        y, X = _dgp(0)
        cf = _fit(y, X, 40, [0, 1, 2, 3])
        with pytest.raises(MlsynthConfigError):
            pda_prediction_intervals(y, X, 40, counterfactual=cf, support=[0],
                                     refit=lambda yb: (cf, [0]), n_boot=1)

    def test_bad_alpha(self):
        y, X = _dgp(0)
        cf = _fit(y, X, 40, [0, 1, 2, 3])
        with pytest.raises(MlsynthConfigError):
            pda_prediction_intervals(y, X, 40, counterfactual=cf, support=[0],
                                     refit=lambda yb: (cf, [0]), n_boot=10, alpha=1.5)

    def test_x_not_2d(self):
        y, X = _dgp(0)
        cf = _fit(y, X, 40, [0, 1, 2, 3])
        with pytest.raises(MlsynthDataError):
            pda_prediction_intervals(y, X[:, 0], 40, counterfactual=cf, support=[0],
                                     refit=lambda yb: (cf, [0]), n_boot=10)


# ---------------------------------------------------------------------------
# Degenerate bootstrap draws (saturated refits)
# ---------------------------------------------------------------------------

# A short pre-period with a wide donor pool, so a support can saturate it.
_SAT_T0, _SAT_T1, _SAT_P = 12, 4, 20
_SAT_SPARSE = [0, 1, 2]
_SAT_FULL = list(range(_SAT_T0 - 1))     # |support| + intercept == T0


def _sat_dgp(seed=11):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((_SAT_T0 + _SAT_T1, _SAT_P))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.standard_normal(_SAT_T0 + _SAT_T1) * 0.5
    return y, X


class _SaturatingRefit:
    """Refit whose support saturates the pre-period on the first ``k`` calls.

    A support of size ``T0 - 1`` plus an intercept interpolates ``T0``
    observations, leaving a residual variance of order 1e-30 -- tiny but not
    exactly zero, so the ``se_star > 0`` guard does not catch it. The
    post-period is a genuine extrapolation and its error stays O(1), so the
    studentized statistic ``e_star / se_star`` reaches ~1e15. Forward selection
    does this on about 5% of draws at ``T0 = 30``; here it is deterministic.
    """

    def __init__(self, X, T0, k):
        self.X, self.T0, self.k, self.calls = X, T0, k, 0

    def __call__(self, y_boot):
        self.calls += 1
        support = _SAT_FULL if self.calls <= self.k else _SAT_SPARSE
        return (_ols_fit(y_boot, self.X, self.T0, support),
                np.asarray(support, dtype=int))


class TestDegenerateBootstrapDraws:
    """A saturated refit carries no information about out-of-sample error.

    Jiang et al.'s theory assumes the selected support stays sparse relative to
    ``T0``; a refit that interpolates violates that, and dividing by its
    vanishing residual scale produces an interval wide enough to cover
    unconditionally. Such draws are excluded from the quantiles and counted in
    the result.
    """

    def _setup(self, k, n_boot=99):
        y, X = _sat_dgp()
        cf = _fit(y, X, _SAT_T0, _SAT_SPARSE)
        return y, X, cf, _SaturatingRefit(X, _SAT_T0, k), n_boot

    # -- premise: the saturated refit is the pathology ------------------------

    def test_saturated_refit_interpolates_but_extrapolates_badly(self):
        y, X = _sat_dgp()
        cf, support = _SaturatingRefit(X, _SAT_T0, k=1)(y)
        s2 = float(np.mean((y[:_SAT_T0] - cf[:_SAT_T0]) ** 2))
        assert len(support) + 1 == _SAT_T0        # saturated, intercept included
        assert 0.0 < s2 < 1e-20                   # tiny, so se_star > 0 passes
        assert np.abs(y[_SAT_T0:] - cf[_SAT_T0:]).mean() > 1.0   # O(1) numerator

    # -- invariants ----------------------------------------------------------

    def test_degenerate_draws_are_counted(self):
        y, X, cf, refit, n_boot = self._setup(k=7)
        out = pda_prediction_intervals(
            y, X, _SAT_T0, counterfactual=cf, support=_SAT_SPARSE, refit=refit,
            n_boot=n_boot, seed=1)
        assert out["n_degenerate"] == 7
        assert out["n_boot_effective"] == n_boot - 7
        assert out["n_boot"] == n_boot          # the requested count is unchanged

    def test_interval_stays_finite_despite_degenerate_draws(self):
        y, X, cf, refit, n_boot = self._setup(k=20)
        out = pda_prediction_intervals(
            y, X, _SAT_T0, counterfactual=cf, support=_SAT_SPARSE, refit=refit,
            n_boot=n_boot, seed=1)
        e = out["effect"]
        for key in ("eq_lower", "eq_upper", "sy_lower", "sy_upper"):
            assert np.all(np.isfinite(e[key])), key
        assert (e["eq_lower"] <= e["eq_upper"] + 1e-9).all()
        assert (e["sy_lower"] <= e["sy_upper"] + 1e-9).all()

    def test_width_comparable_to_a_clean_run(self):
        # Excluding the degenerate draws must leave the interval on the same
        # scale as one where none occurred -- not orders of magnitude wider.
        y, X, cf, refit, n_boot = self._setup(k=20)
        dirty = pda_prediction_intervals(
            y, X, _SAT_T0, counterfactual=cf, support=_SAT_SPARSE, refit=refit,
            n_boot=n_boot, seed=1)
        clean = pda_prediction_intervals(
            y, X, _SAT_T0, counterfactual=cf, support=_SAT_SPARSE,
            refit=_make_refit(X, _SAT_T0, _SAT_SPARSE), n_boot=n_boot, seed=1)
        wd = float(dirty["effect"]["eq_upper"][0] - dirty["effect"]["eq_lower"][0])
        wc = float(clean["effect"]["eq_upper"][0] - clean["effect"]["eq_lower"][0])
        assert 0.2 < wd / wc < 5.0, f"width ratio {wd / wc:.3g}"

    # -- the guard is a no-op when nothing degenerates ------------------------

    def test_clean_path_unchanged(self):
        # Pinned from the implementation before the guard existed: on data where
        # no draw degenerates the intervals must be identical, to the bit.
        y, X = _dgp(7)
        support = [0, 1, 2, 3]
        cf = _fit(y, X, 40, support)
        out = pda_prediction_intervals(
            y, X, 40, counterfactual=cf, support=support,
            refit=_make_refit(X, 40, support), n_boot=199, seed=42)
        assert out["n_degenerate"] == 0
        assert out["n_boot_effective"] == 199
        np.testing.assert_allclose(
            out["effect"]["eq_lower"],
            [-1.148044173333, -0.710756198779, -1.475381568738,
             -1.379849220776, -1.628921193323, -1.473219215610])
        np.testing.assert_allclose(
            out["effect"]["eq_upper"],
            [1.028535794298, 1.477007898191, 0.923063871361,
             0.952722049167, 0.749210506010, 0.664584840508])
        np.testing.assert_allclose(
            out["effect"]["sy_upper"],
            [0.872494477313, 1.413704825787, 0.860925699272,
             0.859676346861, 0.775951061707, 0.548522138988])

    # -- edge cases ----------------------------------------------------------

    def test_all_draws_degenerate_raises(self):
        y, X, cf, refit, n_boot = self._setup(k=99)
        with pytest.raises(MlsynthEstimationError, match="degenerate"):
            pda_prediction_intervals(
                y, X, _SAT_T0, counterfactual=cf, support=_SAT_SPARSE,
                refit=refit, n_boot=n_boot, seed=1)

    def test_single_surviving_draw_raises(self):
        # Two draws are the minimum from which a quantile can be formed.
        y, X, cf, refit, n_boot = self._setup(k=98)
        with pytest.raises(MlsynthEstimationError, match="degenerate"):
            pda_prediction_intervals(
                y, X, _SAT_T0, counterfactual=cf, support=_SAT_SPARSE,
                refit=refit, n_boot=n_boot, seed=1)

    def test_two_surviving_draws_succeed(self):
        y, X, cf, refit, n_boot = self._setup(k=97)
        out = pda_prediction_intervals(
            y, X, _SAT_T0, counterfactual=cf, support=_SAT_SPARSE, refit=refit,
            n_boot=n_boot, seed=1)
        assert out["n_boot_effective"] == 2
        assert np.all(np.isfinite(out["effect"]["eq_lower"]))

    def test_interpolating_point_estimator_raises(self):
        # sigma^2 = 0 leaves no residual scale to studentize against at all.
        y, X = _dgp(3)
        with pytest.raises(MlsynthEstimationError, match="pre-period"):
            pda_prediction_intervals(
                y, X, 40, counterfactual=y.copy(), support=[0, 1, 2, 3],
                refit=_make_refit(X, 40, [0, 1, 2, 3]), n_boot=49, seed=1)

    # -- the failure is reported, not swallowed ------------------------------

    def test_degenerate_count_surfaces_through_the_estimator(self):
        # The engine's count must reach the PDA result, so a user can see that
        # some draws were dropped without instrumenting the engine.
        from mlsynth import PDA

        df = _pda_panel(seed=2)
        res = PDA(_pda_cfg(df, method="fs", prediction_intervals=True,
                           pi_n_boot=99, pi_seed=0)).fit()
        pi = res.fits["fs"].prediction_intervals
        assert "n_degenerate" in pi and "n_boot_effective" in pi
        assert pi["n_degenerate"] >= 0
        assert pi["n_boot_effective"] + pi["n_degenerate"] == pi["n_boot"]


class TestIndependentBootstrap:
    def test_iid_multipliers_path(self):
        y, X = _dgp(0)
        support = [0, 1, 2, 3]
        cf = _fit(y, X, 40, support)
        out = pda_prediction_intervals(
            y, X, 40, counterfactual=cf, support=support,
            refit=_make_refit(X, 40, support), n_boot=199, dependent=False, seed=1,
        )
        assert out["effect"]["eq_lower"].shape == (6,)
        assert (out["effect"]["eq_lower"] <= out["effect"]["eq_upper"] + 1e-9).all()


# ---------------------------------------------------------------------------
# Wiring into the PDA estimator (all three methods)
# ---------------------------------------------------------------------------

def _pda_panel(n_units=8, n_periods=40, t_start=30, effect=5.0, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    periods = np.arange(1, n_periods + 1)
    n_donors = n_units - 1
    donors = {}
    for j in range(n_donors):
        donors[j] = rng.normal(20 + 4 * j, 3) + np.linspace(0, 6 + j, n_periods) \
            + rng.normal(0, 0.6, n_periods)
    w = rng.uniform(0.2, 1.0, n_donors)
    w /= w.sum()
    treated = sum(w[j] * donors[j] for j in range(n_donors)) + rng.normal(0, 0.3, n_periods)
    rows = []
    for i, t in enumerate(periods):
        v = treated[i] + (effect if t >= t_start else 0.0)
        rows.append({"ID": 1, "Period": int(t), "Value": float(v),
                     "IsTreated": int(t >= t_start)})
        for j in range(n_donors):
            rows.append({"ID": j + 2, "Period": int(t),
                         "Value": float(donors[j][i]), "IsTreated": 0})
    return pd.DataFrame(rows)


def _pda_cfg(df, **kw):
    cfg = dict(df=df, outcome="Value", treat="IsTreated", unitid="ID",
               time="Period", display_graphs=False)
    cfg.update(kw)
    return cfg


class TestPDAWiring:
    @pytest.mark.parametrize("method", ["l2", "LASSO", "fs", "hcw"])
    def test_prediction_intervals_attached(self, method):
        from mlsynth import PDA
        from mlsynth.config_models import PDAConfig

        df = _pda_panel()
        res = PDA(PDAConfig(**_pda_cfg(
            df, method=method, prediction_intervals=True, pi_n_boot=99))).fit()
        fit = next(iter(res.fits.values()))
        pi = fit.prediction_intervals
        assert pi is not None
        assert pi["studentization"] in ("sandwich", "sigma2")
        T1 = res.inputs.T2
        assert pi["effect"]["eq_lower"].shape == (T1,)
        assert pi["counterfactual"]["sy_upper"].shape == (T1,)
        assert (pi["effect"]["eq_lower"] <= pi["effect"]["eq_upper"] + 1e-9).all()

    def test_off_by_default(self):
        from mlsynth import PDA
        from mlsynth.config_models import PDAConfig

        df = _pda_panel()
        res = PDA(PDAConfig(**_pda_cfg(df, method="fs"))).fit()
        fit = next(iter(res.fits.values()))
        assert fit.prediction_intervals is None

    def test_reproducible_via_pi_seed(self):
        from mlsynth import PDA
        from mlsynth.config_models import PDAConfig

        df = _pda_panel()
        cfg = _pda_cfg(df, method="fs", prediction_intervals=True, pi_n_boot=99, pi_seed=7)
        a = PDA(PDAConfig(**cfg)).fit().fits["fs"].prediction_intervals
        b = PDA(PDAConfig(**cfg)).fit().fits["fs"].prediction_intervals
        np.testing.assert_allclose(a["effect"]["eq_lower"], b["effect"]["eq_lower"])
