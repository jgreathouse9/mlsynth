"""``optimized`` must not depend on what units a covariate is measured in.

The estimator is defined by a minimisation whose exact solution is
scale-equivariant: rescale ``X`` by ``c`` and ``beta`` scales by ``1/c``, so
``X beta`` -- and therefore the estimate -- is unchanged. The reference does not
solve that minimisation exactly. It descends with a fixed ``1/t`` step schedule
and no curvature scaling, and a fixed step is not scale free: in a direction
whose curvature goes like ``(4e4)^2`` the first step overshoots, the residual
grows, and the next gradient is larger still.

The result was an estimate eleven orders of magnitude out, returned silently:

    no covariates                -46.94
    optimized, unscaled       1.745e+11
    Stata sdid (published)       -53.154

So each covariate is scaled by its standard deviation before the descent and the
coefficient is rescaled back afterwards. That is not a numerical nicety layered
on top of the reference -- it is what makes the fitted coefficient reproduce the
reference at all, on three separate published panels.

These tests pin the invariant (rescaling a column changes nothing), the failure
that motivated it (a wine-shaped panel stays finite and sane), and the degenerate
case the scaling introduces (a constant covariate must not divide by zero).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pd = pytest.importorskip("pandas")


def _panel(n_units=8, n_periods=16, first_treated=11, cov_scale=1.0, seed=0):
    """Block panel with one covariate that genuinely drives the outcome.

    ``cov_scale`` multiplies the covariate only. Because the covariate's effect
    on the outcome is applied before scaling, every ``cov_scale`` describes the
    same data in different units -- which is the whole point.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        level = 10.0 + 2.0 * i
        for t in range(1, n_periods + 1):
            x = rng.normal(3.0, 1.0)
            treated = i == 0 and t >= first_treated
            y = (level + 0.4 * t + 1.5 * x + rng.normal(0, 0.2)
                 - (5.0 if treated else 0.0))
            rows.append((f"u{i}", t, y, x * cov_scale, int(treated)))
    return pd.DataFrame(rows, columns=["unit", "t", "y", "x", "treat"])


def _att(df, covariates=None):
    from mlsynth import SDID

    cfg = {"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
           "time": "t", "vce": "noinference", "display_graphs": False}
    if covariates is not None:
        cfg["covariates"] = covariates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(SDID(cfg).fit().effects.att)


class TestTheEstimateDoesNotDependOnTheCovariatesUnits:
    """The invariant whose absence was the bug."""

    @pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 1e5])
    def test_rescaling_a_covariate_leaves_the_estimate_alone(self, scale):
        base = _att(_panel(cov_scale=1.0), {"optimized": ["x"]})
        got = _att(_panel(cov_scale=scale), {"optimized": ["x"]})
        assert got == pytest.approx(base, rel=1e-6)

    def test_the_estimate_stays_finite_at_every_scale(self):
        for scale in (1e-4, 1.0, 1e4, 1e6):
            got = _att(_panel(cov_scale=scale), {"optimized": ["x"]})
            assert np.isfinite(got), f"diverged at scale {scale}"
            assert abs(got) < 1e3, f"exploded to {got} at scale {scale}"

    def test_the_fitted_coefficient_is_returned_in_the_callers_units(self):
        """beta must come back on the column's own scale, not the internal one.

        The caller residualises the raw covariate with it, so a coefficient
        left in standardised units would silently rescale the outcome.
        """
        from mlsynth.utils.sdid_helpers.setup import prepare_sdid_inputs

        betas = {}
        for scale in (1.0, 1000.0):
            inputs = prepare_sdid_inputs(
                df=_panel(cov_scale=scale), outcome="y", treat="treat",
                unitid="unit", time="t", optimized_covariates=["x"])
            payload = next(iter(inputs.cohorts_dict.values()))
            betas[scale] = float(payload["optimized_beta"][0])
        # x scaled up by 1000 -> beta scaled down by 1000
        assert betas[1000.0] == pytest.approx(betas[1.0] / 1000.0, rel=1e-6)


class TestTheWineShapedPanelThatFoundThis:
    """A covariate whose dispersion is ~70x the outcome's, as in the paper."""

    def _wine_shaped(self):
        rng = np.random.default_rng(3)
        rows = []
        for i in range(11):
            level = 1000.0 + 120.0 * i
            income = 45000.0 + 4000.0 * i
            for t in range(1, 41):
                inc = income + rng.normal(0, 4700)
                treated = i == 0 and t >= 31
                y = (level + 3.0 * t + 0.004 * inc + rng.normal(0, 60)
                     - (40.0 if treated else 0.0))
                rows.append((f"s{i}", t, y, inc, int(treated)))
        return pd.DataFrame(rows, columns=["unit", "t", "y", "x", "treat"])

    def test_it_does_not_diverge(self):
        df = self._wine_shaped()
        plain = _att(df)
        got = _att(df, {"optimized": ["x"]})
        assert np.isfinite(got)
        # the covariate explains part of the outcome, so the two differ -- but
        # by an amount an effect of this size could plausibly move, not by 1e11
        assert abs(got - plain) < 10 * abs(plain)

    def test_the_coefficient_recovers_the_planted_slope(self):
        """Sanity that scaling did not break the fit itself.

        The covariate enters with slope 0.004; the reference stops early by
        design, so this asks for the right sign and order of magnitude.
        """
        from mlsynth.utils.sdid_helpers.setup import prepare_sdid_inputs

        inputs = prepare_sdid_inputs(
            df=self._wine_shaped(), outcome="y", treat="treat", unitid="unit",
            time="t", optimized_covariates=["x"])
        beta = float(next(iter(inputs.cohorts_dict.values()))["optimized_beta"][0])
        assert 0.0 < beta < 0.04


class TestDegenerateCovariates:
    """Scaling introduces a division, so the zero case needs an answer."""

    def test_a_constant_covariate_does_not_divide_by_zero(self):
        df = _panel()
        df["const"] = 7.0
        got = _att(df, {"optimized": ["const"]})
        assert np.isfinite(got)

    def test_a_constant_covariate_leaves_the_estimate_alone(self):
        """It carries no information, so it must not move anything."""
        df = _panel()
        df["const"] = 7.0
        assert _att(df, {"optimized": ["const"]}) == pytest.approx(
            _att(df), rel=1e-9)

    def test_a_unit_invariant_covariate_is_harmless(self):
        """Absorbed by the unit effects, so its coefficient stays at zero."""
        from mlsynth.utils.sdid_helpers.setup import prepare_sdid_inputs

        df = _panel()
        df["fixed"] = df["unit"].map(
            {u: float(i) for i, u in enumerate(sorted(df["unit"].unique()))})
        inputs = prepare_sdid_inputs(
            df=df, outcome="y", treat="treat", unitid="unit", time="t",
            optimized_covariates=["fixed"])
        beta = float(next(iter(inputs.cohorts_dict.values()))["optimized_beta"][0])
        assert abs(beta) < 1e-6


class TestThePinnedPanelMovesToTheBetterNumber:
    """The quota panel is the pinned case; scaling must improve it, not break it."""

    def test_quota_reproduces_more_closely_than_before(self):
        from pathlib import Path

        panel = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
                 / "sdid_quota" / "quota_panel.csv")
        if not panel.exists():  # pragma: no cover - the panel is committed
            pytest.skip(f"missing {panel}")
        from mlsynth import SDID

        d = pd.read_csv(panel).dropna(subset=["lngdp"])
        d["year"] = d["year"].astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = float(SDID({
                "df": d, "outcome": "womparl", "treat": "quota",
                "unitid": "country", "time": "year", "vce": "noinference",
                "display_graphs": False,
                "covariates": {"optimized": ["lngdp"]}}).fit().effects.att)
        # Stata reports 8.051. Unscaled gives 8.0446, which is 0.0064 away;
        # scaling gives 8.0483, which is 0.0027. The threshold sits between
        # them so this states the improvement rather than merely comparing
        # against a rounded literal -- an earlier version did the latter and
        # passed against the unscaled code on rounding noise alone.
        assert abs(got - 8.051) < 0.005
