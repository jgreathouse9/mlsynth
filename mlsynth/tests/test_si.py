"""Tests for the Synthetic Interventions (SI) estimator.

Covers config validation, the SI-PCR estimation math (plain and bias-corrected),
the estimator integration, the asymptotic-normality confidence intervals, and
plotting.

Reference: Agarwal, Shah & Shen (2026), "Synthetic Interventions," Operations
Research 74(2):840-859.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt

from mlsynth import SI
from mlsynth.config_models import SIConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.si_helpers.estimation import (
    bias_corrected_fit,
    donoho_rank,
    resolve_rank,
    select_omega,
    si_pcr_weights,
    variance_estimation,
)
from mlsynth.utils.clustersc_helpers.pcr.hsvt import hsvt
from mlsynth.utils.si_helpers.plotter import plot_si
from mlsynth.utils.si_helpers.structures import SIArm, SIResults


# ----------------------------------------------------------------------
# DGP: a low-rank panel with a focal treated unit and two donor pools
# ----------------------------------------------------------------------

def _simulate(seed=0, N=14, T=30, T0=20, rank=2, noise=0.3):
    rng = np.random.default_rng(seed)
    F = np.column_stack([np.cumsum(rng.normal(0, 1, T)) for _ in range(rank)])
    lam = rng.normal(0, 1, (N, rank))
    Y = lam @ F.T + noise * rng.standard_normal((N, T))
    units = [f"u{j:02d}" for j in range(N)]
    rows = []
    for j, u in enumerate(units):
        inter_a = int(1 <= j <= 6)
        inter_b = int(7 <= j <= N - 1)
        for t in range(T):
            rows.append({
                "unit": u, "time": t, "y": float(Y[j, t]),
                "treat": int(u == "u00" and t >= T0),
                "interA": inter_a, "interB": inter_b,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    return _simulate(0)


_BASE = dict(outcome="y", unitid="unit", time="time", treat="treat",
             inters=["interA", "interB"], display_graphs=False)


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------

class TestSIConfig:
    def test_defaults(self, panel):
        cfg = SIConfig(df=panel, **_BASE)
        assert cfg.rank_method == "donoho" and cfg.bias_correct is True
        assert cfg.variance == "double" and cfg.interval == "confidence"
        assert cfg.alpha == 0.05 and cfg.cumvar_threshold == 0.95

    def test_inters_required_nonempty(self, panel):
        with pytest.raises(Exception):
            SIConfig(df=panel, outcome="y", unitid="unit", time="time",
                     treat="treat", inters=[])

    def test_fixed_rank_requires_rank(self, panel):
        with pytest.raises(MlsynthConfigError):
            SIConfig(df=panel, **{**_BASE, "rank_method": "fixed"})

    def test_alpha_out_of_range_rejected(self, panel):
        with pytest.raises(Exception):
            SIConfig(df=panel, **{**_BASE, "alpha": 1.5})


# ----------------------------------------------------------------------
# Estimation math
# ----------------------------------------------------------------------

class TestEstimationMath:
    def test_si_pcr_weights_reconstruct_pre(self):
        rng = np.random.default_rng(1)
        F = rng.standard_normal((25, 2))
        lam = rng.standard_normal((6, 2))
        donor_pre = F @ lam.T                       # (25, 6), exact rank 2
        target_pre = F @ np.array([0.5, 0.3])       # in the donor span
        w = si_pcr_weights(donor_pre, target_pre, rank=2)
        assert w.shape == (6,)
        # rank-2 reconstruction of an in-span target is near-exact
        assert np.allclose(donor_pre @ w, target_pre, atol=1e-6)

    def test_select_omega_is_rank_complete(self):
        rng = np.random.default_rng(2)
        donor_pre = rng.standard_normal((20, 8))
        omega = select_omega(donor_pre, rank=3)
        assert len(omega) == 3 and len(set(omega)) == 3
        assert np.linalg.matrix_rank(donor_pre[:, omega]) == 3

    def test_bias_corrected_fit_shapes_and_sigma(self):
        rng = np.random.default_rng(3)
        F = rng.standard_normal((25, 2))
        donor_pre = F @ rng.standard_normal((2, 7)) + 0.01 * rng.standard_normal((25, 7))
        target_pre = F @ np.array([0.4, -0.2]) + 0.01 * rng.standard_normal(25)
        omega, w_omega, sigma = bias_corrected_fit(donor_pre, target_pre, rank=2)
        assert len(omega) == 2 and w_omega.shape == (2,)
        assert sigma >= 0.0 and np.isfinite(sigma)

    def test_resolve_rank_modes(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((20, 6))
        assert resolve_rank(X, "fixed", rank=3) == 3
        assert 1 <= resolve_rank(X, "usvt") <= 6
        assert 1 <= resolve_rank(X, "cumvar", cumvar_threshold=0.9) <= 6
        assert 1 <= resolve_rank(X, "donoho") <= 6

    def test_donoho_rank_matches_authors_formula(self):
        # the paper's exact rule: omega(ratio)*median(s), ratio = rows/cols
        rng = np.random.default_rng(5)
        X = (rng.standard_normal((30, 4)) @ rng.standard_normal((4, 8)))
        X = X + 0.05 * rng.standard_normal(X.shape)
        s = np.linalg.svd(X, compute_uv=False)
        m, n = X.shape
        omega = 0.56 * (m / n) ** 3 - 0.95 * (m / n) ** 2 + 1.43 + 1.82 * (m / n)
        expected = max(int(np.sum(s > omega * np.median(s))), 1)
        assert donoho_rank(s, m / n) == expected
        assert resolve_rank(X, "donoho") == min(expected, min(m, n))

    def test_variance_estimators_finite(self):
        rng = np.random.default_rng(6)
        F = rng.standard_normal((40, 3))
        donor_pre = F[:30] @ rng.standard_normal((3, 8)) + 0.1 * rng.standard_normal((30, 8))
        donor_post = F[30:] @ rng.standard_normal((3, 8))
        target_pre = F[:30] @ np.array([0.4, -0.2, 0.1])
        _, U, _, Vt = hsvt(donor_pre, 3)
        double, units, time_iv = variance_estimation(U, Vt.T, target_pre, donor_post)
        assert all(np.isfinite([double, units, time_iv]))
        assert all(v >= 0 for v in (double, units, time_iv))


# ----------------------------------------------------------------------
# Estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    def test_fit_returns_results(self, panel):
        res = SI({"df": panel, **_BASE}).fit()
        assert isinstance(res, SIResults) and res.mode == "si"
        assert set(res.arms) == {"interA", "interB"}
        for arm in res.arms.values():
            assert isinstance(arm, SIArm)
            assert arm.counterfactual.shape == (res.inputs.T,)
            assert np.isfinite(arm.att)
            assert arm.bias_corrected is True

    def test_bias_corrected_has_cis(self, panel):
        res = SI({"df": panel, **_BASE}).fit()
        arm = res.arms["interA"]
        assert arm.att_ci is not None and arm.cf_mean_ci is not None
        lo, hi = arm.att_ci
        assert lo < arm.att < hi
        assert arm.sigma_hat is not None and arm.weight_norm is not None
        # weights supported on the rank-complete subset
        assert set(arm.weights) <= set(arm.omega_names)
        assert len(arm.omega_names) == arm.selected_rank

    def test_plain_pcr_no_cis(self, panel):
        res = SI({"df": panel, **{**_BASE, "bias_correct": False}}).fit()
        arm = res.arms["interA"]
        assert arm.att_ci is None and arm.sigma_hat is None
        # plain SI-PCR uses the full donor pool
        assert arm.omega_names == arm.donor_names

    def test_att_by_intervention(self, panel):
        res = SI({"df": panel, **_BASE}).fit()
        d = res.att_by_intervention
        assert set(d) == {"interA", "interB"}
        assert all(np.isfinite(v) for v in d.values())

    def test_missing_intervention_column_raises(self, panel):
        with pytest.raises(MlsynthConfigError):
            SI({"df": panel, **{**_BASE, "inters": ["interA", "nope"]}}).fit()

    def test_no_donors_raises(self, panel):
        df = panel.copy()
        df["interC"] = 0  # nobody received it
        with pytest.raises(MlsynthDataError):
            SI({"df": df, **{**_BASE, "inters": ["interC"]}}).fit()

    def test_fixed_rank_runs(self, panel):
        res = SI({"df": panel, **{**_BASE, "rank_method": "fixed", "rank": 2}}).fit()
        assert all(arm.selected_rank == 2 for arm in res.arms.values())

    def test_prediction_interval_wider_than_confidence(self, panel):
        ci = SI({"df": panel, **{**_BASE, "interval": "confidence"}}).fit().arms["interA"]
        pi = SI({"df": panel, **{**_BASE, "interval": "prediction"}}).fit().arms["interA"]
        ci_w = ci.att_ci[1] - ci.att_ci[0]
        pi_w = pi.att_ci[1] - pi.att_ci[0]
        assert pi_w > ci_w

    def test_variance_choice_changes_width(self, panel):
        a = SI({"df": panel, **{**_BASE, "variance": "double"}}).fit().arms["interA"]
        b = SI({"df": panel, **{**_BASE, "variance": "units"}}).fit().arms["interA"]
        assert a.sigma_hat != b.sigma_hat


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

class TestPlot:
    def test_plot_runs(self, panel):
        res = SI({"df": panel, **_BASE}).fit()
        plot_si(res)
        _plt.close("all")
