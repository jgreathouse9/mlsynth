"""Coverage + correctness tests for the Shi & Huang (2023) PDA Table-1 DGP."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.pda_helpers.simulation import (
    _factors,
    _shock,
    simulate_pda_panel,
    simulate_pda_shocks,
)

# Unconditional factor variances under the dynamic structure, from the ARMA
# specifications: i.i.d.; AR(1) 0.9; MA(2) (0.8, 0.4); ARMA(1,1) (0.5, 0.5).
# These are the diagonal of Sigma.f.nnd in their FS.simulation.dense.R.
_DYNAMIC_FACTOR_VAR = (
    1.0,
    1.0 / (1.0 - 0.9 ** 2),
    1.0 + 0.8 ** 2 + 0.4 ** 2,
    (1.0 + 2 * 0.5 * 0.5 + 0.5 ** 2) / (1.0 - 0.5 ** 2),
)


class TestSimulatePdaPanel:
    def test_default_iid_shapes_and_labels(self):
        s = simulate_pda_panel(N=20, T1=15, seed=0)
        assert s.T1 == 15 and s.T2 == 15
        assert s.Y_treated.shape == (30,)
        assert s.Y_controls.shape == (20, 30)
        assert s.relevant_donors == ["c000", "c001", "c002", "c003"]
        assert s.shock == "D1" and s.is_null is True
        # long frame: (N + 1 treated) * T rows; one treated unit
        assert isinstance(s.df, pd.DataFrame)
        assert len(s.df) == 21 * 30
        assert set(s.df["unit"]) == {"treated", *[f"c{i:03d}" for i in range(20)]}
        # treatment is absorbing on the treated unit only
        tr = s.df[s.df["unit"] == "treated"].sort_values("time")
        assert tr["treat"].tolist() == [0] * 15 + [1] * 15
        assert (s.df[s.df["unit"] != "treated"]["treat"] == 0).all()

    def test_seed_is_deterministic(self):
        a = simulate_pda_panel(N=10, T1=12, seed=7).Y_treated
        b = simulate_pda_panel(N=10, T1=12, seed=7).Y_treated
        np.testing.assert_array_equal(a, b)

    def test_rng_takes_precedence_over_seed(self):
        rng = np.random.default_rng(3)
        s = simulate_pda_panel(N=8, T1=10, rng=rng, seed=999)
        assert s.Y_controls.shape == (8, 20)

    def test_dynamic_factors_differ_from_iid(self):
        kw = dict(N=12, T1=40, seed=1)
        iid = simulate_pda_panel(dynamic_factors=False, **kw).Y_treated
        dyn = simulate_pda_panel(dynamic_factors=True, **kw).Y_treated
        assert not np.allclose(iid, dyn)

    def test_effect_override_sets_custom_and_shifts_post(self):
        s0 = simulate_pda_panel(N=10, T1=10, effect=0.0, seed=2)
        assert s0.shock == "custom" and s0.is_null is True
        s5 = simulate_pda_panel(N=10, T1=10, effect=5.0, seed=2)
        assert s5.shock == "custom" and s5.is_null is False
        # the only difference is a +5 shift on the treated post-period
        np.testing.assert_allclose(s5.Y_treated[:10], s0.Y_treated[:10])
        np.testing.assert_allclose(s5.Y_treated[10:], s0.Y_treated[10:] + 5.0)

    @pytest.mark.parametrize("shock,is_null", [
        ("D1", True), ("D2", True), ("D3", True),
        ("D4", False), ("D5", False), ("D6", False), ("D7", False),
    ])
    def test_all_shocks_run_and_flag_null(self, shock, is_null):
        s = simulate_pda_panel(N=10, T1=12, shock=shock, seed=4)
        assert s.shock == shock
        assert s.is_null is is_null

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            simulate_pda_panel(N=4, T1=10)           # N must exceed 4 relevant
        with pytest.raises(ValueError):
            simulate_pda_panel(N=10, T1=0)            # T1 positive
        with pytest.raises(ValueError):
            simulate_pda_panel(N=10, T1=10, T2=0)     # T2 positive


class TestHelpers:
    def test_factors_iid_shape(self):
        f = _factors(50, np.random.default_rng(0), dynamic=False)
        assert f.shape == (50, 4)

    def test_factors_dynamic_shape_after_burn(self):
        f = _factors(50, np.random.default_rng(0), dynamic=True)
        assert f.shape == (50, 4)

    def test_shock_null_processes_are_mean_small(self):
        rng = np.random.default_rng(0)
        assert np.allclose(_shock("D1", 30, rng), 0.0)
        # D4/D5 carry positive mean shifts
        assert _shock("D5", 2000, np.random.default_rng(1)).mean() > 0.5

    def test_shock_unknown_raises(self):
        with pytest.raises(ValueError):
            _shock("D9", 10, np.random.default_rng(0))


class TestTheSimulatorMatchesTheReleasedDgp:
    """The moments the DGP is defined by, against ``FS.simulation.dense.R``.

    Every test above this class asserts a shape, a label or a repeat draw. None
    of them can see a factor generated at the wrong scale or noise generated at
    twice the right standard deviation, because those produce a panel of exactly
    the right shape, with the right column names, reproducible from its seed. The
    quantities below are the ones the DGP actually consists of, so they are what
    gets asserted.

    Provenance: ``zhentaoshi/fsPDA`` at ``5f4542c``,
    ``simulation/nonsparse/FS.simulation.dense.R`` (``FS.simulation.dense`` for
    the factors and noise, ``DGP.Lambda`` for the loadings, ``DGP.Delta`` for the
    shocks) and the ``loading.RData`` its driver loads.
    """

    def test_iid_factor_l_has_standard_deviation_l(self):
        """``f = foreach(k=1:K) %do% {rnorm(Tn, sd = k * sigma.u)}``.

        Factor 4 carries sixteen times the variance of factor 1. Drawing all
        four standard normal instead flattens the factor structure, which is the
        thing the estimators are selecting on.
        """
        f = _factors(200_000, np.random.default_rng(0), dynamic=False)
        np.testing.assert_allclose(f.std(0), [1.0, 2.0, 3.0, 4.0], rtol=0.02)

    def test_dynamic_factor_variances_match_their_arma_specifications(self):
        f = _factors(200_000, np.random.default_rng(1), dynamic=True)
        np.testing.assert_allclose(f.var(0), _DYNAMIC_FACTOR_VAR, rtol=0.05)

    def test_idiosyncratic_noise_has_standard_deviation_one_half(self):
        """``sigma.eta = 0.5``, so the variance is 0.25.

        The paper writes ``e_jt ~ N(0, 0.5^2)``. Reading that as a variance of
        0.5 puts the standard deviation at ``sqrt(0.5) = 0.707`` -- half again
        too much noise on every series in the panel.
        """
        s = simulate_pda_panel(N=60, T1=4000, seed=2)
        resid = s.Y_controls - s.loadings[1:] @ s.factors.T
        assert abs(float(resid.std()) - 0.5) < 0.01

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_the_ols_projection_matches_their_closed_form(self, dynamic):
        """The check that ties the pieces together.

        Under the factor model the population projection of the treated outcome
        on the donors is available in closed form -- it is ``beta0.dense`` in
        their script:

            beta0 = (Lam Sigma_f Lam' + sigma_eta^2 I)^-1 Lam Sigma_f lambda_0.

        OLS on a long panel converges to it. The identity involves the factor
        covariance and the noise variance at once, so it fails if either is
        wrong, and it does so whatever the shapes look like.
        """
        var = [1.0, 4.0, 9.0, 16.0] if not dynamic else list(_DYNAMIC_FACTOR_VAR)
        s = simulate_pda_panel(N=12, T1=60_000, T2=1, dynamic_factors=dynamic,
                               seed=3)
        lam0, lam = s.loadings[0], s.loadings[1:]
        sf = np.diag(var)
        expected = np.linalg.solve(lam @ sf @ lam.T + 0.25 * np.eye(len(lam)),
                                   lam @ sf @ lam0)
        X = s.Y_controls[:, :s.T1].T
        fitted = np.linalg.lstsq(X, s.Y_treated[:s.T1], rcond=None)[0]
        np.testing.assert_allclose(fitted, expected, atol=0.02)

    def test_loadings_are_drawn_from_their_two_ranges(self):
        s = simulate_pda_panel(N=400, T1=10, seed=4)
        major, minor = s.loadings[:5], s.loadings[5:]
        assert major.min() >= 1.0 and major.max() <= 2.0
        assert minor.min() >= -0.5 and minor.max() <= 0.5
        # and the irrelevant block really does fill its range
        assert minor.min() < -0.45 and minor.max() > 0.45

    def test_supplied_loadings_are_used_verbatim(self):
        """The benchmark hands in their committed ``loading.RData``."""
        lam = np.arange(4 * 11, dtype=float).reshape(11, 4) / 10.0
        s = simulate_pda_panel(N=10, T1=6, loadings=lam, seed=5)
        np.testing.assert_array_equal(s.loadings, lam)

    def test_supplied_loadings_of_the_wrong_shape_raise(self):
        with pytest.raises(ValueError, match="loadings"):
            simulate_pda_panel(N=10, T1=6, loadings=np.zeros((5, 4)))

    def test_irrelevant_loading_half_width_is_settable(self):
        """The paper's prose says 0.1; the loadings its table was run on say 0.5."""
        s = simulate_pda_panel(N=400, T1=10, irrelevant_loading=0.1, seed=6)
        assert np.abs(s.loadings[5:]).max() <= 0.1


class TestTheSevenShocksShareOneInnovation:
    """``DGP.Delta`` draws ``w`` once and builds all six columns from it.

    Drawing an independent ``w`` per shock gives each column the right marginal
    law and the wrong joint one, which is invisible to a test that looks at one
    column at a time. The size and power cells of Table 1 are read off the same
    replication, so the joint law is the one that matters.
    """

    def test_all_seven_columns_come_back(self):
        d = simulate_pda_shocks(40, seed=0)
        assert d.shape == (40, 7)
        np.testing.assert_array_equal(d[:, 0], 0.0)          # D1

    def test_the_mean_shifts_are_exact_differences(self):
        """D4 - D2 is 0.5 and D5 - D2 is 1.0, period by period: same ``w``."""
        d = simulate_pda_shocks(50, seed=1)
        np.testing.assert_allclose(d[:, 3] - d[:, 1], 0.5)
        np.testing.assert_allclose(d[:, 4] - d[:, 1], 1.0)

    def test_the_serially_correlated_shifts_are_exact_differences(self):
        """D6 and D7 sit above D3 by ``mu / (1 - rho)``, which is what the
        burn-in buys: without it the AR recursion starts at zero and the first
        periods sit below their stationary mean."""
        d = simulate_pda_shocks(50, seed=2)
        np.testing.assert_allclose(d[:, 5] - d[:, 2], 0.35 / 0.7, atol=1e-9)
        np.testing.assert_allclose(d[:, 6] - d[:, 2], 0.7 / 0.7, atol=1e-9)

    def test_the_autoregressive_columns_are_serially_correlated(self):
        d = simulate_pda_shocks(20_000, seed=3)
        for j in (2, 5, 6):
            z = d[:, j] - d[:, j].mean()
            assert 0.2 < float(np.sum(z[1:] * z[:-1]) / np.sum(z * z)) < 0.4
        for j in (1, 3, 4):
            z = d[:, j] - d[:, j].mean()
            assert abs(float(np.sum(z[1:] * z[:-1]) / np.sum(z * z))) < 0.05

    def test_the_means_are_the_paper_s(self):
        d = simulate_pda_shocks(200_000, seed=4)
        np.testing.assert_allclose(d.mean(0), [0, 0, 0, 0.5, 1.0, 0.5, 1.0],
                                   atol=0.02)

    def test_a_single_shock_is_the_matching_column(self):
        rng_a, rng_b = np.random.default_rng(9), np.random.default_rng(9)
        col = {"D1": 0, "D2": 1, "D3": 2, "D4": 3, "D5": 4, "D6": 5, "D7": 6}
        for name, j in col.items():
            np.testing.assert_allclose(
                _shock(name, 30, np.random.default_rng(11)),
                simulate_pda_shocks(30, rng=np.random.default_rng(11))[:, j])
        assert rng_a.random() == rng_b.random()

    def test_a_non_positive_horizon_raises(self):
        with pytest.raises(ValueError, match="T2"):
            simulate_pda_shocks(0, seed=0)


class TestLibellDgp3:
    """Li & Bell (2017) DGP3 simulator (the LASSO-PDA OOS-prediction design)."""

    def test_shape_treated_and_controls(self):
        from mlsynth.utils.pda_helpers.simulation import simulate_libell_panel
        df = simulate_libell_panel(N=31, T1=25, T2=10, seed=0)
        assert df["unit"].nunique() == 31 and df["time"].nunique() == 35
        tr = df[df["unit"] == "treated"].sort_values("time")
        assert tr["treat"].tolist() == [0] * 25 + [1] * 10
        assert (df[df["unit"] != "treated"]["treat"] == 0).all()

    def test_deterministic(self):
        from mlsynth.utils.pda_helpers.simulation import simulate_libell_panel
        a = simulate_libell_panel(seed=7)["y"].to_numpy()
        b = simulate_libell_panel(seed=7)["y"].to_numpy()
        np.testing.assert_array_equal(a, b)

    def test_sigma2_scales_idiosyncratic_variance(self):
        from mlsynth.utils.pda_helpers.simulation import simulate_libell_panel
        lo = simulate_libell_panel(sigma2=0.1, seed=1)["y"].var()
        hi = simulate_libell_panel(sigma2=4.0, seed=1)["y"].var()
        assert hi > lo

    def test_factors_shape(self):
        from mlsynth.utils.pda_helpers.simulation import _libell_factors
        f = _libell_factors(50, np.random.default_rng(0))
        assert f.shape == (50, 3)


class TestShiWangDgp:
    """Shi & Wang (2024) Table-2 simulator (the L2-relaxation size/power design)."""

    def test_shapes_and_default(self):
        from mlsynth.utils.pda_helpers.simulation import simulate_shiwang_panel
        y, Yc, T1 = simulate_shiwang_panel(N=100, T1=50, seed=0)
        assert y.shape == (100,) and Yc.shape == (100, 100) and T1 == 50

    def test_deterministic(self):
        from mlsynth.utils.pda_helpers.simulation import simulate_shiwang_panel
        a = simulate_shiwang_panel(T1=40, seed=7)[0]
        b = simulate_shiwang_panel(T1=40, seed=7)[0]
        np.testing.assert_array_equal(a, b)

    def test_d4_adds_constant_post_shift(self):
        from mlsynth.utils.pda_helpers.simulation import simulate_shiwang_panel
        y0 = simulate_shiwang_panel(T1=50, shock="D1", seed=3)[0]
        y4 = simulate_shiwang_panel(T1=50, shock="D4", seed=3)[0]
        np.testing.assert_allclose(y4[:50], y0[:50])
        np.testing.assert_allclose(y4[50:], y0[50:] + 0.3)

    def test_unknown_shock_raises(self):
        from mlsynth.utils.pda_helpers.simulation import simulate_shiwang_panel
        with pytest.raises(ValueError):
            simulate_shiwang_panel(shock="D99", seed=0)

    def test_factors_and_loadings(self):
        from mlsynth.utils.pda_helpers.simulation import (
            _shiwang_factors, _shiwang_loadings)
        f = _shiwang_factors(60, np.random.default_rng(0))
        assert f.shape == (60, 4)
        lam = _shiwang_loadings(50, np.random.default_rng(0))
        assert lam.shape == (50, 4)
        assert np.all((np.abs(lam) >= 0.3) & (np.abs(lam) <= 0.5))
