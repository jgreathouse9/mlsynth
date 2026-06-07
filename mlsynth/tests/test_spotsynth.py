"""Tests for the SPOTSYNTH estimator (O'Riordan & Gilligan-Lee 2025).

Layered per agents/agents_tests.md:

* Layer 1 (numerical core): the Algorithm 1 screen flags an instantaneous-
  spillover donor and a planted noisy-proxy donor.
* Layer 2 (data utilities): single-treated-unit ingestion + guards.
* Layer 3 (estimator integration): the screen restores the ATT on a planted
  contaminant; the Path-B bias ordering (All > S1, Valid ~ 0) holds.
* Layer 4 (public API contracts): import, frozen results, config validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mlsynth import SPOTSYNTH
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.spotsynth_helpers import (
    SpilloverScreen,
    SpotSynthInputs,
    SpotSynthResults,
    bayesian_simplex_sc,
    prepare_spotsynth_inputs,
    proximal_debias,
    simulate_spillover_panel,
    spillover_screen,
)

# Frequentist SC keeps the structural tests fast; dedicated Bayesian tests live
# in TestBayesAndDebias.
FREQ = {"inference": "frequentist"}


# ----------------------------------------------------------------------
# Layer 1: numerical core -- the screen
# ----------------------------------------------------------------------

class TestScreen:
    def test_flags_instantaneous_spillover(self):
        # 80% invalid donors with a -2 spillover; S1 should put invalid donors
        # among the largest forecast errors.
        df, valid = simulate_spillover_panel(
            n_donors=60, T0=60, n_post=15, sigma_x=0.2, seed=3)
        inp = prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")
        sc = spillover_screen(inp.D, inp.T0, inp.donor_names,
                              selection="S1", forecast="lag", n_donors=12)
        # mean forecast error of valid donors < that of invalid donors
        err = sc.forecast_error
        assert err[valid].mean() < err[~valid].mean()

    def test_flags_noisy_proxy(self):
        # A donor that is a noisy copy of the treated unit (carrying the effect)
        # should be flagged by the leave-one-out screen on a gradual-effect panel.
        rng = np.random.default_rng(0)
        T, T0, n = 40, 28, 8
        f = np.cumsum(rng.normal(0, 1, T))                 # common factor
        rows = []
        y = f + np.r_[np.zeros(T0), np.linspace(0, -6, T - T0)]   # gradual effect
        for t in range(T):
            rows.append({"unit": "tr", "time": t, "Y": float(y[t]),
                         "treated": int(t >= T0)})
            for j in range(n):
                rows.append({"unit": f"d{j}", "time": t,
                             "Y": float(f[t] + rng.normal(0, 0.3)), "treated": 0})
            # noisy proxy of the treated unit (carries the effect)
            rows.append({"unit": "proxy", "time": t,
                         "Y": float(y[t] + rng.normal(0, 0.3)), "treated": 0})
        df = pd.DataFrame(rows)
        inp = prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")
        sc = spillover_screen(inp.D, inp.T0, inp.donor_names,
                              selection="S1", forecast="loo", n_donors=n)
        assert "proxy" in sc.excluded_names

    def test_loo_is_onset_robust_lag_is_not(self):
        # Valid-majority, GRADUAL onset: loo keeps power, first-post-point lag
        # decays toward chance. (Reproduces the power-analysis headline.)
        from mlsynth.utils.spotsynth_helpers import run_forecast_power_analysis
        res = run_forecast_power_analysis(
            n_donors=40, T0=50, n_post=24, invalid_fracs=(0.3,), ramps=(24,),
            n_reps=8, n_factors=8, verbose=False)
        cell = res[(0.3, 24)]
        assert cell["loo"] > 0.8          # onset-robust
        assert cell["loo"] > cell["lag"] + 0.2

    def test_loo_inverts_under_invalid_majority(self):
        # 80% invalid (paper's regime): loo inverts (flags the valid donors),
        # so the package pins 'lag' there. This guards that documented limit.
        from mlsynth.utils.spotsynth_helpers import run_forecast_power_analysis
        res = run_forecast_power_analysis(
            n_donors=40, T0=50, n_post=24, invalid_fracs=(0.8,), ramps=(1,),
            n_reps=8, n_factors=8, verbose=False)
        cell = res[(0.8, 1)]
        assert cell["loo"] < 0.3          # inverted
        assert cell["lag"] > 0.6          # lag retains power for sharp onsets

    def test_all_selection_keeps_everyone(self):
        df, _ = simulate_spillover_panel(n_donors=30, T0=40, n_post=10, seed=1)
        inp = prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")
        sc = spillover_screen(inp.D, inp.T0, inp.donor_names, selection="all")
        assert sc.selected_idx.size == inp.n_donors
        assert sc.excluded_idx.size == 0


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_ingest_shapes(self):
        df, _ = simulate_spillover_panel(n_donors=20, T0=30, n_post=8, seed=0)
        inp = prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")
        assert inp.D.shape == (38, 20)
        assert inp.T0 == 30
        assert inp.treated_name == "target"

    def test_rejects_multiple_treated(self):
        df, _ = simulate_spillover_panel(n_donors=10, T0=20, n_post=6, seed=0)
        # mark a second unit as treated
        df.loc[(df["unit"] == "d0") & (df["time"] >= 20), "treated"] = 1
        with pytest.raises(MlsynthDataError):
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")

    def test_rejects_unbalanced(self):
        df, _ = simulate_spillover_panel(n_donors=10, T0=20, n_post=6, seed=0)
        df = df.iloc[1:]                                   # drop a cell
        with pytest.raises(MlsynthDataError):
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestIntegration:
    def test_screen_restores_att_on_contaminant(self):
        # Build a clean panel, plant a noisy proxy of the treated unit, and check
        # that S1 excludes it and recovers a more negative ATT than 'all'.
        rng = np.random.default_rng(2)
        T, T0, n = 36, 26, 12
        f = np.cumsum(rng.normal(0, 1, T))
        eff = np.r_[np.zeros(T0), np.full(T - T0, -8.0)]
        y = f + eff
        rows = []
        for t in range(T):
            rows.append({"unit": "tr", "time": t, "Y": float(y[t]),
                         "treated": int(t >= T0)})
            for j in range(n):
                rows.append({"unit": f"d{j}", "time": t,
                             "Y": float(f[t] + rng.normal(0, 0.5)), "treated": 0})
            rows.append({"unit": "proxy", "time": t,
                         "Y": float(y[t] + rng.normal(0, 0.5)), "treated": 0})
        df = pd.DataFrame(rows)
        cfg = dict(df=df, outcome="Y", treat="treated", unitid="unit",
                   time="time", display_graphs=False, **FREQ)
        screened = SPOTSYNTH({**cfg, "selection": "S1", "forecast": "loo",
                              "n_donors": n}).fit()
        assert "proxy" in screened.screen.excluded_names
        # the contaminated ('all') ATT is biased toward zero relative to screened
        assert screened.att < screened.att_unscreened
        assert screened.att == pytest.approx(-8.0, abs=2.0)

    def test_pathb_bias_ordering(self):
        # Averaged over a few draws: All is biased up, Valid ~ 0, S1 reduces bias.
        tau = 2.0
        allb, validb, s1b = [], [], []
        for rep in range(6):
            df, valid = simulate_spillover_panel(
                n_donors=60, T0=60, n_post=15, sigma_x=0.3, seed=rep)
            cfg = dict(df=df, outcome="Y", treat="treated", unitid="unit",
                       time="time", display_graphs=False, **FREQ)
            allb.append(SPOTSYNTH({**cfg, "selection": "all"}).fit().att)
            vnames = ["target"] + [n for n, ok in zip(
                sorted(df.loc[df.unit != "target", "unit"].unique()), valid) if ok]
            dfv = df[df.unit.isin(vnames)]
            validb.append(SPOTSYNTH({**cfg, "df": dfv, "selection": "all"}).fit().att)
            # 80%-invalid regime -> the paper's 'lag' anchor (loo inverts here)
            s1b.append(SPOTSYNTH({**cfg, "selection": "S1", "n_donors": 12,
                                  "forecast": "lag"}).fit().att)
        all_bias = np.mean(allb) - tau
        valid_bias = np.mean(validb) - tau
        s1_bias = np.mean(s1b) - tau
        assert all_bias > 1.0                    # using invalid donors biases up
        assert abs(valid_bias) < 0.3             # oracle ~ unbiased
        assert s1_bias < all_bias                # S1 reduces the bias


# ----------------------------------------------------------------------
# Layer 3b: Bayesian Dirichlet SC + proximal debiasing (paper functional forms)
# ----------------------------------------------------------------------

class TestBayesAndDebias:
    def test_bayesian_sc_recovers_att_with_ci(self):
        # Dirichlet simplex SC (NumPyro NUTS) recovers a planted effect; CrI
        # brackets it. Skipped if numpyro is not installed.
        pytest.importorskip("numpyro")
        rng = np.random.default_rng(0)
        T, T0, n = 40, 30, 6
        f = np.cumsum(rng.normal(0, 1, (T, 3)), axis=0)
        load = rng.dirichlet(np.ones(3), n)
        D = f @ load.T + rng.normal(0, 0.2, (T, n))
        beta = np.array([0.4, 0.3, 0.3, 0, 0, 0])
        y = D @ beta + rng.normal(0, 0.2, T)
        y[T0:] += -5.0
        fit = bayesian_simplex_sc(y, D, T0, alpha=0.4, n_samples=2000,
                                  n_warmup=1000, seed=0)
        assert abs(fit.weights.sum() - 1.0) < 1e-6   # on the simplex
        assert fit.att == pytest.approx(-5.0, abs=1.0)
        lo, hi = fit.att_ci
        assert lo < fit.att < hi
        assert fit.accept_prob > 0.5                  # NUTS adapted

    def test_bayesian_fit_through_estimator(self):
        pytest.importorskip("numpyro")
        df, _ = simulate_spillover_panel(n_donors=30, T0=40, n_post=10,
                                         sigma_x=0.3, seed=1)
        res = SPOTSYNTH({"df": df, "outcome": "Y", "treat": "treated",
                         "unitid": "unit", "time": "time", "selection": "S1",
                         "n_donors": 10, "inference": "bayes", "n_samples": 1000,
                         "n_warmup": 600, "display_graphs": False}).fit()
        assert res.inference_method == "bayes"
        assert res.att_ci is not None and res.att_ci[0] < res.att_ci[1]
        assert res.counterfactual_lower is not None
        assert res.counterfactual_lower.shape == res.counterfactual.shape
        assert np.all(res.counterfactual_lower <= res.counterfactual_upper + 1e-9)

    def test_replicate_basque_returns_full_diagnostics(self):
        # The real-data replications return ATTs, weights, pre-treatment fit,
        # and the selection -- on the third (Basque) panel, via loo + NUTS.
        import pathlib
        pytest.importorskip("numpyro")
        from mlsynth.utils.spotsynth_helpers import replicate_basque_spillover
        path = (pathlib.Path(__file__).resolve().parents[2]
                / "basedata" / "basque_data.csv")
        if not path.exists():
            pytest.skip("basque_data.csv not available")
        r = replicate_basque_spillover(str(path), verbose=False)
        for key in ("oracle_att", "contaminated_att", "screened_att", "att_ci",
                    "pre_rmse", "donor_weights", "selected_donors",
                    "excluded_donors", "synthetic_donor_excluded", "results"):
            assert key in r
        assert r["synthetic_donor_excluded"] is True       # loo flags the proxy
        assert isinstance(r["donor_weights"], dict) and r["donor_weights"]
        assert abs(sum(r["donor_weights"].values()) - 1.0) < 0.05
        assert r["pre_rmse"] >= 0.0

    def test_proximal_debias_reduces_eiv_bias(self):
        # Noisy-proxy donors -> attenuation bias; 2SLS with extra proxies helps.
        def trial(seed):
            rng = np.random.default_rng(seed)
            T, T0, k = 50, 40, 3
            sig = np.cumsum(rng.normal(0, 1, (T, k)), axis=0)
            beta = np.array([0.5, 0.3, 0.2])
            y = sig @ beta + rng.normal(0, 0.1, T)        # no treatment effect
            X = sig + rng.normal(0, 1.0, (T, k))
            Z = np.column_stack([sig + rng.normal(0, 1.0, (T, k)) for _ in range(2)])
            Xc = np.column_stack([np.ones(T0), X[:T0]])
            b = np.linalg.lstsq(Xc, y[:T0], rcond=None)[0]
            naive = np.mean((y - (b[0] + X @ b[1:]))[np.arange(T) >= T0])
            deb = proximal_debias(y, X, Z, T0).att
            return abs(naive), abs(deb)
        naive, deb = np.mean([trial(s) for s in range(30)], axis=0)
        assert deb < naive                      # debiasing reduces |bias|


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestAPI:
    def test_results_are_frozen(self):
        df, _ = simulate_spillover_panel(n_donors=20, T0=30, n_post=8, seed=0)
        res = SPOTSYNTH({"df": df, "outcome": "Y", "treat": "treated",
                         "unitid": "unit", "time": "time",
                         "display_graphs": False, **FREQ}).fit()
        assert isinstance(res, SpotSynthResults)
        assert isinstance(res.screen, SpilloverScreen)
        assert isinstance(res.inputs, SpotSynthInputs)
        # SpotSynthResults is now a frozen pydantic EffectResult; mutating a
        # field raises pydantic's ValidationError (not FrozenInstanceError).
        with pytest.raises(ValidationError):
            res.att_unscreened = 0.0

    def test_config_validation(self):
        df, _ = simulate_spillover_panel(n_donors=10, T0=20, n_post=6, seed=0)
        with pytest.raises(MlsynthConfigError):
            SPOTSYNTH({"df": df, "outcome": "Y", "treat": "treated",
                       "unitid": "unit", "time": "time",
                       "selection": "nonsense"})

    def test_metadata_fields(self):
        df, _ = simulate_spillover_panel(n_donors=20, T0=30, n_post=8, seed=0)
        res = SPOTSYNTH({"df": df, "outcome": "Y", "treat": "treated",
                         "unitid": "unit", "time": "time", "selection": "S1",
                         "n_donors": 8, "display_graphs": False, **FREQ}).fit()
        assert res.metadata["n_selected"] == 8
        assert res.metadata["estimator"] == "SPOTSYNTH"
        assert set(res.donor_weights) == set(res.screen.selected_names)
