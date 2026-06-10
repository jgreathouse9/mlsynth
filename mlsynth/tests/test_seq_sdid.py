"""Tests for the Sequential Synthetic Difference-in-Differences estimator.

Follows ``agents/agents_tests.md``'s four-layer testing philosophy.

Reference: Arkhangelsky & Samkov (2025), arXiv:2404.00164v2.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import SequentialSDID
from mlsynth.config_models import SequentialSDIDConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.seq_sdid_helpers.algorithm import (
    _warn_donor_starved_cohorts,
    pooled_event_study,
    run_sequential_sdid,
)
from mlsynth.utils.seq_sdid_helpers.setup import prepare_seq_sdid_inputs
from mlsynth.utils.seq_sdid_helpers.structures import (
    SeqSDIDCohortEffect,
    SeqSDIDEventStudy,
    SeqSDIDInference,
    SeqSDIDInputs,
    SeqSDIDResults,
)
from mlsynth.utils.seq_sdid_helpers.weights import solve_time_qp, solve_unit_qp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _staggered_panel(
    *, seed: int = 0,
    cohort_adoptions=(10, 15, 20, None),  # None = never-treated
    n_per_cohort: int = 30,
    T: int = 30,
    true_effect: float = -5.0,
) -> pd.DataFrame:
    """Synthesize a staggered-adoption panel from the paper's IFE DGP."""
    rng = np.random.default_rng(seed)
    records = []
    unit_id = 0
    for adoption in cohort_adoptions:
        for _ in range(n_per_cohort):
            alpha_i = rng.standard_normal() * 5 + 50
            theta_i = rng.standard_normal() * 2
            for t in range(T):
                psi_t = np.cos(t / 5.0)
                beta_t = 0.5 * t
                mu = alpha_i + beta_t + theta_i * psi_t
                eps = rng.standard_normal() * 0.3
                outcome = mu + eps
                treated = 0
                if adoption is not None and t >= adoption:
                    treated = 1
                    outcome += true_effect
                records.append({
                    "unit": f"u_{unit_id}",
                    "year": 2000 + t,
                    "y": outcome,
                    "treated": treated,
                })
            unit_id += 1
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def staggered_panel() -> pd.DataFrame:
    return _staggered_panel()


# ---------------------------------------------------------------------------
# Layer 1: weight-solver numerical tests
# ---------------------------------------------------------------------------

class TestUnitWeightQP:
    """The unit-weight QP must satisfy its sum constraint and KKT conditions."""

    def test_sum_to_one_with_eta_zero(self):
        rng = np.random.default_rng(0)
        T_pre, J = 10, 4
        Y_pre = rng.standard_normal((T_pre, J))
        y_treated = rng.standard_normal(T_pre)
        pi = np.full(J, 1.0 / J)
        omega, omega_0 = solve_unit_qp(Y_pre, y_treated, pi, eta=0.0)
        assert omega.shape == (J,)
        assert omega.sum() == pytest.approx(1.0, abs=1e-9)

    def test_eta_infinity_limit_recovers_pi_proportional(self):
        # As eta -> infinity, omega_j must converge to pi_j / sum(pi_j).
        rng = np.random.default_rng(1)
        T_pre, J = 6, 3
        Y_pre = rng.standard_normal((T_pre, J))
        y_treated = rng.standard_normal(T_pre)
        pi = np.array([0.5, 0.3, 0.2])
        omega, _ = solve_unit_qp(Y_pre, y_treated, pi, eta=1e8)
        expected = pi / pi.sum()
        assert np.allclose(omega, expected, atol=1e-3)


class TestTimeWeightQP:
    """The time-weight QP must satisfy its sum constraint."""

    def test_sum_to_one(self):
        rng = np.random.default_rng(2)
        T_pre, J = 8, 5
        Y_pre = rng.standard_normal((T_pre, J))
        y_event = rng.standard_normal(J)
        lam, lam_0 = solve_time_qp(Y_pre, y_event, eta=1.0)
        assert lam.shape == (T_pre,)
        assert lam.sum() == pytest.approx(1.0, abs=1e-9)

    def test_eta_infinity_limit_recovers_uniform_lambda(self):
        # As eta -> infinity, lambda_l must converge to 1 / T_pre.
        rng = np.random.default_rng(3)
        T_pre, J = 7, 4
        Y_pre = rng.standard_normal((T_pre, J))
        y_event = rng.standard_normal(J)
        lam, _ = solve_time_qp(Y_pre, y_event, eta=1e8)
        assert np.allclose(lam, 1.0 / T_pre, atol=1e-4)


# ---------------------------------------------------------------------------
# Layer 2: setup / aggregation tests
# ---------------------------------------------------------------------------

class TestSetup:
    """``prepare_seq_sdid_inputs`` must correctly aggregate cohorts."""

    def test_cohort_count_and_shares(self, staggered_panel):
        inputs = prepare_seq_sdid_inputs(
            df=staggered_panel, outcome="y", treat="treated",
            unitid="unit", time="year",
        )
        # 4 cohorts (3 treated + 1 never-treated), all of equal size.
        assert inputs.Y_agg.shape[1] == 4
        assert inputs.pi.sum() == pytest.approx(1.0)
        assert np.allclose(inputs.pi, 0.25, atol=1e-12)

    def test_default_K_fits_in_panel(self, staggered_panel):
        inputs = prepare_seq_sdid_inputs(
            df=staggered_panel, outcome="y", treat="treated",
            unitid="unit", time="year",
        )
        # Default K = T - a_max so a_max + K <= T.
        T = staggered_panel["year"].nunique()
        assert inputs.a_max + inputs.K <= T

    def test_cohort_periods_are_one_based_indices(self, staggered_panel):
        inputs = prepare_seq_sdid_inputs(
            df=staggered_panel, outcome="y", treat="treated",
            unitid="unit", time="year",
        )
        # The first treated cohort adopts in year 2010, which is 1-based
        # position 11 in [2000..2029].
        assert int(inputs.cohort_periods[0]) == 11

    def test_no_treated_unit_rejected(self, staggered_panel):
        df = staggered_panel.copy()
        df["treated"] = 0
        with pytest.raises(MlsynthDataError):
            prepare_seq_sdid_inputs(
                df=df, outcome="y", treat="treated",
                unitid="unit", time="year",
            )


# ---------------------------------------------------------------------------
# Layer 3: integration tests
# ---------------------------------------------------------------------------

class TestStaggeredIntegration:
    """End-to-end fit recovers the true effect on a clean DGP."""

    def test_short_horizon_effects_near_truth(self, staggered_panel):
        res = SequentialSDID({
            "df": staggered_panel, "outcome": "y", "treat": "treated",
            "unitid": "unit", "time": "year",
            "n_bootstrap": 0, "eta": 1.0, "display_graphs": False,
        }).fit()
        # First 5 horizons should be near the true effect of -5.
        for tau in res.event_study.tau[:5]:
            assert tau == pytest.approx(-5.0, abs=0.3)

    def test_bootstrap_inference_well_formed(self, staggered_panel):
        res = SequentialSDID({
            "df": staggered_panel, "outcome": "y", "treat": "treated",
            "unitid": "unit", "time": "year",
            "n_bootstrap": 50, "eta": 1.0, "seed": 7,
            "display_graphs": False,
        }).fit()
        assert res.event_study.bootstrap_draws.shape[0] == 50
        for k in range(len(res.event_study.tau)):
            lo, hi = res.event_study.ci[k]
            tau = res.event_study.tau[k]
            assert lo <= tau <= hi
            assert res.event_study.se[k] >= 0

    def test_cohort_effects_indexed_by_period(self, staggered_panel):
        res = SequentialSDID({
            "df": staggered_panel, "outcome": "y", "treat": "treated",
            "unitid": "unit", "time": "year",
            "n_bootstrap": 0, "display_graphs": False,
        }).fit()
        # Three treated cohorts (positions 11, 16, 21), K horizons each.
        unique_a = sorted({a for (a, _) in res.cohort_effects.keys()})
        assert unique_a == [11, 16, 21]

    def test_sdid_imputation_mode_runs(self, staggered_panel):
        res = SequentialSDID({
            "df": staggered_panel, "outcome": "y", "treat": "treated",
            "unitid": "unit", "time": "year",
            "n_bootstrap": 0, "mode": "sdid_imputation",
            "display_graphs": False,
        }).fit()
        assert res.mode == "sdid_imputation"
        # Even at the eta -> inf limit the recovered effect should be in the
        # ballpark of -5 for short horizons on a clean DGP.
        assert res.event_study.tau[:3].mean() == pytest.approx(-5.0, abs=1.0)

    def test_k_zero_bootstrap_yields_nan_inference(self, staggered_panel):
        res = SequentialSDID({
            "df": staggered_panel, "outcome": "y", "treat": "treated",
            "unitid": "unit", "time": "year",
            "n_bootstrap": 0, "display_graphs": False,
        }).fit()
        assert np.all(np.isnan(res.event_study.se))
        assert np.all(np.isnan(res.event_study.ci))


class TestImputationCascade:
    """The sequential imputation step replaces treated cells in place."""

    def test_inplace_imputation_modifies_treated_cell(self, staggered_panel):
        inputs = prepare_seq_sdid_inputs(
            df=staggered_panel, outcome="y", treat="treated",
            unitid="unit", time="year",
        )
        Y_after, cohort_effects = run_sequential_sdid(
            Y_agg=inputs.Y_agg, pi=inputs.pi,
            cohort_periods=inputs.cohort_periods,
            treated_cohort_indices=inputs.treated_cohort_indices,
            a_min=inputs.a_min, a_max=inputs.a_max, K=inputs.K,
            eta=1.0,
        )
        # The treated cell of the earliest cohort at k=0 should have been
        # replaced with (observed - tau_hat_{a, 0}).
        a = inputs.a_min  # 11
        a_col = int(np.where(inputs.cohort_periods == a)[0][0])
        observed = inputs.Y_agg[a - 1, a_col]
        imputed = Y_after[a - 1, a_col]
        tau = cohort_effects[(a, 0)].tau
        assert imputed == pytest.approx(observed - tau, abs=1e-9)


# ---------------------------------------------------------------------------
# Layer 4: public API tests
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_import(self):
        from mlsynth import SequentialSDID as Imported  # noqa: F401
        assert Imported is SequentialSDID

    def test_results_object_types(self, staggered_panel):
        res = SequentialSDID({
            "df": staggered_panel, "outcome": "y", "treat": "treated",
            "unitid": "unit", "time": "year",
            "n_bootstrap": 10, "display_graphs": False,
        }).fit()
        assert isinstance(res, SeqSDIDResults)
        assert isinstance(res.inputs, SeqSDIDInputs)
        assert isinstance(res.event_study, SeqSDIDEventStudy)
        assert isinstance(res.inference_detail, SeqSDIDInference)
        for effect in res.cohort_effects.values():
            assert isinstance(effect, SeqSDIDCohortEffect)

    def test_event_study_shapes(self, staggered_panel):
        res = SequentialSDID({
            "df": staggered_panel, "outcome": "y", "treat": "treated",
            "unitid": "unit", "time": "year",
            "n_bootstrap": 10, "display_graphs": False,
        }).fit()
        K_plus_1 = res.inputs.K + 1
        assert res.event_study.tau.shape == (K_plus_1,)
        assert res.event_study.se.shape == (K_plus_1,)
        assert res.event_study.ci.shape == (K_plus_1, 2)
        assert res.event_study.horizons.shape == (K_plus_1,)

    def test_dict_vs_config_object(self, staggered_panel):
        cfg_dict = {
            "df": staggered_panel, "outcome": "y", "treat": "treated",
            "unitid": "unit", "time": "year",
            "n_bootstrap": 10, "seed": 99, "display_graphs": False,
        }
        cfg_obj = SequentialSDIDConfig(**cfg_dict)
        r1 = SequentialSDID(cfg_dict).fit()
        r2 = SequentialSDID(cfg_obj).fit()
        assert np.allclose(r1.event_study.tau, r2.event_study.tau)


# ---------------------------------------------------------------------------
# Donor-balance diagnostic + the exact-recovery property it protects
# ---------------------------------------------------------------------------

def _rank1_ife_panel(
    *, adoptions, n_never=30, n_per_cohort=6, T=40, seed=3, true_effect=0.0
) -> pd.DataFrame:
    """Noiseless rank-1 interactive-FE panel: y = a_i + b_t + lambda_i f_t.

    With no idiosyncratic noise, Sequential SDiD must recover the planted
    effect *exactly* for every cohort whose donor pool can span the
    one-dimensional loading -- the property the donor-balance diagnostic
    guards. ``f_t`` is a bounded (stationary) factor so the recovery does not
    rely on a trend.
    """
    rng = np.random.default_rng(seed)
    f = np.sin(np.linspace(0, 8, T)) + 0.4 * np.cos(np.linspace(0, 15, T))
    lam = np.linspace(-2, 2, len(adoptions) * n_per_cohort + n_never)
    rng.shuffle(lam)
    records, uid, li = [], 0, 0
    cohorts = [(a, n_per_cohort) for a in adoptions] + [(None, n_never)]
    for adoption, n in cohorts:
        for _ in range(n):
            L = lam[li]; li += 1
            for t in range(T):
                y = 3.0 + 0.05 * t + L * f[t]
                treated = int(adoption is not None and t >= adoption)
                if treated:
                    y += true_effect
                records.append({"unit": uid, "year": t, "y": y, "treat": treated})
            uid += 1
    return pd.DataFrame(records)


class TestDonorBalanceDiagnostic:
    """A donor-starved late cohort biases the pooled estimate; we must warn."""

    def test_exact_recovery_when_all_cohorts_balanced(self):
        # Many cohorts; cap a_max so every estimated cohort keeps >= 2 donors.
        df = _rank1_ife_panel(adoptions=list(range(20, 32)), true_effect=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any donor warning would fail here
            res = SequentialSDID({
                "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
                "time": "year", "mode": "ssdid", "eta": 1e-3, "K": 4,
                "a_max": 26, "n_bootstrap": 0, "display_graphs": False,
            }).fit()
        # Noiseless rank-1 IFE, balanced donors -> machine-exact zero effect.
        assert np.allclose(res.event_study.tau, 0.0, atol=1e-6)

    def test_warns_and_names_starved_cohort_and_fix(self):
        df = _rank1_ife_panel(adoptions=list(range(20, 32)))
        with pytest.warns(UserWarning, match="fewer than 2 donor cohorts"):
            res = SequentialSDID({
                "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
                "time": "year", "mode": "ssdid", "eta": 1e-3, "K": 4,
                "n_bootstrap": 0, "display_graphs": False,
            }).fit()
        # The starved tail biases the pooled estimate away from zero.
        assert not np.allclose(res.event_study.tau, 0.0, atol=1e-6)

    def test_no_warning_when_capped(self):
        df = _rank1_ife_panel(adoptions=list(range(20, 32)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SequentialSDID({
                "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
                "time": "year", "mode": "ssdid", "eta": 1e-3, "K": 4,
                "a_max": 26, "n_bootstrap": 0, "display_graphs": False,
            }).fit()
        assert not [w for w in caught
                    if "donor cohorts" in str(w.message)]


class TestWarnDonorStarvedHelper:
    """Unit-level coverage of the warning helper's branches."""

    def test_empty_is_noop(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_donor_starved_cohorts({})

    def test_no_starved_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_donor_starved_cohorts({10: 3, 12: 2})

    def test_suggests_a_max_when_some_balanced(self):
        with pytest.warns(UserWarning, match="Lower a_max to 12"):
            _warn_donor_starved_cohorts({10: 3, 12: 2, 18: 1})

    def test_suggests_adding_donors_when_all_starved(self):
        with pytest.warns(UserWarning, match="Add later-adopting"):
            _warn_donor_starved_cohorts({18: 1, 20: 1})
