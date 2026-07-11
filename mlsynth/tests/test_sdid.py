"""Tests for the modernized SDID estimator.

Follows the four-layer testing philosophy in ``agents/agents_tests.md``:

    Layer 1: numerical / helper tests
        - unit/time-weight QPs return simplex-feasible solutions
        - compute_regularization is positive on healthy panels
        - per-cohort cohort estimator returns the expected shapes / keys
        - placebo variance is non-negative
    Layer 2: data utility tests
        - prepare_sdid_inputs handles both dataprep return shapes
          (single-treated and cohorts), surfacing a uniform cohorts_dict
        - panels with no treated unit are rejected at dataprep
    Layer 3: estimator integration tests
        - SDID.fit() reproduces the canonical Prop 99 ATT to the last decimal
        - the cohort-specific event-time effects (Ciccia 2024 Eq. 3) match
          the corresponding pooled effects (Eq. 6) in the single-cohort case
        - high event-time effects are negative for Prop 99 (signed test)
        - B = 0 yields a NaN inference but a well-formed effect estimate
    Layer 4: public API contract tests
        - from mlsynth import SDID works
        - SDIDResults exposes inference / event_study / cohorts as typed
          frozen dataclasses

Reference papers:
  Arkhangelsky et al. (2021), AER.
  Ciccia (2024), arXiv:2407.09565.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import SDID
from mlsynth.config_models import SDIDConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.sdid_helpers.cohort import estimate_cohort_sdid_effects
from mlsynth.utils.sdid_helpers.inference import (
    _fixed_weight_cohort_att,
    estimate_bootstrap_variance,
    estimate_jackknife_variance,
    estimate_placebo_variance,
)
from mlsynth.utils.sdid_helpers.setup import prepare_sdid_inputs
from mlsynth.utils.sdid_helpers.structures import (
    SDIDCohort,
    SDIDEventEffect,
    SDIDEventStudy,
    SDIDInference,
    SDIDInputs,
    SDIDResults,
)
from mlsynth.utils.sdid_helpers.weights import (
    compute_regularization,
    fit_time_weights,
    unit_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMOKING_DATA_URL = (
    "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/"
    "basedata/smoking_data.csv"
)


@pytest.fixture(scope="module")
def smoking_panel() -> pd.DataFrame:
    """Load the canonical Prop 99 panel used as the SDID benchmark."""
    df = pd.read_csv(SMOKING_DATA_URL)
    df["Proposition 99"] = df["Proposition 99"].astype(int)
    return df


def _make_staggered_panel(seed: int = 0) -> pd.DataFrame:
    """Synthesize a 3-cohort staggered-adoption panel.

    Two treated states adopt at t=10, one at t=12, the rest (5 states)
    are never-treated controls. 20 time periods total.
    """
    rng = np.random.default_rng(seed)
    states = [f"s{i}" for i in range(8)]
    T = 20
    records = []
    for i, s in enumerate(states):
        base = rng.standard_normal() * 5 + 50
        for t in range(T):
            outcome = base + 0.5 * t + rng.standard_normal()
            if i in (0, 1):
                treated = 1 if t >= 10 else 0
                if t >= 10:
                    outcome -= 5.0
            elif i == 2:
                treated = 1 if t >= 12 else 0
                if t >= 12:
                    outcome -= 5.0
            else:
                treated = 0
            records.append({"state": s, "year": 2000 + t, "y": float(outcome),
                            "treated": treated})
    return pd.DataFrame(records)


_LOCAL_SMOKING = (
    Path(__file__).resolve().parents[2] / "basedata" / "smoking_data.csv"
)


@pytest.fixture(scope="module")
def block_multitreated_panel() -> pd.DataFrame:
    """A block panel with three treated states (California, Nevada, Utah).

    Built from the local Prop 99 outcomes with a *synthetic* block treatment
    (the three states treated from 1989) so the jackknife and bootstrap
    variances -- which require more than one treated unit -- are defined and can
    be cross-validated against the ``synthdid`` R package on the identical
    matrix (synthdid jackknife_se = 10.557038, point = -8.804942).
    """
    df = pd.read_csv(_LOCAL_SMOKING)
    treated = {"California", "Nevada", "Utah"}
    df["treat"] = ((df["state"].isin(treated)) & (df["year"] >= 1989)).astype(int)
    return df[["state", "year", "cigsale", "treat"]]


def _block_prepped(df: pd.DataFrame) -> dict:
    """``{'cohorts': ...}`` payload for a block panel, for the variance helpers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inputs = prepare_sdid_inputs(
            df=df, outcome="cigsale", treat="treat", unitid="state", time="year"
        )
    return {"cohorts": inputs.cohorts_dict}


def _base_config(df: pd.DataFrame, **overrides) -> dict:
    cfg = dict(
        df=df,
        outcome="cigsale",
        treat="Proposition 99",
        unitid="state",
        time="year",
        B=20,
        display_graphs=False,
    )
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Layer 1: numerical / helper tests
# ---------------------------------------------------------------------------

class TestWeightSolvers:
    """The unit/time-weight QPs return feasible simplex solutions."""

    def test_unit_weights_lie_on_simplex(self):
        rng = np.random.default_rng(0)
        T0, N = 10, 5
        Y_donors_pre = rng.standard_normal((T0, N))
        y_treated_pre = rng.standard_normal(T0)
        intercept, omega = unit_weights(Y_donors_pre, y_treated_pre, 0.1)
        assert omega is not None
        assert omega.shape == (N,)
        assert omega.sum() == pytest.approx(1.0, abs=1e-6)
        assert (omega >= -1e-8).all()

    def test_time_weights_lie_on_simplex(self):
        rng = np.random.default_rng(1)
        T0, N = 10, 5
        Y_donors_pre = rng.standard_normal((T0, N))
        y_donors_post_mean = rng.standard_normal(N)
        intercept, lam = fit_time_weights(Y_donors_pre, y_donors_post_mean)
        assert lam is not None
        assert lam.shape == (T0,)
        assert lam.sum() == pytest.approx(1.0, abs=1e-6)
        assert (lam >= -1e-8).all()

    def test_regularization_positive_on_healthy_panel(self):
        rng = np.random.default_rng(2)
        Y_donors_pre = rng.standard_normal((20, 5)) * 5
        zeta = compute_regularization(Y_donors_pre, num_post_treatment_periods=10)
        assert np.isfinite(zeta)
        assert zeta > 0

    def test_regularization_scales_with_treated_count(self):
        """zeta_omega = (N_tr * T_post)^(1/4) * sigma (Arkhangelsky et al. 2021).

        synthdid's ``eta.omega = ((N - N0)*(T - T0))^(1/4)`` folds the treated
        count into the unit-weight ridge. A single treated unit (the default)
        must leave zeta unchanged; more treated units must scale it by
        ``N_tr^(1/4)``.
        """
        rng = np.random.default_rng(3)
        Y_donors_pre = rng.standard_normal((20, 6)) * 4
        sigma = np.std(np.diff(Y_donors_pre, axis=0).flatten(), ddof=1)
        # Default (single treated) reproduces the (T_post)^(1/4) form.
        z1 = compute_regularization(Y_donors_pre, num_post_treatment_periods=8)
        assert z1 == pytest.approx((8 ** 0.25) * sigma, rel=1e-9)
        # Three treated units scale zeta by 3^(1/4).
        z3 = compute_regularization(Y_donors_pre, num_post_treatment_periods=8,
                                    num_treated_units=3)
        assert z3 == pytest.approx(((3 * 8) ** 0.25) * sigma, rel=1e-9)
        assert z3 == pytest.approx(z1 * (3 ** 0.25), rel=1e-9)
        # A non-positive treated count is rejected.
        with pytest.raises(MlsynthConfigError):
            compute_regularization(Y_donors_pre, num_post_treatment_periods=8,
                                   num_treated_units=0)


class TestCohortEstimator:
    """The single-cohort estimator returns the documented schema."""

    def test_returns_expected_keys(self):
        from collections import defaultdict

        rng = np.random.default_rng(3)
        T, n_treat, n_donor, T0 = 12, 2, 6, 8
        cohort = {
            "y": rng.standard_normal((T, n_treat)),
            "donor_matrix": rng.standard_normal((T, n_donor)),
            "total_periods": T,
            "pre_periods": T0,
            "post_periods": T - T0,
            "treated_indices": list(range(n_treat)),
        }
        accumulator = defaultdict(list)
        out = estimate_cohort_sdid_effects(T0 + 1, cohort, accumulator)
        for key in ("att", "effects", "pre_effects", "post_effects",
                    "actual", "counterfactual", "fitted_counterfactual",
                    "treatment_effects_series", "ell"):
            assert key in out
        assert out["actual"].shape == (T,)
        # The accumulator must have been populated for post events.
        assert any(len(v) > 0 for v in accumulator.values())


class TestPlaceboVariance:
    """Placebo variance is non-negative and respects the panel shape."""

    def test_smoke_on_small_panel(self):
        rng = np.random.default_rng(4)
        T, n_treat, n_donor, T0 = 10, 1, 6, 6
        cohorts = {
            T0 + 1: {
                "y": rng.standard_normal((T, n_treat)),
                "donor_matrix": rng.standard_normal((T, n_donor)),
                "total_periods": T,
                "pre_periods": T0,
                "post_periods": T - T0,
                "treated_indices": [0],
            }
        }
        out = estimate_placebo_variance(
            {"cohorts": cohorts}, num_placebo_iterations=10, seed=42
        )
        # Variance fields can be NaN with very few iterations but must
        # never be negative.
        for v in (out["att_variance"],
                  *out["cohort_variances"].values(),
                  *out["pooled_event_variances"].values()):
            assert np.isnan(v) or v >= 0

    def test_pseudo_treated_unit_is_dropped_from_its_own_donor_pool(self, monkeypatch):
        """A control reassigned as pseudo-treated must leave the donor pool.

        Arkhangelsky et al. (2021), Algorithm 4 (and the ``synthdid`` R
        package's ``vcov.R`` ``placebo_se``) assign a control unit as the
        pseudo-treated unit and re-fit SDID on the *remaining* controls. If the
        pseudo-treated column is left in the donor matrix, the synthetic control
        reconstructs it from itself, the placebo effect collapses toward zero,
        and the placebo variance is deflated -- the root of mlsynth's SDID SE
        (7.6) undershooting synthdid's (~9.2) on Prop 99.

        White-box invariant: for every cohort handed to the per-cohort
        estimator during the placebo loop, the pseudo-treated outcome column
        (its ``y``) must not appear among the columns of its ``donor_matrix``.
        """
        import mlsynth.utils.sdid_helpers.inference as inference_mod

        rng = np.random.default_rng(0)
        T, n_donor, T0 = 12, 8, 7
        cohorts = {
            T0 + 1: {
                "y": rng.standard_normal((T, 1)),
                "donor_matrix": rng.standard_normal((T, n_donor)),
                "total_periods": T,
                "pre_periods": T0,
                "post_periods": T - T0,
                "treated_indices": [0],
            }
        }

        real = inference_mod.estimate_cohort_sdid_effects
        seen: list[bool] = []

        def spy(period, cohort_data, accumulator):
            y_col = np.asarray(cohort_data["y"])[:, 0]
            donor = np.asarray(cohort_data["donor_matrix"])
            # No donor column may equal the pseudo-treated outcome column.
            clash = any(np.allclose(y_col, donor[:, j]) for j in range(donor.shape[1]))
            seen.append(clash)
            return real(period, cohort_data, accumulator)

        monkeypatch.setattr(inference_mod, "estimate_cohort_sdid_effects", spy)
        estimate_placebo_variance(
            {"cohorts": cohorts}, num_placebo_iterations=15, seed=7
        )
        assert seen, "placebo loop never called the per-cohort estimator"
        assert not any(seen), (
            "a pseudo-treated unit remained in its own donor pool during "
            "placebo inference (deflates the SDID variance)"
        )


class TestJackknifeAndBootstrapVariance:
    """The two other Arkhangelsky et al. (2021) variance estimators.

    Cross-validated against the ``synthdid`` R package's ``vcov.R``:
      * jackknife (Algorithm 3) is deterministic and matches value-for-value;
      * bootstrap (Algorithm 2) is stochastic and matches in magnitude;
      * both are NaN for a single treated unit, matching synthdid.
    """

    # synthdid reference on the identical CA+NV+UT block matrix (see
    # scratch/sdid_vce_ref.R): point = -8.804942, jackknife_se = 10.557038.
    _SYNTHDID_JACKKNIFE_SE = 10.557038

    def test_fixed_weight_att_reproduces_cohort_att(self, block_multitreated_panel):
        """The closed-form fixed-weight ATT equals the cohort estimator's ATT.

        The jackknife holds the fitted weights fixed and recomputes the ATT from
        the closed form; that closed form must reproduce the estimator's own ATT
        when handed the estimator's own weights, or the leave-one-out estimates
        are not on the same footing as the point estimate.
        """
        from collections import defaultdict

        prepped = _block_prepped(block_multitreated_panel)
        (period, cohort), = prepped["cohorts"].items()
        fitted = estimate_cohort_sdid_effects(period, cohort, defaultdict(list))
        fw = _fixed_weight_cohort_att(
            cohort["y"], cohort["donor_matrix"],
            np.asarray(fitted["unit_weights"]), np.asarray(fitted["time_weights"]),
            cohort["pre_periods"],
        )
        assert fw == pytest.approx(fitted["att"], abs=1e-9)

    def test_jackknife_matches_synthdid_on_block_panel(self, block_multitreated_panel):
        """Value-for-value jackknife SE against the authors' synthdid R."""
        prepped = _block_prepped(block_multitreated_panel)
        out = estimate_jackknife_variance(prepped)
        se = float(np.sqrt(out["att_variance"]))
        # Matches synthdid to ~1e-3; the residual is the Frank-Wolfe vs
        # active-set QP solver, not a method difference.
        assert se == pytest.approx(self._SYNTHDID_JACKKNIFE_SE, abs=0.05)

    def test_jackknife_is_deterministic(self, block_multitreated_panel):
        """The jackknife has no RNG: repeated calls give the identical SE."""
        prepped = _block_prepped(block_multitreated_panel)
        a = estimate_jackknife_variance(prepped)["att_variance"]
        b = estimate_jackknife_variance(prepped)["att_variance"]
        assert a == b

    def test_bootstrap_is_positive_and_seed_reproducible(self, block_multitreated_panel):
        """Bootstrap SE is finite/positive and reproducible under a fixed seed."""
        prepped = _block_prepped(block_multitreated_panel)
        a = estimate_bootstrap_variance(prepped, 200, seed=2020)["att_variance"]
        b = estimate_bootstrap_variance(prepped, 200, seed=2020)["att_variance"]
        assert np.isfinite(a) and a > 0
        assert a == b  # same seed -> same resamples -> same variance

    def test_jackknife_single_treated_is_nan(self, smoking_panel):
        """A single treated unit leaves the jackknife undefined (synthdid: NA)."""
        prepped = {"cohorts": prepare_sdid_inputs(
            df=smoking_panel, outcome="cigsale", treat="Proposition 99",
            unitid="state", time="year").cohorts_dict}
        out = estimate_jackknife_variance(prepped)
        assert np.isnan(out["att_variance"])

    def test_bootstrap_single_treated_is_nan(self, smoking_panel):
        """Bootstrap is undefined for a single treated unit (synthdid: NA)."""
        prepped = {"cohorts": prepare_sdid_inputs(
            df=smoking_panel, outcome="cigsale", treat="Proposition 99",
            unitid="state", time="year").cohorts_dict}
        out = estimate_bootstrap_variance(prepped, 200, seed=1)
        assert np.isnan(out["att_variance"])

    def test_jackknife_rejects_staggered_design(self):
        """Jackknife is block-design only; staggered adoption must raise."""
        df = _make_staggered_panel()
        prepped = {"cohorts": prepare_sdid_inputs(
            df=df, outcome="y", treat="treated", unitid="state",
            time="year").cohorts_dict}
        assert len(prepped["cohorts"]) >= 2
        with pytest.raises(MlsynthEstimationError):
            estimate_jackknife_variance(prepped)

    def test_bootstrap_rejects_staggered_design(self):
        """Bootstrap is block-design only; staggered adoption must raise."""
        df = _make_staggered_panel()
        prepped = {"cohorts": prepare_sdid_inputs(
            df=df, outcome="y", treat="treated", unitid="state",
            time="year").cohorts_dict}
        with pytest.raises(MlsynthEstimationError):
            estimate_bootstrap_variance(prepped, 50, seed=1)

    def test_sum_normalize_zero_vector_is_uniform(self):
        """A zero weight vector renormalizes to uniform (synthdid fallback)."""
        from mlsynth.utils.sdid_helpers.inference import _sum_normalize

        out = _sum_normalize(np.zeros(4))
        assert out == pytest.approx(np.full(4, 0.25))

    def test_variance_helpers_reject_malformed_payload(self):
        """The block-design guard rejects a payload without a 'cohorts' dict."""
        with pytest.raises(MlsynthDataError):
            estimate_jackknife_variance({"not_cohorts": {}})
        with pytest.raises(MlsynthDataError):
            estimate_jackknife_variance({"cohorts": "nope"})

    def test_bootstrap_rejects_invalid_arguments(self, block_multitreated_panel):
        prepped = _block_prepped(block_multitreated_panel)
        with pytest.raises(MlsynthConfigError):
            estimate_bootstrap_variance(prepped, -1, seed=1)
        with pytest.raises(MlsynthConfigError):
            estimate_bootstrap_variance(prepped, 10, seed="not-an-int")

    def test_dispatch_rejects_unknown_vce(self, block_multitreated_panel):
        """The internal dispatcher rejects an unknown method name."""
        from mlsynth.utils.sdid_helpers.event_study import _dispatch_variance

        prepped = _block_prepped(block_multitreated_panel)
        with pytest.raises(MlsynthConfigError):
            _dispatch_variance(prepped, "mystery", 10, 1)


class TestVceIntegration:
    """``vce`` threads from the config through ``SDID.fit()`` to the results."""

    def test_default_vce_is_placebo(self, block_multitreated_panel):
        res = SDID(_base_config(block_multitreated_panel, treat="treat")).fit()
        assert res.inference_detail.method == "placebo"

    def test_point_estimate_invariant_to_vce(self, block_multitreated_panel):
        """Choosing a variance estimator must not move the point estimate."""
        atts = {}
        for vce in ("placebo", "jackknife", "bootstrap", "noinference"):
            res = SDID(_base_config(block_multitreated_panel, treat="treat",
                                    vce=vce)).fit()
            atts[vce] = res.att
        assert atts["jackknife"] == pytest.approx(atts["placebo"], abs=1e-9)
        assert atts["bootstrap"] == pytest.approx(atts["placebo"], abs=1e-9)
        assert atts["noinference"] == pytest.approx(atts["placebo"], abs=1e-9)

    def test_jackknife_method_and_se_recorded(self, block_multitreated_panel):
        res = SDID(_base_config(block_multitreated_panel, treat="treat",
                                vce="jackknife")).fit()
        assert res.inference_detail.method == "jackknife"
        assert np.isfinite(res.inference_detail.se)
        assert len(res.inference_detail.placebo_att) == 0  # no placebo distribution

    def test_noinference_yields_nan_se(self, block_multitreated_panel):
        res = SDID(_base_config(block_multitreated_panel, treat="treat",
                                vce="noinference")).fit()
        assert np.isnan(res.inference_detail.se)
        assert res.inference_detail.method == "noinference"

    def test_jackknife_nan_se_on_single_treated(self, smoking_panel):
        """SDID.fit with jackknife on Prop 99 (one treated) -> NaN SE, no crash."""
        res = SDID(_base_config(smoking_panel, vce="jackknife")).fit()
        assert np.isnan(res.inference_detail.se)

    def test_invalid_vce_rejected_by_config(self, smoking_panel):
        with pytest.raises(MlsynthConfigError):
            SDID(_base_config(smoking_panel, vce="not-a-method"))


# ---------------------------------------------------------------------------
# Layer 2: data-utility tests
# ---------------------------------------------------------------------------

class TestSetupUnification:
    """``prepare_sdid_inputs`` packages both dataprep shapes uniformly."""

    def test_single_treated_unit_shape(self, smoking_panel):
        inputs = prepare_sdid_inputs(
            df=smoking_panel, outcome="cigsale", treat="Proposition 99",
            unitid="state", time="year",
        )
        assert isinstance(inputs, SDIDInputs)
        assert len(inputs.cohorts_dict) == 1
        (adoption, cohort), = inputs.cohorts_dict.items()
        assert "treated_indices" in cohort
        assert "donor_matrix" in cohort
        assert inputs.n_pre == cohort["pre_periods"]
        assert inputs.n_post == cohort["post_periods"]

    def test_staggered_panel_yields_multiple_cohorts(self):
        df = _make_staggered_panel()
        inputs = prepare_sdid_inputs(
            df=df, outcome="y", treat="treated", unitid="state", time="year",
        )
        assert len(inputs.cohorts_dict) >= 2  # two distinct adoption periods
        for cohort in inputs.cohorts_dict.values():
            assert "treated_indices" in cohort
            assert cohort["donor_matrix"].shape[0] == cohort["total_periods"]

    def test_no_treated_unit_rejected(self, smoking_panel):
        df = smoking_panel.copy()
        df["Proposition 99"] = 0
        # ``dataprep`` itself catches this and raises MlsynthDataError.
        with pytest.raises(MlsynthDataError):
            prepare_sdid_inputs(
                df=df, outcome="cigsale", treat="Proposition 99",
                unitid="state", time="year",
            )


# ---------------------------------------------------------------------------
# Layer 3: estimator integration tests
# ---------------------------------------------------------------------------

class TestProp99Replication:
    """Prop 99 ATT must match the canonical Arkhangelsky et al. (2021) result."""

    def test_overall_att(self, smoking_panel):
        res = SDID(_base_config(smoking_panel, B=20, seed=1400)).fit()
        # Canonical value is -15.6, the headline number in Arkhangelsky
        # et al. (2021) Table 1. The point estimate is deterministic given
        # the regularization parameter; only inference fields move with
        # the placebo seed.
        assert res.inference_detail.att == pytest.approx(-15.6054, abs=5e-3)

    def test_multitreated_block_att_matches_synthdid(self, block_multitreated_panel):
        """A multi-treated block ATT must match synthdid's point estimate.

        With three treated units the unit-weight ridge picks up the
        ``N_tr^(1/4)`` factor (Arkhangelsky et al. 2021); omitting it
        under-regularized omega and pulled the ATT off the authors' R. synthdid
        on the identical CA+NV+UT block matrix gives -8.804942; the residual is
        the Frank-Wolfe vs active-set QP solver, the same ~1e-3 seen on Prop 99.
        """
        res = SDID(_base_config(block_multitreated_panel, treat="treat",
                                B=20, seed=1400)).fit()
        assert res.inference_detail.att == pytest.approx(-8.804942, abs=0.02)

    def test_inference_is_well_formed(self, smoking_panel):
        res = SDID(_base_config(smoking_panel, B=50, seed=1400)).fit()
        assert res.inference_detail.n_placebo == 50
        assert np.isfinite(res.inference_detail.se) and res.inference_detail.se > 0
        lo, hi = res.inference_detail.ci
        assert lo <= res.inference_detail.att <= hi
        assert 0.0 < res.inference_detail.p_value <= 1.0

    def test_event_study_post_period_negative(self, smoking_panel):
        # Prop 99 reduced cigarette sales; the LAST event-time effect should
        # be substantially negative.
        res = SDID(_base_config(smoking_panel, B=20, seed=1400)).fit()
        ells = res.event_study.event_times
        taus = res.event_study.tau
        late_post = taus[ells >= 5]
        assert late_post.size > 0
        assert (late_post < 0).all()

    def test_cohort_att_matches_overall_in_single_cohort_case(self, smoking_panel):
        # With one cohort, the cohort ATT and the overall ATT are identical
        # by construction.
        res = SDID(_base_config(smoking_panel, B=20, seed=1400)).fit()
        assert len(res.cohorts) == 1
        cohort = next(iter(res.cohorts.values()))
        assert cohort.att == pytest.approx(res.inference_detail.att, abs=1e-10)

    def test_zero_placebo_iterations_yield_nan_inference(self, smoking_panel):
        res = SDID(_base_config(smoking_panel, B=0)).fit()
        # Point estimate is still computable.
        assert np.isfinite(res.inference_detail.att)
        # But the variance / CI fields are NaN.
        assert np.isnan(res.inference_detail.se)
        assert np.isnan(res.inference_detail.p_value)


class TestStaggeredAdoption:
    """Staggered-adoption panels exercise the multi-cohort path."""

    def test_multiple_cohorts_in_results(self):
        df = _make_staggered_panel()
        res = SDID({"df": df, "outcome": "y", "treat": "treated",
                    "unitid": "state", "time": "year",
                    "B": 0, "display_graphs": False}).fit()
        assert len(res.cohorts) >= 2

    def test_each_cohort_carries_its_event_effects(self):
        df = _make_staggered_panel()
        res = SDID({"df": df, "outcome": "y", "treat": "treated",
                    "unitid": "state", "time": "year",
                    "B": 0, "display_graphs": False}).fit()
        for cohort in res.cohorts.values():
            # Each cohort exposes at least the ell=0 effect.
            assert 0 in cohort.event_effects
            assert isinstance(cohort.event_effects[0], SDIDEventEffect)


# ---------------------------------------------------------------------------
# Layer 4: public API contract tests
# ---------------------------------------------------------------------------

class TestPublicAPI:
    """SDID exposes the same typed-results contract as the newer estimators."""

    def test_top_level_import(self):
        from mlsynth import SDID as Imported  # noqa: F401
        assert Imported is SDID

    def test_results_object_field_types(self, smoking_panel):
        res = SDID(_base_config(smoking_panel, B=10)).fit()
        assert isinstance(res, SDIDResults)
        assert isinstance(res.inputs, SDIDInputs)
        assert isinstance(res.inference_detail, SDIDInference)
        assert isinstance(res.event_study, SDIDEventStudy)
        for cohort in res.cohorts.values():
            assert isinstance(cohort, SDIDCohort)

    def test_event_study_shapes_align(self, smoking_panel):
        res = SDID(_base_config(smoking_panel, B=10)).fit()
        es = res.event_study
        assert es.event_times.shape == es.tau.shape == es.se.shape
        assert es.ci.shape == (es.event_times.size, 2)

    def test_dict_and_config_inputs_match(self, smoking_panel):
        cfg_dict = _base_config(smoking_panel, B=10, seed=42)
        cfg_obj = SDIDConfig(**cfg_dict)
        r1 = SDID(cfg_dict).fit()
        r2 = SDID(cfg_obj).fit()
        assert r1.inference_detail.att == pytest.approx(r2.inference_detail.att)


# ---------------------------------------------------------------------------
# Exposed unit weights + intercept-adjusted counterfactual toggle
# ---------------------------------------------------------------------------

class TestWeightsAndInterceptToggle:
    """SDID exposes its unit weights, and the counterfactual can optionally be
    intercept-adjusted (off by default)."""

    def test_donor_weights_exposed_on_simplex(self, smoking_panel):
        res = SDID(_base_config(smoking_panel, B=10)).fit()
        dw = res.weights.donor_weights
        assert isinstance(dw, dict) and len(dw) > 0
        vals = np.array(list(dw.values()), dtype=float)
        assert (vals >= -1e-8).all()
        assert vals.sum() == pytest.approx(1.0, abs=1e-6)

    def test_cohort_carries_unit_weights(self, smoking_panel):
        res = SDID(_base_config(smoking_panel, B=10)).fit()
        for cohort in res.cohorts.values():
            assert cohort.unit_weights is not None
            assert np.asarray(cohort.unit_weights).ndim == 1

    def _pre_gap(self, res):
        yr = np.asarray(res.time_series.time_periods).ravel()
        obs = np.asarray(res.time_series.observed_outcome, dtype=float)
        cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
        pre = yr < res.time_series.intervention_time
        return abs(float((obs[pre] - cf[pre]).mean()))

    def test_default_counterfactual_is_not_intercept_adjusted(self, smoking_panel):
        """By default the counterfactual is the raw weighted-donor series, which
        sits at a different level than the treated unit in the pre-period."""
        res = SDID(_base_config(smoking_panel, B=10)).fit()
        assert self._pre_gap(res) > 10.0

    def test_intercept_adjust_tracks_treated(self, smoking_panel):
        """With the toggle on, the counterfactual is level-matched to the treated
        unit over the pre-period."""
        res = SDID(_base_config(smoking_panel, B=10, intercept_adjust=True)).fit()
        assert self._pre_gap(res) < 5.0

    def test_toggle_leaves_att_unchanged(self, smoking_panel):
        raw = SDID(_base_config(smoking_panel, B=10, seed=1400)).fit()
        adj = SDID(_base_config(smoking_panel, B=10, seed=1400,
                                intercept_adjust=True)).fit()
        assert raw.effects.att == pytest.approx(adj.effects.att, abs=1e-9)

    def test_intercept_shift_is_constant(self, smoking_panel):
        """The two counterfactuals differ by a single constant (the intercept)."""
        raw = SDID(_base_config(smoking_panel, B=10)).fit()
        adj = SDID(_base_config(smoking_panel, B=10, intercept_adjust=True)).fit()
        diff = (np.asarray(adj.time_series.counterfactual_outcome, dtype=float)
                - np.asarray(raw.time_series.counterfactual_outcome, dtype=float))
        assert np.ptp(diff) == pytest.approx(0.0, abs=1e-6)
        assert abs(float(diff.mean())) > 1e-6
