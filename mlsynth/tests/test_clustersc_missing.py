"""CLUSTERSC's PCR path on panels with missing donor cells (#1146 step 2).

The algorithm is the forecast procedure of Agarwal, Amjad, Shah and Shen,
"Model Agnostic Time Series Analysis via Matrix Estimation", Proc. ACM Meas.
Anal. Comput. Syst. 2(3), Article 40 (2018), Section 3.3, mapped to synthetic
control: the feature rows are the donors, the target row is the treated unit,
and the training columns are the pre-periods.

1. Matrix estimation (Property 2.1) on the donor pre-period block: missing
   entries set to 0, p-hat the observed fraction floored at 1/(mn)
   (footnote 6), hard singular-value truncation, scaled by 1/p-hat.
2. beta-hat = argmin ||y_pre - M-hat v|| (Forecast step 2).
3. For each period t, alpha_t = argmin ||v_t - M-hat' alpha|| over v_t's observed
   coordinates, v_proj = M-hat' alpha_t, estimate = v_proj' beta-hat (step 3).

Layers, as agents_tests.md lays them out: an independent numpy oracle of the
de-noising step; closed-form reduction to the complete-data path bit for bit;
missing cells never entering as values; refusals for the treated unit's cells
and for the branches the paper does not cover; smoke and edge cases; hypothesis
properties; and the planted-missingness simulation on the latent-variable model
of Amjad, Shah and Shen (2018), Section 5.3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import CLUSTERSC
from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.clustersc_helpers.pcr import run_pcr
from mlsynth.utils.clustersc_helpers.pcr.missing import (
    ObservedDenoise,
    check_missing_supported,
    denoise_observed,
    forecast_counterfactual,
    observed_fraction,
)
from mlsynth.utils.clustersc_helpers.setup import prepare_clustersc_inputs
from mlsynth.utils.clustersc_helpers.simulation import simulate_rsc_panel
from mlsynth.utils.pcr.core import hsvt, pcr_weights

SUPPORTED = dict(method="pcr", clustering=False, compute_shen_ci=False)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _factor(T=20, J=12, r=2, T0=14, noise=0.3, seed=0, tau=1.0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, r))
    lam = rng.standard_normal((J + 1, r))
    Y = F @ lam.T + noise * rng.standard_normal((T, J + 1))
    Y[T0:, 0] += tau
    return Y, T0


def _long(Y, T0):
    T, n = Y.shape
    return pd.DataFrame([
        {"unit": j, "time": t, "y": float(Y[t, j]), "D": int(j == 0 and t >= T0)}
        for j in range(n) for t in range(T)])


def _plant(Y, cells):
    Y = Y.copy()
    for t, j in cells:
        Y[t, j] = np.nan
    return Y


def _fit(df, **cfg):
    return CLUSTERSC(dict(df=df, outcome="y", treat="D", unitid="unit",
                          time="time", display_graphs=False,
                          **{**SUPPORTED, **cfg})).fit()


def _paper_denoise(values, mask, r):
    """Property 2.1 written from the paper, independently of the library."""
    m, n = values.shape
    Y = np.zeros((m, n))
    Y[mask] = values[mask]
    p_hat = max(mask.sum() / (m * n), 1.0 / (m * n))
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    return (U[:, :r] @ np.diag(s[:r]) @ Vt[:r]) / p_hat, p_hat


def _paper_forecast(values, mask, M_hat, beta, r):
    """Forecast step 3 written from the paper: alpha_t on v_t's observed
    coordinates, v_proj = M-hat' alpha_t, estimate v_proj' beta."""
    out = np.empty(values.shape[0])
    Mt = M_hat.T                                   # (J, T0): columns = samples
    for t in range(values.shape[0]):
        o = mask[t]
        alpha, *_ = np.linalg.lstsq(Mt[o], values[t, o], rcond=1e-10)
        out[t] = (Mt @ alpha) @ beta
    return out


# ----------------------------------------------------------------------
# The de-noising step against an independent oracle
# ----------------------------------------------------------------------

class TestDenoiseOracle:
    @pytest.mark.parametrize("seed", range(5))
    def test_matches_the_paper_written_out(self, seed):
        Y, T0 = _factor(seed=seed)
        pre = Y[:T0, 1:]
        rng = np.random.default_rng(100 + seed)
        mask = rng.uniform(size=pre.shape) > 0.25
        values = np.where(mask, pre, np.nan)
        den = denoise_observed(values, mask, 2)
        ref, p_ref = _paper_denoise(values, mask, 2)
        assert den.p_hat == pytest.approx(p_ref, abs=0)
        np.testing.assert_allclose(den.M_hat, ref, rtol=1e-10, atol=1e-10)

    def test_p_hat_is_the_observed_fraction(self):
        mask = np.zeros((4, 5), bool)
        mask[:2, :3] = True
        assert observed_fraction(mask) == 6 / 20

    def test_p_hat_is_floored_at_one_over_mn(self):
        # footnote 6: max{fraction observed, 1/(mn)}
        assert observed_fraction(np.zeros((4, 5), bool)) == 1 / 20

    def test_the_scaling_is_one_over_p_hat(self):
        Y, T0 = _factor()
        pre = Y[:T0, 1:]
        mask = np.ones(pre.shape, bool)
        mask[0, :6] = False
        den = denoise_observed(pre, mask, 2)
        unscaled = hsvt(np.where(mask, pre, 0.0), 2)[0]
        np.testing.assert_allclose(den.M_hat * den.p_hat, unscaled,
                                   rtol=1e-12, atol=1e-12)
        assert den.p_hat == pytest.approx(1 - 6 / pre.size)

    def test_the_subspace_spans_the_rows_of_m_hat(self):
        Y, T0 = _factor()
        pre = Y[:T0, 1:]
        mask = np.random.default_rng(3).uniform(size=pre.shape) > 0.2
        den = denoise_observed(pre, mask, 2)
        assert isinstance(den, ObservedDenoise)
        assert den.V.shape == (pre.shape[1], 2)
        resid = den.M_hat - (den.M_hat @ den.V) @ den.V.T
        assert np.abs(resid).max() < 1e-10 * np.abs(den.M_hat).max()


class TestForecastOracle:
    @pytest.mark.parametrize("seed", range(5))
    def test_step_three_matches_the_paper_written_out(self, seed):
        Y, T0 = _factor(seed=seed)
        D = Y[:, 1:]
        rng = np.random.default_rng(200 + seed)
        mask = rng.uniform(size=D.shape) > 0.2
        den = denoise_observed(D[:T0], mask[:T0], 2)
        beta = pcr_weights(den.M_hat, Y[:T0, 0], 2)
        cf = forecast_counterfactual(np.where(mask, D, np.nan), mask, den.V, beta)
        ref = _paper_forecast(D, mask, den.M_hat, beta, 2)
        np.testing.assert_allclose(cf, ref, rtol=1e-8, atol=1e-8)

    def test_a_complete_period_is_v_t_times_beta(self):
        # beta lies in the span of the rows of M-hat, so v_proj' beta = v_t' beta.
        Y, T0 = _factor()
        D = Y[:, 1:]
        mask = np.ones(D.shape, bool)
        mask[2, 3] = False
        den = denoise_observed(D[:T0], mask[:T0], 2)
        beta = pcr_weights(den.M_hat, Y[:T0, 0], 2)
        cf = forecast_counterfactual(D, mask, den.V, beta)
        full_rows = mask.all(axis=1)
        np.testing.assert_array_equal(cf[full_rows], (D @ beta)[full_rows])
        proj = den.V @ (den.V.T @ D[5])
        assert proj @ beta == pytest.approx(D[5] @ beta, rel=1e-10)


# ----------------------------------------------------------------------
# Closed form first: an all-observed mask is the complete-data path
# ----------------------------------------------------------------------

class TestAllObservedReduction:
    def test_denoise_is_hsvt_bit_for_bit(self):
        Y, T0 = _factor()
        pre = Y[:T0, 1:]
        den = denoise_observed(pre, np.ones(pre.shape, bool), 3)
        assert den.p_hat == 1.0
        np.testing.assert_array_equal(den.M_hat, hsvt(pre, 3)[0])

    def test_forecast_is_the_raw_projection_bit_for_bit(self):
        Y, T0 = _factor()
        D = Y[:, 1:]
        mask = np.ones(D.shape, bool)
        den = denoise_observed(D[:T0], mask[:T0], 2)
        beta = pcr_weights(den.M_hat, Y[:T0, 0], 2)
        np.testing.assert_array_equal(
            forecast_counterfactual(D, mask, den.V, beta), D @ beta)

    def test_complete_panel_fit_carries_no_missing_metadata(self):
        Y, T0 = _factor()
        res = _fit(_long(Y, T0))
        assert "missing_p_hat" not in res.method_details.parameters_used
        assert "missing_p_hat" not in res.pcr.metadata

    def test_run_pcr_complete_equals_hsvt_ols_raw_projection(self):
        Y, T0 = _factor()
        fit, _ = run_pcr(Y[:, 0], Y[:, 1:], [str(j) for j in range(1, 13)], T0,
                         clustering=False, rank=2, compute_shen_ci=False)
        M = hsvt(Y[:T0, 1:], 2)[0]
        beta = pcr_weights(M, Y[:T0, 0], 2)
        np.testing.assert_array_equal(fit.counterfactual, Y[:, 1:] @ beta)


# ----------------------------------------------------------------------
# Missing cells never enter as values
# ----------------------------------------------------------------------

class TestMaskedValuesAreInert:
    @pytest.mark.parametrize("fill", [0.0, 1e6, -3.5, np.nan])
    def test_denoise_ignores_what_sits_under_the_mask(self, fill):
        Y, T0 = _factor()
        pre = Y[:T0, 1:]
        mask = np.random.default_rng(7).uniform(size=pre.shape) > 0.3
        base = denoise_observed(np.where(mask, pre, np.nan), mask, 2)
        other = denoise_observed(np.where(mask, pre, fill), mask, 2)
        np.testing.assert_array_equal(base.M_hat, other.M_hat)
        assert base.p_hat == other.p_hat

    @pytest.mark.parametrize("fill", [0.0, 1e6, -3.5])
    def test_forecast_ignores_what_sits_under_the_mask(self, fill):
        Y, T0 = _factor()
        D = Y[:, 1:]
        mask = np.random.default_rng(8).uniform(size=D.shape) > 0.3
        den = denoise_observed(D[:T0], mask[:T0], 2)
        beta = pcr_weights(den.M_hat, Y[:T0, 0], 2)
        a = forecast_counterfactual(np.where(mask, D, np.nan), mask, den.V, beta)
        b = forecast_counterfactual(np.where(mask, D, fill), mask, den.V, beta)
        np.testing.assert_array_equal(a, b)


# ----------------------------------------------------------------------
# Estimator integration: smoke, scale consistency
# ----------------------------------------------------------------------

class TestEstimator:
    def test_smoke_missing_donor_cells(self):
        Y, T0 = _factor()
        Ym = _plant(Y, [(1, 3), (5, 7), (16, 2), (18, 9)])
        res = _fit(_long(Ym, T0))
        assert np.isfinite(res.att)
        assert np.all(np.isfinite(res.counterfactual))
        params = res.method_details.parameters_used
        assert params["missing_donor_cells"] == 4
        assert params["missing_p_hat"] == pytest.approx(1 - 2 / (T0 * 12))
        assert res.pcr.metadata["missing_p_hat"] == params["missing_p_hat"]

    def test_counterfactual_is_the_paper_forecast(self):
        Y, T0 = _factor(seed=4)
        cells = [(0, 1), (3, 4), (9, 9), (15, 5), (19, 11)]
        Ym = _plant(Y, cells)
        res = _fit(_long(Ym, T0), rank=2, rank_method="fixed")
        D = Ym[:, 1:]
        mask = np.isfinite(D)
        M_hat, _ = _paper_denoise(D[:T0], mask[:T0], 2)
        beta = pcr_weights(M_hat, Y[:T0, 0], 2)
        ref = _paper_forecast(D, mask, M_hat, beta, 2)
        np.testing.assert_allclose(res.counterfactual, ref, rtol=1e-8, atol=1e-8)
        w = np.array([res.donor_weights[str(j)] for j in range(1, 13)])
        np.testing.assert_allclose(w, beta, rtol=1e-8, atol=1e-10)

    def test_weights_scale_with_one_over_p_hat(self):
        # The weights are fitted to M-hat = HSVT(Y)/p-hat, so they carry p-hat:
        # the weights on the unscaled reconstruction are 1/p-hat times larger.
        Y, T0 = _factor(seed=2)
        rng = np.random.default_rng(9)
        Ym = Y.copy()
        Ym[:T0, 1:][rng.uniform(size=(T0, 12)) > 0.7] = np.nan
        res = _fit(_long(Ym, T0), rank=2, rank_method="fixed")
        p_hat = res.pcr.metadata["missing_p_hat"]
        assert p_hat < 0.85
        unscaled = hsvt(np.nan_to_num(Ym[:T0, 1:]), 2)[0]
        beta_unscaled = pcr_weights(unscaled, Y[:T0, 0], 2)
        w = np.array([res.donor_weights[str(j)] for j in range(1, 13)])
        np.testing.assert_allclose(w, p_hat * beta_unscaled, rtol=1e-8, atol=1e-10)

    def test_rank_is_chosen_by_the_configured_rule_on_the_zero_filled_block(self):
        Y, T0 = _factor()
        Ym = _plant(Y, [(2, 2)])
        res = _fit(_long(Ym, T0), rank_method="usvt")
        from mlsynth.utils.pcr.rank import select_rank
        assert res.pcr.metadata["rank"] == select_rank(
            np.nan_to_num(Ym[:T0, 1:]), method="usvt")


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------

class TestRefusals:
    def test_treated_pre_cell_refused_naming_unit_and_period(self):
        Y, T0 = _factor()
        df = _long(_plant(Y, [(4, 0)]), T0)
        with pytest.raises(MlsynthDataError, match=r"treated unit '?0'?.*\b4\b"):
            prepare_clustersc_inputs(df, "y", "D", "unit", "time")

    def test_treated_post_cell_refused_naming_unit_and_period(self):
        Y, T0 = _factor()
        df = _long(_plant(Y, [(17, 0)]), T0)
        with pytest.raises(MlsynthDataError, match=r"\b17\b"):
            _fit(df)

    def test_treated_refusal_counts_every_cell(self):
        Y, T0 = _factor()
        df = _long(_plant(Y, [(3, 0), (16, 0), (18, 0)]), T0)
        with pytest.raises(MlsynthDataError, match="3 missing"):
            prepare_clustersc_inputs(df, "y", "D", "unit", "time")

    def test_donor_cells_are_kept_by_setup(self):
        Y, T0 = _factor()
        df = _long(_plant(Y, [(4, 5)]), T0)
        inputs = prepare_clustersc_inputs(df, "y", "D", "unit", "time")
        assert np.isnan(inputs.donor_outcomes[4, 4])
        assert np.isfinite(inputs.treated_outcome).all()

    @pytest.mark.parametrize("cfg, reason", [
        (dict(method="rpca"), "rpca"),
        (dict(method="both"), "rpca"),
        (dict(clustering=True), "clustering"),
        (dict(estimator="bayesian"), "bayesian"),
        (dict(pcr_objective="SIMPLEX"), "SIMPLEX"),
        (dict(lambda_penalty=0.5), "lambda_penalty"),
        (dict(project_denoised=True), "project_denoised"),
        (dict(compute_shen_ci=True), "compute_shen_ci"),
        (dict(compute_scpi_pi=True), "compute_scpi_pi"),
    ])
    def test_branches_the_paper_does_not_cover_refuse(self, cfg, reason):
        Y, T0 = _factor()
        df = _long(_plant(Y, [(4, 5)]), T0)
        with pytest.raises(MlsynthDataError, match=reason):
            _fit(df, **cfg)

    def test_the_refusal_lists_every_unsupported_option(self):
        with pytest.raises(MlsynthDataError) as info:
            check_missing_supported(
                method="pcr", clustering=True, estimator="bayesian",
                objective="OLS", lambda_penalty=None, project_denoised=False,
                compute_shen_ci=True, compute_scpi_pi=False)
        msg = str(info.value)
        assert "clustering" in msg and "bayesian" in msg and "compute_shen_ci" in msg
        assert "SIMPLEX" not in msg

    def test_supported_options_pass(self):
        check_missing_supported(
            method="pcr", clustering=False, estimator="frequentist",
            objective="OLS", lambda_penalty=0.0, project_denoised=False,
            compute_shen_ci=False, compute_scpi_pi=False)

    def test_run_pcr_refuses_unsupported_options_itself(self):
        Y, T0 = _factor()
        D = _plant(Y, [(4, 5)])[:, 1:]
        with pytest.raises(MlsynthDataError, match="clustering"):
            run_pcr(Y[:, 0], D, [str(j) for j in range(12)], T0,
                    clustering=True, compute_shen_ci=False)

    def test_complete_panel_keeps_every_branch(self):
        # The refusals apply to missing cells only.
        Y, T0 = _factor()
        res = _fit(_long(Y, T0), compute_shen_ci=True, project_denoised=True)
        assert np.isfinite(res.att)


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

class TestEdgeCases:
    def test_period_with_no_observed_donor_is_refused_by_label(self):
        Y, T0 = _factor()
        Ym = Y.copy()
        Ym[17, 1:] = np.nan
        with pytest.raises(MlsynthDataError, match=r"17"):
            _fit(_long(Ym, T0), rank=2, rank_method="fixed")

    def test_period_with_fewer_observed_donors_than_rank_is_refused(self):
        Y, T0 = _factor()
        Ym = Y.copy()
        Ym[16, 2:] = np.nan                       # one donor observed, rank 2
        with pytest.raises(MlsynthDataError, match="rank"):
            _fit(_long(Ym, T0), rank=2, rank_method="fixed")

    def test_forecast_refuses_with_default_labels(self):
        V = np.linalg.qr(np.random.default_rng(0).standard_normal((5, 2)))[0]
        values = np.ones((3, 5))
        mask = np.ones((3, 5), bool)
        mask[1, 1:] = False
        with pytest.raises(MlsynthDataError, match="period 1"):
            forecast_counterfactual(values, mask, V, np.ones(5))

    def test_single_donor(self):
        Y, T0 = _factor(J=1, r=1)
        Ym = _plant(Y, [(3, 1)])
        res = _fit(_long(Ym, T0), rank=1, rank_method="fixed")
        assert np.isfinite(res.att)
        assert res.pcr.metadata["missing_p_hat"] == pytest.approx(1 - 1 / T0)

    def test_collinear_donors_with_missing_cells(self):
        Y, T0 = _factor()
        Y[:, 2] = Y[:, 1]
        Ym = _plant(Y, [(1, 1), (6, 2)])
        res = _fit(_long(Ym, T0))
        assert np.all(np.isfinite(res.counterfactual))

    def test_every_pre_donor_cell_missing_is_an_error(self):
        Y, T0 = _factor()
        D = Y[:, 1:].copy()
        D[:T0] = np.nan
        with pytest.raises((MlsynthEstimationError, MlsynthDataError)):
            run_pcr(Y[:, 0], D, [str(j) for j in range(12)], T0,
                    clustering=False, rank=2, compute_shen_ci=False)

    def test_denoise_refuses_a_mask_of_the_wrong_shape(self):
        with pytest.raises(MlsynthEstimationError, match="shape"):
            denoise_observed(np.ones((3, 4)), np.ones((4, 3), bool), 1)

    def test_forecast_refuses_a_mask_of_the_wrong_shape(self):
        with pytest.raises(MlsynthEstimationError, match="shape"):
            forecast_counterfactual(np.ones((3, 4)), np.ones((4, 3), bool),
                                    np.eye(4)[:, :1], np.ones(4))


# ----------------------------------------------------------------------
# Hypothesis properties over random low-rank panels with random missingness
# ----------------------------------------------------------------------

@st.composite
def _lowrank(draw):
    m = draw(st.integers(4, 18))
    n = draw(st.integers(2, 14))
    r = draw(st.integers(1, min(3, m, n)))
    seed = draw(st.integers(0, 2 ** 31 - 1))
    q = draw(st.floats(0.0, 0.6))
    level = draw(st.floats(-50.0, 50.0))
    rng = np.random.default_rng(seed)
    X = level + rng.standard_normal((m, r)) @ rng.standard_normal((r, n)) \
        + 0.1 * rng.standard_normal((m, n))
    mask = rng.uniform(size=(m, n)) >= q
    return X, mask, r


@settings(max_examples=80, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(_lowrank())
def test_property_p_hat_is_the_floored_observed_fraction(case):
    X, mask, r = case
    den = denoise_observed(np.where(mask, X, np.nan), mask, r)
    assert den.p_hat == max(mask.sum() / mask.size, 1 / mask.size)


@settings(max_examples=80, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(_lowrank())
def test_property_denoised_matrix_is_finite(case):
    X, mask, r = case
    den = denoise_observed(np.where(mask, X, np.nan), mask, r)
    assert np.all(np.isfinite(den.M_hat))
    assert np.all(np.isfinite(den.V))


@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(_lowrank())
def test_property_all_observed_is_the_complete_path(case):
    X, _, r = case
    full = np.ones(X.shape, bool)
    den = denoise_observed(X, full, r)
    np.testing.assert_array_equal(den.M_hat, hsvt(X, r)[0])
    beta = np.random.default_rng(0).standard_normal(X.shape[1])
    np.testing.assert_array_equal(
        forecast_counterfactual(X, full, den.V, beta), X @ beta)


@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(_lowrank(), st.floats(-1e3, 1e3))
def test_property_masked_cells_are_inert(case, garbage):
    X, mask, r = case
    a = denoise_observed(np.where(mask, X, np.nan), mask, r)
    b = denoise_observed(np.where(mask, X, garbage), mask, r)
    np.testing.assert_array_equal(a.M_hat, b.M_hat)


@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(_lowrank())
def test_property_observed_coordinates_of_a_subspace_vector_are_recovered(case):
    # A donor vector that lies in the denoised subspace is reproduced exactly
    # from its observed coordinates, whatever is missing (step 3 identifies it).
    X, mask, r = case
    den = denoise_observed(X, np.ones(X.shape, bool), r)
    k = den.V.shape[1]
    rng = np.random.default_rng(1)
    v = den.V @ rng.standard_normal(k)
    o = rng.uniform(size=X.shape[1]) > 0.3
    if o.sum() < k or np.linalg.svd(den.V[o], compute_uv=False).min() < 1e-6:
        return
    beta = rng.standard_normal(X.shape[1])
    cf = forecast_counterfactual(v[None, :], o[None, :], den.V, beta)
    assert cf[0] == pytest.approx(v @ beta, rel=1e-6, abs=1e-6)


# ----------------------------------------------------------------------
# Planted missingness on the latent-variable model of ASS (2018) Section 5.3
# ----------------------------------------------------------------------

def _sim_error(N, T0, p_pre, p_post, seed, rank=2):
    T = int(T0 * 1.25)
    panel = simulate_rsc_panel(N=N, T=T, T0=T0, noise=1.0, seed=seed)
    X = panel.observed.T
    M = panel.means.T
    y, D = X[:, 0], X[:, 1:].copy()
    rng = np.random.default_rng(10_000 + seed)
    u = rng.uniform(size=D.shape)
    D[:T0][u[:T0] > p_pre] = np.nan
    D[T0:][u[T0:] > p_post] = np.nan
    mask = np.isfinite(D)
    den = denoise_observed(D[:T0], mask[:T0], rank)
    beta = pcr_weights(den.M_hat, y[:T0], rank)
    cf = forecast_counterfactual(D, mask, den.V, beta)
    e = cf[T0:] - M[T0:, 0]
    return float(np.mean(e)), float(np.sqrt(np.mean(e ** 2)))


@pytest.mark.parametrize("p_pre, p_post", [
    (1.0, 1.0), (0.9, 0.9), (0.8, 0.8), (0.7, 0.7),
    (0.7, 1.0), (1.0, 0.7), (0.9, 0.6),
])
def test_simulation_counterfactual_is_unbiased_across_missing_rates(p_pre, p_post):
    errs = [_sim_error(100, 800, p_pre, p_post, s) for s in range(8)]
    bias = np.mean([b for b, _ in errs])
    se = np.std([b for b, _ in errs]) / np.sqrt(len(errs))
    rmse = np.mean([r for _, r in errs])
    assert abs(bias) < max(4 * se, 0.05), (bias, se)
    assert rmse < 0.25, rmse


def test_simulation_error_falls_with_the_panel_length():
    for p in (1.0, 0.8, 0.7):
        rmse = [np.mean([_sim_error(100, T0, p, p, s)[1] for s in range(6)])
                for T0 in (100, 1600)]
        assert rmse[1] < rmse[0], (p, rmse)


def test_simulation_error_grows_with_the_missing_fraction():
    rmse = [np.mean([_sim_error(100, 400, p, p, s)[1] for s in range(6)])
            for p in (1.0, 0.9, 0.8, 0.7)]
    assert rmse == sorted(rmse), rmse
    assert rmse[-1] < 1.5 * rmse[0], rmse
