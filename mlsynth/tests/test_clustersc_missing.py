"""CLUSTERSC's PCR path on panels with missing donor cells (#1146 step 2).

The algorithm is the two-block PCR of Agarwal, Shah and Shen, "On Model
Identification and Out-of-Sample Prediction of Principal Component
Regression: Applications to Synthetic Controls", JMLR 26 (2025), Section 2.2,
in the synthetic-control mapping of their Section 7.1 (Z = donor pre-periods,
Z' = donor post-periods, y = treated pre-periods):

I.  Z-tilde = zero-filled Z / rho-hat (rho-hat = observed fraction of Z),
    beta-hat = (Z-tilde^k)^+ y.
II. Z-tilde' = zero-filled Z' / rho-hat' (its own observed fraction),
    y-hat' = Z-tilde'^ell beta-hat, with ell full rank in CLUSTERSC, so
    y-hat' = Z-tilde' beta-hat.

Layers, as agents_tests.md lays them out: an independent numpy oracle;
closed-form reduction to the complete-data path bit for bit; missing cells
never entering as values; refusals for the treated unit's cells, for empty
periods and for the branches the paper does not cover; smoke and edge cases;
hypothesis properties; and planted missingness on the latent-variable model of
Amjad, Shah and Shen (2018), Section 5.3, with different rates pre and post.
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
    observed_fraction,
    refuse_empty_periods,
    rescale_observed,
    two_block_counterfactual,
)
from mlsynth.utils.clustersc_helpers.setup import prepare_clustersc_inputs
from mlsynth.utils.clustersc_helpers.simulation import simulate_rsc_panel
from mlsynth.utils.pcr.core import hsvt, pcr_weights
from mlsynth.utils.pcr.rank import select_rank

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


def _paper(values, T0, y_pre, k):
    """JMLR 2025 Section 2.2, Steps I and II (ell full), from the paper."""
    def tilde(B):
        obs = ~np.isnan(B)
        rho = obs.sum() / obs.size
        Z = np.zeros(B.shape)
        Z[obs] = B[obs]
        return Z / rho, rho
    Zt, rho = tilde(values[:T0])
    U, s, Vt = np.linalg.svd(Zt, full_matrices=False)
    beta = Vt[:k].T @ np.diag(1 / s[:k]) @ U[:, :k].T @ y_pre     # eq. (3)
    Zp, rho_p = tilde(values[T0:])
    return beta, np.concatenate([Zt @ beta, Zp @ beta]), rho, rho_p


# ----------------------------------------------------------------------
# Against an independent oracle written from the paper
# ----------------------------------------------------------------------

class TestOracle:
    @pytest.mark.parametrize("seed", range(5))
    def test_weights_and_counterfactual_match_the_paper(self, seed):
        Y, T0 = _factor(seed=seed)
        rng = np.random.default_rng(100 + seed)
        D = np.where(rng.uniform(size=Y[:, 1:].shape) > 0.2, Y[:, 1:], np.nan)
        D[:, 0] = Y[:, 1]          # one complete donor, so no period is empty
        mask = ~np.isnan(D)
        den = denoise_observed(D[:T0], mask[:T0], 2)
        beta = pcr_weights(den.M_hat, Y[:T0, 0], 2)
        cf, p_post = two_block_counterfactual(D, mask, T0, beta)
        b_ref, cf_ref, rho, rho_p = _paper(D, T0, Y[:T0, 0], 2)
        assert den.p_hat == pytest.approx(rho, abs=0)
        assert p_post == pytest.approx(rho_p, abs=0)
        np.testing.assert_allclose(beta, b_ref, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(cf, cf_ref, rtol=1e-9, atol=1e-9)

    def test_fit_matches_the_paper_end_to_end(self):
        Y, T0 = _factor(seed=4)
        Ym = _plant(Y, [(0, 1), (3, 4), (9, 9), (15, 5), (19, 11), (17, 3)])
        res = _fit(_long(Ym, T0), rank=2, rank_method="fixed")
        b_ref, cf_ref, rho, rho_p = _paper(Ym[:, 1:], T0, Y[:T0, 0], 2)
        np.testing.assert_allclose(res.counterfactual, cf_ref, rtol=1e-9, atol=1e-9)
        w = np.array([res.donor_weights[str(j)] for j in range(1, 13)])
        np.testing.assert_allclose(w, b_ref, rtol=1e-9, atol=1e-10)
        assert res.pcr.metadata["missing_p_hat"] == pytest.approx(rho)
        assert res.pcr.metadata["missing_p_hat_post"] == pytest.approx(rho_p)

    def test_p_hat_is_the_observed_fraction(self):
        mask = np.zeros((4, 5), bool)
        mask[:2, :3] = True
        assert observed_fraction(mask) == 6 / 20

    def test_p_hat_is_floored_at_one_over_mn(self):
        assert observed_fraction(np.zeros((4, 5), bool)) == 1 / 20

    def test_rescaling_divides_the_zero_filled_block_by_p_hat(self):
        values = np.arange(12.0).reshape(3, 4) + 1
        mask = np.ones((3, 4), bool)
        mask[0, 0] = mask[2, 3] = False
        Z, p = rescale_observed(values, mask)
        assert p == 10 / 12
        expect = np.where(mask, values, 0.0) * 12 / 10
        np.testing.assert_allclose(Z, expect, rtol=1e-15)
        assert Z[0, 0] == 0.0 and Z[2, 3] == 0.0

    def test_each_block_has_its_own_p_hat(self):
        Y, T0 = _factor()
        D = Y[:, 1:].copy()
        D[:T0][np.random.default_rng(0).uniform(size=(T0, 12)) > 0.9] = np.nan
        D[T0:][np.random.default_rng(1).uniform(size=(6, 12)) > 0.6] = np.nan
        D[:, 0] = Y[:, 1]
        mask = ~np.isnan(D)
        den = denoise_observed(D[:T0], mask[:T0], 2)
        beta = pcr_weights(den.M_hat, Y[:T0, 0], 2)
        cf, p_post = two_block_counterfactual(D, mask, T0, beta)
        assert den.p_hat == pytest.approx(mask[:T0].mean())
        assert p_post == pytest.approx(mask[T0:].mean())
        assert abs(den.p_hat - p_post) > 0.05
        np.testing.assert_allclose(
            cf[T0:], (np.where(mask[T0:], D[T0:], 0) / p_post) @ beta, rtol=1e-12)

    def test_in_sample_fit_equals_the_truncated_fit(self):
        # beta lies in the row space of Z-tilde^k, so Z-tilde beta = Z-tilde^k beta.
        Y, T0 = _factor(seed=3)
        D = _plant(Y, [(1, 2), (4, 6), (8, 8)])[:, 1:]
        mask = ~np.isnan(D)
        den = denoise_observed(D[:T0], mask[:T0], 2)
        beta = pcr_weights(den.M_hat, Y[:T0, 0], 2)
        cf, _ = two_block_counterfactual(D, mask, T0, beta)
        np.testing.assert_allclose(cf[:T0], den.M_hat @ beta, rtol=1e-9, atol=1e-10)

    def test_weights_carry_p_hat(self):
        # The weights are fitted to Z-tilde = Y / rho-hat, so they are rho-hat
        # times the weights on the unscaled zero-filled block.
        Y, T0 = _factor(seed=2)
        Ym = Y.copy()
        Ym[:T0, 1:][np.random.default_rng(9).uniform(size=(T0, 12)) > 0.7] = np.nan
        Ym[:T0, 1] = Y[:T0, 1]
        res = _fit(_long(Ym, T0), rank=2, rank_method="fixed")
        p_hat = res.pcr.metadata["missing_p_hat"]
        assert p_hat < 0.85
        unscaled = hsvt(np.nan_to_num(Ym[:T0, 1:]), 2)[0]
        beta_unscaled = pcr_weights(unscaled, Y[:T0, 0], 2)
        w = np.array([res.donor_weights[str(j)] for j in range(1, 13)])
        np.testing.assert_allclose(w, p_hat * beta_unscaled, rtol=1e-8, atol=1e-10)

    def test_rank_rule_reads_the_rescaled_block(self):
        Y, T0 = _factor()
        Ym = _plant(Y, [(2, 2)])
        res = _fit(_long(Ym, T0), rank_method="usvt")
        Z, _ = rescale_observed(Ym[:T0, 1:], ~np.isnan(Ym[:T0, 1:]))
        assert res.pcr.metadata["rank"] == select_rank(Z, method="usvt")
        assert isinstance(denoise_observed(Z, np.ones(Z.shape, bool), 1),
                          ObservedDenoise)


# ----------------------------------------------------------------------
# Closed form first: an all-observed mask is the complete-data path
# ----------------------------------------------------------------------

class TestAllObservedReduction:
    def test_denoise_is_hsvt_bit_for_bit(self):
        Y, T0 = _factor()
        pre = Y[:T0, 1:]
        den = denoise_observed(pre, np.ones(pre.shape, bool), 3)
        assert den.p_hat == 1.0
        np.testing.assert_array_equal(den.Z_tilde, pre)
        np.testing.assert_array_equal(den.M_hat, hsvt(pre, 3)[0])

    def test_counterfactual_is_the_raw_projection_bit_for_bit(self):
        Y, T0 = _factor()
        D = Y[:, 1:]
        mask = np.ones(D.shape, bool)
        beta = pcr_weights(hsvt(D[:T0], 2)[0], Y[:T0, 0], 2)
        cf, p_post = two_block_counterfactual(D, mask, T0, beta)
        assert p_post == 1.0
        np.testing.assert_array_equal(cf[:T0], D[:T0] @ beta)
        np.testing.assert_array_equal(cf[T0:], D[T0:] @ beta)

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
    def test_counterfactual_ignores_what_sits_under_the_mask(self, fill):
        Y, T0 = _factor()
        D = Y[:, 1:]
        mask = np.random.default_rng(8).uniform(size=D.shape) > 0.3
        beta = np.random.default_rng(1).standard_normal(12)
        a, _ = two_block_counterfactual(np.where(mask, D, np.nan), mask, T0, beta)
        b, _ = two_block_counterfactual(np.where(mask, D, fill), mask, T0, beta)
        np.testing.assert_array_equal(a, b)

    def test_fit_ignores_what_sits_under_the_mask(self):
        # Through the estimator: a planted NaN against the same cell refilled
        # by run_pcr's caller with garbage is not possible (NaN is the mask),
        # so the check is that two panels differing only in masked cells'
        # original values give the same fit.
        Y, T0 = _factor(seed=5)
        cells = [(2, 3), (16, 7)]
        Y2 = Y.copy()
        for t, j in cells:
            Y2[t, j] += 50.0
        a = _fit(_long(_plant(Y, cells), T0))
        b = _fit(_long(_plant(Y2, cells), T0))
        np.testing.assert_array_equal(a.counterfactual, b.counterfactual)


# ----------------------------------------------------------------------
# Estimator integration
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
        assert params["missing_p_hat_post"] == pytest.approx(1 - 2 / (6 * 12))
        assert res.pcr.metadata["missing_p_hat"] == params["missing_p_hat"]

    def test_only_post_cells_missing(self):
        Y, T0 = _factor()
        res = _fit(_long(_plant(Y, [(16, 2)]), T0))
        assert res.pcr.metadata["missing_p_hat"] == 1.0
        assert res.pcr.metadata["missing_p_hat_post"] < 1.0

    def test_two_donors_one_missing_cell(self):
        Y, T0 = _factor(J=2, r=1)
        res = _fit(_long(_plant(Y, [(3, 1)]), T0), rank=1, rank_method="fixed")
        assert np.isfinite(res.att)
        assert res.pcr.metadata["missing_p_hat"] == pytest.approx(1 - 1 / (2 * T0))


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------

class TestRefusals:
    def test_treated_pre_cell_refused_naming_unit_and_period(self):
        Y, T0 = _factor()
        df = _long(_plant(Y, [(4, 0)]), T0)
        with pytest.raises(MlsynthDataError, match=r"treated unit '0'.*period\(s\) 4\b"):
            prepare_clustersc_inputs(df, "y", "D", "unit", "time")

    def test_treated_post_cell_refused_naming_unit_and_period(self):
        Y, T0 = _factor()
        df = _long(_plant(Y, [(17, 0)]), T0)
        with pytest.raises(MlsynthDataError, match=r"period\(s\) 17\b"):
            _fit(df)

    def test_treated_refusal_counts_every_cell(self):
        Y, T0 = _factor()
        df = _long(_plant(Y, [(3, 0), (16, 0), (18, 0)]), T0)
        with pytest.raises(MlsynthDataError, match="3 missing.*3, 16, 18"):
            prepare_clustersc_inputs(df, "y", "D", "unit", "time")

    def test_run_pcr_refuses_a_missing_treated_cell(self):
        Y, T0 = _factor()
        y = Y[:, 0].copy()
        y[2] = np.nan
        with pytest.raises(MlsynthDataError, match="treated outcome"):
            run_pcr(y, Y[:, 1:], [str(j) for j in range(12)], T0,
                    clustering=False, compute_shen_ci=False)

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
        assert "SIMPLEX" not in msg and "project_denoised" not in msg

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
        Y, T0 = _factor()
        res = _fit(_long(Y, T0), compute_shen_ci=True, project_denoised=True)
        assert np.isfinite(res.att)


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

class TestEdgeCases:
    def test_post_period_with_no_observed_donor_is_refused_by_label(self):
        Y, T0 = _factor()
        Ym = Y.copy()
        Ym[17, 1:] = np.nan
        with pytest.raises(MlsynthDataError, match=r"period\(s\) 17 have no observed"):
            _fit(_long(Ym, T0))

    def test_pre_period_with_no_observed_donor_is_refused_by_label(self):
        Y, T0 = _factor()
        Ym = Y.copy()
        Ym[3, 1:] = np.nan
        with pytest.raises(MlsynthDataError, match=r"period\(s\) 3 have no observed"):
            _fit(_long(Ym, T0))

    def test_empty_period_refusal_with_default_labels(self):
        mask = np.ones((3, 4), bool)
        mask[1] = False
        with pytest.raises(MlsynthDataError, match=r"period\(s\) 1 have"):
            refuse_empty_periods(mask)
        refuse_empty_periods(np.ones((3, 4), bool))

    def test_single_donor_missing_cell_leaves_its_period_empty(self):
        Y, T0 = _factor(J=1, r=1)
        with pytest.raises(MlsynthDataError, match="no observed"):
            _fit(_long(_plant(Y, [(3, 1)]), T0), rank=1, rank_method="fixed")

    def test_collinear_donors_with_missing_cells(self):
        Y, T0 = _factor()
        Y[:, 2] = Y[:, 1]
        res = _fit(_long(_plant(Y, [(1, 1), (6, 2)]), T0))
        assert np.all(np.isfinite(res.counterfactual))

    def test_every_pre_donor_cell_but_one_column_missing(self):
        # A design of rank one: the rank-2 request keeps one direction.
        Y, T0 = _factor()
        Ym = Y.copy()
        Ym[:T0, 2:] = np.nan
        res = _fit(_long(Ym, T0), rank=2, rank_method="fixed")
        assert np.all(np.isfinite(res.counterfactual))
        assert res.pcr.metadata["missing_p_hat"] == pytest.approx(1 / 12)

    def test_rescale_refuses_a_mask_of_the_wrong_shape(self):
        with pytest.raises(MlsynthEstimationError, match="shape"):
            rescale_observed(np.ones((3, 4)), np.ones((4, 3), bool))


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


_PROP = dict(deadline=None, suppress_health_check=[HealthCheck.too_slow])


@settings(max_examples=80, **_PROP)
@given(_lowrank())
def test_property_p_hat_is_the_floored_observed_fraction(case):
    X, mask, r = case
    den = denoise_observed(np.where(mask, X, np.nan), mask, r)
    assert den.p_hat == max(mask.sum() / mask.size, 1 / mask.size)


@settings(max_examples=80, **_PROP)
@given(_lowrank())
def test_property_denoised_matrix_is_finite(case):
    X, mask, r = case
    den = denoise_observed(np.where(mask, X, np.nan), mask, r)
    assert np.all(np.isfinite(den.M_hat)) and np.all(np.isfinite(den.Z_tilde))


@settings(max_examples=60, **_PROP)
@given(_lowrank(), st.integers(1, 17))
def test_property_all_observed_is_the_complete_path(case, T0):
    X, _, r = case
    T0 = min(T0, X.shape[0] - 1)
    full = np.ones(X.shape, bool)
    den = denoise_observed(X[:T0], full[:T0], r)
    np.testing.assert_array_equal(den.M_hat, hsvt(X[:T0], r)[0])
    beta = np.random.default_rng(0).standard_normal(X.shape[1])
    cf, _ = two_block_counterfactual(X, full, T0, beta)
    np.testing.assert_array_equal(cf[:T0], X[:T0] @ beta)
    np.testing.assert_array_equal(cf[T0:], X[T0:] @ beta)


@settings(max_examples=60, **_PROP)
@given(_lowrank(), st.floats(-1e3, 1e3))
def test_property_masked_cells_are_inert(case, garbage):
    X, mask, r = case
    a = denoise_observed(np.where(mask, X, np.nan), mask, r)
    b = denoise_observed(np.where(mask, X, garbage), mask, r)
    np.testing.assert_array_equal(a.M_hat, b.M_hat)
    beta = np.ones(X.shape[1])
    T0 = X.shape[0] // 2
    ca, pa = two_block_counterfactual(np.where(mask, X, np.nan), mask, T0, beta)
    cb, pb = two_block_counterfactual(np.where(mask, X, garbage), mask, T0, beta)
    np.testing.assert_array_equal(ca, cb)
    assert pa == pb


@settings(max_examples=60, **_PROP)
@given(_lowrank())
def test_property_rescaled_block_is_unbiased_for_the_observed_values(case):
    # E[Z-tilde] = X (Section 2.4.1): averaged over the cells of the block,
    # sum(Z-tilde) = sum over observed cells / rho-hat, an exact identity.
    X, mask, _ = case
    Z, p = rescale_observed(X, mask)
    assert Z.sum() == pytest.approx(X[mask].sum() / p, rel=1e-9, abs=1e-9)
    assert np.all(Z[~mask] == 0.0)


# ----------------------------------------------------------------------
# Planted missingness on the latent-variable model of ASS (2018) Section 5.3
# ----------------------------------------------------------------------

def _sim_error(N, T0, p_pre, p_post, seed, rank=2):
    T = int(T0 * 1.25)
    panel = simulate_rsc_panel(N=N, T=T, T0=T0, noise=1.0, seed=seed)
    X = panel.observed.T
    M = panel.means.T
    y, D = X[:, 0], X[:, 1:].copy()
    u = np.random.default_rng(10_000 + seed).uniform(size=D.shape)
    D[:T0][u[:T0] > p_pre] = np.nan
    D[T0:][u[T0:] > p_post] = np.nan
    mask = ~np.isnan(D)
    den = denoise_observed(D[:T0], mask[:T0], rank)
    beta = pcr_weights(den.M_hat, y[:T0], rank)
    cf, _ = two_block_counterfactual(D, mask, T0, beta)
    e = cf[T0:] - M[T0:, 0]
    return float(np.mean(e)), float(np.sqrt(np.mean(e ** 2)))


@pytest.mark.parametrize("p_pre, p_post", [
    (1.0, 1.0), (0.9, 0.9), (0.8, 0.8), (0.7, 0.7),
    (0.7, 1.0), (1.0, 0.7), (0.9, 0.6),
])
def test_simulation_counterfactual_is_unbiased_across_missing_rates(p_pre, p_post):
    errs = [_sim_error(100, 400, p_pre, p_post, s) for s in range(8)]
    bias = np.array([b for b, _ in errs])
    se = bias.std() / np.sqrt(bias.size)
    rmse = np.mean([r for _, r in errs])
    assert abs(bias.mean()) < max(4 * se, 0.05), (bias.mean(), se)
    assert rmse < 0.3, rmse


def test_simulation_error_falls_with_the_panel_length():
    for p in (1.0, 0.8, 0.6):
        rmse = [np.mean([_sim_error(100, T0, p, p, s)[1] for s in range(6)])
                for T0 in (100, 800)]
        assert rmse[1] < rmse[0], (p, rmse)


def test_simulation_error_grows_with_the_missing_fraction():
    rmse = [np.mean([_sim_error(100, 400, p, p, s)[1] for s in range(6)])
            for p in (1.0, 0.8, 0.6, 0.4)]
    assert rmse == sorted(rmse), rmse
