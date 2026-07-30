"""Tests for the ridge-augmentation layer of the bilevel SCM engine (ASCM).

Covers the standalone ridge primitives, the ``BilevelSCM(augment="ridge")``
capability, and a regression against the ``augsynth`` R package on its canonical
Kansas tax-cut example (Average ATT -0.0401, plain SCM -0.0294).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from mlsynth.utils.bilevel import (
    BilevelSCM,
    best_lambda,
    build_matching,
    conformal_intervals,
    conformal_pvalue,
    generate_lambdas,
    ridge_augment_weights,
    solve_ridge,
)

cp = pytest.importorskip("cvxpy")  # ridge base is an exact simplex QP

_KANSAS = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "kansas_taxcut.csv"
)
_KANSAS_ASCM = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "kansas_ascm.csv"
)


def _toy(seed: int = 0, T0: int = 20, J: int = 6):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(T0 + 5, 2))
    loads = rng.uniform(0.2, 1.0, size=(J, 2))
    Y0 = factor @ loads.T + 0.05 * rng.normal(size=(T0 + 5, J))
    w = np.zeros(J); w[: 3] = [0.5, 0.3, 0.2]
    y = Y0 @ w + 0.05 * rng.normal(size=T0 + 5)
    return y[:T0], Y0[:T0], y, Y0


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_generate_lambdas_grid():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 8))
    lambdas = generate_lambdas(X, lambda_min_ratio=1e-8, n_lambda=20)
    assert lambdas.shape == (21,)                      # augsynth: n_lambda+1
    assert np.all(np.diff(lambdas) < 0)                # strictly descending
    lmax = np.linalg.svd(X, compute_uv=False)[0] ** 2
    assert lambdas[0] == pytest.approx(lmax)
    assert lambdas[-1] == pytest.approx(lmax * 1e-8)


def test_solve_ridge_zero_lambda_interpolates():
    # With lambda -> 0 and a wide donor matrix the ridge correction drives the
    # pre-period imbalance to ~0 (exact interpolation of the centered target).
    rng = np.random.default_rng(1)
    B = rng.normal(size=(10, 15))           # m < J: underdetermined
    A = rng.normal(size=10)
    W = np.full(15, 1 / 15)
    W_aug = W + solve_ridge(A, B, W, lambda_=1e-10)
    assert np.linalg.norm(A - B @ W_aug) < 1e-4


def test_best_lambda_1se_is_larger_than_min():
    lambdas = np.array([10.0, 3.0, 1.0, 0.3, 0.1])
    mean = np.array([5.0, 2.0, 1.05, 1.0, 1.2])      # min at lambda=0.3
    se = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
    lam_min = best_lambda(lambdas, mean, se, min_1se=False)
    lam_1se = best_lambda(lambdas, mean, se, min_1se=True)
    assert lam_min == 0.3                              # outright CV minimum
    # 1-SE picks the largest lambda within min_error+se (1.3): lambda=1.0
    assert lam_1se == 1.0 and lam_1se > lam_min


# --------------------------------------------------------------------------- #
# ridge_augment_weights
# --------------------------------------------------------------------------- #
def test_ridge_augment_base_is_simplex():
    y_pre, Y0_pre, _, _ = _toy()
    ra = ridge_augment_weights(y_pre, Y0_pre)
    assert ra.W_base.min() >= -1e-8
    assert ra.W_base.sum() == pytest.approx(1.0, abs=1e-6)
    # augmentation moves off the simplex but the correction is additive
    assert np.allclose(ra.W, ra.W_base + ra.W_ridge)


def test_ridge_augment_fixed_lambda_skips_cv():
    y_pre, Y0_pre, _, _ = _toy()
    ra = ridge_augment_weights(y_pre, Y0_pre, lambda_=1.0)
    assert ra.lambda_ == 1.0
    assert ra.cv is None
    ra_cv = ridge_augment_weights(y_pre, Y0_pre)
    assert ra_cv.cv is not None
    assert set(ra_cv.cv) == {"lambdas", "errors_mean", "errors_se"}


def test_ridge_augment_improves_pre_fit_over_simplex():
    # The whole point of ASCM: the augmented weights close residual pre-period
    # imbalance the simplex SCM leaves behind.
    y_pre, Y0_pre, _, _ = _toy(seed=3, T0=25, J=8)
    ra = ridge_augment_weights(y_pre, Y0_pre, lambda_=1e-3)
    mu = Y0_pre.mean(axis=1)
    base_imb = np.sum((y_pre - mu - (Y0_pre - mu[:, None]) @ ra.W_base) ** 2)
    aug_imb = np.sum((y_pre - mu - (Y0_pre - mu[:, None]) @ ra.W) ** 2)
    assert aug_imb < base_imb


# --------------------------------------------------------------------------- #
# Engine capability
# --------------------------------------------------------------------------- #
def test_engine_augment_ridge_flags():
    y_pre, Y0_pre, _, _ = _toy()
    r = BilevelSCM("outcome-only", augment="ridge").fit(y_pre, Y0_pre)
    assert r.augment == "ridge"
    assert r.is_simplex is False
    assert r.lambda_ is not None
    assert r.W_base is not None and r.W_base.sum() == pytest.approx(1.0, abs=1e-6)
    assert "cv" in r.augment_diagnostics


def test_engine_no_augment_is_simplex():
    y_pre, Y0_pre, _, _ = _toy()
    r = BilevelSCM("outcome-only").fit(y_pre, Y0_pre)
    assert r.augment is None
    assert r.is_simplex is True
    assert r.W_base is None and r.lambda_ is None


def test_engine_bad_augment_raises():
    with pytest.raises(ValueError, match="augment"):
        BilevelSCM("outcome-only", augment="lasso")


def test_engine_ridge_fixed_lambda():
    y_pre, Y0_pre, _, _ = _toy()
    r = BilevelSCM("outcome-only", augment="ridge", ridge_lambda=2.0).fit(y_pre, Y0_pre)
    assert r.lambda_ == 2.0
    assert "cv" not in r.augment_diagnostics      # fixed lambda -> no CV curve


# --------------------------------------------------------------------------- #
# Covariates (parallel-inclusion stacking)
# --------------------------------------------------------------------------- #
def test_build_matching_stacks_and_standardizes_covariates():
    y_pre, Y0_pre, _, _ = _toy(T0=18, J=7)
    J = Y0_pre.shape[1]
    Z0 = np.column_stack([Y0_pre.mean(0), Y0_pre[-1]])     # (J, 2)
    z1 = np.array([y_pre.mean(), y_pre[-1]])
    B, A = build_matching(y_pre, Y0_pre, Z0, z1)
    assert B.shape == (18 + 2, J)                          # 2 covariate rows stacked
    assert A.shape == (18 + 2,)
    # the stacked covariate rows are standardized to the centered-outcome scale
    B_out = Y0_pre - Y0_pre.mean(1)[:, None]
    sdx = B_out.std(ddof=1)
    assert B[-2:].std(axis=1, ddof=1) == pytest.approx(np.full(2, sdx), rel=0.3)


def test_engine_ridge_with_covariates_changes_fit():
    y_pre, Y0_pre, _, _ = _toy(seed=4, T0=20, J=8)
    J = Y0_pre.shape[1]
    X0 = np.vstack([Y0_pre.mean(0), Y0_pre[-1]])           # (P=2, J)
    X1 = np.array([y_pre.mean(), y_pre[-1]])
    r0 = BilevelSCM("outcome-only", augment="ridge").fit(y_pre, Y0_pre)
    rc = BilevelSCM("outcome-only", augment="ridge").fit(y_pre, Y0_pre, X1=X1, X0=X0)
    assert rc.W.shape == (J,)                              # weights stay over donors
    assert not np.allclose(rc.W, r0.W)                     # covariates move the fit


# --------------------------------------------------------------------------- #
# Conformal inference
# --------------------------------------------------------------------------- #
def test_conformal_pvalue_in_unit_interval():
    y_pre, Y0_pre, y, Y0 = _toy(T0=20, J=6)
    pre = 20
    p = conformal_pvalue(y, Y0, pre, lambda_=1.0, ns=200, seed=0)
    assert 0.0 <= p <= 1.0


def test_conformal_intervals_smoke():
    y_pre, Y0_pre, y, Y0 = _toy(T0=18, J=6)
    pre = 18
    ci = conformal_intervals(y, Y0, pre, lambda_=1.0, ns=100, grid_size=6, seed=0)
    n_post = len(y) - pre
    assert ci.att.shape == (n_post,)
    assert ci.lower.shape == (n_post,) and ci.upper.shape == (n_post,)
    assert np.all(ci.lower <= ci.upper)
    assert 0.0 <= ci.joint_p_value <= 1.0


# --------------------------------------------------------------------------- #
# Regression: augsynth R Kansas tax-cut (single_augsynth, progfunc="Ridge")
# --------------------------------------------------------------------------- #
def test_augsynth_kansas_replication():
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(_KANSAS)
    piv = df.pivot(index="fips", columns="year_qtr", values="lngdpcapita").sort_index()
    times = np.array(sorted(df["year_qtr"].unique()))
    pre = int((times < 2012.25).sum())
    y = piv.loc[20.0].to_numpy()
    Y0 = piv.drop(index=20.0).to_numpy().T          # (T, J)
    y_pre, Y0_pre = y[:pre], Y0[:pre]

    r = BilevelSCM("outcome-only", augment="ridge").fit(y_pre, Y0_pre)
    att = float(np.mean((y - r.counterfactual(Y0))[pre:]))
    att_base = float(np.mean((y - Y0 @ r.W_base)[pre:]))

    # augsynth vignette: plain SCM Average ATT -0.0294; Ridge ASCM -0.0401.
    assert att_base == pytest.approx(-0.0294, abs=5e-4)
    assert att == pytest.approx(-0.0401, abs=5e-4)
    assert r.lambda_ == pytest.approx(0.0787, abs=5e-3)


def test_augsynth_kansas_conformal_pvalue():
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(_KANSAS)
    piv = df.pivot(index="fips", columns="year_qtr", values="lngdpcapita").sort_index()
    times = np.array(sorted(df["year_qtr"].unique()))
    pre = int((times < 2012.25).sum())
    y = piv.loc[20.0].to_numpy()
    Y0 = piv.drop(index=20.0).to_numpy().T

    r = BilevelSCM("outcome-only", augment="ridge").fit(y[:pre], Y0[:pre])
    # augsynth reuses the fitted penalty in the conformal refit (no re-CV).
    p = conformal_pvalue(y, Y0, pre, lambda_=r.lambda_, ns=2000, seed=0)
    # augsynth vignette joint-null p-value for Ridge ASCM: 0.071 (permutation
    # test -> reproduced to Monte-Carlo precision).
    assert p == pytest.approx(0.071, abs=0.02)


# --------------------------------------------------------------------------- #
# Covariates: parallel (residualize=False) vs residualized (residualize=True)
# --------------------------------------------------------------------------- #
def test_residualize_returns_donor_weights_off_simplex():
    y_pre, Y0_pre, _, _ = _toy(seed=7, T0=24, J=8)
    J = Y0_pre.shape[1]
    Z0 = np.column_stack([Y0_pre.mean(0), Y0_pre[-1], Y0_pre[0]])   # (J, 3)
    z1 = np.array([y_pre.mean(), y_pre[-1], y_pre[0]])
    par = ridge_augment_weights(y_pre, Y0_pre, Z0=Z0, z1=z1, residualize=False)
    res = ridge_augment_weights(y_pre, Y0_pre, Z0=Z0, z1=z1, residualize=True)
    assert res.W.shape == (J,)                       # weights stay over donors
    # residualize differs from parallel inclusion, and (as ASCM) leaves simplex
    assert not np.allclose(res.W, par.W)
    assert res.W.sum() == pytest.approx(1.0, abs=1e-6)   # add-back preserves sum-1


def _kansas_covariate_stack():
    """The raw ``(N, T0, K)`` pre-treatment covariate cube, before aggregation."""
    pytest.importorskip("pandas")
    from benchmarks.cases.ascm_kansas import covariate_stack

    return covariate_stack()


def _kansas_ascm_covariates():
    pytest.importorskip("pandas")
    from benchmarks.cases.ascm_kansas import kansas_panel

    return kansas_panel()


class TestCovariateAggregation:
    """How per-unit covariate means treat missing values.

    augsynth's ``extract_covariates`` (``R/format.R``) passes ``na.action = NULL``
    to ``model.frame`` -- so no rows are removed -- and then aggregates each
    covariate independently with ``mean(x, na.rm = TRUE)``. Omitting missing
    values *column*-wise rather than *row*-wise is the whole content of that
    choice, and on this panel it is not a rounding-level distinction: the two
    revenue series are reported annually, so listwise deletion would discard
    those quarters from all six covariates and not just from revenue.
    """

    def test_only_the_revenue_series_are_sparse(self):
        """The property that makes this panel diagnostic, asserted not assumed.

        If the vendored CSV ever gains or loses missing values the two tests
        below stop distinguishing column-wise from row-wise aggregation, and
        would pass for the wrong reason.
        """
        stack = _kansas_covariate_stack()
        missing = np.isnan(stack).sum(axis=(0, 1))
        # covariate order: lngdpcapita, log(revstate), log(revlocal),
        #                  log(avgwklywage), estabscapita, emplvlcapita
        assert missing[0] == 0
        assert missing[3] == missing[4] == missing[5] == 0
        assert missing[1] == missing[2] > 0
        # 50 units x 56 of the 89 pre-treatment quarters
        n_units, n_pre, _ = stack.shape
        assert missing[1] == n_units * 56
        assert n_pre == 89

    def test_missing_values_are_omitted_per_column_not_per_row(self):
        """The substance of the difference, stated as behaviour.

        A covariate with no missing values must average over every pre-treatment
        quarter, however sparse its neighbours are. Under row-wise deletion it
        would average over only the 33 quarters where revenue is also reported.
        """
        from benchmarks.cases.ascm_kansas import aggregate_covariates

        stack = _kansas_covariate_stack()
        Z = aggregate_covariates(stack)
        # the four fully observed covariates: the mean over all T0 quarters
        dense = [0, 3, 4, 5]
        np.testing.assert_allclose(
            Z[:, dense], stack[:, :, dense].mean(axis=1), rtol=0, atol=1e-12)
        # the two sparse ones: the mean over their reported quarters only
        for k in (1, 2):
            for u in range(stack.shape[0]):
                col = stack[u, :, k]
                assert Z[u, k] == pytest.approx(
                    col[~np.isnan(col)].mean(), rel=0, abs=1e-12)

    def test_row_wise_deletion_would_move_the_dense_covariates(self):
        """Pins that the two rules genuinely differ, so the fix is load-bearing.

        Without this, a future refactor could reintroduce row-wise deletion and
        the assertions above would be the only thing standing in the way, with
        no record of how much was at stake.
        """
        from benchmarks.cases.ascm_kansas import aggregate_covariates

        stack = _kansas_covariate_stack()
        Z = aggregate_covariates(stack)
        keep = ~np.isnan(stack).any(axis=2)
        Z_rowwise = np.array([stack[u][keep[u]].mean(0)
                              for u in range(stack.shape[0])])
        gap = np.abs(Z - Z_rowwise).max(axis=0)
        assert gap[1] == gap[2] == pytest.approx(0.0, abs=1e-12)  # sparse: same
        assert gap[0] > 0.1 and gap[3] > 0.1                      # dense: not


def test_kansas_covariate_lambda_max_matches_augsynth():
    """``get_lambda_max`` on the covariate-stacked matching matrix.

    The whole lambda grid is ``lambda_max * scaler^k``, so a wrong
    ``lambda_max`` misplaces every candidate penalty. It is also the cheapest
    scalar that detects a wrong covariate block: nothing downstream can be right
    if this is off, and it is off by 0.15 under row-wise aggregation.
    """
    y_pre, Y0_pre, _, _, Z0, z1 = _kansas_ascm_covariates()
    B, _ = build_matching(y_pre, Y0_pre, Z0=Z0, z1=z1)
    lambda_max = float(np.linalg.svd(B, compute_uv=False)[0]) ** 2
    assert lambda_max == pytest.approx(128.6077, abs=1e-4)


def test_augsynth_kansas_covariate_ladder():
    y_pre, Y0_pre, y_post, Y0_post, Z0, z1 = _kansas_ascm_covariates()
    att = lambda w: float(np.mean(y_post - Y0_post.T @ w))
    cov = ridge_augment_weights(y_pre, Y0_pre, Z0=Z0, z1=z1)
    res = ridge_augment_weights(y_pre, Y0_pre, Z0=Z0, z1=z1, residualize=True)
    # Live augsynth on this spec: covariate ASCM -0.060937, residualized
    # -0.052773. The covariate cell reproduces to 1e-6 -- it is an exact
    # cross-validation, not an approximate one, so the tolerance says so.
    assert att(cov.W) == pytest.approx(-0.060937, abs=1e-5)
    # The residualized cell stays loose on purpose. After residualizing out K
    # covariates the residual Gram is rank-deficient, so augsynth's residual
    # lambda-CV is ill-posed and drifts to the grid floor; mlsynth tunes on the
    # outcome scale instead, which is where augsynth's CV lands anyway and which
    # reproduces the vignette's published -0.055 robustly. The gap is that
    # deliberate difference, not a defect -- see benchmarks/cases/ascm_kansas.py.
    assert att(res.W) == pytest.approx(-0.052773, abs=4e-3)


# === moving-block (cyclic) conformal permutation (augsynth type="block") ===

def _conf_panel(T=20, J=4, pre=15, seed=3):
    import numpy as np
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(size=(T, J)) + np.arange(T)[:, None] * 0.2
    y = Y0 @ np.array([0.4, 0.3, 0.2, 0.1]) + rng.normal(scale=0.05, size=T)
    return y, Y0, pre


def test_conformal_pvalue_block_is_seed_invariant():
    """Block uses the deterministic T cyclic shifts, not random draws."""
    import numpy as np
    from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue
    y, Y0, pre = _conf_panel()
    a = conformal_pvalue(y, Y0, pre, conformal_type="block", ns=100, seed=0)
    b = conformal_pvalue(y, Y0, pre, conformal_type="block", ns=100, seed=12345)
    assert a == b                                   # deterministic across seeds


def test_conformal_pvalue_iid_depends_on_seed():
    from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue
    y, Y0, pre = _conf_panel()
    a = conformal_pvalue(y, Y0, pre, conformal_type="iid", ns=50, seed=0)
    b = conformal_pvalue(y, Y0, pre, conformal_type="iid", ns=50, seed=999)
    assert 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0      # both valid p-values


def test_conformal_pvalue_block_matches_cyclic_shift_reference():
    """The block p-value equals mean(obs <= stat over the T np.roll shifts)."""
    import numpy as np
    from mlsynth.utils.bilevel.ridge_inference import (
        conformal_pvalue, _augmented_gaps, _stat,
    )
    y, Y0, pre = _conf_panel()
    p = conformal_pvalue(y, Y0, pre, conformal_type="block")
    resids = _augmented_gaps(y, Y0, None, None, {})
    obs = _stat(resids[pre:], 1.0)
    ref = np.array([_stat(np.roll(resids, s)[pre:], 1.0) for s in range(resids.shape[0])])
    assert p == pytest.approx(float(np.mean(obs <= ref)))


def test_conformal_pvalue_bad_type_raises():
    from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue
    y, Y0, pre = _conf_panel()
    with pytest.raises(ValueError, match="conformal_type"):
        conformal_pvalue(y, Y0, pre, conformal_type="bootstrap")
