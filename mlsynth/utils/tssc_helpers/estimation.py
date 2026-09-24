"""Constrained-LS estimation of the four SC-class variants for TSSC.

Each variant minimizes the pre-treatment sum of squares
``sum_{t<=T1} (y_{1t} - x_t' beta)^2`` subject to its constraint set
(see :mod:`.structures`). The solve is delegated to the project's shared
``Opt.SCopt`` cvxpy wrapper. The post-period counterfactual is
``y_hat^0_{1t} = x_t' beta_hat`` and the ATT is the mean post-period gap
(Li & Shankar, 2023, Eqs. (2.1)/(2.5)).

ATT confidence intervals (per variant) use the subsampling procedure of
Li (2020), which combines two sources of uncertainty: (i) donor-weight
estimation error -- captured by refitting the variant on size-``m``
permuted pre-treatment subsamples whose treated outcome is regenerated
from the fitted weights plus pre-period noise -- and (ii) post-period
idiosyncratic prediction error. The interval is
``[ATT - q_{1-a/2}, ATT - q_{a/2}]`` where the ``q`` are quantiles of the
normalized statistic's subsampling distribution.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthEstimationError
from ..resultutils import effects
from ..weights import WeightConstraint, WeightSolution, solve_weights
from .structures import MSCA, MSCB, MSCC, SC, TSSCInputs, TSSCVariantFit

# Li and Shankar (2023) define the four variants by their constraint set and
# nothing else, so each variant is one WeightConstraint and the choice among
# them is the whole method.
_CONSTRAINT = {
    SC:   WeightConstraint(),                                      # w >= 0, sum w = 1
    MSCA: WeightConstraint(intercept=True),                        # + free intercept
    MSCB: WeightConstraint(sum_to_one=False),                      # w >= 0 only
    MSCC: WeightConstraint(sum_to_one=False, intercept=True),      # + free intercept
}
# Which variants carry a free intercept (beta_1).
_HAS_INTERCEPT = {SC: False, MSCA: True, MSCB: False, MSCC: True}
# scpi weight-constraint family per variant. SC / MSCa map exactly (simplex, with
# a free constant for MSCa); MSCb / MSCc carry bare non-negativity, which scpi's
# family has no member for, so they use scpi's ols set -- the band does not
# re-impose w >= 0 and is therefore slightly conservative.
_SCPI_CONSTRAINT = {SC: "simplex", MSCA: "simplex", MSCB: "ols", MSCC: "ols"}

# Pre-treatment subsample is T1 minus this when forming ATT-CI subsamples.
_ATT_CI_SUBSAMPLE_ADJUSTMENT = 5


def _solve_certified(
    method: str, donor_pre: np.ndarray, y_pre: np.ndarray, n_pre: int, n_donors: int
) -> Optional[WeightSolution]:
    """Solve one SC-class variant, with its optimality certificate.

    The four variants differ only in which polyhedron the weights are drawn
    from, so the solve is :func:`~mlsynth.utils.weights.solve_weights` under
    the variant's :class:`WeightConstraint`. Each polyhedron here has non-empty
    relative interior, so the returned KKT residual settles optimality outright
    (Boyd and Vandenberghe 2004, section 5.5.3).

    The intercept of MSCa and MSCc is a field on the constraint set, not a
    donor column carrying hand-written slices. It is free in sign and profiled
    out by centring, which is what it means for the treated unit to be allowed
    to sit below its donors.

    Returns ``None`` when the panel cannot be solved at all, which is the
    contract the subsampling loop and Step-1 selection branch on to skip a bad
    refit instead of losing the whole interval.
    """
    try:
        return solve_weights(
            np.asarray(donor_pre, dtype=float)[:n_pre],
            np.asarray(y_pre, dtype=float)[:n_pre],
            _CONSTRAINT[method],
        )
    except (MlsynthEstimationError, MlsynthConfigError):
        return None


def _solve(
    method: str, donor_pre: np.ndarray, y_pre: np.ndarray, n_pre: int, n_donors: int
) -> Optional[np.ndarray]:
    """One variant's coefficient vector, intercept first, or ``None``.

    The flat layout matches :func:`_features`: length ``n_donors + 1`` with the
    intercept leading for MSCa and MSCc, length ``n_donors`` otherwise.
    """
    solution = _solve_certified(method, donor_pre, y_pre, n_pre, n_donors)
    if solution is None:
        return None
    if _HAS_INTERCEPT[method]:
        return np.r_[solution.intercept, solution.weights]
    return np.array(solution.weights, dtype=float)


def fit_mscc_beta(
    donor_pre: np.ndarray, y_pre: np.ndarray, n_pre: int, n_donors: int
) -> Optional[np.ndarray]:
    """MSC(c) coefficient vector ``beta`` (length ``N``, intercept first).

    This is the benchmark estimator the Step-1 tests build on, and the
    workhorse re-fit inside the subsampling loop.
    """
    return _solve(MSCC, donor_pre, y_pre, n_pre, n_donors)


def _features(method: str, donor_matrix: np.ndarray) -> np.ndarray:
    """Donor design matrix, prepending an intercept column when needed."""
    if _HAS_INTERCEPT[method]:
        return np.c_[np.ones((donor_matrix.shape[0], 1)), donor_matrix]
    return donor_matrix


def bootstrap_att_ci(
    inputs: TSSCInputs,
    method: str,
    weights: np.ndarray,
    counterfactual: np.ndarray,
    att: float,
    n_bootstrap: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Subsampling confidence interval for a variant's ATT (Li, 2020).

    Parameters
    ----------
    inputs : TSSCInputs
    method : str
        SC-class variant name.
    weights : np.ndarray
        The variant's fitted coefficient vector.
    counterfactual : np.ndarray
        Length-``T`` fitted/counterfactual path for the variant.
    att : float
        The variant's point ATT.
    n_bootstrap : int
        Number of subsampling replications.
    confidence_level : float
        E.g. ``0.95`` for a 95% interval.
    rng : numpy.random.Generator
        RNG for reproducible subsampling (no global-seed side effects).
    """
    T0, T2 = inputs.T0, inputs.T2
    n = inputs.n_donors
    m = T0 - _ATT_CI_SUBSAMPLE_ADJUSTMENT
    if m <= 0 or T2 <= 0:
        return (float("nan"), float("nan"))

    donor_pre = inputs.donor_matrix[:T0]
    donor_post = inputs.donor_matrix[T0:]
    y_pre = inputs.y[:T0]

    pre_resid_std = float(np.sqrt(np.mean((y_pre - counterfactual[:T0]) ** 2)))
    post_gap = inputs.y[T0:] - counterfactual[T0:]
    post_var = float(np.mean((post_gap - post_gap.mean()) ** 2))
    post_sd = float(np.sqrt(max(post_var, 0.0)))

    feats_pre = _features(method, donor_pre)
    feats_post = _features(method, donor_post)

    stats: list = []
    for _ in range(n_bootstrap):
        idx = rng.permutation(T0)[:m]
        y_star = feats_pre[idx] @ weights + pre_resid_std * rng.standard_normal(m)
        w_star = _solve(method, donor_pre[idx], y_star, m, n)
        if w_star is None or not np.all(np.isfinite(w_star)):
            continue
        comp_weights = -np.mean(feats_post @ (w_star - weights)) * np.sqrt(
            (T2 * m) / T0
        )
        comp_noise = np.sqrt(T2) * np.mean(post_sd * rng.standard_normal(T2))
        stats.append((comp_weights + comp_noise) / np.sqrt(T2))

    if len(stats) < 2:
        return (float("nan"), float("nan"))

    alpha = 1.0 - confidence_level
    lower_q = float(np.quantile(stats, alpha / 2))
    upper_q = float(np.quantile(stats, 1 - alpha / 2))
    return (att - upper_q, att - lower_q)


def _scpi_for_variant(inputs, method, donor_coefs, intercept, *, sims, alpha,
                      e_method, seed):
    """scpi prediction-interval band for one variant, under its mapped
    constraint (see ``_SCPI_CONSTRAINT``). Returns ``None`` on failure or when
    there are no post-treatment periods."""
    if inputs.T <= inputs.T0:
        return None
    from ..clustersc_helpers.scpi_pi import scpi_pi_inference
    from ...exceptions import MlsynthEstimationError
    constraint = _SCPI_CONSTRAINT[method]
    has_const = _HAS_INTERCEPT[method]
    # scpi expects the constant coefficient last; TSSC stores it first.
    W = (np.concatenate([np.asarray(donor_coefs, float), [float(intercept)]])
         if has_const else np.asarray(donor_coefs, float))
    try:
        return scpi_pi_inference(
            np.asarray(inputs.y, float), np.asarray(inputs.donor_matrix, float),
            int(inputs.T0), W, constraint=constraint, constant=has_const,
            sims=int(sims), alpha=float(alpha), e_method=e_method,
            seed=int(seed), periods=list(inputs.time_labels[inputs.T0:]),
        )
    except (MlsynthEstimationError, ValueError, ImportError):
        return None


def fit_variant(
    inputs: TSSCInputs,
    method: str,
    n_bootstrap: int,
    confidence_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
    compute_scpi_pi: bool = False,
    scpi_sims: int = 200,
    scpi_alpha: float = 0.05,
    scpi_e_method: str = "gaussian",
    scpi_seed: int = 0,
    compute_ci: bool = True,
) -> TSSCVariantFit:
    """Fit one SC-class variant and assemble its :class:`TSSCVariantFit`.

    ``compute_ci=False`` skips the subsampling confidence interval and returns
    ``att_ci = (nan, nan)``. The weights, counterfactual and ATT are a
    deterministic QP and are unaffected -- only the interval is dropped. The
    subsampling dominates the runtime, so this is the difference between a fit
    that is affordable inside a simulation loop and one that is not.
    """

    if rng is None:
        rng = np.random.default_rng()

    n = inputs.n_donors
    T0 = inputs.T0
    donor_pre = inputs.donor_matrix[:T0]
    y_pre = inputs.y[:T0]

    solution = _solve_certified(method, donor_pre, y_pre, T0, n)
    if solution is None:
        raise MlsynthEstimationError(
            f"TSSC: optimization failed for variant {method!r}."
        )

    has_intercept = _HAS_INTERCEPT[method]
    donor_coefs = np.array(solution.weights, dtype=float)
    if has_intercept:
        intercept = float(solution.intercept)
        weights = np.r_[intercept, donor_coefs]
    else:
        intercept = None
        weights = donor_coefs

    counterfactual = _features(method, inputs.donor_matrix) @ weights

    att_results, fit_diag, _ = effects.calculate(
        observed_outcome_series=inputs.y,
        counterfactual_outcome_series=counterfactual,
        num_pre_treatment_periods=T0,
        num_actual_post_periods=inputs.T2,
    )
    att = float(att_results["ATT"])

    att_ci = (
        bootstrap_att_ci(
            inputs=inputs, method=method, weights=weights,
            counterfactual=counterfactual, att=att, n_bootstrap=n_bootstrap,
            confidence_level=confidence_level, rng=rng,
        )
        if compute_ci
        else (float("nan"), float("nan"))
    )

    donor_weights = {
        str(name): float(round(coef, 4))
        for name, coef in zip(inputs.donor_names, donor_coefs)
        if abs(coef) > 1e-3
    }

    scpi_band = None
    if compute_scpi_pi:
        scpi_band = _scpi_for_variant(
            inputs, method, donor_coefs, intercept,
            sims=scpi_sims, alpha=scpi_alpha, e_method=scpi_e_method,
            seed=scpi_seed)

    return TSSCVariantFit(
        method=method,
        weights=weights,
        intercept=intercept,
        donor_weights=donor_weights,
        counterfactual=counterfactual,
        gap=inputs.y - counterfactual,
        att=att,
        att_ci=att_ci,
        rmse_pre=float(fit_diag["T0 RMSE"]),
        rmse_post=float(fit_diag["T1 RMSE"]),
        r2_pre=float(fit_diag["R-Squared"]),
        scpi=scpi_band,
        weights_unique=solution.unique,
        kkt_residual=solution.kkt_residual,
        solver=solution.solver,
    )
