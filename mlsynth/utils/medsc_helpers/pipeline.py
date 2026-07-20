"""MEDSC pipeline: total / direct / indirect decomposition and placebo inference.

The total effect is an ordinary synthetic control on the treated outcome. The
direct effect is a cross-world control that, in each post period, also matches
the treated unit's post-treatment mediator path up to that period (paper
Section 3.2); the indirect effect is the difference. With covariates the
predictor weights are cross-validated by the bilevel mscmt search; without
covariates the fit matches the pre-treatment outcome path directly (the
specification under which the paper's decomposition reproduces).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError
from ..bilevel.active_set import solve_simplex_qp
from ..bilevel.mscmt import solve_mscmt, _inner_weights
from ..bilevel.structure import BilevelProblem


def _zscore(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scale each row of ``A`` (and the matching entry of ``b``) to unit std.

    The std is taken across donors (columns of ``A``); constant rows are left
    unscaled. This conditions the predictor blocks before the simplex/bilevel
    solve so no single row dominates by units alone.
    """
    sd = A.std(axis=1, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return A / sd, b / sd.ravel()


def _resolve_backend(backend: str, has_covariates: bool) -> str:
    """Map the ``"auto"`` backend to the concrete one."""
    if backend == "auto":
        return "mscmt" if has_covariates else "outcome-only"
    if backend == "mscmt" and not has_covariates:
        raise MlsynthConfigError(
            "backend='mscmt' needs covariates; use 'outcome-only' (or 'auto') "
            "for outcome-path matching.")
    return backend


def _lag_indices(predictor_lags: Optional[List[Any]], time_labels: np.ndarray,
                 T0: int) -> List[int]:
    """Row indices of the outcome/mediator matching lags.

    ``None`` -> every pre-treatment period (the full outcome path). Otherwise
    the positions of the requested time labels, which must be pre-treatment.
    """
    if predictor_lags is None:
        return list(range(T0))
    label_to_idx = {lab: i for i, lab in enumerate(time_labels)}
    idx = []
    for lab in predictor_lags:
        if lab not in label_to_idx:
            raise MlsynthConfigError(
                f"predictor_lags label {lab!r} is not a time period in df.")
        i = label_to_idx[lab]
        if i >= T0:
            raise MlsynthConfigError(
                f"predictor_lags label {lab!r} is not in the pre-treatment period.")
        idx.append(i)
    return idx


def _predictor_block(
    donor_outcomes: np.ndarray, donor_mediators: Optional[np.ndarray],
    treated_outcome: np.ndarray, treated_mediator: Optional[np.ndarray],
    donor_cov: Optional[np.ndarray], treated_cov: Optional[np.ndarray],
    lag_i: List[int], with_mediator_and_cov: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble the z-scored predictor block ``(K, J)`` and treated vector ``(K,)``.

    Outcome-lag rows always; pre-treatment mediator lags and covariate rows only
    when ``with_mediator_and_cov`` (the mscmt/covariate path).
    """
    Ao, bo = _zscore(donor_outcomes[lag_i, :], treated_outcome[lag_i])
    rows_A = [Ao]
    rows_b = [bo]
    if with_mediator_and_cov:
        if donor_mediators is not None:
            Am, bm = _zscore(donor_mediators[lag_i, :], treated_mediator[lag_i])
            rows_A.append(Am)
            rows_b.append(bm)
        if donor_cov is not None and donor_cov.size:
            Ac, bc = _zscore(donor_cov, treated_cov)
            rows_A.append(Ac)
            rows_b.append(bc)
    return np.vstack(rows_A), np.concatenate(rows_b)


def _outcome_path_weights(donor_pre: np.ndarray, treated_pre: np.ndarray,
                          lag_i: List[int]) -> np.ndarray:
    """Simplex weights matching the treated outcome path at ``lag_i``."""
    A, b = _zscore(donor_pre[lag_i, :], treated_pre[lag_i])
    return solve_simplex_qp(A, b)


def _total_fit(inputs, config, backend: str, lag_i: List[int]):
    """Total-effect synthetic control: donor weights and the counterfactual."""
    y1 = inputs.treated_outcome
    Y_tot = inputs.total_donor_outcomes
    pre = slice(0, inputs.T0)
    v_pre = None
    if backend == "outcome-only":
        w = _outcome_path_weights(Y_tot, y1, lag_i)
    else:
        X0, X1 = _predictor_block(
            Y_tot, inputs.total_donor_outcomes, y1, inputs.treated_mediator,
            inputs.total_covariates, inputs.treated_covariates, lag_i,
            with_mediator_and_cov=True)
        prob = BilevelProblem(y1_pre=y1[pre], Y0_pre=Y_tot[pre, :], X1=X1, X0=X0)
        sol = solve_mscmt(prob, seed=config.seed, maxiter=config.mscmt_maxiter,
                          popsize=config.mscmt_popsize)
        w = sol.W
    cf = Y_tot @ w
    return w, cf


def _direct_v_pre(inputs, config, lag_i: List[int]) -> np.ndarray:
    """Cross-validated pre-block predictor weights V on the direct pool (mscmt)."""
    y1 = inputs.treated_outcome
    Y_dir = inputs.direct_donor_outcomes
    pre = slice(0, inputs.T0)
    X0, X1 = _predictor_block(
        Y_dir, inputs.direct_donor_mediators, y1, inputs.treated_mediator,
        inputs.direct_covariates, inputs.treated_covariates, lag_i,
        with_mediator_and_cov=True)
    prob = BilevelProblem(y1_pre=y1[pre], Y0_pre=Y_dir[pre, :], X1=X1, X0=X0)
    sol = solve_mscmt(prob, seed=config.seed, maxiter=config.mscmt_maxiter,
                      popsize=config.mscmt_popsize)
    v = sol.V
    s = v.sum()
    return v / s if s > 0 else v


def _direct_fit(inputs, config, backend: str, lag_i: List[int]):
    """Per-period cross-world (mediator-matched) control.

    Returns the counterfactual ``(T,)`` (NaN pre-treatment) and the donor
    weights at the final post period.
    """
    y1 = inputs.treated_outcome
    m1 = inputs.treated_mediator
    Y_dir = inputs.direct_donor_outcomes
    M_dir = inputs.direct_donor_mediators
    T, T0 = inputs.T, inputs.T0
    pre_w = config.pre_weight

    cf = np.full(T, np.nan)
    last_w: Optional[np.ndarray] = None

    if backend == "outcome-only":
        A_pre, b_pre = _zscore(Y_dir[:T0, :], y1[:T0])
        A_pre, b_pre = A_pre[lag_i, :], b_pre[lag_i]
        n_pre = A_pre.shape[0]
    else:
        v_pre = _direct_v_pre(inputs, config, lag_i)
        X0_pre, X1_pre = _predictor_block(
            Y_dir, M_dir, y1, m1, inputs.direct_covariates,
            inputs.treated_covariates, lag_i, with_mediator_and_cov=True)

    for ti in range(T0, T):
        med_idx = list(range(T0, ti + 1))
        Am, bm = _zscore(M_dir[med_idx, :], m1[med_idx])
        n_med = len(med_idx)
        if backend == "outcome-only":
            wp = np.sqrt(pre_w / n_pre)
            wo = np.sqrt((1.0 - pre_w) / n_med)
            A = np.vstack([wp * A_pre, wo * Am])
            b = np.concatenate([wp * b_pre, wo * bm])
            w = solve_simplex_qp(A, b)
        else:
            X0 = np.vstack([X0_pre, Am])
            X1 = np.concatenate([X1_pre, bm])
            V = np.concatenate([pre_w * v_pre,
                                np.full(n_med, (1.0 - pre_w) / n_med)])
            prob = BilevelProblem(
                y1_pre=y1[:T0], Y0_pre=Y_dir[:T0, :], X1=X1, X0=X0)
            w = _inner_weights(prob, V)
        cf[ti] = Y_dir[ti, :] @ w
        last_w = w
    return cf, last_w


def _placebo_inference(inputs, lag_i: List[int], cutoff: float
                       ) -> Tuple[Optional[float], Dict[str, Any]]:
    """In-space placebo test on the total effect (outcome-path SC).

    Refits the outcome-path synthetic control treating each total-pool donor as
    pseudo-treated, screens out donors whose pre-RMSPE exceeds ``cutoff`` times
    the treated unit's, and ranks the treated unit's post/pre RMSPE ratio among
    the survivors. Returns the p-value and a details dict.
    """
    y1 = inputs.treated_outcome
    Y_tot = inputs.total_donor_outcomes
    names = inputs.total_donor_names
    T0, T = inputs.T0, inputs.T
    J = Y_tot.shape[1]

    def _ratio(y: np.ndarray, cf: np.ndarray) -> Tuple[float, float, float]:
        pre = np.sqrt(np.mean((y[:T0] - cf[:T0]) ** 2))
        post = np.sqrt(np.mean((y[T0:] - cf[T0:]) ** 2))
        ratio = post / pre if pre > 0 else np.inf
        return pre, post, ratio

    w_t = _outcome_path_weights(Y_tot, y1, lag_i)
    pre_t, _, ratio_t = _ratio(y1, Y_tot @ w_t)

    placebo_ratios: List[float] = []
    kept: List[Any] = []
    for d in range(J):
        y_d = Y_tot[:, d]
        others = [k for k in range(J) if k != d]
        Y_o = Y_tot[:, others]
        if Y_o.shape[1] == 0:  # pragma: no cover - needs a single-donor pool
            continue
        w_d = _outcome_path_weights(Y_o, y_d, lag_i)
        pre_d, _, ratio_d = _ratio(y_d, Y_o @ w_d)
        if pre_t > 0 and pre_d > cutoff * pre_t:
            continue
        placebo_ratios.append(float(ratio_d))
        kept.append(names[d])

    n = len(placebo_ratios)
    if n == 0:  # pragma: no cover - defensive; cutoff screened everything
        return None, {"n_placebos": 0, "treated_rmspe_ratio": float(ratio_t)}
    n_ge = int(np.sum(np.asarray(placebo_ratios) >= ratio_t))
    p_value = (1 + n_ge) / (1 + n)
    details = {
        "n_placebos": n,
        "treated_rmspe_ratio": float(ratio_t),
        "placebo_rmspe_ratios": dict(zip(kept, placebo_ratios)),
        "cutoff": float(cutoff),
    }
    return float(p_value), details


def run_medsc_core(inputs, config):
    """Compute the mediation decomposition and placebo inference.

    Returns ``(decomposition, inference_p, inference_details, metadata)``.
    """
    from .structures import MediationDecomposition

    backend = _resolve_backend(config.backend, bool(inputs.covariate_names))
    lag_i = _lag_indices(config.predictor_lags, inputs.time_labels, inputs.T0)
    T0, T = inputs.T0, inputs.T
    y1 = inputs.treated_outcome

    w_tot, cf_tot = _total_fit(inputs, config, backend, lag_i)
    total = y1 - cf_tot

    cf_dir, w_dir = _direct_fit(inputs, config, backend, lag_i)
    direct = y1 - cf_dir
    indirect = total - direct

    post = slice(T0, T)
    pre_rmse = float(np.sqrt(np.mean((y1[:T0] - cf_tot[:T0]) ** 2)))
    decomposition = MediationDecomposition(
        total=total,
        direct=direct,
        indirect=indirect,
        counterfactual_total=cf_tot,
        counterfactual_direct=cf_dir,
        att_total=float(np.mean(total[post])),
        att_direct=float(np.nanmean(direct[post])),
        att_indirect=float(np.nanmean(indirect[post])),
        pre_rmse_total=pre_rmse,
        total_weights={n: float(w) for n, w in
                       zip(inputs.total_donor_names, w_tot)},
        direct_weights_final=({n: float(w) for n, w in
                               zip(inputs.direct_donor_names, w_dir)}
                              if w_dir is not None else {}),
    )

    inf_p: Optional[float] = None
    inf_details: Dict[str, Any] = {}
    if config.inference:
        inf_p, inf_details = _placebo_inference(inputs, lag_i, config.placebo_cutoff)

    metadata = {
        "backend": backend,
        "n_total_donors": len(inputs.total_donor_names),
        "n_direct_donors": len(inputs.direct_donor_names),
        "n_covariates": inputs.L,
        "pre_weight": float(config.pre_weight),
    }
    return decomposition, inf_p, inf_details, metadata
