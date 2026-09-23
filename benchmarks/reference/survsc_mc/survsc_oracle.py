"""Readable oracle port of Synthetic Survival Control (Han & Shah 2025).

Provenance: arXiv:2511.14133v1, Section 3.4.2 (estimator) and Section 4
(simulation design). Every step below cites the paper.

The design in Section 4 makes both DGPs exponential: the Cox specification
fixes the baseline hazard at ``h(t) = lambda`` and the Aalen specification
fixes ``h'(t) = lambda_0``, so in each case the hazard is constant in ``t``
and the potential event time under control is ``Exp(rate[p, n])``. That
gives a closed form for the estimand,

    S^(0)_{p,n}(t) = exp(-rate[p, n] * t),

which is what the sup-norm error in Table 1 is measured against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize

from mlsynth.utils.pcr import pcr_weights, select_rank, spectral_rank

# --- Section 4 design constants -------------------------------------------

R_LATENT = 4            # r = 4 latent factors
N_UNITS = 20            # N = 20 units, one of them treated
N_PERIODS = 2           # p in {0, 1}
TREATED = 0             # "one unit designated as the treated unit"
LAMBDA_COX = 0.05       # Cox baseline hazard, h(t) = lambda
AALEN_FLOOR = 0.05      # lambda_0 chosen so the minimum hazard is 0.05
GRID_POINTS = 100       # T0 = 100 equally spaced timestamps
HORIZON_Q = 0.90        # tau_tilde = quantile_0.90 of observed times
CENSOR_TARGET = 0.10    # "empirical censoring proportion ~ 10%"


@dataclass(frozen=True)
class Design:
    """One draw of the latent structure, shared by both periods."""

    rate: np.ndarray        # (N_PERIODS, N_UNITS) exponential rates under control
    covariates: np.ndarray  # (N_PERIODS, N_UNITS, 2 * R_LATENT) true [V_n, U_p]


def draw_design(rng: np.random.Generator, dgp: str) -> Design:
    """Draw latent factors and coefficients, then form the control hazards.

    Cox (Section 4, model 1):  h^(0)_{p,n}(t) = lambda * exp(b1'V_n + b2'U_p).
    Aalen (Section 4, model 2): h^(0)_{p,n}(t) = lambda_0 + b1'V_n + b2'U_p,
    with lambda_0 = 0.05 - min_{p,n}(b1'V_n + b2'U_p) so hazards stay positive.
    """
    V = rng.standard_normal((N_UNITS, R_LATENT))
    U = rng.standard_normal((N_PERIODS, R_LATENT))
    b1 = rng.standard_normal(R_LATENT)
    b2 = rng.standard_normal(R_LATENT)

    linear = (V @ b1)[None, :] + (U @ b2)[:, None]      # (P, N)
    if dgp == "cox":
        rate = LAMBDA_COX * np.exp(linear)
    elif dgp == "aalen":
        rate = (AALEN_FLOOR - linear.min()) + linear
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown dgp {dgp!r}")

    covariates = np.empty((N_PERIODS, N_UNITS, 2 * R_LATENT))
    for p in range(N_PERIODS):
        for n in range(N_UNITS):
            covariates[p, n] = np.concatenate([V[n], U[p]])
    return Design(rate=rate, covariates=covariates)


def solve_censoring_rate(rate: np.ndarray, target: float = CENSOR_TARGET) -> float:
    """Find nu with C ~ Exp(nu) giving the target pooled censoring share.

    For a cell with event rate ``r``, P(C < tau) = nu / (nu + r); the paper
    calibrates nu so the pooled share is about 10 percent.
    """
    def share(nu: float) -> float:
        return float(np.mean(nu / (nu + rate)) - target)

    lo, hi = 1e-12, 1.0
    while share(hi) < 0.0:
        hi *= 2.0
        if hi > 1e12:  # pragma: no cover - unreachable for positive rates
            raise RuntimeError("censoring calibration failed to bracket")
    return float(brentq(share, lo, hi, xtol=1e-14, rtol=1e-12))


def sample_cell(rate: float, nu: float, size: int, rng: np.random.Generator):
    """Right-censored draws for one unit-period cohort.

    Inverse transform exactly as in Section 4: tau = -log(W) / rate.
    """
    W = rng.uniform(size=size)
    tau = -np.log(W) / rate
    censor = rng.exponential(scale=1.0 / nu, size=size)
    observed = np.minimum(tau, censor)
    event = (tau <= censor).astype(float)
    return observed, event


def kaplan_meier(times: np.ndarray, events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival curve evaluated on ``grid`` (Appendix D).

    S(t) = prod_{t_(j) <= t} (1 - d(t_(j)) / n(t_(j))), with S(0) = 1.
    """
    event_times = np.unique(times[events > 0])
    if event_times.size == 0:
        return np.ones_like(grid)
    deaths = np.array([(times[events > 0] == e).sum() for e in event_times], dtype=float)
    at_risk = np.array([(times >= e).sum() for e in event_times], dtype=float)
    factors = np.cumprod(1.0 - deaths / at_risk)
    steps = np.concatenate([[1.0], factors])
    # number of distinct event times at or before each grid point
    return steps[np.searchsorted(event_times, grid, side="right")]


def ssc_counterfactual(
    donors_pre: np.ndarray,
    treated_pre: np.ndarray,
    donors_post: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Steps 3-4 of Section 3.4.2.

    ``pcr_weights`` computes w = V_r diag(1/s_r) U_r' y, which is the paper's
    closed form  w-hat = sum_{i<=r0} (1/s_i) v_i u_i' S-hat_{0,n}  verbatim.
    """
    weights = pcr_weights(donors_pre, treated_pre, rank)
    return donors_post @ weights


def _exponential_loglik(rate: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """Observed-data log-likelihood for exponential event times under censoring."""
    return float(np.sum(events * np.log(rate) - rate * times))


def confounder_aware_parametric(
    dgp: str,
    covariates: np.ndarray,
    times: list[np.ndarray],
    events: list[np.ndarray],
    target_covariate: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """The Table 1 comparator: true latent factors plus the correct model form.

    ``covariates`` stacks the true [V_n, U_p] for each observed cohort, and the
    fitted model is the same parametric family that generated the data, so this
    is the correctly-specified oracle the paper compares SSC against.
    """
    flat_x = np.concatenate([
        np.repeat(covariates[i][None, :], times[i].size, axis=0)
        for i in range(len(times))
    ])
    flat_t = np.concatenate(times)
    flat_d = np.concatenate(events)
    design = np.column_stack([np.ones(flat_x.shape[0]), flat_x])

    if dgp == "cox":
        def negll(theta: np.ndarray) -> float:
            return -_exponential_loglik(np.exp(design @ theta), flat_t, flat_d)

        theta0 = np.zeros(design.shape[1])
        theta0[0] = np.log(max(flat_d.sum(), 1.0) / max(flat_t.sum(), 1e-12))
        fit = minimize(negll, theta0, method="L-BFGS-B")
        rate = float(np.exp(np.concatenate([[1.0], target_covariate]) @ fit.x))
    else:
        def negll(theta: np.ndarray) -> float:
            lin = design @ theta
            if np.any(lin <= 0):
                return 1e12
            return -_exponential_loglik(lin, flat_t, flat_d)

        theta0 = np.zeros(design.shape[1])
        theta0[0] = max(flat_d.sum(), 1.0) / max(flat_t.sum(), 1e-12)
        fit = minimize(negll, theta0, method="Nelder-Mead",
                       options={"maxiter": 20000, "fatol": 1e-10, "xatol": 1e-10})
        rate = float(np.concatenate([[1.0], target_covariate]) @ fit.x)
        rate = max(rate, 1e-12)

    return np.exp(-rate * grid)


def replication(
    dgp: str,
    K: int,
    seed: int,
    rank_rule: str = "cumvar",
    fixed_rank: int | None = None,
    run_oracle: bool = True,
    horizon_mode: str = "pooled",
    design_seed: int | None = None,
    weight_mode: str = "pcr",
) -> dict:
    """One Monte-Carlo replication; returns sup-norm errors.

    ``design_seed`` separates the latent draw from the patient draw. Holding it
    fixed across ``K`` gives common random numbers, so the K-comparison in
    Figure 4 is not swamped by design variability -- under this DGP the latent
    draw moves the evaluation horizon over two orders of magnitude, which is
    far larger than the effect of K.
    """
    design_rng = np.random.default_rng(seed if design_seed is None else design_seed)
    rng = np.random.default_rng(seed)
    design = draw_design(design_rng, dgp)
    nu = solve_censoring_rate(design.rate)

    donors = [n for n in range(N_UNITS) if n != TREATED]

    # Cohorts the estimator may see: every donor in both periods, and the
    # treated unit in the pre-period only. The treated unit's post-period is
    # the counterfactual, so it is never generated.
    times: dict[tuple[int, int], np.ndarray] = {}
    events: dict[tuple[int, int], np.ndarray] = {}
    for p in range(N_PERIODS):
        for n in range(N_UNITS):
            if p == 1 and n == TREATED:
                continue
            t, d = sample_cell(design.rate[p, n], nu, K, rng)
            times[(p, n)] = t
            events[(p, n)] = d

    # tau_tilde. The paper writes quantile_0.90({T_{p,n,i}}) with all three
    # indices free, which reads as the pooled sample ("pooled" below). Under
    # the Cox design that quantile is set by the slowest cells and ranges over
    # two orders of magnitude across draws, so the alternative reading -- the
    # treated unit's own observed times -- is carried as a sensitivity.
    if horizon_mode == "pooled":
        reference = np.concatenate(list(times.values()))
    elif horizon_mode == "treated":
        reference = times[(0, TREATED)]
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown horizon_mode {horizon_mode!r}")
    horizon = float(np.quantile(reference, HORIZON_Q))
    grid = np.linspace(0.0, horizon, GRID_POINTS)

    curves = {key: kaplan_meier(times[key], events[key], grid) for key in times}

    donors_pre = np.column_stack([curves[(0, n)] for n in donors])    # (T0, N0)
    donors_post = np.column_stack([curves[(1, n)] for n in donors])
    treated_pre = curves[(0, TREATED)]

    if fixed_rank is not None:
        rank = min(fixed_rank, min(donors_pre.shape))
    elif rank_rule == "spectral":
        rank = spectral_rank(np.linalg.svd(donors_pre, compute_uv=False))
    else:
        rank = select_rank(donors_pre, method=rank_rule)
    # Numerical-rank cap. mlsynth's pcr_weights divides by s_r unguarded, so a
    # rule that asks for more directions than the donor matrix numerically has
    # returns NaN. The dispersion in the Cox design makes that reachable.
    singular = np.linalg.svd(donors_pre, compute_uv=False)
    tol = singular[0] * max(donors_pre.shape) * np.finfo(float).eps
    numerical_rank = int(np.sum(singular > max(tol, 1e-10)))
    rank = max(1, min(int(rank), numerical_rank))

    if weight_mode == "pcr":
        estimate = ssc_counterfactual(donors_pre, treated_pre, donors_post, rank)
    elif weight_mode == "simplex":
        # Not the paper's estimator. Assumption 6 asks only for a linear span,
        # but Abadie's convex-hull condition makes the output a survival
        # function by construction: a convex combination of monotone [0, 1]
        # curves is monotone in [0, 1].
        from mlsynth.utils.inferutils import _outcome_only_simplex

        weights = _outcome_only_simplex(treated_pre, donors_pre)
        estimate = donors_post @ weights
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown weight_mode {weight_mode!r}")
    truth = np.exp(-design.rate[1, TREATED] * grid)

    out = {
        "ssc": float(np.max(np.abs(estimate - truth))),
        "rank": float(rank),
        "censored_share": float(1.0 - np.concatenate(list(events.values())).mean()),
        "horizon": horizon,
        "monotone": float(np.all(np.diff(estimate) <= 1e-12)),
        "in_unit_range": float(np.all((estimate >= -1e-12) & (estimate <= 1 + 1e-12))),
    }

    if run_oracle:
        keys = list(times.keys())
        oracle = confounder_aware_parametric(
            dgp,
            [design.covariates[p, n] for (p, n) in keys],
            [times[k] for k in keys],
            [events[k] for k in keys],
            design.covariates[1, TREATED],
            grid,
        )
        out["oracle"] = float(np.max(np.abs(oracle - truth)))
    return out
