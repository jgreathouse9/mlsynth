"""The MSCMT inner solver: exactness against cvxpy, and the work it costs.

Cross-validation of a solver rather than an estimator. The MSCMT backend prices
tens of thousands of candidate predictor weightings, each needing donor weights
that minimise the ``V``-weighted predictor discrepancy on the simplex, and it
solves them with a batched active set on the Gram form
(:mod:`mlsynth.utils.bilevel.minnorm`). Two things about that solver need a
durable record, and neither belongs in a unit test.

Exactness. The reference is cvxpy's interior-point solver (CLARABEL) on the same
programs, an implementation that shares no code with the active set. They are
compared on the objective, not the weights: the donor discrepancy matrix ``R`` is
``13 x 17`` on this specification, so it is rank deficient and the optimum is
generally a face, where which point a solver returns is not identified by the
data. The gap pinned below is the active set's objective excess over cvxpy's,
divided by ``trace(G)`` so it is free of the problem's scale.

Work. The active set's cost is its iteration count, which is machine independent
and so pins as a number, where wall-clock would flake. It scales with how many
donors carry the fit: across the eighteen Basque panels (the treated unit and
each donor cast as treated in turn) it runs from nine iterations where two donors
carry the fit to thirty-three where six do.

Search budget. The outer search stops when the population's spread in pre-fit
MSPE falls below ``mscmt_tol`` of its mean, and the default (``1e-6``) is a
statement about estimate precision: the donor weights settle by generation 93 and
the following 120 generations move them by 1e-8. The case pins the consequence --
what the default's estimate differs from an exhaustive search's -- so loosening
the default until it starts to matter fails here.

Data ship as ``basedata/basque_mscmt.csv``; the specification is Abadie &
Gardeazabal's (2003), the thirteen predictors of the MSCMT vignette with the
outcome fit over 1960-1969.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "basque_mscmt.csv")

_W16 = (1964, 1969)
_W19 = (1961, 1969)
_COVS = ["school.illit", "school.prim", "school.med", "school.higher", "invest",
         "gdpcap", "sec.agriculture", "sec.energy", "sec.industry",
         "sec.construction", "sec.services.venta", "sec.services.nonventa",
         "popdens"]
_WINDOWS = {
    "school.illit": _W16, "school.prim": _W16, "school.med": _W16,
    "school.higher": _W16, "invest": _W16, "gdpcap": (1960, 1969),
    "sec.agriculture": _W19, "sec.energy": _W19, "sec.industry": _W19,
    "sec.construction": _W19, "sec.services.venta": _W19,
    "sec.services.nonventa": _W19, "popdens": (1969, 1969),
}
# Candidate predictor weightings, drawn over the search's own box
# (log10 V in [-8, 0]^13). Fixed seed: the pins below are exact counts.
_N_CANDIDATES = 250
_SEED = 0


def _panel() -> pd.DataFrame:
    d = pd.read_csv(os.path.abspath(_DATA))
    d["treat"] = ((d.regionname == "Basque Country (Pais Vasco)")
                  & (d.year >= 1970)).astype(int)
    return d


def _predictors(d: pd.DataFrame) -> np.ndarray:
    """The scaled ``(13, 18)`` predictor matrix: treated unit then donors."""
    from mlsynth.utils.datautils import dataprep
    from mlsynth.utils.vanillasc_helpers.pipeline import (_covariate_means,
                                                          _scale_unit_variance)
    prep = dataprep(df=d, unit_id_column_name="regionname",
                    time_period_column_name="year", outcome_column_name="gdpcap",
                    treatment_indicator_column_name="treat")
    labels = np.asarray(prep["time_labels"])
    pre = int(prep["pre_periods"])
    units = [prep["treated_unit_name"], *prep["donor_names"]]
    return _scale_unit_variance(
        _covariate_means(d, units, _COVS, _WINDOWS, list(labels[:pre]),
                         "regionname", "year"))


def _grams(X1: np.ndarray, X0: np.ndarray, logv: np.ndarray) -> np.ndarray:
    """One Gram per candidate weighting: ``G(V) = sum_p v_p r_p r_p'``."""
    R = np.ascontiguousarray(X1[:, None] - X0)
    K, J = R.shape
    rank_one = np.einsum("kj,kl->kjl", R, R).reshape(K, J * J)
    return (np.power(10.0, logv).T @ rank_one).reshape(-1, J, J)


def _cvxpy_objective(G: np.ndarray) -> float:
    """Reference optimum of ``min w'Gw`` on the simplex, via CLARABEL."""
    import cvxpy as cp

    w = cp.Variable(G.shape[0])
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(G))),
                      [w >= 0, cp.sum(w) == 1])
    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        return float("nan")
    v = np.clip(np.asarray(w.value, dtype=float).ravel(), 0.0, None)
    v = v / v.sum()
    return float(v @ G @ v)


def run() -> dict:
    try:
        import cvxpy  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional reference
        raise BenchmarkSkipped(f"cvxpy is unavailable ({exc})")

    from mlsynth import VanillaSC
    from mlsynth.utils.bilevel.minnorm import solve_simplex_minnorm_batch

    d = _panel()
    Xs = _predictors(d)
    K = Xs.shape[0]
    logv = np.random.default_rng(_SEED).uniform(-8.0, 0.0,
                                                size=(K, _N_CANDIDATES))

    # --- exactness and work, on the treated panel -------------------------- #
    G = _grams(Xs[:, 0], Xs[:, 1:], logv)
    W, info = solve_simplex_minnorm_batch(G, return_info=True)
    ours = np.einsum("sj,sjk,sk->s", W, G, W)
    trace = np.einsum("sjj->s", G)
    # cvxpy on every candidate is slow; a fixed stride is a fair sample and
    # keeps the case under a minute.
    sample = np.arange(0, _N_CANDIDATES, 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = np.array([_cvxpy_objective(G[i]) for i in sample])
    finite = np.isfinite(ref)
    excess = ((ours[sample] - ref) / trace[sample])[finite]

    feasible = float(max(abs(W.sum(axis=1) - 1.0).max(), -W.min()))
    treated_iters = int(info["iterations"])
    all_certified = float(info["converged"].all())

    # --- work across every panel the in-space placebo refits --------------- #
    per_panel = []
    for j in range(Xs.shape[1] - 1):
        others = [k for k in range(Xs.shape[1] - 1) if k != j]
        Gj = _grams(Xs[:, 1 + j], Xs[:, [1 + k for k in others]], logv)
        Wj, ij = solve_simplex_minnorm_batch(Gj, return_info=True)
        per_panel.append((int(ij["iterations"]), float((Wj > 1e-9).sum(1).mean())))
    iters = np.array([p[0] for p in per_panel])
    support = np.array([p[1] for p in per_panel])

    # --- the search budget the default tolerance buys ---------------------- #
    cfg = dict(df=d, outcome="gdpcap", treat="treat", unitid="regionname",
               time="year", backend="mscmt", covariates=_COVS,
               covariate_windows=_WINDOWS, fit_window=(1960, 1969), seed=0,
               inference=False, display_graphs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = VanillaSC(dict(cfg)).fit()
        exhaustive = VanillaSC(dict(cfg, mscmt_tol=1e-10)).fit()
    w_def = np.array([float(v) for v in default.weights.donor_weights.values()])
    w_exh = np.array([float(v) for v in exhaustive.weights.donor_weights.values()])

    return {
        # exactness: the active set never lands above cvxpy's optimum
        "max_objective_excess_over_cvxpy": float(np.max(excess)),
        "mean_objective_excess_over_cvxpy": float(np.mean(excess)),
        "max_simplex_violation": feasible,
        "all_candidates_certified": all_certified,
        # work: iterations, machine independent
        "treated_panel_iterations": float(treated_iters),
        "max_panel_iterations": float(iters.max()),
        "mean_panel_iterations": float(iters.mean()),
        # the geometry that sets the work
        "iterations_support_correlation": float(np.corrcoef(iters, support)[0, 1]),
        # the search budget: what the default tolerance costs in estimate terms
        "default_vs_exhaustive_max_weight_gap": float(np.abs(w_def - w_exh).max()),
        "default_vs_exhaustive_att_gap": float(abs(default.effects.att
                                                   - exhaustive.effects.att)),
    }


EXPECTED = {
    # The active set is exact: it never finishes above the interior-point
    # optimum, and sits below it by the amount cvxpy's own tolerance leaves on
    # the table. Scale-free (divided by trace(G)).
    "max_objective_excess_over_cvxpy": (0.0, 1e-9),
    "mean_objective_excess_over_cvxpy": (0.0, 1e-9),
    "max_simplex_violation": (0.0, 1e-9),
    "all_candidates_certified": (1.0, 0.0),
    # Work bounds. Exact counts for this data and candidate set; the tolerance
    # absorbs BLAS/LAPACK differences in the corral solves, not an algorithm
    # change, which would move these by much more.
    "treated_panel_iterations": (11.0, 4.0),
    "max_panel_iterations": (28.0, 8.0),
    "mean_panel_iterations": (17.6, 4.0),
    # Iterations track the size of the optimal support, which is what makes a
    # panel expensive -- not conditioning, and not the donor count. Every panel
    # here has the same 16 donors and a discrepancy matrix of the same rank, and
    # the work still varies threefold.
    "iterations_support_correlation": (0.81, 0.15),
    # The default tolerance's estimate cost, which is the reason it was chosen.
    # Three orders below the four decimals the MSCMT replication compares to.
    "default_vs_exhaustive_max_weight_gap": (0.0, 1e-4),
    "default_vs_exhaustive_att_gap": (0.0, 1e-4),
}
