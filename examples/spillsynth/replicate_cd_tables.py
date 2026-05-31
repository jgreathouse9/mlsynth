"""Monte Carlo replication of Cao & Dowd (2023) Tables 1 and 2.

Implements the DGPs in Sections 6.1.1 (stationary common factors) and
6.1.2 (cointegrated I(1) common factors), then reports per-cell
empirical bias and variance of the treatment-effect estimator for both
vanilla Abadie 2010 SCM and the spillover-aware SP estimator from Cao &
Dowd (2023).

The Cao-Dowd Section 6 spec we replicate:

* Pre-period length :math:`T \\in \\{15, 50, 200\\}`, panel size
  :math:`N \\in \\{10, 30, 50\\}`, one post-period.
* Treatment effect :math:`\\alpha_1 = 5` on unit 1.
* Spillover effect on each affected control unit :math:`= 3`.
* Three scenarios per cell:
    * ``no_spillover`` -- no spillover in DGP; SP runs *conservatively*
      declaring ``round((N-1)/3)`` controls as potentially affected.
    * ``concentrated`` -- ``round((N-1)/3)`` controls actually receive
      spillover; SP declares the same set (correctly specified).
    * ``spreadout`` -- ``round(2*(N-1)/3)`` controls actually receive
      spillover; SP declares the same set.
* 1000 Monte Carlo replications per cell.
* Loadings :math:`\\mu_i` drawn once and fixed across reps:
    * Table 1 (stationary): each entry iid Uniform[0, 1].
    * Table 2 (I(1)): Condition CO. :math:`\\mu_1 = e_1`,
      :math:`\\mu_2 = e_2`, :math:`\\mu_3 = e_1`, :math:`\\mu_4 = e_2`;
      for :math:`j \\geq 5` draw Uniform[0, 1]^3 and normalize so the
      three entries sum to one.

Usage::

    python -m examples.spillsynth.replicate_cd_tables --reps 100 --table 1
    python -m examples.spillsynth.replicate_cd_tables --reps 1000 --table both

For pilot/smoke-testing pass ``--cells "(10,15),(10,50)"`` to restrict
the grid.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from mlsynth import SPILLSYNTH


# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------


def loadings_stationary(N: int, rng: np.random.Generator) -> np.ndarray:
    """Each entry of ``mu_i`` is iid Uniform[0, 1]. Fixed across reps."""
    return rng.uniform(0.0, 1.0, size=(N, 3))


def loadings_I1(N: int, rng: np.random.Generator) -> np.ndarray:
    """Condition CO loadings.

    ``mu_1 = (1,0,0)``, ``mu_2 = (0,1,0)``, ``mu_3 = (1,0,0)``,
    ``mu_4 = (0,1,0)``; for ``j >= 5`` draw Uniform[0, 1]^3 and
    normalize so entries sum to one.
    """
    if N < 4:
        raise ValueError("I(1) DGP requires N >= 4 to seed condition CO.")
    mu = np.zeros((N, 3))
    mu[0] = [1.0, 0.0, 0.0]
    mu[1] = [0.0, 1.0, 0.0]
    mu[2] = [1.0, 0.0, 0.0]
    mu[3] = [0.0, 1.0, 0.0]
    if N > 4:
        v = rng.uniform(0.0, 1.0, size=(N - 4, 3))
        mu[4:] = v / v.sum(axis=1, keepdims=True)
    return mu


def dgp_stationary(
    N: int, T: int, mu: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Section 6.1.1 DGP. Returns potential outcomes ``Y(0)`` of shape ``(N, T+1)``."""
    Ttot = T + 1
    eta = np.zeros(Ttot)
    lam = np.zeros((Ttot, 3))
    nu0 = rng.standard_normal(Ttot)
    nu1 = rng.standard_normal(Ttot)
    nu2 = rng.standard_normal(Ttot)
    nu3 = rng.standard_normal(Ttot)
    # period 0 initial draws (AR(1) processes need a starting point)
    eta[0] = nu0[0]
    lam[0, 0] = nu1[0]
    lam[0, 1] = 1.0 + nu2[0]
    lam[0, 2] = nu3[0]
    for t in range(1, Ttot):
        eta[t] = 1.0 + 0.5 * eta[t - 1] + nu0[t]
        lam[t, 0] = 0.5 * lam[t - 1, 0] + nu1[t]
        lam[t, 1] = 1.0 + nu2[t] + 0.5 * nu2[t - 1]
        lam[t, 2] = 0.5 * lam[t - 1, 2] + nu3[t] + 0.5 * nu3[t - 1]
    eps = rng.standard_normal((N, Ttot))
    Y0 = eta[None, :] + mu @ lam.T + eps
    return Y0


def dgp_I1(
    N: int, T: int, mu: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Section 6.1.2 DGP (cointegrated I(1) factors)."""
    Ttot = T + 1
    lam = np.zeros((Ttot, 3))
    nu1 = rng.standard_normal(Ttot)
    nu2 = rng.standard_normal(Ttot)
    nu3 = rng.standard_normal(Ttot)
    lam[0, 0] = nu1[0]
    lam[0, 1] = nu2[0]
    lam[0, 2] = nu3[0]
    for t in range(1, Ttot):
        lam[t, 0] = lam[t - 1, 0] + 0.5 * nu1[t]
        lam[t, 1] = lam[t - 1, 1] + 0.5 * nu2[t]
        lam[t, 2] = 0.5 * lam[t - 1, 2] + nu3[t]
    eps = rng.standard_normal((N, Ttot))
    Y0 = mu @ lam.T + eps
    return Y0


# ---------------------------------------------------------------------------
# Original Abadie 2010 SCM (no intercept) -- the "SCM" column in Tables 1-2
# ---------------------------------------------------------------------------


def vanilla_abadie_scm_weights(
    Y_donors_pre: np.ndarray, y_treated_pre: np.ndarray,
) -> np.ndarray:
    """Abadie 2010 SCM: simplex weights on raw levels, no intercept.

    Parameters
    ----------
    Y_donors_pre : np.ndarray
        Shape ``(T0, K)`` -- donor units' pre-treatment outcomes.
    y_treated_pre : np.ndarray
        Length-``T0`` treated-unit pre-treatment outcomes.

    Returns
    -------
    w : np.ndarray
        Length-``K`` simplex weights.
    """
    T0, K = Y_donors_pre.shape
    w = cp.Variable(K, nonneg=True)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(y_treated_pre - Y_donors_pre @ w)),
        [cp.sum(w) == 1],
    )
    prob.solve(solver="CLARABEL")
    w_val = np.asarray(w.value).flatten()
    w_val = np.clip(w_val, 0.0, None)
    s = w_val.sum()
    if s <= 0:
        raise RuntimeError("Abadie SCM: degenerate (all-zero) weights.")
    return w_val / s


# ---------------------------------------------------------------------------
# Monte Carlo driver
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    N: int
    T: int
    scenario: str
    scm_bias: float
    scm_var: float
    sp_bias: float
    sp_var: float
    seconds: float


def _affected_counts(N: int) -> Tuple[int, int]:
    """``round((N-1)/3)`` for concentrated, ``round(2(N-1)/3)`` for spreadout."""
    n_conc = int(round((N - 1) / 3))
    n_spr = int(round(2 * (N - 1) / 3))
    return n_conc, n_spr


def _scenario_setup(N: int, scenario: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(alpha, declared_idx)`` for the given scenario.

    ``alpha`` is length-``N``: the true post-period effect vector. The
    treated unit is index 0 with ``alpha[0] = 5``; affected controls
    receive ``alpha[i] = 3``.

    ``declared_idx`` is the integer index of the controls the
    econometrician declares as potentially affected (passed into A).
    """
    n_conc, n_spr = _affected_counts(N)
    alpha = np.zeros(N)
    alpha[0] = 5.0
    if scenario == "no_spillover":
        # DGP has no spillover; SP declares 1/3 of controls (conservative).
        declared_idx = np.arange(1, 1 + n_conc, dtype=int)
    elif scenario == "concentrated":
        affected = np.arange(1, 1 + n_conc, dtype=int)
        alpha[affected] = 3.0
        declared_idx = affected
    elif scenario == "spreadout":
        affected = np.arange(1, 1 + n_spr, dtype=int)
        alpha[affected] = 3.0
        declared_idx = affected
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return alpha, declared_idx


def _panel_to_df(
    Y: np.ndarray, T0: int,
) -> "tuple[pd.DataFrame, list[str]]":
    """Pack an ``(N, Ttot)`` outcome matrix into the long-form DataFrame
    SPILLSYNTH consumes, with treatment indicator on row 0 from ``T0``.
    """
    N, Ttot = Y.shape
    unit_labels = [f"u{i}" for i in range(N)]
    rows = []
    for i in range(N):
        for t in range(Ttot):
            rows.append({
                "unit": unit_labels[i],
                "year": t,
                "y": float(Y[i, t]),
                "treat": int(i == 0 and t >= T0),
            })
    return pd.DataFrame(rows), unit_labels


def _one_rep(
    Y0: np.ndarray, alpha: np.ndarray, declared_idx: np.ndarray,
) -> Tuple[float, float]:
    """One Monte Carlo replication. Returns ``(scm_bias, sp_bias)`` for unit 1.

    **Path-B contract compliance.** The SP estimator is invoked through
    the public ``SPILLSYNTH(config).fit()`` API so the entire
    config-validation -> panel-prep -> estimation -> inference pipeline
    is exercised end-to-end. The vanilla Abadie 2010 SCM column is
    computed via a tiny independent helper (no intercept; this is the
    paper's comparator and is NOT what mlsynth's ``res.att_scm``
    returns, which is the Ferman-Pinto-demeaned variant).
    """
    N, Ttot = Y0.shape
    T0 = Ttot - 1
    Y = Y0.copy()
    Y[:, -1] = Y[:, -1] + alpha                       # add effects at t = T+1
    Y_pre = Y[:, :T0]
    Y_post = Y[:, T0:]                                # (N, 1)
    y_treat_post = float(Y[0, -1])

    # Abadie SCM (no intercept) on raw levels -- paper's comparator.
    w_scm = vanilla_abadie_scm_weights(Y_pre[1:].T, Y_pre[0])
    cf_scm = float(Y_post[1:, 0] @ w_scm)
    scm_bias = (y_treat_post - cf_scm) - alpha[0]

    # SP via the public estimator.
    df, unit_labels = _panel_to_df(Y, T0=T0)
    affected_labels = [unit_labels[i] for i in declared_idx]
    res = SPILLSYNTH({
        "df": df, "outcome": "y", "treat": "treat",
        "unitid": "unit", "time": "year",
        "method": "cd",
        "affected_units": affected_labels,
        "display_graphs": False,
    }).fit()
    # T1 = 1 by construction -- one post-period.
    sp_bias = float(res.cd.alpha[0, 0]) - alpha[0]
    return scm_bias, sp_bias


def run_cell(
    *, dgp_fn, loadings_fn, N: int, T: int, scenario: str,
    reps: int, seed: int,
) -> CellResult:
    rng_master = np.random.default_rng(seed)
    mu = loadings_fn(N, rng_master)
    alpha, declared_idx = _scenario_setup(N, scenario)

    biases_scm = np.empty(reps)
    biases_sp = np.empty(reps)
    t0 = time.time()
    for r in range(reps):
        rng_r = np.random.default_rng(seed * 1_000_003 + r)
        Y0 = dgp_fn(N, T, mu, rng_r)
        sb, pb = _one_rep(Y0, alpha, declared_idx)
        biases_scm[r] = sb
        biases_sp[r] = pb
    elapsed = time.time() - t0

    return CellResult(
        N=N, T=T, scenario=scenario,
        scm_bias=float(biases_scm.mean()),
        scm_var=float(biases_scm.var(ddof=1)),
        sp_bias=float(biases_sp.mean()),
        sp_var=float(biases_sp.var(ddof=1)),
        seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


DEFAULT_NS = (10, 30, 50)
DEFAULT_TS = (15, 50, 200)
SCENARIOS = ("no_spillover", "concentrated", "spreadout")


def _parse_cells(s: str) -> List[Tuple[int, int]]:
    s = s.strip()
    if not s:
        return [(N, T) for N in DEFAULT_NS for T in DEFAULT_TS]
    pairs = []
    for chunk in s.replace(" ", "").split("),("):
        chunk = chunk.strip("()")
        N, T = chunk.split(",")
        pairs.append((int(N), int(T)))
    return pairs


def _format_table(results: Sequence[CellResult], title: str) -> str:
    by_scenario: dict[str, dict[Tuple[int, int], CellResult]] = {
        sc: {} for sc in SCENARIOS
    }
    Ns_seen, Ts_seen = set(), set()
    for r in results:
        by_scenario[r.scenario][(r.N, r.T)] = r
        Ns_seen.add(r.N); Ts_seen.add(r.T)
    Ns = sorted(Ns_seen)
    Ts = sorted(Ts_seen)

    lines = [title, "=" * len(title), ""]
    header_top = " " * 14 + "  ".join(f"  N = {N:<14}" for N in Ns)
    header_t = " " * 14 + "  ".join(
        "  ".join(f"T={T:<6}" for T in Ts) for _ in Ns
    )
    lines.append(header_top)
    lines.append(header_t)
    for sc_label, sc in [
        ("No spillover effects", "no_spillover"),
        ("Concentrated spillover effects", "concentrated"),
        ("Spreadout spillover effects", "spreadout"),
    ]:
        lines.append(sc_label)
        for est in ("SCM", "SP"):
            row_bias = [f"  {est:<3}"]
            row_var = ["     "]
            for N in Ns:
                for T in Ts:
                    r = by_scenario[sc].get((N, T))
                    if r is None:
                        row_bias.append(f"  {'--':>8}")
                        row_var.append(f"  {'':>8}")
                        continue
                    bias = r.scm_bias if est == "SCM" else r.sp_bias
                    var = r.scm_var if est == "SCM" else r.sp_var
                    row_bias.append(f"  {bias:+8.3f}")
                    row_var.append(f"  ({np.sqrt(var):.3f}) ")
            lines.append("".join(row_bias))
            lines.append("".join(row_var))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20230308)
    parser.add_argument("--table", choices=("1", "2", "both"), default="both")
    parser.add_argument(
        "--cells", type=str, default="",
        help="Comma-list of (N,T) pairs e.g. '(10,15),(30,50)'. Empty = full grid.",
    )
    args = parser.parse_args()

    cells = _parse_cells(args.cells)
    tables_to_run = []
    if args.table in ("1", "both"):
        tables_to_run.append(("Table 1: stationary common factors",
                              dgp_stationary, loadings_stationary))
    if args.table in ("2", "both"):
        tables_to_run.append(("Table 2: I(1) common factors",
                              dgp_I1, loadings_I1))

    for title, dgp_fn, loadings_fn in tables_to_run:
        print(f"\n>>> {title}  ({args.reps} reps/cell)")
        results = []
        for N, T in cells:
            for sc in SCENARIOS:
                r = run_cell(
                    dgp_fn=dgp_fn, loadings_fn=loadings_fn,
                    N=N, T=T, scenario=sc,
                    reps=args.reps, seed=args.seed,
                )
                results.append(r)
                print(
                    f"  cell N={N:<3} T={T:<3} {sc:<14}  "
                    f"SCM bias={r.scm_bias:+.3f} ({np.sqrt(r.scm_var):.3f})  "
                    f"SP  bias={r.sp_bias:+.3f} ({np.sqrt(r.sp_var):.3f})  "
                    f"[{r.seconds:.1f}s]"
                )
        print()
        print(_format_table(results, title))


if __name__ == "__main__":
    main()
