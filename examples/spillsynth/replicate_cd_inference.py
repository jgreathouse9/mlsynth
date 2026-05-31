"""Monte Carlo replication of Cao & Dowd (2023) Tables 3 and 4.

Tables 3 and 4 report empirical rejection rates of three test
procedures for the treatment-effect hypothesis :math:`H_0: \\alpha_1 = 0`
on the treated unit:

* **Placebo test** (Abadie, Diamond & Hainmueller 2010) -- rank the
  treated unit's post-period residual :math:`\\widehat u_{1, T+1}`
  against the distribution of donor residuals
  :math:`\\{\\widehat u_{j, T+1}\\}_{j \\geq 2}`. Reject if the treated
  unit's :math:`|\\widehat u|` exceeds the :math:`(1-\\tau)`-quantile.

* **Andrews' P-test** (Andrews 2003; Section 4.1 of Cao-Dowd) --
  compare :math:`\\widehat u_{1, T+1}^2` against the empirical CDF of
  :math:`\\{\\widehat u_{1, t}^2\\}_{t=1}^T` (the treated unit's own
  pre-period squared residuals). Reject if the post statistic is
  above the :math:`(1-\\tau)`-quantile.

* **SP test** (this paper, Section 4.2) -- the spillover-adjusted
  P-test from :mod:`mlsynth.utils.spillsynth_helpers.cd.inference`,
  with :math:`C = e_1^\\prime`, :math:`d = 0`, :math:`W_T = I`.

The DGP is the stationary factor model from Section 6.1.1 (the same as
Table 1). Table 3 has :math:`\\alpha_1 = 0` so rejection probability
should converge to the nominal level :math:`\\tau = 5\\%` -- it measures
**size**. Table 4 has :math:`\\alpha_1 = 5` so rejection probability
measures **power**.

Usage::

    python -m examples.spillsynth.replicate_cd_inference --reps 1000
    python -m examples.spillsynth.replicate_cd_inference --reps 500 --cells "(10,15),(30,50)"

Reference: Cao, J., & Dowd, C. (2023), "Estimation and Inference for
Synthetic Control Methods with Spillover Effects", Sections 4 and 6.2.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from mlsynth.utils.spillsynth_helpers.cd.inference import (
    G_matrix, compute_pre_residuals, p_test,
)
from mlsynth.utils.spillsynth_helpers.cd.scm_core import fit_leave_one_out_sc
from mlsynth.utils.spillsynth_helpers.cd.estimation import build_M, sp_estimate
from mlsynth.utils.spillsynth_helpers.setup import build_A_example3

# Re-use the DGPs and scenario setup from the bias replication script.
from examples.spillsynth.replicate_cd_tables import (
    _affected_counts, dgp_stationary, loadings_stationary,
)


SCENARIOS = ("no_spillover", "concentrated", "spreadout")
DEFAULT_NS = (10, 30, 50)
DEFAULT_TS = (15, 50, 200)


# ---------------------------------------------------------------------------
# Three inference procedures
# ---------------------------------------------------------------------------


def _placebo_reject(
    Y_pre: np.ndarray, Y_post: np.ndarray, a: np.ndarray, B: np.ndarray,
    tau: float = 0.05,
) -> bool:
    """Abadie-2010-style placebo test on the post-period residual.

    Computes :math:`\\widehat u_{i, T+1} = y_{i, T+1} - (a_i + b_i' Y_{T+1})`
    for every unit using its leave-one-out SCM fit, then rejects if the
    treated unit's :math:`|\\widehat u_{1, T+1}|` is in the top :math:`\\tau`
    of the across-unit distribution.
    """
    N = Y_pre.shape[0]
    # u_{i, T+1} = (I - B) Y_post - a, single column
    u_post = (np.eye(N) - B) @ Y_post[:, 0] - a               # (N,)
    abs_u = np.abs(u_post)
    cutoff = np.quantile(abs_u, 1 - tau)
    return bool(abs_u[0] > cutoff)


def _andrews_reject(
    Y_pre: np.ndarray, Y_post: np.ndarray, a: np.ndarray, B: np.ndarray,
    tau: float = 0.05,
) -> bool:
    """Andrews-2003 P-test on the treated unit alone.

    Reference distribution: pre-period squared residuals of the treated
    unit only, :math:`\\{\\widehat u_{1, t}^2\\}_{t=1}^T`.
    """
    N = Y_pre.shape[0]
    u_pre = (np.eye(N) - B) @ Y_pre - a[:, None]              # (N, T0)
    u_post = (np.eye(N) - B) @ Y_post[:, 0] - a               # (N,)
    P_pre = u_pre[0] ** 2
    P_post = float(u_post[0] ** 2)
    cutoff = np.quantile(P_pre, 1 - tau)
    return bool(P_post > cutoff)


def _sp_reject(
    Y_pre: np.ndarray, Y_post: np.ndarray, a: np.ndarray, B: np.ndarray,
    A: np.ndarray, alpha_hat: np.ndarray, tau: float = 0.05,
) -> bool:
    """Cao-Dowd Section 4.2 P-test for ``H_0: alpha_1(T+1) = 0``."""
    N = Y_pre.shape[0]
    U_pre = compute_pre_residuals(Y_pre, a, B)
    G_hat = G_matrix(A, B)
    e_treat = np.zeros((1, N)); e_treat[0, 0] = 1.0
    res = p_test(
        alpha_hat=alpha_hat, U_pre=U_pre, G_hat=G_hat, C=e_treat,
    )
    return bool(res.P_post[0] > np.quantile(res.P_pre, 1 - tau))


# ---------------------------------------------------------------------------
# Per-rep / per-cell drivers
# ---------------------------------------------------------------------------


def _one_rep_inference(
    Y0: np.ndarray, alpha: np.ndarray, declared_idx: np.ndarray,
) -> Tuple[bool, bool, bool]:
    """One Monte Carlo replication. Returns (placebo, andrews, sp) reject flags."""
    N, Ttot = Y0.shape
    T0 = Ttot - 1
    Y = Y0.copy()
    Y[:, -1] = Y[:, -1] + alpha
    Y_pre = Y[:, :T0]
    Y_post = Y[:, T0:]

    a, B = fit_leave_one_out_sc(Y_pre)
    A = build_A_example3(N, len(declared_idx))
    M = build_M(B)
    _gamma, alpha_hat, _cond = sp_estimate(Y_post, a=a, B=B, M=M, A=A)

    placebo = _placebo_reject(Y_pre, Y_post, a, B)
    andrews = _andrews_reject(Y_pre, Y_post, a, B)
    sp = _sp_reject(Y_pre, Y_post, a, B, A, alpha_hat)
    return placebo, andrews, sp


def _scenario_setup_inference(
    N: int, scenario: str, alpha_1: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Like ``replicate_cd_tables._scenario_setup`` but with configurable alpha_1.

    Table 3 uses ``alpha_1 = 0`` (null is true). Table 4 uses
    ``alpha_1 = 5`` (alternative).
    """
    n_conc, n_spr = _affected_counts(N)
    alpha = np.zeros(N)
    alpha[0] = alpha_1
    if scenario == "no_spillover":
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
        raise ValueError(scenario)
    return alpha, declared_idx


@dataclass
class CellRejectionRates:
    N: int
    T: int
    scenario: str
    alpha_1: float
    placebo: float
    andrews: float
    sp: float
    reps: int
    seconds: float


def run_cell_inference(
    *, N: int, T: int, scenario: str, alpha_1: float, reps: int, seed: int,
) -> CellRejectionRates:
    rng_master = np.random.default_rng(seed)
    mu = loadings_stationary(N, rng_master)
    alpha, declared_idx = _scenario_setup_inference(N, scenario, alpha_1)

    counts = np.zeros(3, dtype=int)
    t0 = time.time()
    for r in range(reps):
        rng_r = np.random.default_rng(seed * 1_000_003 + r)
        Y0 = dgp_stationary(N, T, mu, rng_r)
        pl, an, sp = _one_rep_inference(Y0, alpha, declared_idx)
        counts += np.array([pl, an, sp], dtype=int)
    elapsed = time.time() - t0
    rates = counts / reps
    return CellRejectionRates(
        N=N, T=T, scenario=scenario, alpha_1=alpha_1,
        placebo=float(rates[0]), andrews=float(rates[1]), sp=float(rates[2]),
        reps=reps, seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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


def _format_inference_table(
    results: Sequence[CellRejectionRates], title: str,
) -> str:
    by_scenario: dict[str, dict[Tuple[int, int], CellRejectionRates]] = {
        sc: {} for sc in SCENARIOS
    }
    Ns, Ts = set(), set()
    for r in results:
        by_scenario[r.scenario][(r.N, r.T)] = r
        Ns.add(r.N); Ts.add(r.T)
    Ns_l = sorted(Ns); Ts_l = sorted(Ts)

    lines = [title, "=" * len(title), ""]
    cols = " " * 12 + "  ".join(
        "  ".join(f"T={T:<6}" for T in Ts_l) + "    "
        for _ in Ns_l
    )
    Nhdr = " " * 12 + "  ".join(
        f"  N = {N:<{8*len(Ts_l)+6*(len(Ts_l)-1)}}" for N in Ns_l
    )
    lines.append(Nhdr)
    lines.append(cols)
    for sc_label, sc in [
        ("No spillover effects", "no_spillover"),
        ("Concentrated spillover effects", "concentrated"),
        ("Spreadout spillover effects", "spreadout"),
    ]:
        lines.append(sc_label)
        for est, attr in [("Placebo", "placebo"),
                          ("Andrews", "andrews"),
                          ("SP",      "sp")]:
            row = [f"  {est:<7}"]
            for N in Ns_l:
                for T in Ts_l:
                    r = by_scenario[sc].get((N, T))
                    if r is None:
                        row.append(f"  {'--':>6}  ")
                    else:
                        row.append(f"  {getattr(r, attr):.3f}  ")
            lines.append("".join(row))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20230308)
    parser.add_argument(
        "--tables", default="3,4",
        help="Comma list: '3' for size (alpha1=0), '4' for power (alpha1=5).",
    )
    parser.add_argument(
        "--cells", type=str, default="",
        help="Comma-list of (N,T) pairs e.g. '(10,15),(30,50)'. Empty = full grid.",
    )
    args = parser.parse_args()

    cells = _parse_cells(args.cells)
    runs = []
    for t in args.tables.split(","):
        t = t.strip()
        if t == "3":
            runs.append((3, 0.0, "Table 3: rejection rates under H_0: alpha_1 = 0 (size)"))
        elif t == "4":
            runs.append((4, 5.0, "Table 4: rejection rates under alpha_1 = 5 (power)"))
        else:
            raise ValueError(f"unknown table {t!r}")

    for tbl, alpha_1, title in runs:
        print(f"\n>>> {title}  ({args.reps} reps/cell)")
        results = []
        for N, T in cells:
            for sc in SCENARIOS:
                r = run_cell_inference(
                    N=N, T=T, scenario=sc, alpha_1=alpha_1,
                    reps=args.reps, seed=args.seed,
                )
                results.append(r)
                print(
                    f"  cell N={N:<3} T={T:<3} {sc:<14}  "
                    f"placebo={r.placebo:.3f}  andrews={r.andrews:.3f}  "
                    f"sp={r.sp:.3f}   [{r.seconds:.1f}s]"
                )
        print()
        print(_format_inference_table(results, title))


if __name__ == "__main__":
    main()
