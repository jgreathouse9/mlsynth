"""Solving allsynth Example 12's 566 synthetic controls as six matrix problems.

The stacked design in ``sdid_allsynth_walmart.py`` fits one synthetic control per
treated county -- 566 independent simplex solves, about a minute. Almost all of
that work is redundant, and an event-time reshape shows why.

Write ``e = year - super_year``. A treated county's normalized series is its own
row of the event-time cube; a donor, having no adoption year of its own, gets one
normalized series *per cohort* -- because allsynth's ``transform(..., normalize)``
indexes every series to the treated county's final pre-treatment year, and that
base year is a property of the cohort, not of the donor. So the donor design
matrix takes only as many distinct values as there are cohorts. Six, here, not
566. Every treated county in a cohort is solving

    min_w  || A_g w - b_j ||^2      s.t.  w >= 0,  1'w = 1

against the *same* ``A_g``, differing only in the right-hand side ``b_j``. That is
a multiple-right-hand-side problem, and the accelerated projected-gradient
iteration batches over it exactly: form the Gram matrix ``A_g' A_g`` and the
Lipschitz constant once per cohort, then every iteration is one
``(39, 39) @ (39, N_g)`` matmul plus a column-wise projection onto the simplex.
Counterfactuals are a second matmul, ``D_g W``, for the whole cohort at once.

Section 3 is what the exercise turned up on the way, and it matters more than the
speed. Each cohort has 5 to 10 pre-treatment periods against 39 donors, so
``A_g`` has rank equal to its row count and a null space of dimension 29 or more.
Moving ``w`` along that null space changes the post-treatment prediction without
changing the pre-treatment fit at all: the per-county weights are not identified,
and two solvers that both reach the optimum can disagree by several percentage
points on a single county. The aggregate is unaffected -- that is why the batched
and looped event paths agree to 0.04pp while individual counties differ by up to
8pp -- but no per-county weight vector here should be read as *the* synthetic
control.

Section 4 notes the same trick for SDID, where more is shared than the design
matrix: the time weights and the ridge penalty are fit on the donors alone, so
they are identical for every treated county in a cohort.

Takes about 80 seconds, nearly all of it the looped reference fit.
Run from the repository root:

    python examples/stacked_sc_vectorized.py
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from mlsynth import SDID, VanillaSC

PANEL = "basedata/allsynth_walmart.parquet"
K = 5


# ------------------------------------------------------------- batched primitives
def project_simplex_cols(V: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Project every column of ``V`` onto ``{w >= 0, sum(w) = z}``.

    The column-wise form of the exact sort-based algorithm (Held, Wolfe & Crowder
    1974; Duchi et al. 2008) that
    :func:`mlsynth.utils.bilevel.simplex.project_simplex` applies to one vector.
    The threshold ``theta`` is found per column by a single argmax over the
    reversed condition array rather than a Python loop.
    """
    n, k = V.shape
    U = -np.sort(-V, axis=0)                        # descending within each column
    cssv = np.cumsum(U, axis=0) - z
    ind = np.arange(1, n + 1)[:, None]
    cond = (U - cssv / ind) > 0                     # a True block then all False
    rho = n - 1 - np.argmax(cond[::-1], axis=0)     # index of the last True
    theta = cssv[rho, np.arange(k)] / (rho + 1.0)
    return np.maximum(V - theta, 0.0)


def simplex_lstsq_batch(A, B, *, ridge=0.0, tol=1e-12, check_every=25,
                        max_iter=50000):
    """``min_W ||A W - B||_F^2`` with every column of ``W`` on the simplex.

    FISTA over all right-hand sides at once. ``A`` is shared, so the Gram matrix,
    the ridge and the step size are formed once and amortised over ``B``'s
    columns.

    ``ridge`` adds ``rho * mean(diag(G))`` to the Gram diagonal. It is not a
    solver tolerance: it changes the estimand, shrinking the donor weights toward
    uniform exactly as SDID's ``zeta`` does. Section 3 measures how far.

    Convergence is judged on the objective, not on the step norm. With a rank
    deficient ``A`` the iterate keeps drifting along the optimal face long after
    the objective has settled, so a step-norm rule never fires.
    """
    A = np.asarray(A, float)
    B = np.atleast_2d(np.asarray(B, float))
    n, k = A.shape[1], B.shape[1]
    if n == 1:
        return np.ones((1, k)), 0
    G = A.T @ A
    if ridge:
        G = G + ridge * float(np.mean(np.diag(G))) * np.eye(n)
    C = A.T @ B
    bb = float((B * B).sum())
    step = 1.0 / (2.0 * float(np.linalg.eigvalsh(G)[-1]) + 1e-300)

    W = np.full((n, k), 1.0 / n)
    Z = W.copy()
    t = 1.0
    prev = None
    for it in range(max_iter):
        W_new = project_simplex_cols(Z - step * 2.0 * (G @ Z - C))
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = W_new + ((t - 1.0) / t_new) * (W_new - W)
        W, t = W_new, t_new
        if (it + 1) % check_every == 0:
            obj = float(((G @ W) * W).sum() - 2.0 * (C * W).sum() + bb)
            if prev is not None and prev - obj <= tol * max(1.0, abs(prev)):
                return W, it + 1
            prev = obj
    return W, max_iter


# -------------------------------------------------------------- the event-time cube
class EventPanel:
    """The panel keyed by cohort, with the normalization already applied."""

    def __init__(self, path: str = PANEL) -> None:
        d = pd.read_parquet(path)
        self.Y = d.pivot(index="year", columns="cty_fips", values="emps_n10").sort_index()
        meta = d.groupby("cty_fips").agg(sc=("supercenter", "max"),
                                         sy=("super_year", "max"),
                                         pop90=("pop90", "max"))
        self.meta = meta.loc[self.Y.columns]
        self.years = self.Y.index.to_numpy()
        self.Yv = self.Y.to_numpy(float)
        self.donors = np.flatnonzero((self.meta.sc == 0).to_numpy())
        self.treated = np.flatnonzero((self.meta.sc == 1).to_numpy())
        self.cohort_of = self.meta.sy.to_numpy()

    def cohorts(self):
        """Yield ``(g, treated_columns, donor_block, treated_block, pre_mask)``.

        Both blocks are indexed to year ``g - 1``, which is what makes the donor
        block cohort-specific and the whole cohort a single design matrix.
        """
        for g in sorted({int(v) for v in self.cohort_of[self.treated]}):
            idx = self.treated[self.cohort_of[self.treated].astype(int) == g]
            base = self.Yv[self.years == g - 1][0]
            yield (g, idx,
                   100.0 * self.Yv[:, self.donors] / base[self.donors],
                   100.0 * self.Yv[:, idx] / base[idx],
                   self.years < g)

    def event_frame(self, gaps_by_cohort) -> pd.DataFrame:
        """Stack per-cohort gap blocks onto the common event clock."""
        rows = []
        for (g, idx), gaps in gaps_by_cohort:
            sel = np.isin(self.years, np.arange(g - K, g + K + 1))
            offsets = self.years[sel] - g
            for col, j in enumerate(idx):
                rows.append({"cty": int(self.Y.columns[j]),
                             "pop90": float(self.meta.pop90.iloc[j]),
                             **dict(zip(offsets, gaps[sel, col]))})
        return pd.DataFrame(rows).sort_values("cty").reset_index(drop=True)


def solve_batched(panel: EventPanel, ridge: float = 0.0):
    """The whole stacked design: one multi-RHS solve per cohort."""
    out, iters = [], []
    for g, idx, donor_block, treated_block, pre in panel.cohorts():
        W, n_it = simplex_lstsq_batch(donor_block[pre], treated_block[pre], ridge=ridge)
        out.append(((g, idx), treated_block - donor_block @ W))
        iters.append(n_it)
    return panel.event_frame(out), iters


def _row(label, values):
    return f"{label:32s}" + " ".join(f"{v:6.2f}" for v in values)


def _header(label, offsets):
    return f"{label:32s}" + " ".join(f"{e:6d}" for e in offsets)


# ---------------------------------------------------------------- 1. does it agree
def section_1_agreement(panel: EventPanel):
    print("\n=== 1. Batched vs looped, same objective ===")
    t0 = time.time()
    batched, iters = solve_batched(panel)
    t_batch = time.time() - t0

    d = pd.read_parquet(PANEL)
    d["treat"] = ((d.supercenter == 1) & (d.year >= d.super_year)).astype(int)
    donor_ids = list(panel.meta.index[panel.meta.sc == 0])
    t0 = time.time()
    rows = []
    for j in panel.meta.index[panel.meta.sc == 1]:
        g = int(panel.meta.sy.loc[j])
        sub = d[d.cty_fips.isin([j] + donor_ids)].copy()
        base = sub[sub.year == g - 1].set_index("cty_fips").emps_n10
        sub["ynorm"] = 100.0 * sub.emps_n10.values / base.reindex(sub.cty_fips).values
        res = VanillaSC({"df": sub, "outcome": "ynorm", "treat": "treat",
                         "unitid": "cty_fips", "time": "year",
                         "display_graphs": False}).fit()
        path = np.asarray(res.time_series.estimated_gap, float)
        gaps = {y - g: path[i] for i, y in enumerate(sorted(sub.year.unique()))}
        rows.append({"cty": int(j), **{e: gaps.get(e, np.nan) for e in range(-K, K + 1)}})
    looped = pd.DataFrame(rows).sort_values("cty").reset_index(drop=True)
    t_loop = time.time() - t0

    cols = list(range(-K, K + 1))
    print(f"  batched  {t_batch:6.2f}s   FISTA iterations per cohort: {iters}")
    print(f"  looped   {t_loop:6.2f}s   566 separate VanillaSC fits")
    print(f"  speedup  {t_loop / t_batch:6.1f}x")
    print(_header("event time", cols))
    print(_row("batched", [batched[e].mean() for e in cols]))
    print(_row("looped VanillaSC", [looped[e].mean() for e in cols]))
    delta = np.abs(batched[cols].to_numpy() - looped[cols].to_numpy())
    print(f"  aggregate difference: {np.abs(np.nanmean(batched[cols].to_numpy(), 0) - np.nanmean(looped[cols].to_numpy(), 0)).max():.3f}pp")
    print(f"  worst single county:  {np.nanmax(delta):.2f}pp   -- see section 3")
    return batched, looped


# --------------------------------------------------------------- 2. where the work went
def section_2_structure(panel: EventPanel):
    print("\n=== 2. Why six solves suffice ===")
    print(f"  {'cohort':>7} {'treated':>8} {'T_pre':>6} {'donors':>7} {'rank(A_g)':>10} "
          f"{'null space':>11}")
    for g, idx, donor_block, _, pre in panel.cohorts():
        A = donor_block[pre]
        rank = int(np.linalg.matrix_rank(A))
        print(f"  {g:>7} {len(idx):>8} {int(pre.sum()):>6} {A.shape[1]:>7} "
              f"{rank:>10} {A.shape[1] - rank:>11}")
    print("\n  One Gram matrix and one Lipschitz constant per cohort, amortised over"
          "\n  76-114 right-hand sides; each iteration is a (39, 39) @ (39, N_g) matmul.")


# ------------------------------------------------------- 3. the weights aren't identified
def section_3_identification(panel: EventPanel, batched, looped):
    print("\n=== 3. The per-county weights are not identified ===")
    cols = list(range(-K, K + 1))
    delta = np.abs(batched[cols].to_numpy() - looped[cols].to_numpy())
    per_county = np.nanmax(delta, axis=1)
    print(f"  per-county max gap difference: median {np.median(per_county):.2f}pp, "
          f"90th pct {np.percentile(per_county, 90):.2f}pp, max {per_county.max():.2f}pp")
    print("  Both solutions are optimal -- A_g's null space lets the weights move"
          "\n  without touching the pre-treatment fit. Only the average is pinned down.")

    print("\n  A ridge restores uniqueness, but it is not free -- it shrinks toward"
          "\n  uniform weights, which is what SDID's zeta does:")
    print(_header("  event time", range(0, K + 1)))
    for rho in (0.0, 1e-4, 1e-2):
        t0 = time.time()
        frame, iters = solve_batched(panel, ridge=rho)
        print(_row(f"  rho = {rho:g}", [frame[e].mean() for e in range(0, K + 1)])
              + f"   [{time.time() - t0:.2f}s, {max(iters)} iters]")
    print("  (rho = 1e-2 lands on SDID's own -1.9 at e = 5; the SC/SDID gap in"
          "\n   sdid_allsynth_walmart.py is this penalty, not a different design.)")


# ------------------------------------------------------------- 4. the same trick for SDID
def section_4_sdid(panel: EventPanel):
    print("\n=== 4. SDID shares more than the design matrix ===")
    d = pd.read_parquet(PANEL)
    d["treat"] = ((d.supercenter == 1) & (d.year >= d.super_year)).astype(int)
    donor_ids = list(panel.meta.index[panel.meta.sc == 0])
    cohort = [c for c in panel.meta.index[panel.meta.sc == 1]
              if int(panel.meta.sy.loc[c]) == 1996][:3]
    print("  SDID fits its time weights and its ridge on the donors alone, so within"
          "\n  a cohort they do not depend on which county is treated:")
    for j in cohort:
        sub = d[d.cty_fips.isin([j] + donor_ids)].copy()
        base = sub[sub.year == 1995].set_index("cty_fips").emps_n10
        sub["ynorm"] = 100.0 * sub.emps_n10.values / base.reindex(sub.cty_fips).values
        res = SDID({"df": sub, "outcome": "ynorm", "treat": "treat",
                    "unitid": "cty_fips", "time": "year", "vce": "noinference",
                    "display_graphs": False}).fit()
        lam = np.asarray(res.cohorts[max(res.cohorts)].time_weights, float)
        print(f"    county {j}: lambda = {np.round(lam, 5)}")
    print("  So a batched SDID is one time-weight solve per cohort plus one"
          "\n  multi-RHS unit-weight solve, with the intercept concentrated out by"
          "\n  centring the pre-period rows and zeta folded into the Gram diagonal.")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    pd.set_option("display.width", 200)
    p = EventPanel()
    b, l = section_1_agreement(p)
    section_2_structure(p)
    section_3_identification(p, b, l)
    section_4_sdid(p)
