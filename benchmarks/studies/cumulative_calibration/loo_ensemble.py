"""Leave-one-out ensemble calibration for MAREX's cumulative band.

MAREX calibrates its cumulative band on the blank window, which supports
``Tb // H = 3`` sub-windows. Since the band's binding constraint is the number of
windows, the obvious lever is to make more of them, and EnbPI (Xu & Xie 2021)
supplies a construction that does: fit an ensemble of control-weight vectors, each
on a bootstrap sample of the pre-periods, and score period ``t`` using only the
members whose sample excluded it. Every pre-period then carries an out-of-sample
residual and the count rises from ``Tb // H = 3`` to ``T0 // H = 8``.

What is held fixed decides what this costs. MAREX's design step chooses the
treated set ``S`` and the treated weights ``w`` by mixed-integer program; that is
the expensive part and it is not bootstrapped. Each member takes ``S`` and ``w``
as given and refits only the control weights over the remaining units, which is a
least-squares fit on the probability simplex -- milliseconds against seconds for
the design solve, so a few hundred members is cheap. See ``design_cost.py``.

Fixing ``S`` leaves one leak. ``S`` was chosen having seen the fitting periods, so
a residual there is out of sample for the control weights but not for the choice
of ``S``; the blank periods carry no such leak, having been withheld from the
design entirely. The script reports both spreads so the leak's size is visible.

Four bands are scored per draw, arranged so each comparison isolates one thing:

  ship    the design's own blank-window gap, m = Tb // H   -- what MAREX does
  loo_b   the ensemble residuals, blank window only, m = 3 -- isolates the ensemble
  loo     the ensemble residuals, whole pre-period, m = 8  -- isolates the count
  loo_s   as ``loo``, with the fitted stretch rescaled onto the withheld
          stretch's spread -- the leak, corrected as far as rescaling can

    MLSYNTH_CAL_PANEL=<panel.csv> python loo_ensemble.py <n_draws> <first_seed> <out.jsonl>
    python loo_ensemble.py --summarise <out.jsonl> [<out.jsonl> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

# run from anywhere: the repo root supplies mlsynth, this directory supplies panel
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import sys
import warnings

import numpy as np

from mlsynth import MAREX
from mlsynth.utils.bilevel.simplex import simplex_lstsq
from mlsynth.utils.conformal import block_error_paths
import panel
from panel import ALPHA, DELTA, H, T0, TB, TE, draw, frame

B = 200
J = 20


def design(YN):
    """The one mixed-integer solve: which units to treat, and with what weights."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MAREX({"df": frame(YN), "outcome": "y", "unitid": "unit",
                     "time": "time", "T0": T0, "blank_periods": TB, "T_post": H,
                     "m_eq": 3, "inference": True, "alpha": ALPHA}).fit()
    return (np.asarray(res.globres.treated_weights_agg, dtype=float),
            np.asarray(res.globres.control_weights_agg, dtype=float))


def ensemble(target, donors, rng, n_train, n_members=B):
    """Bootstrap ensemble of simplex fits, with the periods each member saw.

    ``seen[b, t]`` is what makes the aggregation in :func:`loo_residuals` out of
    sample: a member that sampled period ``t`` is barred from predicting it.
    """
    V = np.empty((n_members, donors.shape[0]))
    seen = np.zeros((n_members, n_train), dtype=bool)
    for b in range(n_members):
        idx = rng.integers(0, n_train, size=n_train)
        seen[b, np.unique(idx)] = True
        V[b] = simplex_lstsq(donors[:, idx].T, target[idx])
    return V, seen


def loo_residuals(target, donors, V, seen, n_train):
    """Out-of-sample residual at every training period.

    A period seen by every member has no honest prediction and comes back as
    ``nan``, which the caller drops and counts.
    """
    P = V @ donors[:, :n_train]
    pred = np.full(n_train, np.nan)
    for t in range(n_train):
        keep = ~seen[:, t]
        if keep.any():
            pred[t] = np.median(P[keep, t])
    return target[:n_train] - pred


def band(series, m, n_sim, seed):
    paths = block_error_paths(series, horizon=H, block=0, n_sim=n_sim,
                              rng=np.random.default_rng(seed),
                              n_windows=m if m >= 2 else 0)
    return float(np.quantile(np.abs(paths.sum(axis=1)), 1 - ALPHA))


def one(seed, n_sim=2000):
    YN = draw(seed, J=J)
    w, v_design = design(YN)
    S = np.flatnonzero(w > 1e-8)
    rest = np.array([j for j in range(J) if j not in set(S.tolist())])

    Y = YN.copy()
    Y[S, T0:] += DELTA
    target, donors = w @ Y, Y[rest]

    rng = np.random.default_rng(seed)
    V, seen = ensemble(target, donors, rng, T0)
    e = loo_residuals(target, donors, V, seen, T0)
    ok = np.isfinite(e)

    # the ensemble's own post-period counterfactual, consistent with its residuals
    err_loo = abs(float((target[T0:] - np.median(V @ donors[:, T0:], axis=0)).sum())
                  - DELTA * H)
    gap_design = target - v_design @ Y
    err_ship = abs(float(gap_design[T0:].sum()) - DELTA * H)

    out = {"seed": seed, "n_dropped": int((~ok).sum()),
           "err_ship": err_ship, "err_loo": err_loo}

    h = band(gap_design[T0 - TB:T0], TB // H, n_sim, seed)
    out.update(m_ship=TB // H, half_ship=h, cover_ship=float(err_ship <= h))

    eb = e[T0 - TB:][np.isfinite(e[T0 - TB:])]
    h = band(eb, eb.size // H, n_sim, seed)
    out.update(m_loo_b=eb.size // H, half_loo_b=h, cover_loo_b=float(err_loo <= h))

    ea = e[ok]
    h = band(ea, ea.size // H, n_sim, seed)
    out.update(m_loo=ea.size // H, half_loo=h, cover_loo=float(err_loo <= h))

    sd_fit = float(np.nanstd(e[:TE]))
    sd_blank = float(np.nanstd(e[TE:T0]))
    adj = e.copy()
    if sd_fit > 0:
        adj[:TE] *= sd_blank / sd_fit
    es = adj[np.isfinite(adj)]
    h = band(es, es.size // H, n_sim, seed)
    out.update(m_loo_s=es.size // H, half_loo_s=h, cover_loo_s=float(err_loo <= h),
               leak_ratio=sd_fit / sd_blank if sd_blank else float("nan"),
               sd_loo_fit=sd_fit, sd_loo_blank=sd_blank,
               sd_design_blank=float(gap_design[T0 - TB:T0].std()))
    return out


def summarise(rows):
    n = len(rows)
    col = lambda k: np.array([r[k] for r in rows], dtype=float)   # noqa: E731
    print(f"draws {n}\n")
    print(f"{'band':<8}{'m':>4}{'coverage':>11}{'se':>8}{'median half':>14}"
          f"{'ratio vs ship':>15}")
    hs = col("half_ship")
    for tag in ("ship", "loo_b", "loo", "loo_s"):
        c, h = col(f"cover_{tag}"), col(f"half_{tag}")
        p = c.mean()
        print(f"{tag:<8}{int(np.median(col(f'm_{tag}'))):>4}{p:>11.4f}"
              f"{np.sqrt(p * (1 - p) / n):>8.4f}{np.median(h):>14.4f}"
              f"{np.median(h / hs):>15.3f}")

    print("\npaired comparison on the same draws (McNemar discordant pairs)")
    cov = {t: col(f"cover_{t}") for t in ("ship", "loo_b", "loo", "loo_s")}
    for a, b in (("ship", "loo_b"), ("ship", "loo_s"),
                 ("loo_b", "loo"), ("loo_b", "loo_s")):
        n01 = int(((cov[a] == 0) & (cov[b] == 1)).sum())
        n10 = int(((cov[a] == 1) & (cov[b] == 0)).sum())
        d = n01 + n10
        z = (n01 - n10) / np.sqrt(d) if d else float("nan")
        print(f"  {b:>6} vs {a:<6}  {b} alone {n01:2d}, {a} alone {n10:2d}"
              f"   z = {z:+.2f}")

    print(f"\nleak: sd(LOO residual) on fitted periods / on withheld periods")
    print(f"  median ratio {np.median(col('leak_ratio')):.3f}   "
          f"fitted {np.median(col('sd_loo_fit')):.4f}   "
          f"withheld {np.median(col('sd_loo_blank')):.4f}")
    print(f"periods with no honest prediction: max {int(col('n_dropped').max())}")
    print(f"median |error|: design {np.median(col('err_ship')):.4f}   "
          f"ensemble {np.median(col('err_loo')):.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "--summarise":
        summarise([json.loads(l) for f in sys.argv[2:] for l in open(f)])
    else:
        n, start, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
        with open(path, "a") as fh:
            for s in range(start, start + n):
                # Stamp which panel produced the row, so a results file cannot be
                # read as real-panel numbers when the stand-in supplied them.
                fh.write(json.dumps({**one(s), "panel": panel.source()}) + "\n")
                fh.flush()
                print(f"seed {s}", flush=True)
