"""Issue #770's lead 2: certificates borrowed from candidates already fitted.

Two kinds, both valid for every candidate whatever their source:

* separating directions. A fitted candidate's residual ``y = B_S w - B_C v``,
  normalised, bounds every other candidate's hull distance by
  ``min over i in S of y'b_i - max over j in C of y'b_j`` (the bound of
  :mod:`bound`, with ``x`` borrowed). The pool holds the residuals of the 256
  candidates with the smallest closed-form bound, fitted in full, as the
  search's first pass would; the bound is ``lb + beta * sep^2``, best over the
  pool and both signs.
* a dual pair. :func:`bound.fenchel_bound` at the optimal design's own
  ``(lam, mu)``, applied to 300 random candidates.

From the repository root, after ``search.py``::

    python benchmarks/studies/marex_coupling_bound/borrowed.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from benchmarks.studies.marex_coupling_bound import bound, panel  # noqa: E402
from mlsynth.utils.marex_helpers import exact  # noqa: E402

POOL = 256
POOL_SIZES = (1, 16, 64, 256)
RANDOM = 300
SEED = 1


def separation(P, sets, m):
    """Best over the rows of ``P`` and their negatives of min over S less max off S."""
    P = np.vstack([P, -P])
    top = np.argsort(-P, axis=1)[:, :m + 1]
    top_val = np.take_along_axis(P, top, axis=1)
    out = np.empty(len(sets))
    for lo in range(0, len(sets), 5000):
        s = sets[lo:lo + 5000]
        mins = P[:, s].min(axis=2).T
        in_s = (top[None, :, :, None] == s[:, None, None, :]).any(-1)
        first = np.argmax(~in_s, axis=2)
        off = np.take_along_axis(np.broadcast_to(top_val, in_s.shape),
                                 first[..., None], axis=2)[..., 0]
        out[lo:lo + 5000] = (mins - off).max(axis=1)
    return out


def residual(B, A, S, beta):
    n = B.shape[1]
    C = [j for j in range(n) if j not in S]
    value, w, v = exact.coupled_fit(B, A, list(S), C, beta)
    return value, B[:, list(S)] @ w, B[:, C] @ v


def measure(B, A, beta):
    n = B.shape[1]
    fstar = panel.optimum(beta)
    cut = fstar + exact.PRUNE_MARGIN * (1.0 + abs(fstar))
    sets = np.array(list(itertools.combinations(range(n), panel.M)))
    lb = exact.treated_fit_bound(B, A, sets)
    dirs = []
    for k in np.argsort(lb, kind="stable")[:POOL]:
        _, t, c = residual(B, A, sets[k], beta)
        d = t - c
        if np.linalg.norm(d) > 0:
            dirs.append(d / np.linalg.norm(d))
    Y = np.array(dirs)
    out = {"optimum": fstar, "candidates": len(sets),
           "under_optimum_closed_form": int((lb < cut).sum()), "pool": {}}
    for size in POOL_SIZES:
        sep = separation(Y[:size] @ B, sets, panel.M)
        with_pool = lb + beta * np.maximum(sep, 0.0) ** 2
        out["pool"][str(size)] = int((with_pool < cut).sum())

    best = json.loads((panel.RESULTS / "search.json").read_text())
    S0 = best["runs"][str(beta)]["shipped"]["treated"]
    C0 = [j for j in range(n) if j not in S0]
    f0, t, c = residual(B, A, S0, beta)
    lam, mu = 2 * (A - t), 2 * beta * (t - c)
    at_source = bound.fenchel_bound(B, A, S0, C0, lam, mu, beta) / f0
    rng = np.random.default_rng(SEED)
    shares = []
    for _ in range(RANDOM):
        S = sorted(rng.choice(n, panel.M, replace=False).tolist())
        C = [j for j in range(n) if j not in S]
        f = residual(B, A, S, beta)[0]
        shares.append(bound.fenchel_bound(B, A, S, C, lam, mu, beta) / f)
    out["dual_pair"] = {"share_at_source": at_source,
                        "share_on_random_max": float(np.max(shares)),
                        "share_on_random_median": float(np.median(shares))}
    return out


def main():
    B, A = panel.geolift()
    out = {str(beta): measure(B, A, beta) for beta in panel.BETAS}
    (panel.RESULTS / "borrowed.json").write_text(json.dumps(out, indent=1) + "\n")
    for beta, r in out.items():
        print(f"beta={beta}: under the optimum, closed form "
              f"{r['under_optimum_closed_form']}, with pools {r['pool']}; "
              f"dual pair {r['dual_pair']}")


if __name__ == "__main__":
    main()
