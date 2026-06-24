#!/usr/bin/env python3
"""Reference run for the ``linf_prop99`` benchmark case.

Runs the authors' own L-infinity SC implementation -- Wang, Xing & Ye (2025),
the ``LinfinitySC`` repository (https://github.com/BioAlgs/LinfinitySC),
``utils/synth.py`` -- on the Abadie-Diamond-Hainmueller Proposition 99 tobacco
panel (``basedata/smoking_data.csv``), exactly as their ``Tobacco.ipynb``
notebook does:

  * pre-period = years <= 1988 (T0 = 19), post-period = 1989-2000 (T1 = 12);
  * California is the treated unit, the other 38 states are donors;
  * the L-infinity corner case ``our(method='inf')`` with an intercept and no
    standardisation, at the penalty ``lambda`` selected by the authors' own
    deterministic cross-validation ``param_selector(method='inf', n_folds=10)``
    (``method='inf'`` pins ``alpha = 1/J`` and the KFold CV uses a fixed
    ``random_state`` per repeat, so the selection is reproducible).

These are the genuine ``LinfinitySC`` outputs -- donor weights, intercept, the
counterfactual effect path and the post-period ATT -- that the Python case pins
against, captured live, not transcribed.

The reference repo is fetched on demand (git clone, else the codeload tarball)
and pinned at ``clone_linfinitysc._COMMIT`` for reproducibility; the L-infinity
QP is solved by ``cvxopt`` (open source -- no commercial solver needed). If the
clone or ``cvxopt`` is unavailable the script exits without emitting the
``== REFERENCE VALUES ==`` block, so ``generate.py`` skips the bundle cleanly.

Run from the repository root::

    python benchmarks/reference/linf_prop99/reference.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

INTERVENTION = 1989       # Proposition 99 takes effect; pre-period is <= 1988
N_FOLDS = 10              # the authors' Tobacco.ipynb CV setting
N_REPEATS = 2            # param_selector default (deterministic random_state per repeat)


def main() -> int:
    try:
        from benchmarks.reference.clone_linfinitysc import import_synth
        synth = import_synth()          # raises BenchmarkSkipped if clone absent
    except Exception as exc:            # noqa: BLE001 - report and skip
        print(f"[skip] reference clone unavailable: {exc}", file=sys.stderr)
        return 0
    try:
        import cvxopt  # noqa: F401     (reference solver backend)
    except Exception as exc:            # noqa: BLE001
        print(f"[skip] cvxopt unavailable: {exc}", file=sys.stderr)
        return 0

    d = pd.read_csv(ROOT / "basedata" / "smoking_data.csv")
    states = sorted(d["state"].unique())
    donors = [s for s in states if s != "California"]
    years = sorted(d["year"].unique())
    T0 = years.index(INTERVENTION)      # 19 pre-treatment periods (1970..1988)

    piv = d.pivot(index="year", columns="state", values="cigsale")
    Y1 = piv["California"].to_numpy(dtype=float)        # (T,)
    Y0 = piv[donors].to_numpy(dtype=float)             # (T, J)
    Y1_pre, Y0_pre = Y1[:T0], Y0[:T0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The authors' own deterministic CV (method='inf' fixes alpha = 1/J).
        alpha, lam = synth.param_selector(
            Y1_pre, Y0_pre, method="inf",
            n_folds=N_FOLDS, n_repeats=N_REPEATS, max_workers=4,
        )
        vec = np.asarray(
            synth.our(Y1_pre, Y0_pre, alpha, lam, method="inf",
                      std=False, intercept=True),
            dtype=float,
        )

    intercept, w = float(vec[0]), vec[1:]
    cf = intercept + Y0 @ w
    tau = Y1 - cf                       # full-sample effect path
    att = float(np.mean(tau[T0:]))      # post-period ATT
    pre_rmspe = float(np.sqrt(np.mean(tau[:T0] ** 2)))

    ti = {yr: float(tau[years.index(yr)]) for yr in (1990, 1995, 2000)}

    print("== REFERENCE VALUES ==")
    for name, wj in zip(donors, w):
        print(f"weight\t{name}\t{wj:.6f}")
    print(f"intercept\t{intercept:.6f}")
    print(f"att_mean_post\t{att:.6f}")
    print(f"ite_1990\t{ti[1990]:.6f}")
    print(f"ite_1995\t{ti[1995]:.6f}")
    print(f"ite_2000\t{ti[2000]:.6f}")
    print(f"pre_rmspe\t{pre_rmspe:.6f}")
    print(f"n_donors\t{len(donors):d}")
    print(f"n_negative\t{int(np.sum(w < -1e-3)):d}")
    print(f"lam_ref\t{float(lam):.10f}")
    print(f"alpha_ref\t{float(alpha):.10f}")
    print("== ENVIRONMENT ==")
    print(f"python\t{sys.version.split()[0]}")
    print(f"numpy\t{np.__version__}")
    print(f"cvxopt\t{cvxopt.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
