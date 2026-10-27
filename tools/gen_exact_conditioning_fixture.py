"""Regenerate ``mlsynth/tests/fixtures/exact/conditioning_basque.json``.

Runs VanillaSC's exact backend (positive definite class) on Abadie and Gardeazabal's
Basque specification with ``school.higher`` dropped (the four schooling shares sum to
a constant, so the 13 predictors span 12 dimensions) and the outcome fit over
1955-1969. Every call to ``lsq.simplex_ls`` is recorded; the faces whose validation
block has a condition number above 1e4 are kept, and each one's optimum is computed
in 50-digit arithmetic: the equality-constrained least squares problem on the active
rows of the double-precision answer, accepted only when its KKT conditions hold in
50 digits (Björck, Numerical Methods in Matrix Computations, p. 371: once the active
set is known, the problem has equality constraints only).

Needs ``mpmath``, which mlsynth does not depend on; run it by hand:

    python tools/gen_exact_conditioning_fixture.py
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sys
import warnings

import mpmath as mp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmarks.cases.mscmt_solver as spec  # noqa: E402
from mlsynth import VanillaSC  # noqa: E402
from mlsynth.utils.bilevel.exact import lsq, search  # noqa: E402

OUT = ROOT / "mlsynth" / "tests" / "fixtures" / "exact" / "conditioning_basque.json"
COND_FLOOR = 1e4
N_KEEP = 6
mp.mp.dps = 50


def _constraints(m, Gin, hin):
    """Rows A b >= a: the simplex's b >= 0, then the piece's own rows."""
    A, a = np.eye(m), np.zeros(m)
    if Gin is not None and len(Gin):
        A, a = np.vstack([A, Gin]), np.r_[a, hin]
    return A, a


def _reference(z, Z, Gin, hin, b):
    """The optimum on the active rows of b, in 50 digits, or None when its KKT conditions fail."""
    m = Z.shape[1]
    A, a = _constraints(m, Gin, hin)
    act = np.flatnonzero(A @ b - a <= 1e-9 * (1.0 + np.abs(a)))
    C = np.vstack([np.ones(m), A[act]])
    d = np.r_[1.0, a[act]]
    k = len(C)
    Zm, zm = mp.matrix(Z.tolist()), mp.matrix(z.tolist())
    K, rhs = mp.zeros(m + k, m + k), mp.zeros(m + k, 1)
    ZtZ, Ztz = Zm.T * Zm, Zm.T * zm
    for i in range(m):
        rhs[i] = Ztz[i]
        for j in range(m):
            K[i, j] = ZtZ[i, j]
    for r in range(k):
        rhs[m + r] = d[r]
        for j in range(m):
            K[m + r, j] = K[j, m + r] = C[r, j]
    sol = mp.lu_solve(K, rhs)
    bs = [sol[i] for i in range(m)]
    lam = [-sol[m + 1 + r] for r in range(k - 1)]          # inequality multipliers; the equality's is free
    slack = [sum(mp.mpf(A[i, j]) * bs[j] for j in range(m)) - a[i] for i in range(len(A))]
    tiny = -mp.mpf(10) ** -30
    if min(slack + [0]) < tiny or min(lam + [0]) < tiny:
        return None
    loss = sum((zm[t] - sum(Zm[t, j] * bs[j] for j in range(m))) ** 2 for t in range(len(z)))
    return [mp.nstr(v, 30) for v in bs], mp.nstr(loss, 30)


def main():
    warnings.simplefilter("ignore")
    cap = lsq.COND_MAX if hasattr(lsq, "COND_MAX") else None
    if cap is not None:
        lsq.COND_MAX = 1e12                               # capture past the old refusal
    seen, real = [], lsq.simplex_ls

    def spy(z, Z, Gin=None, hin=None):
        loss, b = real(z, Z, Gin=Gin, hin=hin)
        m = Z.shape[1]
        if b is not None and m > 1:
            N = lsq._simplex_basis(m)[1]
            sv = np.linalg.svd(Z @ N, compute_uv=False)
            if sv[0] / sv[-1] > COND_FLOOR:
                seen.append((sv[0] / sv[-1], z, Z, Gin, hin, b))
        return loss, b

    lsq.simplex_ls = search.simplex_ls = spy
    covs = [c for c in spec._COVS if c != "school.higher"]
    win = {k: v for k, v in spec._WINDOWS.items() if k != "school.higher"}
    VanillaSC(dict(df=spec._panel(), outcome="gdpcap", treat="treat", unitid="regionname", time="year",
                   covariates=covs, covariate_windows=win, fit_window=(1955, 1969), backend="exact",
                   exact_v_class="pd", inference=False, display_graphs=False)).fit()
    seen.sort(key=lambda s: -s[0])
    cases, keys = [], set()
    for cond, z, Z, Gin, hin, b in seen:
        key = (round(cond, 6), Z.shape)
        if key in keys:
            continue
        ref = _reference(z, Z, Gin, hin, b)
        if ref is None:
            continue
        keys.add(key)
        cases.append({"cond": cond, "z": z.tolist(), "Z": Z.tolist(),
                      "Gin": None if Gin is None else Gin.tolist(),
                      "hin": None if hin is None else hin.tolist(),
                      "b_ref": ref[0], "loss_ref": ref[1]})
        if len(cases) == N_KEEP:
            break
    OUT.write_text(json.dumps({
        "provenance": (
            "Faces of VanillaSC's exact backend (positive definite class) on Abadie and Gardeazabal's "
            "Basque specification with school.higher dropped and the outcome fit over 1955-1969, "
            "captured from lsq.simplex_ls: the faces whose validation block has a condition number "
            "above 1e4, which the reduction's former cap COND_MAX = 1e4 refused. 'b_ref' and "
            "'loss_ref' are the optimum in 50-digit arithmetic, KKT-verified; 'z', 'Z', 'Gin', 'hin' "
            "are the arguments of lsq.simplex_ls. Generated by tools/gen_exact_conditioning_fixture.py "
            f"on {datetime.date.today().isoformat()}."),
        "cases": cases}, indent=1))
    print(f"{len(seen)} faces above {COND_FLOOR:.0e}; kept {len(cases)}, conditions "
          + ", ".join(f"{c['cond']:.2e}" for c in cases))


if __name__ == "__main__":
    main()
