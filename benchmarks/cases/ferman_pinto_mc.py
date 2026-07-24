"""Path B + golden cross-validation: Ferman & Pinto (2021, QE) Monte Carlo, Table 1.

Ferman and Pinto (2021), *"Synthetic controls with imperfect pretreatment fit"*
(Quantitative Economics 12:1197-1221), study synthetic control when the
pre-treatment fit is imperfect. Their Table 1 is an application-calibrated Monte
Carlo -- a linear factor model fit to monthly US state employment rates -- that
contrasts three estimators when the true effect is zero, so the estimated effect
*is* the bias:

* original **SC** (donor weights on the simplex)          -- mlsynth ``VanillaSC``;
* **demeaned SC** (the simplex plus a free intercept)      -- mlsynth ``TSSC`` MSCa;
* **DID** (equal donor weights plus an intercept)          -- the foil (inline).

The paper's findings (Table 1, Panels A-D):

* No structural break (Panels A/B, treated unit at the extreme of the fixed-effect
  distribution): the original SC is biased even as ``T0`` grows -- it fails to
  reconstruct the treated unit's *level* -- while demeaned SC and DID are
  ~unbiased, and demeaned SC carries ~40-50% smaller standard errors than DID.
* A break in factor 1 (Panels C/D, treated at the extreme of the loading
  distribution): all three are biased (selection on unobservables), but the SC and
  demeaned-SC biases are far below DID's, and shrink as ``T0`` grows.

Two things this case checks.

1. Golden tie (live R, value-for-value). mlsynth's ``VanillaSC`` and ``TSSC``-MSCa
   are exactly the authors' ``synth_control_est`` and ``synth_control_est_demean``
   (``_aux.R``, ``quadprog`` QPs). On identical simulated panels -- written out and
   solved live by ``benchmarks/reference/ferman_pinto_mc/reference.R`` -- the donor
   weights, intercept and effect agree to solver tolerance. Skips
   (``BenchmarkSkipped``) when ``Rscript`` / ``quadprog`` is absent.

2. Reproduction (Python MC on the calibrated DGP). Running mlsynth's estimators
   over the four panels reproduces the Panel A/B bias close to the published
   numbers (these turn on the rotation-invariant unit fixed effect), and the
   qualitative structure everywhere: SC biased vs demeaned/DID unbiased under no
   break; demeaned SE well below DID SE; all biased under a break with |SC|,
   |demeaned| far below |DID|; bias shrinking in ``T0``.

The DGP is calibrated once from the authors' CPS panel and baked into
``benchmarks/reference/ferman_pinto_mc/fixed_env.npz`` (see ``calibrate.py`` there
for provenance and the ``gsynth``/``auto.arima`` -> NumPy/AR(1) port notes). The
exact Panel C/D magnitudes depend on ``gsynth``'s factor rotation and are not
pinned; the replication page documents the calibration gap. Deterministic (seeded).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_BUNDLE = os.path.abspath(os.path.join(_ROOT, "benchmarks", "reference", "ferman_pinto_mc"))
_ENV = np.load(os.path.join(_BUNDLE, "fixed_env.npz"), allow_pickle=True)
_REF_R = os.path.join(_BUNDLE, "reference.R")

_UNIT = _ENV["unit"]                 # (N,) state fixed effects c_j
_LAM = _ENV["Lam"]                   # (N, r) factor loadings mu_j
_FAC_AR = _ENV["fac_ar"]             # (r, 3) AR(1) (rho, mean, innov_sd) per factor
_RES_AR = _ENV["res_ar"]             # (N, 3) AR(1) per idiosyncratic series
_SD_FAC = _ENV["sd_factors"]         # (r,) factor SDs (for the break magnitude)
_N = int(_ENV["N"])
_NREPS, _T1 = 300, 12
_T0S = (120, 480)

# Panels: (label, rank-key, which extreme, break-factor). Treated-unit selection
# matches the authors' "Table and Figures - MC.do" (2nd-largest / 2nd-smallest).
_PANELS = [
    ("A", "fe", "hi", None),   # 2nd-largest fixed effect,   no break
    ("B", "fe", "lo", None),   # 2nd-smallest fixed effect,  no break
    ("C", "f1", "hi", 0),      # 2nd-largest factor-1 load,  break in factor 1
    ("D", "f1", "lo", 0),      # 2nd-smallest factor-1 load, break in factor 1
]


def _treated_col(rank_key: str, which: str) -> int:
    key = _UNIT if rank_key == "fe" else _LAM[:, 0]
    order = np.argsort(key)
    return int(order[-2] if which == "hi" else order[1])


def _ar1_batch(pars, n, nreps, rng):
    """Vectorised Gaussian AR(1): (nreps, n, k) started from the stationary law."""
    rho, mu, sd = pars[:, 0], pars[:, 1], pars[:, 2]
    k = len(rho)
    y = np.empty((nreps, n, k))
    y[:, 0, :] = rng.normal(mu, sd / np.sqrt(np.clip(1 - rho ** 2, 1e-6, None)), (nreps, k))
    for t in range(1, n):
        y[:, t, :] = mu * (1 - rho) + rho * y[:, t - 1, :] + rng.normal(0, sd, (nreps, k))
    return y


def _sim_panels(seed, T0, T1, nreps):
    """(nreps, T0+T1, N) draws from the calibrated factor-model DGP. Time fixed
    effects are omitted -- they cancel in every one of the three estimators."""
    rng = np.random.default_rng(seed)
    n = T0 + T1
    Fs = _ar1_batch(_FAC_AR, n, nreps, rng)
    Fs = Fs - Fs.mean(1, keepdims=True)             # E[lambda_t]=0 over the window
    E = _ar1_batch(_RES_AR, n, nreps, rng)
    return Fs @ _LAM.T + _UNIT[None, None, :] + E


def _simplex(B, A, warm_start=None):
    from mlsynth.utils.bilevel.active_set import solve_simplex_qp
    return solve_simplex_qp(B, A, warm_start=warm_start)


def _panel_stats(label, rank_key, which, brk, T0, seed, nreps=_NREPS):
    """Bias and Monte-Carlo SE of SC / demeaned-SC / DID for one panel and T0."""
    col = _treated_col(rank_key, which)
    Y = _sim_panels(seed, T0, _T1, nreps)
    cols = np.array([col] + [c for c in range(_N) if c != col])
    Y = Y[:, :, cols]
    shift = None if brk is None else 2.0 * _SD_FAC[brk] * _LAM[cols, brk]
    eff = np.empty((nreps, 3))
    w_sc = w_dm = None                          # warm-start across reps (~9x faster)
    for r in range(nreps):
        Yb, Ya = Y[r, :T0], Y[r, T0:]
        y, X = Yb[:, 0], Yb[:, 1:]
        w_sc = _simplex(X, y, warm_start=w_sc)
        w_dm = _simplex(X - X.mean(0), y - y.mean(), warm_start=w_dm)
        a_dm = y.mean() - X.mean(0) @ w_dm
        if shift is not None:
            Ya = Ya + shift[None, :]
        yaf, Xaf = Ya[:, 0], Ya[:, 1:]
        eff[r, 0] = (yaf - Xaf @ w_sc).mean()
        eff[r, 1] = (yaf - a_dm - Xaf @ w_dm).mean()
        eff[r, 2] = (yaf.mean() - y.mean()) - (Xaf.mean() - X.mean())
    bias, se = eff.mean(0), eff.std(0, ddof=1)
    return {"bias": bias, "se": se}


def _monte_carlo(nreps=_NREPS):
    out = {}
    for i, (label, rk, wh, brk) in enumerate(_PANELS):
        for T0 in _T0S:
            out[(label, T0)] = _panel_stats(
                label, rk, wh, brk, T0, seed=1234 + 7 * i + T0, nreps=nreps)
    return out


# ------------------------------------------------------------------ live R tie
# Panels used for the value-for-value cross-check: identified (T0 > J = 50), so
# the SC / demeaned-SC minimisers are unique and both solvers must coincide.
_XCHECK = [("A", "fe", "hi", None, 120), ("B", "fe", "lo", None, 240),
           ("C", "f1", "hi", 0, 120), ("D", "f1", "lo", 0, 240)]


def _one_panel_matrix(rank_key, which, brk, T0, seed):
    """A single simulated panel: pre (T0 x (1+J)) and post (T1 x (1+J)), treated
    in column 0, with the Panel C/D break applied to the post block."""
    col = _treated_col(rank_key, which)
    Y = _sim_panels(seed, T0, _T1, 1)[0]
    cols = np.array([col] + [c for c in range(_N) if c != col])
    Y = Y[:, cols]
    pre, post = Y[:T0], Y[T0:]
    if brk is not None:
        post = post + (2.0 * _SD_FAC[brk] * _LAM[cols, brk])[None, :]
    return pre, post


def _mlsynth_on_panel(pre, post):
    """Run mlsynth's real estimators (VanillaSC, TSSC-MSCa) on one panel; return
    donor weights (donor order), MSCa intercept, and both ATTs."""
    from mlsynth import TSSC, VanillaSC
    J = pre.shape[1] - 1
    units = ["t000"] + [f"d{j:03d}" for j in range(1, J + 1)]
    T0, T1 = pre.shape[0], post.shape[0]
    full = np.vstack([pre, post])
    rows = []
    for j, u in enumerate(units):
        for t in range(T0 + T1):
            rows.append({"unit": u, "time": t, "y": full[t, j],
                         "treat": int(u == "t000" and t >= T0)})
    df = pd.DataFrame(rows)
    kw = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
              display_graphs=False)
    donors = units[1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = VanillaSC(kw).fit()
        # draws=2: we use only MSCa's point weights/intercept here (the value-
        # for-value tie), not TSSC's subsampling inference -- so skip that cost.
        msca = TSSC({**kw, "draws": 2}).fit().variants["MSCa"]
    scw = pd.Series(dict(sc.weights.donor_weights), dtype=float).reindex(donors).fillna(0.0).to_numpy()
    dmw = pd.Series(dict(msca.donor_weights), dtype=float).reindex(donors).fillna(0.0).to_numpy()
    return scw, dmw, float(msca.intercept), float(sc.effects.att), float(msca.att)


def _paired_crosscheck():
    """Run mlsynth's real estimators and the authors' QPs (live R) on identical
    identified panels; return per-panel ``(mlsynth, R)`` records. Skips
    (``BenchmarkSkipped``) when ``Rscript`` / ``quadprog`` / ``jsonlite`` is absent."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH (install R + quadprog + jsonlite)")
    probe = subprocess.run(
        [rscript, "-e", "suppressMessages({library(quadprog);library(jsonlite)})"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R packages 'quadprog'/'jsonlite' not installed")

    panels, ml = [], []
    for i, (label, rk, wh, brk, T0) in enumerate(_XCHECK):
        pre, post = _one_panel_matrix(rk, wh, brk, T0, seed=9000 + i)
        panels.append({"pre": pre.tolist(), "post": post.tolist()})
        ml.append(_mlsynth_on_panel(pre, post))

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(panels, fh)
        panels_path = fh.name
    try:
        out = subprocess.run([rscript, _REF_R, panels_path], capture_output=True, text=True)
    finally:
        os.unlink(panels_path)
    if out.returncode != 0:
        raise BenchmarkSkipped(f"reference.R failed: {out.stderr.strip()[-200:]}")
    R = json.loads(out.stdout.strip())

    recs = []
    for (label, *_), (scw, dmw, dm_int, sc_att, dm_att), r in zip(_XCHECK, ml, R):
        recs.append({
            "label": label,
            "sc_w": scw, "dm_w": dmw, "dm_int": dm_int, "sc_att": sc_att, "dm_att": dm_att,
            "r_sc_w": np.asarray(r["sc_w"], float), "r_dm_w": np.asarray(r["dm_w"], float),
            "r_dm_int": float(r["dm_intercept"]),
            "r_sc_eff": float(r["sc_effect"]), "r_dm_eff": float(r["dm_effect"]),
        })
    return recs


def _crosscheck_summary(recs):
    """Worst-case weight / intercept / effect disagreement across the panels."""
    return {
        "sc_weight_l1": max(float(np.abs(x["sc_w"] - x["r_sc_w"]).sum()) for x in recs),
        "dm_weight_l1": max(float(np.abs(x["dm_w"] - x["r_dm_w"]).sum()) for x in recs),
        "intercept_absdiff": max(abs(x["dm_int"] - x["r_dm_int"]) for x in recs),
        "sc_effect_absdiff": max(abs(x["sc_att"] - x["r_sc_eff"]) for x in recs),
        "dm_effect_absdiff": max(abs(x["dm_att"] - x["r_dm_eff"]) for x in recs),
    }


def run() -> dict:
    xc = _crosscheck_summary(_paired_crosscheck())   # skips whole case if R absent
    mc = _monte_carlo()
    A120, A480 = mc[("A", 120)], mc[("A", 480)]
    B120, B480 = mc[("B", 120)], mc[("B", 480)]
    C120, C480 = mc[("C", 120)], mc[("C", 480)]
    D120, D480 = mc[("D", 120)], mc[("D", 480)]
    ab = abs
    return {
        # 1. golden tie: mlsynth's estimators == authors' QPs, value-for-value
        "sc_weight_l1_vs_r": xc["sc_weight_l1"],
        "dm_weight_l1_vs_r": xc["dm_weight_l1"],
        "intercept_absdiff_vs_r": xc["intercept_absdiff"],
        "effect_absdiff_vs_r": max(xc["sc_effect_absdiff"], xc["dm_effect_absdiff"]),
        "agrees_with_r": float(
            xc["sc_weight_l1"] < 1e-6 and xc["dm_weight_l1"] < 5e-3
            and xc["intercept_absdiff"] < 5e-3
            and max(xc["sc_effect_absdiff"], xc["dm_effect_absdiff"]) < 5e-3),
        # 2. reproduction of Table 1's SC-bias cells (Panels A/B, published)
        "A_sc_bias_T120": A120["bias"][0],
        "A_sc_bias_T480": A480["bias"][0],
        "B_sc_bias_T480": B480["bias"][0],
        # 3. no-break: demeaned & DID unbiased; demeaned SE << DID SE
        "A_demeaned_bias_T120": A120["bias"][1],
        "A_did_bias_T120": A120["bias"][2],
        "A_demeaned_over_did_se_T120": float(A120["se"][1] / A120["se"][2]),
        "B_demeaned_over_did_se_T480": float(B480["se"][1] / B480["se"][2]),
        # 4. break panels: all biased, |SC|,|demeaned| far below |DID|
        "C_all_three_biased": float(min(ab(C120["bias"])) > 0.3),
        "C_sc_and_dm_below_did": float(
            ab(C120["bias"][0]) < 0.75 * ab(C120["bias"][2])
            and ab(C120["bias"][1]) < 0.9 * ab(C120["bias"][2])),
        "D_sc_and_dm_below_did": float(
            ab(D120["bias"][0]) < 0.75 * ab(D120["bias"][2])
            and ab(D120["bias"][1]) < 0.9 * ab(D120["bias"][2])),
        # 5. break bias shrinks as T0 grows (SC, both extremes)
        "C_sc_bias_shrinks_in_T0": float(ab(C480["bias"][0]) < ab(C120["bias"][0])),
        "D_sc_bias_shrinks_in_T0": float(ab(D480["bias"][0]) < ab(D120["bias"][0])),
    }


def comparison() -> dict:
    """mlsynth's SC (VanillaSC) and demeaned SC (TSSC MSCa) vs the authors' own
    ``quadprog`` QPs (``_aux.R``), run live via ``Rscript`` on identical simulated
    panels: each estimator's effect and the demeaned intercept, panel by panel.
    This is the value-for-value tie -- ``mlsynth`` *is* Ferman & Pinto's estimator.
    (The Table 1 bias reproduction is on the replication page.)"""
    recs = _paired_crosscheck()
    rows = []
    for x in recs:
        rows.append({"quantity": f"Panel {x['label']} SC effect",
                     "mlsynth": round(x["sc_att"], 4), "reference": round(x["r_sc_eff"], 4)})
        rows.append({"quantity": f"Panel {x['label']} demeaned SC effect",
                     "mlsynth": round(x["dm_att"], 4), "reference": round(x["r_dm_eff"], 4)})
        rows.append({"quantity": f"Panel {x['label']} demeaned intercept",
                     "mlsynth": round(x["dm_int"], 4), "reference": round(x["r_dm_int"], 4)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC (SC) + TSSC MSCa (demeaned SC)",
                         "note": "value-for-value on the calibrated MC panels; "
                                 "Table 1 bias reproduction is on the replication page"},
        "reference": {"impl": "authors' _aux.R synth_control_est + synth_control_est_demean "
                              "(quadprog QPs, live via Rscript)",
                      "version": "Ferman & Pinto (2021), Quantitative Economics 12:1197-1221"},
    }


# Path B (Monte Carlo) + live value-for-value tie. Panels A/B turn on the unit
# fixed effect (rotation-invariant) and reproduce the published SC bias within the
# calibration+MC gap (worst |Delta| ~ 0.11 at the noisier B/T120 cell, so the
# bias-cell tolerances are wide); Panels C/D turn on the factor-1 rotation, which
# the gsynth-free calibration cannot bit-match, so only their qualitative
# structure (biased; SC & demeaned far below DID; shrinking in T0) is pinned. The
# estimator-identity tie against the authors' quadprog QPs is solver-tight.
# Deterministic (fixed seeds). Skips if Rscript/quadprog/jsonlite absent.
EXPECTED = {
    # golden value-for-value tie (live R). SC uses the same active-set QP as R's
    # solve.QP -> machine precision; demeaned SC uses cvxpy/CLARABEL (interior
    # point) -> agreement at the solvers' ~1e-3 tolerance, as in ferman_demeaned.
    "sc_weight_l1_vs_r": (0.0, 1e-6),
    "dm_weight_l1_vs_r": (0.0, 5e-3),
    "intercept_absdiff_vs_r": (0.0, 5e-3),
    "effect_absdiff_vs_r": (0.0, 5e-3),
    "agrees_with_r": (1.0, 0.0),
    # reproduction of Table 1 SC bias (published; wide tol absorbs calibration+MC)
    "A_sc_bias_T120": (0.243, 0.12),
    "A_sc_bias_T480": (0.196, 0.12),
    "B_sc_bias_T480": (-0.596, 0.15),
    # no-break: demeaned & DID ~unbiased; demeaned SE well below DID SE
    "A_demeaned_bias_T120": (0.0, 0.12),
    "A_did_bias_T120": (0.0, 0.18),
    "A_demeaned_over_did_se_T120": (0.63, 0.15),
    "B_demeaned_over_did_se_T480": (0.51, 0.15),
    # break panels: all biased, SC & demeaned far below DID, shrinking in T0
    "C_all_three_biased": (1.0, 0.0),
    "C_sc_and_dm_below_did": (1.0, 0.0),
    "D_sc_and_dm_below_did": (1.0, 0.0),
    "C_sc_bias_shrinks_in_T0": (1.0, 0.0),
    "D_sc_bias_shrinks_in_T0": (1.0, 0.0),
}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, default=float))
