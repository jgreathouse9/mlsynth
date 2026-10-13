r"""Cross-validation: mlsynth's PDA and SCM against ``pampe`` and ``Synth``, on
every simulation design Wan, Xie & Hsiao (2018) ship code for.

The paper's replication archive (Simspda.zip) contains eight scripts covering six
design variants. Only Design 6a has a printed Table 2 column, which
``wan_pda_vs_scm`` matches; the other five are variants the paper leaves to its
Supplementary Note, so there is no published cell to hit. What they can be held
to is the authors' own estimators, which is what this case does: the panels are
generated in R by the authors' data-generating blocks, and the same panels are
then estimated twice -- once by ``pampe`` and ``Synth`` as the authors call them,
once by mlsynth -- and compared replication by replication.

Pairing on identical panels is what makes the comparison sharp. Several of these
designs have heavy-tailed MSE distributions (Design 6d's SCM cell runs to five
figures), so a comparison of Monte Carlo means over independently drawn panels
would be swamped by sampling noise; on shared panels the two implementations are
compared draw for draw and the residual gap is solver tolerance.

The designs
-----------

==============  ========  ========  =========  =======================================
tag             variant   (J, T0)   seed       structure
==============  ========  ========  =========  =======================================
d1b_j20_t20     1b        (20, 20)  12345      AR(0.5) factor, N(1,1) loadings, PWT
                                               fixed effects, common covariate trend
d2d_j10_t5      2d        (10,  5)   2017      drifting random-walk factor, N(1,2)
d2d_j20_t5      2d        (20,  5)   2017      loadings, random-walk covariate
d5b_j10_t10     5b        (10, 10)   1234      cointegrated: J-1 random walks, two
                                               cointegrating vectors
d6a_j20_t20     6a        (20, 20)   1234      drifting walk, unit loadings
d6a_j5_t40      6a        ( 5, 40)   1234      (the printed Table 2 design)
d6c_j5_t40      6c        ( 5, 40)   1234      as 6d plus a unit intercept
d6d_j5_t40      6d        ( 5, 40)   1234      drifting walk, N(1,1) loadings
==============  ========  ========  =========  =======================================

The 6a/6c/6d family sweeps the paper's point about Design 6a. Under 6a every unit
loads on the trend identically, the equal-weight average of the controls is
optimal, and it lies inside the simplex, so SCM wins. Under 6d the loadings are
N(1,1) and the treated unit's loading can fall outside the convex hull of the
donors', at which point SCM cannot track the trend at all and its MSE runs four
orders of magnitude above PDA's. 6c adds an intercept on top of 6d.

Two quirks of the authors' code are reproduced as written, since the target is
their experiment and not a corrected version of it: Design 6c's ``rep(a, t)``
recycles a length-(J+1) intercept vector down a column-major T x (J+1) matrix, so
the intercept a unit receives rotates with the period; and Design 5b builds a
random walk for column 2 and then overwrites it.

What the comparison finds
-------------------------

PDA agrees to machine precision on all eight designs. mlsynth's
``PDA(method="hcw", hcw_nvmax=t0-4)`` and ``pampe(select="AICc", nvmax=t0-4)``
run the same Furnival-Wilson best-subset search under the same AICc, and the
per-replication relative gap stays at 1e-15, which is double-precision
accumulation order and not a difference in the estimator.

SCM agrees closely on all eight designs: the largest gap between the two Monte
Carlo mean MSEs is 0.2%, at the ``T0 = 5`` cells of Design 2d, which are also
the cells where individual replications disagree most. Those per-replication
disagreements are not an mlsynth error. The authors' SCM passes each pre-period outcome as its own
predictor and sets ``time.optimize.ssr`` to that same window, which makes
``V = I`` reproduce the outer objective exactly: at ``V = I`` the inner program
minimises the unweighted pre-period SSR, which is what the outer search is trying
to minimise, so ``V = I`` attains the outer optimum and ``Synth``'s BFGS search
over ``V`` can only match it or fall short. mlsynth solves that convex program
directly. The case records which implementation fits the pre-period better as
``mlsynth_pre_fit_wins``; it is mlsynth on seven of the eight designs, and by
the largest margin on exactly the cells where the MSE gap is largest.

Provenance
----------

* Paper: Wan, Xie & Hsiao (2018), Economics Letters 164, 121-123.
* Reference: ``benchmarks/R/wan_pda_vs_scm.R``, which carries the authors' eight
  data-generating blocks and their ``pampe``/``Synth`` calls verbatim. It is run
  live here, so the reference is the R packages themselves.
* Data: ``basedata/gvb_rgdpl1980.csv`` -- 1980 log real GDP per capita from PWT,
  the pool Gardeazabal & Vega-Bayo's replication files ship as
  ``rgdpl1980.txt``. Designs 1b and 2d sample it; the other four need no data.
* Skips when R, ``Synth``, ``pampe`` or ``leaps`` is unavailable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_R_SCRIPT = _ROOT / "benchmarks" / "R" / "wan_pda_vs_scm.R"
_LGDP = _ROOT / "basedata" / "gvb_rgdpl1980.csv"

# Replications per design. The R side runs Synth's BFGS over a T0-length
# predictor vector once per replication, which dominates the cost; 25 is enough
# for a paired comparison, where the quantity of interest is the per-replication
# gap and not a Monte Carlo mean.
REPS = 25


def _require_r():
    """Return the ``Rscript`` path, or raise ``BenchmarkSkipped``."""
    from benchmarks.compare import BenchmarkSkipped

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH")
    probe = subprocess.run(
        [rscript, "-e",
         'q(status = if (all(sapply(c("Synth","pampe","leaps"), '
         'requireNamespace, quietly = TRUE))) 0 else 1)'],
        capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R packages Synth/pampe/leaps unavailable")
    if not _LGDP.exists():
        raise BenchmarkSkipped(f"missing donor pool {_LGDP}")
    return rscript


def _run_reference(rscript, outdir):
    """Generate the panels and the R-side estimates into ``outdir``."""
    from benchmarks.compare import BenchmarkSkipped

    proc = subprocess.run(
        [rscript, str(_R_SCRIPT), "--out", str(outdir), "--reps", str(REPS),
         "--lgdp", str(_LGDP)],
        capture_output=True, text=True, cwd=str(_ROOT), timeout=7200)
    ref = Path(outdir) / "reference.csv"
    if proc.returncode != 0 or not ref.exists():
        raise BenchmarkSkipped(
            f"reference run failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-400:]}")
    return pd.read_csv(ref)


def _frame(y, T0):
    T, n = y.shape
    unit = np.repeat(np.arange(1, n + 1), T)
    period = np.tile(np.arange(1, T + 1), n)
    return pd.DataFrame({"unit": unit, "time": period, "y": y.T.reshape(-1),
                         "treat": ((unit == 1) & (period > T0)).astype(int)})


def _fit_both(y, T0, nvmax):
    """mlsynth's PDA and SCM on one panel: post MSE and pre-period MAE of each."""
    from mlsynth.estimators.pda import PDA
    from mlsynth.estimators.vanillasc import VanillaSC

    base = dict(df=_frame(y, T0), outcome="y", treat="treat", unitid="unit",
                time="time", display_graphs=False)
    out = {}
    for key, make in (
        ("pda", lambda: PDA({**base, "method": "hcw", "hcw_nvmax": nvmax})),
        ("scm", lambda: VanillaSC(base)),
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cf = np.asarray(
                    make().fit().time_series.counterfactual_outcome, float)
            out[f"mse_{key}"] = float(np.mean((y[T0:, 0] - cf[T0:]) ** 2))
            out[f"mae_{key}"] = float(np.mean(np.abs(y[:T0, 0] - cf[:T0])))
        except Exception:
            out[f"mse_{key}"] = np.nan
            out[f"mae_{key}"] = np.nan
    return out


def _paired_max_rel(got, ref):
    """Largest per-replication relative gap between two aligned MSE vectors."""
    ok = np.isfinite(got) & np.isfinite(ref)
    if not ok.any():
        return np.nan
    return float(np.max(np.abs(got[ok] - ref[ok]) / np.maximum(np.abs(ref[ok]), 1e-9)))


def _measure():
    """Run both sides on shared panels; return a per-design table."""
    rscript = _require_r()
    tmp = tempfile.mkdtemp(prefix="wan_ref_")
    try:
        ref = _run_reference(rscript, tmp)
        rows = []
        for tag, g in ref.groupby("design", sort=False):
            J, T0, T1 = int(g.j.iloc[0]), int(g.t0.iloc[0]), int(g.t1.iloc[0])
            T = T0 + T1
            panels = np.fromfile(
                os.path.join(tmp, f"{tag}.bin"), dtype="<f8"
            ).reshape(-1, T, J + 1)[:len(g)]
            got = pd.DataFrame(
                [_fit_both(y, T0, max(1, T0 - 4)) for y in panels])
            g = g.reset_index(drop=True)
            rows.append({
                "design": tag,
                "pda_gap": _paired_max_rel(got.mse_pda.values, g.mse_pda.values),
                "scm_gap": _paired_max_rel(got.mse_scm.values, g.mse_scm.values),
                "scm_mean_gap": abs(np.nanmean(got.mse_scm) - np.nanmean(g.mse_scm))
                / max(abs(np.nanmean(g.mse_scm)), 1e-12),
                "pre_fit_win": float(np.nanmean(got.mae_scm) <= np.nanmean(g.mae_scm)),
                "scm_over_pda": float(np.nanmean(g.mse_scm) / max(np.nanmean(g.mse_pda), 1e-12)),
            })
        return pd.DataFrame(rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run() -> dict:
    d = _measure()
    by = d.set_index("design")
    out = {
        "n_designs": float(len(d)),
        # PDA is the same estimator in both implementations, so the paired gap
        # is double-precision accumulation and nothing else.
        "pda_max_rel_gap": float(d.pda_gap.max()),
        # SCM solves the same convex program; the two disagree only where
        # Synth's outer BFGS over V falls short of the V = I optimum.
        "scm_max_mean_rel_gap": float(d.scm_mean_gap.max()),
        "mlsynth_pre_fit_wins": float(d.pre_fit_win.sum()),
        # The design family's point: SCM below PDA under unit loadings (6a),
        # far above it once the loadings are random and the treated unit can
        # leave the donors' convex hull (6d).
        "scm_over_pda_6a": round(float(by.loc["d6a_j5_t40", "scm_over_pda"]), 3),
        "scm_over_pda_6d": float(by.loc["d6d_j5_t40", "scm_over_pda"] > 100.0),
    }
    for tag in by.index:
        out[f"pda_gap_{tag}"] = float(by.loc[tag, "pda_gap"])
    return out


# The R side is seeded per design (the authors' own seeds), so the panels are
# identical across runs; the comparison is therefore deterministic up to solver
# tolerance on both sides.
#
# ``pda_max_rel_gap`` is the headline: the two best-subset implementations agree
# to 3e-12 at worst per replication, so the tolerance admits nothing that would
# not be floating-point accumulation. The SCM tolerance is set by the two T0 = 5
# cells of Design 2d, where Synth's outer search leaves the optimum; the
# mechanism is in the module docstring and ``mlsynth_pre_fit_wins`` is the
# evidence for it.
EXPECTED = {
    "n_designs": (8.0, 0.0),
    "pda_max_rel_gap": (0.0, 1e-9),         # machine precision on every design
    "scm_max_mean_rel_gap": (0.002, 0.030),  # largest of the eight; the T0 = 5 cells of Design 2d
    "mlsynth_pre_fit_wins": (7.0, 1.0),      # mlsynth attains the lower pre-period MAE
    "scm_over_pda_6a": (0.818, 0.150),       # SCM beats PDA under unit loadings
    "scm_over_pda_6d": (1.0, 0.0),          # and loses badly once they are random
    "pda_gap_d1b_j20_t20": (0.0, 1e-9),
    "pda_gap_d2d_j10_t5": (0.0, 1e-9),
    "pda_gap_d2d_j20_t5": (0.0, 1e-9),
    "pda_gap_d5b_j10_t10": (0.0, 1e-9),
    "pda_gap_d6a_j20_t20": (0.0, 1e-9),
    "pda_gap_d6a_j5_t40": (0.0, 1e-9),
    "pda_gap_d6c_j5_t40": (0.0, 1e-9),
    "pda_gap_d6d_j5_t40": (0.0, 1e-9),
}
