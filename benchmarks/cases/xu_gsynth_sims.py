r"""Cross-validation: mlsynth's GSYNTH against gsynth 1.0 on Xu's own simulations.

Xu, Y. (2017), "Generalized Synthetic Control Method: Causal Inference with
Interactive Fixed Effects Models", Political Analysis 25(1):57-76. The
Dataverse replication archive ships ``gsynth_1.0.tar.gz`` alongside seven
simulation scripts, each targeting a table in the Online Appendix:

==================  ==========  ===============================================
script              target      what it varies
==================  ==========  ===============================================
``sim_TN.R``        Table A1    finite-sample bias over ``T0``/``Nco``/``Ntr``
``sim_DID.R``       Table A2    against difference-in-differences, over ``w``
``sim_inter.R``     Table A3    against interactive fixed effects
``sim_adh.R``       Table A4    against ADH synthetic control, over ``r``, ``w``
``sim_factor.R``    Table A5    does cross-validation find the factor count
``sim_coverage.R``  --          parametric bootstrap interval coverage
``sim_sampling.R``  --          the shared data-generating process
==================  ==========  ===============================================

The case runs ``sim_factor.R``'s design against Table A5, and cross-validates
both implementations against each other on the way. The cross-validation is the
sharper of the two: the author's own implementation, at the version the paper
ran, on the author's own data-generating process, with both sides seeing the
same panels, so its residual is implementation difference and not Monte Carlo
noise. Table A5 supplies the level the rates should sit at.

The design is ``sim_factor.R``'s: a Bai (2009) interactive fixed-effects panel
with two factors, two covariates, unit and time effects, treated/control loading
overlap ``w = 0.5``, and a treatment effect of 1..10 over ten post periods. Four
of its eighteen cells are run, the ones at ``Ntr = 5`` where the factor-count
choice is under most pressure.

What the comparison finds
-------------------------

The two implementations are the same estimator up to one constant.

* Rank agreement is 97.5% over 600 draws.
* Where the rank agrees, the ATT agrees to 3.6e-14, which is accumulation order
  and not a difference in the estimator.
* Every one of the fifteen disagreements is the guard, and nothing else.

That last point is the mechanism. Algorithm 1 does not take
the rank that minimises cross-validated MSPE; it walks upward and takes a larger
rank only when the improvement beats the running minimum by a relative margin::

    gsynth, R/gsynth.R:524
        if ((min(CV.out[,"MSPE"]) - MSPE) > tol * min(CV.out[,"MSPE"])) r.cv <- r

``gsynth 1.0`` defaults that ``tol`` to 0.01. ``gsynth 1.2.1`` defaults it to
0.001, and mlsynth follows 1.2.1 (``CV_GUARD`` in
``mlsynth/utils/gsynth_helpers/pipeline.py``, which records why: raising it to
0.01 costs five of the sixteen rank agreements in ``gsynth_av_laws``, pinned
against a live 1.2.1 run).

So the two should differ exactly when a rank's improvement lands between 0.1%
and 1%, and on nothing else. Measured over the fifteen disagreements the
improvement runs from 0.1011% to 0.9393%, median 0.4900% -- all fifteen inside
the band, none above 1%, none below 0.1%. The case asserts that share, so a
change to either the guard or the cross-validation shows up as a mechanism
failure and not as a drifting agreement rate.

Where the ranks do differ the ATT on that draw differs materially, up to 0.50,
which is why the guard is a modelling choice and not a formality. The Monte
Carlo means still agree to about 0.004, since the disagreements are rare and
unsigned.

Table A5, reproduced
--------------------

``sim_factor.R`` reports how often cross-validation recovers the true rank of
two. Table A5's ``Ntr = 5`` column, over 5,000 samples a cell, against sixty
here through gsynth 1.0:

==============  ========  ==========  ========
cell            Table A5  gsynth 1.0   mlsynth
==============  ========  ==========  ========
T0=10, Nco=40      0.801       0.840     0.847
T0=30, Nco=40      0.921       0.920     0.887
T0=15, Nco=80      0.896       0.893     0.867
T0=15, Nco=120     0.895       0.880     0.867
==============  ========  ==========  ========

Every cell is inside its Monte Carlo error. A rate near 0.9 on a hundred and
fifty draws carries a standard error of 0.027; the largest gap to the published
value is 0.039 for gsynth 1.0 and 0.046 for mlsynth, so under two of those. The
expectations below are the published numbers with bands of 0.12, which leaves
room for the count without leaving room for a drift in the cross-validation.

The shape reproduces too. Table A5 has recovery rising with ``T0`` (0.801 at ten
pre-periods against 0.921 at thirty) and with the donor pool, because both give
the validation step more to work with; the measured cells move the same way.

Provenance
----------

* Paper: Xu (2017), Political Analysis 25(1):57-76, with the Online Appendix
  supplying Table A5. The empirical side is already covered by
  ``gsynth_xu_turnout`` (Table 2 columns 3 and 4).
* Archive: Dataverse replication files, ``gsynth_1.0.tar.gz`` plus the seven
  simulation scripts. ``benchmarks/R/xu_gsynth_sims.R`` carries the relevant
  part of ``sim_sampling.R``'s generator and the ``sim_factor.R`` call.
* Skips when R, or gsynth, is unavailable. gsynth 1.0 needs a compiler and
  LAPACK/BLAS headers; the reference script does not install it.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_R_SCRIPT = _ROOT / "benchmarks" / "R" / "xu_gsynth_sims.R"

# Replications per cell. The quantities that decide the case -- the ATT
# agreement at a shared rank, and whether every disagreement is the guard -- are
# per-draw and need few. The rank-recovery rates are proportions measured
# against Table A5, and there the count is what decides how tight the band can
# be: at sixty a rate near 0.9 carries a standard error of 0.046, at a hundred
# and fifty it carries 0.029. The whole case runs in about a minute either way.
SIMS = 150

# gsynth 1.0 accepts a larger rank on a 1% improvement, mlsynth on 0.1%.
GUARD_GSYNTH_10 = 0.01
GUARD_MLSYNTH = 0.001


def _require_r():
    from benchmarks.compare import BenchmarkSkipped

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH")
    probe = subprocess.run(
        [rscript, "-e", 'q(status = if (requireNamespace("gsynth", quietly = TRUE)) 0 else 1)'],
        capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R package gsynth unavailable")
    return rscript


def _reference(rscript, outdir):
    from benchmarks.compare import BenchmarkSkipped

    proc = subprocess.run(
        [rscript, str(_R_SCRIPT), "--out", str(outdir), "--sims", str(SIMS)],
        capture_output=True, text=True, cwd=str(_ROOT), timeout=3600)
    ref = Path(outdir) / "reference.csv"
    if proc.returncode != 0 or not ref.exists():
        raise BenchmarkSkipped(
            f"reference run failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}")
    return pd.read_csv(ref)


def _fit(panel):
    """mlsynth's GSYNTH with the rank left to Algorithm 1."""
    from mlsynth.estimators.gsynth import GSYNTH

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = GSYNTH({"df": panel, "outcome": "Y", "treat": "D", "unitid": "id",
                      "time": "time", "covariates": ["X1", "X2"],
                      "r": None, "r_max": 5, "force": "two-way",
                      "inference": False, "display_graphs": False}).fit()
    first = res[0] if isinstance(res, list) else res
    params = first.method_details.parameters_used or {}
    return params.get("r"), float(first.effects.att)


def _measure():
    rscript = _require_r()
    tmp = tempfile.mkdtemp(prefix="xu_gsynth_")
    try:
        ref = _reference(rscript, tmp)
        rows = []
        for case, grp in ref.groupby("case", sort=False):
            panels = pd.read_csv(Path(tmp) / f"{case}.csv")
            for rep, panel in panels.groupby("rep", sort=True):
                g = grp[grp.rep == rep].iloc[0]
                try:
                    r_my, att_my = _fit(panel)
                except Exception:
                    r_my, att_my = np.nan, np.nan
                rows.append({
                    "case": case, "rep": rep,
                    "r_my": r_my, "r_gs": int(g.r_cv),
                    "att_my": att_my, "att_gs": float(g.att_avg),
                    **{f"m{k}": g[f"mspe{k}"] for k in range(6)}})
        return pd.DataFrame(rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _guard_band_share(disagree):
    """Share of rank disagreements explained by the two versions' guards.

    For each, the improvement gsynth 1.0 declined to take, relative to the MSPE
    at the rank it kept. Inside ``(0.001, 0.01]`` the two constants disagree by
    construction and nothing else needs explaining.
    """
    inside = []
    for _, row in disagree.iterrows():
        a, b = int(row.r_gs), int(row.r_my)
        m_a, m_b = row[f"m{a}"], row[f"m{b}"]
        if not (np.isfinite(m_a) and np.isfinite(m_b)) or m_a <= 0:
            continue
        imp = (m_a - m_b) / m_a
        inside.append(GUARD_MLSYNTH < imp <= GUARD_GSYNTH_10)
    return (float(np.mean(inside)) if inside else float("nan")), len(inside)


def run() -> dict:
    d = _measure()
    same = d[d.r_my == d.r_gs]
    diff = d[(d.r_my != d.r_gs) & d.r_my.notna()]
    band, n_band = _guard_band_share(diff)

    out = {
        "n_draws": float(len(d)),
        "rank_agreement": round(float(len(same) / len(d)), 3),
        # The estimator itself: at a shared rank the two are the same program.
        "att_max_gap_same_rank": float(np.abs(same.att_my - same.att_gs).max()),
        # and every disagreement is the guard, not a second mechanism
        "disagreements_in_guard_band": band,
        "n_disagreements": float(n_band),
        # a rank disagreement moves that draw's ATT materially, so the guard is
        # a modelling choice and not a formality
        "att_max_gap_diff_rank": round(
            float(np.abs(diff.att_my - diff.att_gs).max()) if len(diff) else 0.0, 3),
        # but the Monte Carlo means still line up
        "att_mean_gap_max": round(float(max(
            abs(g.att_my.mean() - g.att_gs.mean()) for _, g in d.groupby("case"))), 3),
    }
    # Table A5's quantity: how often cross-validation recovers the true rank.
    for case, g in d.groupby("case", sort=False):
        out[f"gs_correct_{case}"] = round(float(np.mean(g.r_gs == 2)), 3)
        out[f"my_correct_{case}"] = round(float(np.mean(g.r_my == 2)), 3)
    out["mlsynth_correct_rank_min"] = round(
        float(min(np.mean(g.r_my == 2) for _, g in d.groupby("case"))), 3)
    out["correct_rank_max_case_gap"] = round(float(max(
        abs(np.mean(g.r_my == 2) - np.mean(g.r_gs == 2)) for _, g in d.groupby("case"))), 3)
    return out


# The R side seeds once with the archive's ``set.seed(123)`` and both sides see
# the same panels, so the per-draw quantities are deterministic given a gsynth
# build. The rate tolerances absorb Monte Carlo error at 60 draws a cell, where
# a proportion near 0.9 carries a standard error of about 0.04.
EXPECTED = {
    "n_draws": (600.0, 0.0),
    "rank_agreement": (0.975, 0.045),
    # A shared rank makes the two the same computation.
    "att_max_gap_same_rank": (0.0, 1e-8),
    # Every disagreement sits between the two guards. This is the mechanism, so
    # it is pinned at exactly 1 with no slack.
    "disagreements_in_guard_band": (1.0, 0.0),
    "n_disagreements": (15.0, 9.0),
    "att_max_gap_diff_rank": (0.50, 0.45),
    "att_mean_gap_max": (0.004, 0.030),
    # Table A5, Ntr = 5, taken as the target. A proportion near 0.9 on sixty
    # draws has a standard error of about 0.04; the bands are three of those.
    # gsynth 1.0's measured rates are beside each.
    "gs_correct_T10_Nco40": (0.801, 0.12),   # measured 0.840
    "gs_correct_T30_Nco40": (0.921, 0.12),   # measured 0.920
    "gs_correct_T15_Nco80": (0.896, 0.12),   # measured 0.893
    "gs_correct_T15_Nco120": (0.895, 0.12),  # measured 0.880
    # and mlsynth's own, against the same table. It differs from gsynth 1.0 only
    # where the guard splits them, so it is held to the same band.
    "my_correct_T10_Nco40": (0.801, 0.12),
    "my_correct_T30_Nco40": (0.921, 0.12),
    "my_correct_T15_Nco80": (0.896, 0.12),
    "my_correct_T15_Nco120": (0.895, 0.12),
    "mlsynth_correct_rank_min": (0.847, 0.12),
    "correct_rank_max_case_gap": (0.033, 0.060),
}
