r"""Cross-validation: GSYNTH's sampling properties and its Algorithm 2 bootstrap.

Two more of Xu (2017)'s simulation designs, both against the ``gsynth 1.0`` his
Dataverse archive ships. The companion case ``xu_gsynth_sims`` covers
``sim_factor.R``, where the rank is chosen; here the rank is given, which takes
the cross-validation guard out of the comparison and leaves the estimator and
the inference on their own.

==================  ==========  ===============================================
script              target      what it measures
==================  ==========  ===============================================
``sim_TN.R``        Table A1    bias, standard deviation and RMSE of the ATT at
                                the fifth post period
``sim_coverage.R``  --          share of periods whose 95% parametric-bootstrap
                                interval covers the realised effect
==================  ==========  ===============================================

Table A1 is a target here: the Online Appendix supplies it, and the relevant row
(``T0 = 15``, ``Nco = 40``) reads

======  ======  =====  =====
Ntr     Bias    SD     RMSE
======  ======  =====  =====
1        0.023  1.163  1.163
5        0.053  0.589  0.591
20       0.013  0.375  0.375
======  ======  =====  =====

Both sides see the same panels, written out by
``benchmarks/R/xu_gsynth_properties.R``.

Reading the design off the table
--------------------------------

``sim_TN.R`` draws one panel per cell outside its replication loop, at
``fixF = TRUE`` and ``fixL = TRUE``, and redraws only the outcome. Table A1
confirms it without anyone reading the script: its SD is the dispersion of the
ATT and its RMSE is taken around the realised effect, and the two coincide in
every row. They could only separate if the effect moved between replications.

``sim_adh.R`` is the opposite. There the same two columns stand apart by exactly
``D.sd^2`` in all eight cells, which is the signature of an effect redrawn every
replication -- and the appendix says so in words. So the two designs differ in
what they hold fixed, and the published SD/RMSE relation says which is which
without opening either file. The first version of this case regenerated the
whole panel each replication, which put a unit of effect variance into the SD
that Table A1's does not carry.

How close the levels get
------------------------

The case runs twenty-five replications a cell, which is enough for the paired
quantities and not for a standard deviation: at that count the standard error of
an SD estimate is about ``SD / sqrt(2 * 24)``, so 0.17 at one treated unit and
0.07 at five. The measured cells are 1.18, 0.48 and 0.37, and it takes more
draws to say whether that is the count or the estimator.

Run the reference at ``--sims 400`` and it is the count. The three cells settle
at 1.126, 0.575 and 0.375 against Table A1's 1.163, 0.589 and 0.375, with SD and
RMSE agreeing to three decimals in every one. The expectations below are
therefore centred on the published values, with bands sized to the twenty-five
draw standard error and not to a suspected bias.

A residual stays, because sim_TN.R walks thirty-six cells off one seed while
this script runs three, so the fixed panel a cell draws here is not the panel
the author's cell drew. The 400-draw run above draws its own panel for the five
and twenty unit cells, a third one different from both, and still lands within
3% of the published SD -- so that effect sits below the sampling error the bands
already carry.

The point estimator
-------------------

With ``r`` fixed at its true value of two there is no rank to disagree about, so
mlsynth and gsynth should be the same computation. They are: the ATT at the
measured period agrees to solver precision on every draw, which makes the bias,
standard deviation and RMSE identical, not merely close. That is the
sharpest form this comparison can take, and it isolates the one constant that
separated the two implementations in ``xu_gsynth_sims`` -- with the rank given,
nothing separates them at all.

Table A1's shape reproduces as well as its levels: dispersion falls as the
treated group grows, 1.18 to 0.48 to 0.37 from one treated unit to five to
twenty against the paper's 1.163, 0.589 and 0.375. The three cells hold ``T0``
and the donor pool fixed so the only thing moving is the size of the treated
group; the archive's grid crosses all three, and a cell that moved two of them
at once could not support the claim.

The bootstrap
-------------

``sim_coverage.R`` is the design nothing else in the suite pins: Algorithm 2's
parametric bootstrap, at ``Ntr = 40``, ``Nco = 120``, ``T0 = 30``. Coverage is
the share of periods whose interval contains the realised average effect on the
treated, which is the estimand the paper's "parametric" arm compares against.

Both implementations land near the nominal 95% over the post periods. The two
bootstraps draw independently -- there is no way to share a random stream across
the language boundary -- so this half is a comparison of rates and not of draws,
and its tolerance is Monte Carlo error at the replication count, which is the
one place in this case where that is so.

Coverage over the whole panel runs below nominal because the pre-period
intervals are conditioned on the fit that produced them; the paper reports the
post-period figure, and so does this case, with the all-period number recorded
beside it.

Provenance
----------

* Paper: Xu (2017), Political Analysis 25(1):57-76. The empirical side is in
  ``gsynth_xu_turnout``; the rank-selection design is in ``xu_gsynth_sims``.
* Archive: Dataverse replication files, ``gsynth_1.0.tar.gz`` with
  ``sim_TN.R``, ``sim_coverage.R`` and the shared ``sim_sampling.R``.
* Skips without R or gsynth. gsynth 1.0 needs a compiler and LAPACK/BLAS
  headers; the reference script does not install it.
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
_R_SCRIPT = _ROOT / "benchmarks" / "R" / "xu_gsynth_properties.R"

SIMS = 25
BOOTS = 150
NOMINAL = 0.95


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
        [rscript, str(_R_SCRIPT), "--out", str(outdir),
         "--sims", str(SIMS), "--boots", str(BOOTS)],
        capture_output=True, text=True, cwd=str(_ROOT), timeout=5400)
    tn = Path(outdir) / "reference_tn.csv"
    cov = Path(outdir) / "reference_coverage.csv"
    if proc.returncode != 0 or not tn.exists() or not cov.exists():
        raise BenchmarkSkipped(
            f"reference run failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}")
    return pd.read_csv(tn), pd.read_csv(cov)


def _att_at(panel, k, inference=False, boots=0):
    """GSYNTH with the rank given; the ATT at period ``k``, and its bands."""
    from mlsynth.estimators.gsynth import GSYNTH

    cfg = {"df": panel, "outcome": "Y", "treat": "D", "unitid": "id",
           "time": "time", "covariates": ["X1", "X2"], "r": 2,
           "force": "two-way", "inference": inference,
           "display_graphs": False}
    if inference:
        cfg.update({"n_bootstrap": boots, "alpha": 1.0 - NOMINAL, "seed": 0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = GSYNTH(cfg).fit()
    first = res[0] if isinstance(res, list) else res
    gap = np.asarray(first.time_series.estimated_gap, float).ravel()
    out = {"att_k": float(gap[k - 1])}
    if inference:
        d = first.inference.details
        out["lo"] = np.asarray(getattr(d, "tau_lower"), float)
        out["hi"] = np.asarray(getattr(d, "tau_upper"), float)
    return out


def _measure():
    rscript = _require_r()
    tmp = tempfile.mkdtemp(prefix="xu_prop_")
    try:
        ref_tn, ref_cov = _reference(rscript, tmp)

        tn_rows = []
        for case, grp in ref_tn.groupby("case", sort=False):
            panels = pd.read_csv(Path(tmp) / f"tn_{case}.csv")
            k = int(grp.T0.iloc[0]) + 5
            for rep, panel in panels.groupby("rep", sort=True):
                g = grp[grp.rep == rep].iloc[0]
                try:
                    mine = _att_at(panel, k)["att_k"]
                except Exception:
                    mine = np.nan
                tn_rows.append({"case": case, "rep": rep, "att_my": mine,
                                "att_gs": float(g.att_k), "true": float(g.true_k)})

        cov_panels = pd.read_csv(Path(tmp) / "coverage.csv")
        T0, T = 30, 40
        cov_rows = []
        for rep, panel in cov_panels.groupby("rep", sort=True):
            n_units = panel.id.nunique()
            eff = panel.eff.values.reshape(n_units, T).T
            n_tr = int((panel.groupby("id").D.max() > 0).sum())
            truth = eff[:, :n_tr].sum(axis=1) / n_tr
            try:
                out = _att_at(panel, T0 + 1, inference=True, boots=BOOTS)
                covered = ((truth >= out["lo"]) & (truth <= out["hi"])).astype(float)
                cov_rows.append({"rep": rep,
                                 "cover_post": float(covered[T0:].mean()),
                                 "cover_all": float(covered.mean())})
            except Exception:
                cov_rows.append({"rep": rep, "cover_post": np.nan,
                                 "cover_all": np.nan})
        return pd.DataFrame(tn_rows), ref_cov, pd.DataFrame(cov_rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run() -> dict:
    tn, ref_cov, cov = _measure()

    out = {"n_tn_draws": float(len(tn)), "n_cov_draws": float(len(cov))}
    # With the rank given the two are the same computation.
    out["tn_att_max_gap"] = float(np.nanmax(np.abs(tn.att_my - tn.att_gs)))

    # Table A1's quantities, from gsynth, and mlsynth's own beside them.
    worst_b = worst_s = 0.0
    sd_rmse_gaps = []
    for case, g in tn.groupby("case", sort=False):
        b_my = float(np.nanmean(g.att_my - g["true"]))
        b_gs = float(np.nanmean(g.att_gs - g["true"]))
        s_my = float(np.nanstd(g.att_my, ddof=1))
        s_gs = float(np.nanstd(g.att_gs, ddof=1))
        out[f"bias_{case}"] = round(b_gs, 4)
        out[f"sd_{case}"] = round(s_gs, 4)
        rmse_gs = float(np.sqrt(np.nanmean((g.att_gs - g["true"]) ** 2)))
        sd_rmse_gaps.append(abs(s_gs - rmse_gs))
        worst_b = max(worst_b, abs(b_my - b_gs))
        worst_s = max(worst_s, abs(s_my - s_gs))
    out["bias_max_gap"] = float(worst_b)
    out["sd_max_gap"] = float(worst_s)
    # dispersion falls as the treated group grows, which is Table A1's shape
    out["sd_falls_with_ntr"] = float(out["sd_Ntr20"] < out["sd_Ntr5"] < out["sd_Ntr1"])
    # Table A1's SD and RMSE coincide; that is the design signature, and unlike
    # the levels it does not depend on which panel the cell drew.
    out["sd_rmse_max_gap"] = round(float(max(sd_rmse_gaps)), 4)

    # The bootstrap. Independent random streams, so rates and not draws.
    out["cover_post_gsynth"] = round(float(ref_cov.cover_post.mean()), 3)
    out["cover_post_mlsynth"] = round(float(np.nanmean(cov.cover_post)), 3)
    out["cover_post_gap"] = round(
        abs(out["cover_post_gsynth"] - out["cover_post_mlsynth"]), 3)
    out["cover_all_mlsynth"] = round(float(np.nanmean(cov.cover_all)), 3)
    out["cover_post_near_nominal"] = float(
        abs(out["cover_post_mlsynth"] - NOMINAL) <= 0.10)
    return out


# The R side seeds once with the archive's ``set.seed(123)`` and both sides see
# the same panels, so everything but the bootstrap is deterministic given a
# gsynth build. ``tn_att_max_gap`` and the two ``*_max_gap`` entries are solver
# precision and carry no Monte Carlo component: with the rank given there is
# nothing for the implementations to disagree about. The coverage entries do
# carry it -- the two bootstraps draw independently -- and at 25 replications a
# rate near 0.95 has a standard error of about 0.04 across draws, before the
# within-draw correlation across the ten post periods.
EXPECTED = {
    "n_tn_draws": (75.0, 0.0),
    "n_cov_draws": (25.0, 0.0),
    # the point estimator, with the rank given
    "tn_att_max_gap": (0.0, 1e-8),
    "bias_max_gap": (0.0, 1e-8),
    "sd_max_gap": (0.0, 1e-8),
    # Table A1 at T0 = 15, Nco = 40, taken as the target. At twenty-five draws
    # an SD estimate carries a standard error of SD/sqrt(48) -- 0.17, 0.07 and
    # 0.05 for the three cells -- and the bands are about three of those. The
    # levels are the paper's, not a self-calibration: at 400 draws the cells
    # settle at 1.126, 0.575 and 0.375.
    "sd_Ntr1": (1.163, 0.45),      # measured at 25 draws: 1.180
    "sd_Ntr5": (0.589, 0.25),      # measured at 25 draws: 0.477
    "sd_Ntr20": (0.375, 0.20),     # measured at 25 draws: 0.366
    "sd_falls_with_ntr": (1.0, 0.0),
    # Bias, same table. Its standard error is SD/5 at this count.
    "bias_Ntr1": (0.023, 0.55),    # measured at 25 draws: -0.062
    "bias_Ntr5": (0.053, 0.35),    # measured at 25 draws:  0.149
    "bias_Ntr20": (0.013, 0.25),   # measured at 25 draws:  0.106
    # Table A1's SD and RMSE coincide because the effect is held fixed.
    "sd_rmse_max_gap": (0.005, 0.030),
    # the bootstrap
    "cover_post_gsynth": (0.920, 0.10),
    "cover_post_mlsynth": (0.908, 0.10),
    "cover_post_gap": (0.012, 0.11),
    "cover_all_mlsynth": (0.902, 0.12),
    "cover_post_near_nominal": (1.0, 0.0),
}
