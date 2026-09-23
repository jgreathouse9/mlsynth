r"""Xu (2017) Table A4: the factor model against the convex hull.

Path B. Xu, Y. (2017), "Generalized Synthetic Control Method", Political
Analysis 25(1):57-76, Online Appendix Table A4, "Comparison with the Synthetic
Control Estimator (ADH 2010)". One treated unit, forty donors, fifteen
pre-periods, ten post, and eight cells. Cases 1-4 hold the loading overlap
``w`` at one and give the estimator a rank of 1, 2, 3, 4 against a process that
always has two factors, so only case 2 is correctly specified. Cases 5-8 keep
the rank right and pull the treated unit's loadings away from the donors' until,
at ``w = 0``, the supports do not overlap and no convex combination of donors
can reach the treated unit.

What the table is for
---------------------

The synthetic control's counterfactual is a weighted average of donors with
non-negative weights summing to one, so it lives in the convex hull of the donor
paths. When the treated unit's factor loadings sit outside the donors', nothing
in that hull matches it, and no amount of pre-period fit repairs that. The
factor model has no such constraint: it estimates the loadings and can place the
treated unit anywhere. Table A4 is that difference measured.

===  ===  ====  =========  =========  ===========  ===========
k     r      w  GSC bias   GSC SD     Synth bias   Synth SD
===  ===  ====  =========  =========  ===========  ===========
1      1  1.00     -0.010      1.494       -0.011        1.739
2      2  1.00      0.000      1.571       -0.022        2.029
3      3  1.00     -0.003      1.581        0.021        2.368
4      4  1.00      0.013      1.610       -0.013        2.345
5      2  0.75      0.014      1.595        0.707        2.096
6      2  0.50      0.026      1.602        1.331        2.327
7      2  0.25      0.000      1.729        1.630        2.492
8      2  0.00      0.033      1.822        2.127        2.610
===  ===  ====  =========  =========  ===========  ===========

The synthetic control is unbiased while the supports overlap, including when the
factor model is given the wrong rank, and its bias then climbs to 2.13 as the
overlap closes. The factor model's bias never leaves the third decimal and it is
the tighter of the two in every cell. The estimators split on the hull, not on
the rank.

Where the archive's script departs from the table
-------------------------------------------------

``sim_adh.R`` does not reproduce Table A4, in two ways, and both were settled by
measurement. This matters beyond the case: the script is the obvious starting
point, and running it as shipped produces numbers that look like a failed
replication.

The panel. The script draws one panel per cell outside its replication loop and
then redraws only the outcome, which holds the treatment effect fixed. The
appendix says the opposite -- "the treatment effect, regressors, factor
loadings, and error terms are drawn repeatedly" -- and the table agrees: across
the eight cells ``SD^2 - RMSE^2`` is 1.01, 1.05, 1.01, 1.02, 1.02, 0.99, 1.05,
1.01, which is ``D.sd^2`` for an effect redrawn every replication and would be
zero for one held. (Its companion ``sim_TN.R`` is the other case: there SD and
RMSE coincide in every row of Table A1, and ``xu_gsynth_properties`` runs that
design with the panel held.)

The covariates. The script's header sets ``p <- 0  # no covariates`` and then
calls the generator with ``p = 2`` and ``beta = c(1, 3)``, while leaving the
covariates out of ``gsynth``'s formula. Table A4 was generated at ``p = 0``, and
the arithmetic says so twice. With ``p = 2`` the independent part of ``X``
carries variance ``1 + 9`` into the outcome with nothing to absorb it, and the
factor model's SD comes out near 4 against the table's 1.5 to 1.8. And ``X``
contains ``0.5 * F %*% t(lambda)``, so at ``beta = (1, 3)`` the outcome's factor
term is tripled and the loading-mismatch bias with it:

=====  =========  =========  ==========
w      published  at p = 2   at p = 0
=====  =========  =========  ==========
0.75       0.707      2.248       0.697
0.50       1.331      3.273       1.209
0.25       1.630      5.839       1.655
0.00       2.127      6.048       2.146
=====  =========  =========  ==========

Synth's bias at ``w = 0`` under ``p = 2`` is 2.8 times the published figure,
which is what an outcome carrying three times the factor signal does to an
estimator that cannot match the loadings. At ``p = 0`` the path lands on the
table. Adding the covariates to the estimator's formula instead -- the other way
to close the SD gap -- leaves the Synth arm untouched and still 2.8x out, so it
is not the explanation.

The comparison arm
------------------

Synth runs through the R package with the script's own sparse specification:
three special predictors, the outcome at periods 1, 8 and 15, optimised on
2-6 and 9-14. That is what produces the table's small but non-zero failure rate,
where no solution is found. mlsynth's ``VanillaSC`` is also run, on the same
panels, as the plain outcome-matching simplex a user of this library would
reach for. It is a different specification from the table's, so its levels are
recorded and not compared to a published number; what is compared is the shape,
because the hull argument does not depend on which predictors the weights are
chosen with. The shape is the same: its bias is 0.31 where the supports overlap
and climbs monotonically to 1.23 as they separate.

It also carries less of that bias than the table's specification does -- 1.23
against 2.18 at ``w = 0``, on the same panels. Matching the whole fifteen-period
path pins down weights that matching the outcome at periods 1, 8 and 15 leaves
free, and where the hull fails that freedom is spent on combinations that hit
three points while diverging from the rest. The hull constraint is what neither
specification can escape; how much bias it produces depends on what the weights
were chosen to match.

Provenance
----------

* Paper: Xu (2017), Political Analysis 25(1):57-76, Online Appendix Table A4.
* Archive: Dataverse replication files, ``gsynth_1.0.tar.gz``, ``sim_adh.R`` and
  the shared ``sim_sampling.R``. The factors come from the archive's
  ``FLSource.RData``, kept as ``basedata/xu_gsynth_FLSource.RData``.
* Reference: ``benchmarks/R/xu_gsynth_vs_scm.R``, which carries the generator at
  the appendix's reading and drives both gsynth 1.0 and Synth.
* Skips without R, gsynth or Synth. gsynth 1.0 needs a compiler and LAPACK/BLAS
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
_R_SCRIPT = _ROOT / "benchmarks" / "R" / "xu_gsynth_vs_scm.R"

SIMS = 60
POST_INDEX = 5          # the fifth post period, Table A4's ATT_{T0+5}
RANKS = (1, 2, 3, 4, 2, 2, 2, 2)
OVERLAPS = (1.0, 1.0, 1.0, 1.0, 0.75, 0.50, 0.25, 0.0)

# Table A4, by case.
PAPER_GSC_BIAS = (-0.010, 0.000, -0.003, 0.013, 0.014, 0.026, 0.000, 0.033)
PAPER_GSC_SD = (1.494, 1.571, 1.581, 1.610, 1.595, 1.602, 1.729, 1.822)
PAPER_SYNTH_BIAS = (-0.011, -0.022, 0.021, -0.013, 0.707, 1.331, 1.630, 2.127)
PAPER_SYNTH_SD = (1.739, 2.029, 2.368, 2.345, 2.096, 2.327, 2.492, 2.610)


def _require_r():
    from benchmarks.compare import BenchmarkSkipped

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH")
    probe = subprocess.run(
        [rscript, "-e",
         'q(status = if (requireNamespace("gsynth", quietly = TRUE) && '
         'requireNamespace("Synth", quietly = TRUE)) 0 else 1)'],
        capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R packages gsynth and Synth unavailable")
    return rscript


def _reference(rscript, outdir):
    from benchmarks.compare import BenchmarkSkipped

    proc = subprocess.run(
        [rscript, str(_R_SCRIPT), "--out", str(outdir), "--sims", str(SIMS)],
        capture_output=True, text=True, cwd=str(_ROOT), timeout=7200)
    ref = Path(outdir) / "reference.csv"
    if proc.returncode != 0 or not ref.exists():
        raise BenchmarkSkipped(
            f"reference run failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}")
    return pd.read_csv(ref)


def _gap_at(result, pre_periods):
    """The estimated gap at Table A4's fifth post period.

    ``estimated_gap`` and not ``observed_outcome - counterfactual_outcome``:
    under ``force="two-way"`` the two series are reported on the demeaned
    scale the fit works on, so their difference is not the ATT.
    """
    first = result[0] if isinstance(result, list) else result
    gap = np.asarray(first.time_series.estimated_gap, dtype=float).ravel()
    return float(gap[pre_periods + POST_INDEX - 1])


def _gsynth(panel, rank, pre_periods):
    """mlsynth's GSYNTH with the rank given, as Table A4 gives it."""
    from mlsynth.estimators.gsynth import GSYNTH

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = GSYNTH({"df": panel, "outcome": "Y", "treat": "D", "unitid": "id",
                      "time": "time", "r": rank, "force": "two-way",
                      "inference": False, "display_graphs": False}).fit()
    return _gap_at(res, pre_periods)


def _vanilla(panel, pre_periods):
    """mlsynth's plain simplex SC, outcome-matched, on the same panel."""
    from mlsynth import VanillaSC

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({"df": panel, "outcome": "Y", "treat": "D",
                         "unitid": "id", "time": "time",
                         "display_graphs": False}).fit()
    return _gap_at(res, pre_periods)


def _measure():
    rscript = _require_r()
    tmp = tempfile.mkdtemp(prefix="xu_adh_")
    try:
        ref = _reference(rscript, tmp)
        rows = []
        for case, grp in ref.groupby("case", sort=True):
            panels = pd.read_csv(Path(tmp) / f"case{int(case)}.csv")
            rank = RANKS[int(case) - 1]
            for rep, panel in panels.groupby("rep", sort=True):
                g = grp[grp.rep == rep].iloc[0]
                panel = panel.drop(columns=["rep"])
                # Per panel, not per file: summing D over every replication at
                # once counts the same treated periods sixty times over.
                pre = int(panel.time.nunique() - panel.groupby("id").D.sum().max())
                try:
                    att_my = _gsynth(panel, rank, pre)
                except Exception:
                    att_my = np.nan
                try:
                    att_vs = _vanilla(panel, pre)
                except Exception:
                    att_vs = np.nan
                rows.append({
                    "case": int(case), "rep": int(rep),
                    "att_my": att_my, "att_gs": float(g.att_gsynth),
                    "att_vs": att_vs, "att_synth": float(g.att_synth),
                    "true": float(g.true_k),
                    "synth_failed": int(g.synth_failed)})
        return pd.DataFrame(rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run() -> dict:
    d = _measure()
    out: dict = {"n_draws": float(len(d))}
    # Count the fits that returned before anything is aggregated over them. A
    # gap or a dispersion taken over an empty slice comes back 0 or nan and
    # reads like agreement; the first version of this case swallowed an index
    # error on every draw and still reported a max gap of exactly zero.
    out["n_gsynth_fits"] = float(d.att_my.notna().sum())
    out["n_vanillasc_fits"] = float(d.att_vs.notna().sum())

    gsc_bias, gsc_sd, syn_bias, syn_sd, vs_bias = [], [], [], [], []
    agree = 0.0
    for case in sorted(d.case.unique()):
        g = d[d.case == case]
        gsc_bias.append(float(np.nanmean(g.att_gs - g["true"])))
        gsc_sd.append(float(np.nanstd(g.att_gs, ddof=1)))
        syn_bias.append(float(np.nanmean(g.att_synth - g["true"])))
        syn_sd.append(float(np.nanstd(g.att_synth, ddof=1)))
        vs_bias.append(float(np.nanmean(g.att_vs - g["true"])))
        paired = (g.att_my - g.att_gs).abs().dropna()
        agree = max(agree, float(paired.max())) if len(paired) else float("nan")

    # With the rank given there is no cross-validation guard to separate them.
    out["gsynth_att_max_gap"] = agree

    # Table A4's four columns, worst cell against the published value.
    out["gsc_bias_max_gap"] = round(
        float(max(abs(a - b) for a, b in zip(gsc_bias, PAPER_GSC_BIAS))), 3)
    out["gsc_sd_max_gap"] = round(
        float(max(abs(a - b) for a, b in zip(gsc_sd, PAPER_GSC_SD))), 3)
    out["synth_bias_max_gap"] = round(
        float(max(abs(a - b) for a, b in zip(syn_bias, PAPER_SYNTH_BIAS))), 3)
    out["synth_sd_max_gap"] = round(
        float(max(abs(a - b) for a, b in zip(syn_sd, PAPER_SYNTH_SD))), 3)

    # The bias path as the overlap closes, which is the table's content.
    for k in (5, 6, 7, 8):
        out[f"synth_bias_w{int(100 * OVERLAPS[k - 1]):03d}"] = round(syn_bias[k - 1], 3)
    out["gsc_bias_max_abs"] = round(float(max(abs(b) for b in gsc_bias)), 3)
    out["synth_bias_rises_as_overlap_closes"] = float(
        syn_bias[4] < syn_bias[5] < syn_bias[6] < syn_bias[7])
    # and the wrong rank does not do the same thing: cases 1-4 vary r at w = 1.
    out["gsc_bias_max_abs_wrong_rank"] = round(
        float(max(abs(gsc_bias[k]) for k in (0, 2, 3))), 3)

    # Efficiency: the factor model is tighter in every cell of the table.
    out["gsc_tighter_than_synth_cells"] = float(
        sum(a < b for a, b in zip(gsc_sd, syn_sd)))
    # Cases 1-4 vary the rank the factor model is given, and Synth does not use
    # a rank, so its dispersion is the same quantity four times over. The
    # published column spreads 0.629 across them at 5,000 draws, which says the
    # SD of a simplex gap is heavy-tailed and slow to settle. Measured here so
    # the spread is on the record on both sides.
    out["synth_sd_spread_r_cells"] = round(
        float(max(syn_sd[:4]) - min(syn_sd[:4])), 3)

    # mlsynth's own outcome-matching SC on the same panels. A different
    # specification from the table's, so the levels are recorded, not compared.
    out["vanillasc_bias_w100"] = round(vs_bias[1], 3)
    out["vanillasc_bias_w000"] = round(vs_bias[7], 3)
    out["vanillasc_bias_rises_as_overlap_closes"] = float(
        vs_bias[4] < vs_bias[5] < vs_bias[6] < vs_bias[7])
    # Matching the whole pre-period path against matching three points of it,
    # on the same panels, where no convex combination reaches the treated unit.
    out["vanillasc_bias_w000_vs_synth"] = round(vs_bias[7] - syn_bias[7], 3)

    # Synth fails to find a solution on a small share of draws, as the table's
    # own Fail column reports; gsynth never does.
    out["synth_fail_rate"] = round(float(d.synth_failed.mean()), 3)
    return out


# 60 draws a cell against the paper's 5,000. At an SD near 1.6 the standard
# error of a mean is 0.21 and of the SD itself about 0.15, and the bands below
# are set from those. The one exact quantity is the agreement between mlsynth
# and gsynth on a shared panel at a given rank, which carries no Monte Carlo
# component at all.
EXPECTED = {
    "n_draws": (480.0, 0.0),
    # Every fit has to have returned, or the aggregates below mean nothing.
    "n_gsynth_fits": (480.0, 0.0),
    "n_vanillasc_fits": (480.0, 0.0),
    # Rank given, same panels: one computation, as xu_gsynth_properties finds.
    "gsynth_att_max_gap": (0.0, 1e-8),
    # Table A4's four columns. At sixty draws a mean with SD near 1.6 carries a
    # standard error of 0.21 and one near 2.3 carries 0.30, and the worst of
    # eight cells runs about twice that, which is where these bands come from.
    "gsc_bias_max_gap": (0.33, 0.30),
    "gsc_sd_max_gap": (0.30, 0.30),
    "synth_bias_max_gap": (0.45, 0.35),
    # The loosest entry in the case, and the table says why: see
    # synth_sd_spread_r_cells below.
    "synth_sd_max_gap": (0.60, 0.45),
    # The path the table exists to show: 0.707, 1.331, 1.630, 2.127.
    "synth_bias_w075": (0.707, 0.45),
    "synth_bias_w050": (1.331, 0.45),
    "synth_bias_w025": (1.630, 0.45),
    "synth_bias_w000": (2.127, 0.45),
    "synth_bias_rises_as_overlap_closes": (1.0, 0.0),
    # while the factor model's bias never leaves the noise, at any rank.
    "gsc_bias_max_abs": (0.33, 0.30),
    "gsc_bias_max_abs_wrong_rank": (0.32, 0.30),
    "gsc_tighter_than_synth_cells": (8.0, 0.0),
    # Synth ignores the rank, so cases 1-4 measure one quantity four times. The
    # published column spreads 0.629 across them at 5,000 draws.
    "synth_sd_spread_r_cells": (0.50, 0.45),
    # mlsynth's own simplex SC, calibration: the table's spec is Synth's.
    "vanillasc_bias_w100": (0.0, 0.40),
    "vanillasc_bias_w000": (1.23, 0.60),
    "vanillasc_bias_rises_as_overlap_closes": (1.0, 0.0),
    # and it carries less of that bias than the table's sparse spec does.
    "vanillasc_bias_w000_vs_synth": (-0.95, 0.60),
    # the table's Fail column runs 0.008 to 0.016
    "synth_fail_rate": (0.03, 0.05),
}
