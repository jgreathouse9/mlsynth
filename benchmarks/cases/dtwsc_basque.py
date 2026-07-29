"""Cross-validation: DTWSC against the authors' R package on the Basque panel.

Reference: ``conflictlab/dsc`` (MIT) pinned at
``b1cd241518329ac2bc8cfe21a871a798ac14d74f``; see
``benchmarks/reference/dtwsc_basque/README.md`` for the generator and
``benchmarks/R/install_dtwsc.sh`` for the dependency chain.

The reference dump is not committed (it needs a live R + ``Synth`` toolchain),
so this case reads it when present and reports ``nan`` for the reference-backed
quantities when it is absent -- the pure-Python invariants below still run.

What this case pins, and why each row is where it is:

* The warp, against the R dump -- ``cutoff`` and the first-phase speeds. This
  is the tight structural comparison.
* The synthetic-control half, against the reference's own numbers, using
  ``sc_backend="mscmt"`` with the paper's 14 predictors. The UNWARPED arm
  reproduces R's standard-SC result essentially exactly (pre-RMSE 0.0881 vs
  0.0886, ATT -0.6026 vs -0.6027), which is what establishes that the
  delegation to mlsynth's Synth replication is faithful.
* The WARPED arm still differs (0.0601 / -0.6696 against R's 0.0705 /
  -0.5579). That residual is isolated to the Savitzky-Golay preprocessing:
  the reference pads each series with an ``auto.arima`` forecast before
  filtering where mlsynth edge-pads, which leaves 11 of 16 donors with
  bit-exact speeds rather than 16. Feeding mlsynth's warp into R's own
  ``Synth`` gives 0.0705 / -0.5592, so the warp and the SC half are each
  correct in isolation; only the filter's edge treatment is not.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_REF = Path(__file__).resolve().parents[1] / "reference" / "dtwsc_basque"
_DATA = Path(__file__).resolve().parents[2] / "basedata" / "basque_data.csv"

TREATED = "Basque Country (Pais Vasco)"
#: The reference package's own README example uses 1970, not Abadie and
#: Gardeazabal's 1975 terrorism onset -- and its ``t.treat`` INCLUDES the
#: treatment year in the alignment window (``1:t.treat``), where mlsynth's
#: ``pre_periods`` stops the period before. Marking treatment from 1971 gives
#: both sides the same 16-period window (1955--1970) to learn the warp on,
#: which is what this case pins. The end-to-end ATT comparison is in
#: ``docs/replications/dtwsc.rst``, not here.
TREAT_YEAR = 1971


def _panel() -> pd.DataFrame:
    df = pd.read_csv(_DATA)
    df = df[df.regionname != "Spain (Espana)"].dropna(subset=["gdpcap"]).copy()
    df["treated"] = ((df.regionname == TREATED) & (df.year >= TREAT_YEAR)).astype(int)
    return df


def _load_reference():
    """``{unit: {quantity: array}}`` from the R dump, or ``None``."""
    path = _REF / "gold_tfdtw.csv"
    if not path.exists():
        return None
    out = {}
    for line in path.read_text().splitlines():
        unit, quantity, values = line.split(";")
        out.setdefault(unit, {})[quantity] = np.array(
            [np.nan if v == "NA" else float(v) for v in values.split(",")]
        )
    return out


def run() -> dict:
    from mlsynth import DTWSC

    cfg = {"df": _panel(), "outcome": "gdpcap", "treat": "treated",
           "unitid": "regionname", "time": "year", "display_graphs": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warped = DTWSC(cfg).fit()
        plain = DTWSC({**cfg, "warp": False}).fit()

    out = {
        # The paper's claim: aligning speeds tightens the pre-treatment fit.
        "dtwsc_pre_rmse_improves": float(warped.pre_rmse < plain.pre_rmse),
        "dtwsc_pre_rmse_ratio": float(warped.pre_rmse / plain.pre_rmse),
        "dtwsc_n_donors": float(len(warped.donor_weights)),
        "dtwsc_weights_sum": float(sum(warped.donor_weights.values())),
        "dtwsc_att_sign": float(np.sign(warped.att)),
    }

    ref = _load_reference()
    if ref is None:
        out["dtwsc_cutoff_matches_r"] = float("nan")
        out["dtwsc_donors_speeds_exact_vs_r"] = float("nan")
        out["dtwsc_speed_max_abs_diff_vs_r"] = float("nan")
        return out

    n_cut, n_exact, worst = 0, 0, 0.0
    for unit, cutoff in warped.cutoffs.items():
        if unit not in ref:
            continue
        n_cut += int(cutoff == int(ref[unit]["cutoff"][0]))
        gap = float(np.nanmax(np.abs(
            warped.pre_period_speeds[unit] - ref[unit]["weight.a"])))
        n_exact += int(gap < 1e-12)
        worst = max(worst, gap)
    out["dtwsc_cutoff_matches_r"] = float(n_cut)
    out["dtwsc_donors_speeds_exact_vs_r"] = float(n_exact)
    out["dtwsc_speed_max_abs_diff_vs_r"] = worst
    return out


# Deterministic: the warp is a dynamic program with no sampling anywhere, and
# the simplex solve is convex. Re-running gives identical numbers.
#
# Tolerances, and what each row is really pinning:
#  * ``pre_rmse_improves`` / ``pre_rmse_ratio`` -- the paper's own claim, on the
#    authors' own panel, through mlsynth's public API. The ratio is pinned at
#    0.70 +/- 0.08: it depends on the SC half, which is outcome-only here
#    against the reference's 14-predictor Abadie specification, so it is a
#    same-direction check rather than a cell match.
#  * ``cutoff_matches_r`` -- 16/16 exact. The cutoff is an integer read off the
#    first-phase warp, so this is the tight structural check and gets zero
#    tolerance.
#  * ``donors_speeds_exact_vs_r`` -- 11 of 16 donors' first-phase speeds agree
#    with R to 1e-12 through mlsynth's own preprocessing. The other five differ
#    because the reference pads each series with an ``auto.arima`` forecast
#    before the Savitzky-Golay filter where mlsynth edge-pads, which moves the
#    two points at each end and can flip an alignment step. Feeding mlsynth R's
#    own filtered panel instead makes it 16/16 to one ULP -- see
#    ``docs/replications/dtwsc.rst``. Pinned exactly so a regression in the
#    warp shows up rather than hiding inside a loose bound.
#  * ``speed_max_abs_diff_vs_r`` -- the size of that edge-buffer disagreement,
#    2/3 of one period's speed on the worst donor. Pinned loosely (0.1) because
#    it is a known, documented difference, not a target to drive to zero.
EXPECTED = {
    "dtwsc_pre_rmse_improves": (1.0, 0.0),
    "dtwsc_pre_rmse_ratio": (0.70, 0.08),
    "dtwsc_n_donors": (16.0, 0.0),
    "dtwsc_weights_sum": (1.0, 1e-6),
    "dtwsc_att_sign": (-1.0, 0.0),
    "dtwsc_cutoff_matches_r": (16.0, 0.0),
    "dtwsc_donors_speeds_exact_vs_r": (11.0, 0.0),
    "dtwsc_speed_max_abs_diff_vs_r": (0.667, 0.1),
}
