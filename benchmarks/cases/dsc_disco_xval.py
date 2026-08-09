"""Cross-validation: DSC against the ``DiSCos`` R package on the Dube panel.

The reference implementation of Distributional Synthetic Controls is
Gunsilius & Van Dijcke's ``DiSCos``
(`Davidvandijcke/DiSCos <https://github.com/Davidvandijcke/DiSCos>`_). Until now
mlsynth had no case comparing against it, because a naive comparison does not
work: ``DiSCo_weights_reg`` draws its quadrature points with ``runif``, so its
fitted weights are a Monte Carlo estimate. Measured here at ``M = 10,000`` on
the Dube panel, the package disagrees with *itself* across seeds by up to 0.016
under the simplex and 0.030 without it. Pinning a tolerance against one R run
would be pinning that noise.

Averaging removes it. This case scores mlsynth against the mean of 40 seeded
reference runs and reads the across-seed standard deviation as the yardstick:
the question is not whether the two agree exactly, which they cannot, but
whether mlsynth sits closer to the reference's centre than the reference sits to
itself.

It does, in both feasible sets:

===================  ==================  ==================  ==============
Mode                 max abs weight gap  reference seed sd   gap / sd
===================  ==================  ==================  ==============
simplex              0.0079              0.0160              0.50
sum-to-one           0.0103              0.0300              0.34
===================  ==================  ==================  ==============

Both feasible sets are covered because the two implementations default to
different ones. mlsynth constrains weights to the simplex, the set
:math:`\\mathcal H` of Zhang, Zhang & Zhang (2026); ``DiSCos`` passes
``lb = NULL`` unless ``simplex = TRUE``, so its default is sum-to-one with an
upper bound of one and negatives allowed. ``weight_constraint="sum_to_one"``
reaches that second set, and the agreement there is the closer of the two.

What this settles
-----------------

Issue #304 recorded mlsynth and ``DiSCos`` disagreeing on donor weights by up to
0.074 on this same panel, with the two reaching pre-period objective values 4
percent apart, and ``dsc_dube`` treats that as a reason its rows cannot be
cross-validated. The disagreement was a measurement artifact: one seed, at the
package's default ``M``. Raise ``M`` and average the seed noise away and the gap
is 0.0079.

Two other results triangulate the same conclusion. ``disco_tenure`` reproduces
the deterministic Stata implementation's published weights bit-for-bit, so
mlsynth's grid is not the odd one out. And handed bit-identical design matrices,
mlsynth's solver and the reference's ``pracma::lsqlincon`` return weights
agreeing to 3e-9 (``benchmarks/reference/dsc_mc/``), so the solver was never a
candidate either. What remained was the quadrature rule, and that is what this
case measures directly.

Provenance
----------

* Reference: ``DiSCos`` 0.1.4 at ``ed2b3d94``, built by
  ``benchmarks/R/install_discos.sh``, captured to
  ``benchmarks/reference/dsc_disco_xval/gold_weights.csv`` by that directory's
  ``reference.R``. The dump is committed, so this case needs no R to run.
* Data: ``basedata/dube_minwage.parquet``, the package's own ``dube`` dataset --
  652,870 rows, 34 states over 1998-2004, Alaska (``fips = 2``) treated from
  2003. The same file and setup ``dsc_dube`` uses.
* Donor identity travels with the weights: the reference records
  ``control_ids`` and this case joins on it, so neither side depends on the
  other's ordering.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_PANEL = _ROOT / "basedata" / "dube_minwage.parquet"
_GOLD = _ROOT / "benchmarks" / "reference" / "dsc_disco_xval" / "gold_weights.csv"

_M = 10_000            # matches the reference capture
_T0 = 2003
_TARGET = 2.0
_MODES = ("simplex", "sum_to_one")


def _panel() -> pd.DataFrame:
    df = pd.read_parquet(_PANEL)
    df["treat"] = ((df.id_col == _TARGET) & (df.time_col >= _T0)).astype(int)
    return df


def _fit(mode: str) -> pd.Series:
    """mlsynth's donor weights under one feasible set, indexed by donor id."""
    from mlsynth import DSC

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = DSC({
            "df": _panel(), "outcome": "y_col", "treat": "treat",
            "unitid": "id_col", "time": "time_col", "M": _M,
            "grid_method": "equidistant", "weight_constraint": mode,
            "display_graphs": False,
        }).fit()
    weights = res.weights.donor_weights
    return pd.Series({float(k): float(v) for k, v in weights.items()}).sort_index()


def _aligned(mode: str, gold: pd.DataFrame):
    """mlsynth and reference weights on a common donor index."""
    ref = gold[gold["mode"] == mode].set_index(
        gold[gold["mode"] == mode]["donor_id"].astype(float)).sort_index()
    mine = _fit(mode)
    if list(mine.index) != list(ref.index):
        raise AssertionError(
            f"donor sets differ for mode {mode}: "
            f"{sorted(set(mine.index) ^ set(ref.index))}"
        )
    return mine.to_numpy(), ref["mean"].to_numpy(), ref["sd"].to_numpy()


def run() -> dict:
    gold = pd.read_csv(_GOLD)
    out: dict[str, float] = {"n_seeds": float(gold["n_seeds"].iloc[0])}
    for mode in _MODES:
        mine, ref_mean, ref_sd = _aligned(mode, gold)
        diff = np.abs(mine - ref_mean)
        out[f"n_donors_{mode}"] = float(mine.size)
        out[f"weights_sum_{mode}"] = float(mine.sum())
        out[f"max_abs_diff_{mode}"] = float(diff.max())
        out[f"mean_abs_diff_{mode}"] = float(diff.mean())
        out[f"ref_max_seed_sd_{mode}"] = float(ref_sd.max())
        # The headline. Reading the gap in units of the reference's own seed
        # spread is what makes a stochastic reference usable as a target: below
        # 1 means mlsynth sits nearer the reference's centre than a single
        # reference run does.
        out[f"diff_over_ref_sd_{mode}"] = float(diff.max() / ref_sd.max())
        out[f"corr_{mode}"] = float(np.corrcoef(mine, ref_mean)[0, 1])
    # The relaxation reaches the reference's own default, and lands nearer it.
    out["sum_to_one_has_negative_weights"] = float(
        _fit("sum_to_one").min() < -0.01)
    return out


def comparison() -> dict:
    """Per-donor weights, mlsynth against the 40-seed reference mean."""
    gold = pd.read_csv(_GOLD)
    rows = []
    for mode in _MODES:
        mine, ref_mean, _ = _aligned(mode, gold)
        ids = gold[gold["mode"] == mode]["donor_id"].astype(float).sort_values()
        for donor, m, r in zip(ids, mine, ref_mean):
            rows.append({"quantity": f"w[{mode}][fips={donor:g}]",
                         "mlsynth": round(float(m), 6),
                         "reference": round(float(r), 6)})
    return {
        "rows": rows,
        "mlsynth_call": {
            "estimator": "DSC",
            "config": {"M": _M, "grid_method": "equidistant",
                       "weight_constraint": "simplex | sum_to_one",
                       "unitid": "id_col", "time": "time_col",
                       "outcome": "y_col", "t0": _T0},
        },
        "reference": {
            "impl": "Davidvandijcke/DiSCos DiSCo(), mean of 40 seeds at M = 10,000 "
                    "(the package's runif quadrature makes a single run a "
                    "Monte Carlo draw)",
            "version": "DiSCos 0.1.4 @ ed2b3d94 "
                       "(benchmarks/reference/dsc_disco_xval/)",
        },
    }


# Tolerances.
#
# The reference is stochastic, so the pins are stated against its own spread and
# not against an absolute idea of agreement. max_abs_diff carries a band of one
# reference seed standard deviation, which is the resolution the reference
# actually offers: tightening below that would pin the particular 40 seeds this
# dump captured. diff_over_ref_sd is the metric that has to stay below 1 for the
# case to mean anything, and its band keeps it there.
#
# mlsynth's side is deterministic at these settings (equidistant grid, fixed M),
# so all of the movement a re-run could show comes from the reference.
EXPECTED = {
    "n_seeds": (40.0, 0.0),
    "n_donors_simplex": (33.0, 0.0),
    "n_donors_sum_to_one": (33.0, 0.0),
    "weights_sum_simplex": (1.0, 1e-6),
    "weights_sum_sum_to_one": (1.0, 1e-6),
    # Agreement, in absolute weight units.
    "max_abs_diff_simplex": (0.0079, 0.016),
    "max_abs_diff_sum_to_one": (0.0103, 0.030),
    "mean_abs_diff_simplex": (0.0022, 0.005),
    "mean_abs_diff_sum_to_one": (0.0042, 0.010),
    # Agreement, against the reference's own seed noise. Below 1 is the claim.
    "diff_over_ref_sd_simplex": (0.50, 0.45),
    "diff_over_ref_sd_sum_to_one": (0.34, 0.45),
    "corr_simplex": (0.9958, 0.01),
    "corr_sum_to_one": (0.9983, 0.01),
    # The reference's spread, pinned so a change in its noise shows up here
    # instead of widening the effective tolerance above without a signal.
    "ref_max_seed_sd_simplex": (0.0160, 0.006),
    "ref_max_seed_sd_sum_to_one": (0.0300, 0.010),
    # The relaxation does what it is for: it leaves the simplex.
    "sum_to_one_has_negative_weights": (1.0, 0.0),
}
