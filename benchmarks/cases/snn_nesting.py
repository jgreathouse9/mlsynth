"""Differential cross-validation: SNN's anchor cross nests SI and PCR-RSC.

SNN (Agarwal, Dahleh, Shah & Shen, *Synthetic Nearest Neighbors*), SI (Agarwal,
Shah & Shen, *Synthetic Interventions*, Oper. Res. 74(2)) and RSC (Amjad, Shah &
Shen, *Robust Synthetic Control*, JMLR 19) run one kernel: truncate the SVD of a
fully observed block, regress the target onto it by principal component
regression, apply the weights out of sample. They differ in which block the
design hands them. SI's own Corollary 1 discussion closes half the chain -- at
``d = 0`` SI-PCR "exactly recovers the SC-PCR estimator of Amjad et al." -- and
SNN's contribution is to make the block a per-entry choice instead of a fixture
of the treatment geometry.

On the block missingness of a comparative case study the three blocks coincide:
anchor rows are the donor pool, anchor columns are the pre-periods. The nesting
is an algebraic claim, so data cannot prove it; what data can do is refute it.
This case runs the refutation attempt on three real panels.

Four things are measured, and three of them are guards.

Recovery. SNN's own anchor search, given only the mask, returns exactly the
donor x pre-period block on all three case studies -- the step that makes the
other rows a statement about SNN and not about a block handed to it.

Equality. At a matched truncation rank, mlsynth's ``SNN``, ``SI`` (control arm,
``d = 0``) and ``CLUSTERSC`` (the PCR-RSC leg, clustering off) reproduce the
authors' own vertical PCR, path for path. The row-side and column-side
syntheses also agree, which is SNN section 3.1's identity, itself citing the
Shen-Ding-Sekhon-Yu paper whose code is the reference here.

Controls. Three perturbations that have to break the equality, each pinned at
the separation it creates: shrinking the anchor cross, moving the rank by one,
and projecting the counterfactual through the denoised full donor matrix
(``project_denoised=True``) as RSC's Algorithm 1 does. Without these rows the
equality rows would pass on an estimator that ignored its anchor sets, its rank
and its projection alike.

The three controls also locate where the estimators genuinely diverge. The
nesting is exact at a *matched* rank and a raw post-period projection; the
shipped defaults differ on both, so out of the box these estimators do not agree
and are not supposed to.

Path: differential cross-validation, mlsynth against the authors' code and
against itself. Not Path A or B -- no paper publishes a number for "SNN equals
SI equals RSC on Basque", because the claim is structural. What is checkable is
that the structure holds on real panels and breaks when it should.

Provenance
----------
* Data and reference: ``deshen24/panel-data-regressions`` @ ``51e2170``, the
  replication repo for Shen, Ding, Sekhon & Yu, *Same Root Different Leaves*
  (arXiv:2207.14481). Fetched on demand into the gitignored
  ``benchmarks/reference/.cache`` (see
  ``benchmarks/reference/clone_panel_data_regressions.py``); the case skips if
  neither git nor codeload is reachable.
* Panels, with the authors' own pre/post splits: Basque terrorism (16 donors,
  1955-1969 pre), West German reunification (16 donors, 1960-1989 pre),
  California Proposition 99 (38 donors, 1970-1987 pre).
* Reference estimator: the authors' ``regr.pcr`` at the rank their
  ``case_study.py`` selects, ``rank.spectral_rank(s, t=0.999)``.

The network-free companion is ``mlsynth/tests/test_pcr_nesting.py``, which pins
the same identity on a planted low-rank panel.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from benchmarks.reference.clone_panel_data_regressions import (
    TREATED,
    import_panel_data_regressions,
)

#: Spectral-energy threshold ``case_study.py`` uses to pick the PCR rank.
ENERGY = 0.999

DATASETS = ("prop99", "basque", "germany")


def _load(data_dir, data):
    """The authors' panel, their pre/post split, as wide arrays and a long frame."""
    pre = pd.read_csv(data_dir / data / "pre_outcomes.csv")
    post = pd.read_csv(data_dir / data / "post_outcomes.csv")
    treated = TREATED[data]
    wide = pre.merge(post, on="unit").set_index("unit")
    wide.columns = [int(c) for c in wide.columns]
    pre_years = [int(c) for c in pre.columns if c != "unit"]
    post_years = [int(c) for c in post.columns if c != "unit"]
    donors = [u for u in wide.index if u != treated]

    long = (wide.stack().rename("y").reset_index()
                .rename(columns={"level_1": "year"}))
    long["treat"] = ((long.unit == treated)
                     & (long.year.isin(post_years))).astype(int)
    long["control_arm"] = long.unit.isin(donors).astype(int)
    return wide, long, treated, donors, pre_years, post_years


def _fit_all(long, k, post_years, observed_post):
    """The three mlsynth estimators' post-period counterfactuals at rank ``k``."""
    from mlsynth import CLUSTERSC, SI, SNN

    cfg = {"df": long, "outcome": "y", "treat": "treat", "unitid": "unit",
           "time": "year", "display_graphs": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snn = SNN({**cfg, "max_rank": k, "clip": False}).fit()
        si = SI({**cfg, "inters": ["control_arm"], "rank_method": "fixed",
                 "rank": k, "bias_correct": False}).fit()
        rsc = CLUSTERSC({**cfg, "method": "pcr", "clustering": False,
                         "pcr_objective": "OLS", "rank": k,
                         "rank_method": "fixed",
                         "standardize_for_rank": False}).fit()
        # RSC's Algorithm 1 denoises the full donor matrix and projects the
        # counterfactual through it; mlsynth keeps the post-period raw by
        # default. The opt-in flag recovers the paper-strict variant.
        rsc_paper = CLUSTERSC({**cfg, "method": "pcr", "clustering": False,
                               "pcr_objective": "OLS", "rank": k,
                               "rank_method": "fixed",
                               "standardize_for_rank": False,
                               "project_denoised": True}).fit()

    n = len(post_years)
    return {
        "snn": observed_post - np.array([snn.att_by_period[y] for y in post_years]),
        "si": np.asarray(si.arms["control_arm"].counterfactual, float)[-n:],
        "rsc": np.asarray(rsc.counterfactual, float)[-n:],
        "rsc_paper": np.asarray(rsc_paper.counterfactual, float)[-n:],
        "att": float(snn.att),
    }


def _rel_pct(a, b):
    """Max absolute gap between two paths, as a percentage of their level."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(np.abs(a).mean(), np.abs(b).mean(), 1e-12)
    return float(100.0 * np.abs(a - b).max() / scale)


def _per_dataset():
    """Every measured quantity, one row per case study."""
    from mlsynth.utils.pcr import pcr_weights
    from mlsynth.utils.snn_helpers.completion import _find_anchors

    regr, rank_mod, data_dir = import_panel_data_regressions()
    rows = []
    for data in DATASETS:
        wide, long, treated, donors, pre_years, post_years = _load(data_dir, data)
        Y0 = wide.loc[donors, pre_years].to_numpy(float)          # N0 x T0
        y_target_pre = wide.loc[treated, pre_years].to_numpy(float)
        Y_post = wide.loc[donors, post_years].to_numpy(float)     # N0 x T1
        observed_post = wide.loc[treated, post_years].to_numpy(float)
        N0, T0 = Y0.shape
        k = int(rank_mod.spectral_rank(
            np.linalg.svd(Y0, compute_uv=False), t=ENERGY))

        # Reference: the authors' vertical PCR (weights over donors) and
        # horizontal PCR (weights over pre-periods), at their own rank.
        ref_vt = np.array([float(Y_post[:, t] @ regr.pcr(Y0.T, y_target_pre, max_rank=k))
                           for t in range(len(post_years))])
        ref_hz = np.array([float(y_target_pre @ regr.pcr(Y0, Y_post[:, t], max_rank=k))
                           for t in range(len(post_years))])

        # Does SNN's own anchor search return the synthetic-control cross?
        order = list(wide.index)
        i = order.index(treated)
        mask = np.ones(wide.shape, dtype=int)
        mask[i, T0:] = 0
        AR, AC = _find_anchors(mask, i, T0)
        donor_rows = [r for r in range(len(order)) if r != i]
        anchors_exact = (sorted(AR) == donor_rows and sorted(AC) == list(range(T0)))

        fits = _fit_all(long, k, post_years, observed_post)

        # SNN's column-side synthesis (weights over anchor columns).
        col_side = np.array([float(y_target_pre @ pcr_weights(Y0, Y_post[:, t], k))
                             for t in range(len(post_years))])

        # Control: a strictly smaller anchor cross.
        keep_r, keep_c = np.arange(N0 - 3), np.arange(3, T0)
        S2 = Y0[np.ix_(keep_r, keep_c)]
        k2 = int(rank_mod.spectral_rank(
            np.linalg.svd(S2, compute_uv=False), t=ENERGY))
        shrunk = np.array([float(Y_post[keep_r, t]
                                 @ pcr_weights(S2.T, y_target_pre[keep_c], k2))
                           for t in range(len(post_years))])

        # Control: the rank moved by one.
        off_rank = np.array([float(Y_post[:, t] @ pcr_weights(Y0.T, y_target_pre, k + 1))
                             for t in range(len(post_years))])

        rows.append({
            "dataset": data, "treated": treated, "n_donors": N0,
            "n_pre": T0, "n_post": len(post_years), "rank": k,
            "anchors_exact": bool(anchors_exact),
            "att": fits["att"],
            "snn_vs_ref": _rel_pct(fits["snn"], ref_vt),
            "si_vs_ref": _rel_pct(fits["si"], ref_vt),
            "rsc_vs_ref": _rel_pct(fits["rsc"], ref_vt),
            "col_side_vs_ref_hz": _rel_pct(col_side, ref_hz),
            "row_vs_col": _rel_pct(fits["snn"], col_side),
            "control_shrunk_cross": _rel_pct(shrunk, ref_vt),
            "control_off_rank": _rel_pct(off_rank, ref_vt),
            "control_paper_projection": _rel_pct(fits["rsc_paper"], ref_vt),
        })
    return rows


def run() -> dict:
    """Aggregate the per-dataset rows into the pinned metrics.

    Equality rows are reported as the worst case across the three panels, so a
    single panel breaking the identity fails the case. Control rows are reported
    as the *best* case (the smallest separation any panel produces), so a control
    collapsing on any panel fails the case too. Both directions are chosen so
    that one panel cannot hide behind the other two.
    """
    rows = _per_dataset()
    worst = lambda key: max(r[key] for r in rows)   # noqa: E731
    best = lambda key: min(r[key] for r in rows)    # noqa: E731
    return {
        "anchor_blocks_exact": float(sum(r["anchors_exact"] for r in rows)),
        "snn_vs_reference_pct": worst("snn_vs_ref"),
        "si_vs_reference_pct": worst("si_vs_ref"),
        "rsc_vs_reference_pct": worst("rsc_vs_ref"),
        "column_side_vs_reference_pct": worst("col_side_vs_ref_hz"),
        "row_vs_column_pct": worst("row_vs_col"),
        "control_shrunk_cross_pct": best("control_shrunk_cross"),
        "control_off_rank_pct": best("control_off_rank"),
        "control_paper_projection_pct": best("control_paper_projection"),
        "prop99_att": next(r["att"] for r in rows if r["dataset"] == "prop99"),
    }


def comparison() -> dict:
    """mlsynth's three estimators against the authors' PCR, case study by case study.

    Returns ``{"rows": [...], "mlsynth_call": {...}, "reference": {...}}`` with
    one row per case study carrying the panel's shape, the selected rank, the
    shared ATT, each estimator's gap to the reference, and the three controls.
    All gaps are percentages of the counterfactual's own level, so the three
    panels -- packs per capita, thousands of dollars of GDP -- are comparable.
    """
    rows = []
    for r in _per_dataset():
        rows.append({
            "case study": r["dataset"],
            "treated": r["treated"],
            "shape (N0 x T0 x T1)": f"{r['n_donors']} x {r['n_pre']} x {r['n_post']}",
            "rank": r["rank"],
            "anchor cross recovered": r["anchors_exact"],
            "shared ATT": round(r["att"], 6),
            "SNN vs authors' PCR (%)": f"{r['snn_vs_ref']:.2e}",
            "SI vs authors' PCR (%)": f"{r['si_vs_ref']:.2e}",
            "PCR-RSC vs authors' PCR (%)": f"{r['rsc_vs_ref']:.2e}",
            "row vs column side (%)": f"{r['row_vs_col']:.2e}",
            "control: shrunk cross (%)": round(r["control_shrunk_cross"], 3),
            "control: rank k+1 (%)": round(r["control_off_rank"], 3),
            "control: paper projection (%)": round(r["control_paper_projection"], 3),
        })
    return {
        "rows": rows,
        "mlsynth_call": {
            "estimators": ["SNN", "SI", "CLUSTERSC"],
            "matched": "rank = spectral_rank(svd(Y0), t=0.999), the authors' rule",
            "config": {
                "SNN": {"max_rank": "k", "clip": False},
                "SI": {"inters": ["control_arm"], "rank_method": "fixed",
                       "rank": "k", "bias_correct": False},
                "CLUSTERSC": {"method": "pcr", "clustering": False,
                              "pcr_objective": "OLS", "rank_method": "fixed",
                              "rank": "k", "standardize_for_rank": False},
            },
        },
        "reference": {
            "impl": "deshen24/panel-data-regressions regr.pcr (live run), "
                    "vertical and horizontal PCR at rank.spectral_rank(t=0.999)",
            "version": "deshen24/panel-data-regressions @ 51e2170",
        },
    }


# Tolerances.
#
# The equality rows compare calls into the same closed-form PCR expression on the
# same arrays, so they are zero up to floating-point reassociation, not merely
# close. Observed worst case across the three panels is ~1e-13 percent of the
# counterfactual's level (the largest absolute gap, 3e-11, is on German GDP at a
# level of 26,000). 1e-8 percent records that the claim is bit-level agreement
# while leaving room for a BLAS that reorders a reduction -- and it is five
# orders tighter than the smallest control below, so no real regression in
# anchor selection, rank handling or projection can hide under it.
#
# `anchor_blocks_exact` counts the case studies on which SNN's own search
# returns the full donor x pre-period block. All three, exactly; a partial block
# on any panel would make the equality rows a statement about a block handed to
# SNN, not one it found.
#
# The three control rows are guards, not matches: each fails if the perturbation
# stops separating the estimates, which would make every equality row above
# vacuous. Each is pinned at the smallest separation the three panels produce,
# with half of it as the tolerance -- the geox_sdid_equivalence convention. The
# separations are deterministic given the pinned data and rank rule, so the
# two-sided band is a feature: a control drifting either way means the
# perturbation changed meaning.
#
#   shrunk cross     1.946% (germany; prop99 2.146, basque 2.247)
#   rank k+1         0.455% (germany; basque 1.229, prop99 3.595)
#   paper projection 0.241% (germany; basque 1.145, prop99 2.061)
#
# `prop99_att` is the shared estimate all three estimators return on the authors'
# 1970-1987/1988-2000 split, in packs per capita. It is not the ADH headline
# (their split ends the pre-period in 1988), so it is pinned as this case's own
# anchor: a number every row above has to be consistent with. 1e-6 is the
# reassociation band, since the three estimators agree on it exactly.
EXPECTED = {
    "anchor_blocks_exact": (3.0, 0),
    "snn_vs_reference_pct": (0.0, 1e-8),
    "si_vs_reference_pct": (0.0, 1e-8),
    "rsc_vs_reference_pct": (0.0, 1e-8),
    "column_side_vs_reference_pct": (0.0, 1e-8),
    "row_vs_column_pct": (0.0, 1e-8),
    "control_shrunk_cross_pct": (1.9459, 0.97),
    "control_off_rank_pct": (0.4548, 0.227),
    "control_paper_projection_pct": (0.2412, 0.12),
    "prop99_att": (-20.687434, 1e-6),
}
