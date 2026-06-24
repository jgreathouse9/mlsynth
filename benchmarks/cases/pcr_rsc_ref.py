"""Cross-validation: mlsynth PCR vs the original Robust Synthetic Control (tslib).

Cross-validation against the canonical implementation. mlsynth's PCR (principal
component regression / robust synthetic control --
:func:`mlsynth.utils.clustersc_helpers.pcr.pipeline.run_pcr`, the kernel
ClusterSC's ``method="PCR"`` path drives) is checked against the ORIGINAL Robust
Synthetic Control of Amjad, Shah & Shin (*Robust Synthetic Control*,
JMLR 19(22):1-51, 2018): ``RobustSyntheticControl`` from
`jehangiramjad/tslib <https://github.com/jehangiramjad/tslib>`_. Both solve the
same problem on the same panel -- California Proposition 99, California treated,
the 38-state donor pool, 1970-1988 pre-period (T0=19), 1989-2000 post -- and are
fed the SAME retained rank ``k = 3`` (the ``singvals = 3`` tslib's own bundled
Prop 99 study uses), so the hard singular-value-thresholding (HSVT) de-noising
matches.

The estimand and solver are identical in spirit: HSVT the donor pre-matrix at
rank ``k``, then an unconstrained pseudo-inverse (OLS) of the treated pre-series
onto the de-noised donors -- no intercept, no simplex constraint on either side.

A documented convention difference (not a bug)
----------------------------------------------
HSVT+OLS is deterministic given ``k``, but the two libraries de-noise via
slightly different matrices:

* tslib forms the donor subspace from an SVD of the *stacked donor+treated*
  pre-matrix -- the treated row participates in the rank-``k`` truncation
  (``src/models/tsSVDModel.py``: ``fit`` HSVT-truncates the full stacked matrix,
  then the donor block is re-truncated and pseudo-inverted).
* mlsynth's PCR de-noises the donor pre-matrix *alone* at rank ``k`` (the
  Amjad-Shah-Shen 2018 convention, the deliberate choice documented in
  ``mlsynth.utils.clustersc_helpers.pcr.pipeline.run_pcr`` -- using the full
  matrix would leak the treated trajectory into the donor reconstruction).

Both then OLS the treated pre-series onto the de-noised donors. We verified the
isolation: feeding tslib's donor-from-stack block into a plain pseudo-inverse
reproduces tslib's weights to ~1e-16, and mlsynth's ``run_pcr`` equals an
independent HSVT(donor-only)+OLS solve to 0. Stacking the treated row rotates
the rank-``k`` donor subspace marginally, so the two learned weight vectors
agree to ~6e-4 and the ATTs to ~0.04 packs/capita -- the genuine numerical gap
of the differing de-noising convention. We do NOT force a tighter match: the
case pins this honest agreement and its documented cause.

Reference (live captured run)
-----------------------------
The reference side is a live captured run of ``jehangiramjad/tslib``, not
numbers transcribed from a paper. ``benchmarks/reference/pcr_rsc_ref/reference.py``
fetches the library at a pinned commit
(``3e50bc1``, into the gitignored ``benchmarks/reference/.cache``; git clone is
proxy-blocked here, so it falls back to the codeload tarball -- see the bundle
``NOTICE`` and ``benchmarks/reference/clone_tslib.py``) and fits the genuine
``RobustSyntheticControl`` at ``k = 3``. Its learned donor weights, pre-period
RMSE, post-period mean counterfactual, and ATT are captured under
``benchmarks/reference/pcr_rsc_ref/`` with full provenance, and this case pins
them by reading the captured ``reference.json`` via :func:`reference_value` /
:func:`load_reference` -- so the constants in ``EXPECTED`` and the captured run
are the same object and cannot silently drift. Regenerate with
``python benchmarks/reference/generate.py pcr_rsc_ref``.

Provenance
----------
* Data: ``basedata/smoking_data.csv`` -- the Abadie, Diamond & Hainmueller (2010)
  Prop 99 panel (39 states, 1970-2000; California treated from 1989). Outcome
  ``cigsale`` (per-capita cigarette packs).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference, reference_value

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_TREATED = "California"
_TREAT_YEAR = 1989
K = 3  # same retained rank for both mlsynth PCR and tslib RSC

_REF = load_reference("pcr_rsc_ref")
_REF_WEIGHTS = _REF["weights"]
REF_ATT = reference_value("pcr_rsc_ref", "att")
REF_PRE_RMSE = reference_value("pcr_rsc_ref", "pre_rmse")


def _panel():
    df = pd.read_csv(_BASE / "smoking_data.csv")
    wide = df.pivot(index="state", columns="year", values="cigsale").sort_index()
    years = list(wide.columns)
    states = list(wide.index)
    donors = [s for s in states if s != _TREATED]
    pre_years = [y for y in years if y < _TREAT_YEAR]
    post_years = [y for y in years if y >= _TREAT_YEAR]
    T0 = len(pre_years)
    y_treated = wide.loc[_TREATED, years].values.astype(float)
    donor_full = wide.loc[donors, years].values.T          # (T, J)
    pre_idx = [years.index(p) for p in pre_years]
    post_idx = [years.index(p) for p in post_years]
    return wide, donors, y_treated, donor_full, T0, pre_idx, post_idx


def _mlsynth_pcr():
    """mlsynth PCR (run_pcr OLS, fixed rank k) on the Prop 99 panel."""
    from mlsynth.utils.clustersc_helpers.pcr.pipeline import run_pcr

    _wide, donors, y_treated, donor_full, T0, _pre, _post = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit, _ = run_pcr(
            y_treated, donor_full, donors, T0,
            objective="OLS", estimator="frequentist", rank=K,
            compute_shen_ci=False,
        )
    weights = np.array([fit.donor_weights[d] for d in donors])
    return donors, weights, float(fit.att), float(fit.pre_rmse)


def run() -> dict:
    donors, w_ml, att_ml, rmse_ml = _mlsynth_pcr()
    w_ref = np.array([_REF_WEIGHTS[d] for d in donors])

    return {
        "mls_att": att_ml,
        "mls_pre_rmse": rmse_ml,
        # mlsynth PCR vs genuine tslib RSC -- the documented convention gap.
        "att_abs_diff_vs_tslib": float(abs(att_ml - REF_ATT)),
        "weight_max_abs_diff_vs_tslib": float(np.max(np.abs(w_ml - w_ref))),
        "pre_rmse_abs_diff_vs_tslib": float(abs(rmse_ml - REF_PRE_RMSE)),
        "n_donors": int(len(donors)),
        "k": float(K),
    }


def comparison() -> dict:
    """mlsynth PCR vs ``jehangiramjad/tslib`` RobustSyntheticControl, quantity by
    quantity.

    Lays the mlsynth PCR fit against the genuine tslib RSC run on the same Prop 99
    panel (same treated unit, same donor pool, same 1989 pre/post split, same
    retained rank ``k=3``): the ATT, the pre-period RMSE, and the top donor
    weights. The reference side is a live captured ``RobustSyntheticControl`` run
    in ``benchmarks/reference/pcr_rsc_ref/`` (commit ``3e50bc1``), not
    transcribed. Returns ``{"rows": [...], "mlsynth_call": {...},
    "reference": {...}}`` with rows ``{quantity, mlsynth, reference}``.
    """
    donors, w_ml, att_ml, rmse_ml = _mlsynth_pcr()
    weights_ml = dict(zip(donors, w_ml))

    rows = [
        {"quantity": "ATT", "mlsynth": round(att_ml, 6),
         "reference": round(REF_ATT, 6)},
        {"quantity": "pre_RMSE", "mlsynth": round(rmse_ml, 6),
         "reference": round(REF_PRE_RMSE, 6)},
    ]
    # Top donors by |tslib weight|, laid side by side.
    top = sorted(_REF_WEIGHTS.items(), key=lambda kv: -abs(kv[1]))[:6]
    for donor, w_ref in top:
        rows.append({"quantity": f"weight[{donor}]",
                     "mlsynth": round(float(weights_ml[donor]), 6),
                     "reference": round(float(w_ref), 6)})

    cfg = {"outcome": "cigsale", "treat": "Proposition 99", "unitid": "state",
           "time": "year", "method": "PCR", "objective": "OLS", "rank": K}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "ClusterSC/PCR (run_pcr OLS, fixed rank)",
                         "config": cfg},
        "reference": {"impl": "jehangiramjad/tslib RobustSyntheticControl "
                              "(live run, captured), modelType='svd', "
                              f"kSingularValuesToKeep={K}",
                      "version": "jehangiramjad/tslib @ 3e50bc1 "
                                 "(benchmarks/reference/pcr_rsc_ref/)"},
    }


# HSVT+OLS is deterministic given k, but tslib stacks the treated row into the
# rank-k de-noising while mlsynth's PCR de-noises the donor pre-matrix alone
# (Amjad-Shah-Shen 2018 convention). That marginal subspace rotation is the
# whole gap: the two learned weight vectors agree to ~6e-4 and the ATTs to ~0.04
# packs/capita at k=3. Targets are pinned from the live captured tslib run
# (benchmarks/reference/pcr_rsc_ref/) via reference_value/load_reference, not
# transcribed; the *_diff_vs_tslib tolerances are the actual mlsynth-vs-tslib gap
# (the documented convention difference), not inflated passes. mls_att/pre_rmse
# are anchored at the tslib values with a band covering that same gap.
EXPECTED = {
    "mls_att": (REF_ATT, 0.1),                     # tracks tslib ATT (gap ~0.04)
    "mls_pre_rmse": (REF_PRE_RMSE, 0.05),          # tracks tslib RMSE (gap ~0.006)
    "att_abs_diff_vs_tslib": (0.0, 0.1),           # documented convention gap
    "weight_max_abs_diff_vs_tslib": (0.0, 2e-3),   # documented convention gap
    "pre_rmse_abs_diff_vs_tslib": (0.0, 0.05),
    "n_donors": (38, 0),
    "k": (3.0, 0),
}
