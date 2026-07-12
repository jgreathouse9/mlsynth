"""Cross-validation: mlsynth SBC vs the authors' own R (Hong Kong handover).

Cross-validation against the canonical implementation. mlsynth's SBC
(:class:`mlsynth.estimators.sbc.SBC`, Synthetic Business Cycle -- Shi, Xi & Xie
2025, arXiv:2505.22388) is checked against the authors' own replication script
``SBC_HK/SBC_HK.R`` (`jinxi-atlas/Synthetic-business-cycle-code
<https://github.com/jinxi-atlas/Synthetic-business-cycle-code>`_): linear-
projection (Hamilton) detrending at horizon ``h = 4`` with ``p = 2`` lags, then
the Abadie synthetic-control optimizer applied to the cyclical component. This
is the SBC method's *second* empirical application (the first, German
reunification, is pinned by ``sbc_germany`` / ``test_sbc_reference.py``); the
Hong Kong handover panel is a distinct dataset and treatment.

Both run the identical estimand on the identical panel: FRED/World Bank PPP GDP
per capita (``basedata/hong_kong_handover.csv``, 1961-2010), Hong Kong treated
at the 1997 handover, the authors' 11-donor pool (Australia, Austria, Korea,
Canada, Denmark, France, Germany, Italy, Netherlands, New Zealand, US). The
treated series is detrended on the pre-period (1961-1996) and the donors on the
full sample, exactly as ``SBC_HK.R`` does.

Two independent checks (matching the German-reunification finding)
------------------------------------------------------------------
1. Detrending is shared and exact. mlsynth's Hamilton filter reproduces the
   authors' ``lsq`` trend coefficients and cyclical residuals to ~5e-11 -- the
   SBC-specific machinery (detrend, then SC the cycle) is the same on both sides.

2. On the synthetic-control step the two diverge, and mlsynth is the more
   accurate one -- the same result the German case documents. The cyclical
   simplex least-squares is strictly convex, so its optimum is unique; mlsynth's
   in-house projected-gradient solver attains it (pre-period cyclical
   SSE ~1.648e7), while the authors' ``Synth::synth`` (kernlab ``ipop``)
   converges to a point ~6% worse in SSE (~1.749e7) at any tolerance. Both
   recover an Italy/Germany-dominant cyclical synthetic Hong Kong and a negative
   post-handover effect; the ATTs (mlsynth -5111, authors -4993, per-capita PPP
   USD) differ only through ipop's sub-optimal weights.

Reference (live captured run)
-----------------------------
The reference side is a live captured R run of the authors' method, not numbers
transcribed from the paper. ``benchmarks/reference/sbc_hongkong/reference.R``
runs the ``lsq`` detrending + ``trend_predict`` forecast + ``Synth::synth`` ipop
on the in-repo panel; its ATT, classical-SC ATT, attained cyclical SSE and SBC
donor weights are captured under ``benchmarks/reference/sbc_hongkong/`` with full
provenance, and this case reads the captured ``reference.json`` via
:func:`reference_value` / :func:`load_reference`. Regenerate with
``python benchmarks/reference/generate.py sbc_hongkong``.

The upstream repo carries no licence, so the authors' method is reproduced on the
in-repo public FRED data rather than vendored (mirroring ``sbc_germany``).

Provenance
----------
* Data: ``basedata/hong_kong_handover.csv`` -- FRED/World Bank PPP GDP per capita
  (constant USD), 12 economies, 1961-2010; Hong Kong treated from the 1997
  handover (``Handover`` column).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference, reference_value

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_TREATED = "Hong Kong"
_TREAT_YEAR = 1997
_H, _P = 4, 2
_DONORS = ["Australia", "Austria", "Korea", "Canada", "Denmark", "France",
           "Germany", "Italy", "Netherlands", "New Zealand", "US"]

_REF = load_reference("sbc_hongkong")
_REF_WEIGHTS = _REF["weights"]
REF_ATT = reference_value("sbc_hongkong", "att")
REF_CYC_SSE = reference_value("sbc_hongkong", "sbc_cyc_sse")


def _mlsynth_sbc():
    """mlsynth SBC (simplex) on the Hong Kong handover panel."""
    from mlsynth import SBC

    df = pd.read_csv(_BASE / "hong_kong_handover.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SBC({
            "df": df, "outcome": "gdp", "treat": "Handover",
            "unitid": "country", "time": "year",
            "h": _H, "p": _P, "weights_mode": "simplex",
            "display_graphs": False,
        }).fit()
    weights = {str(k): float(v) for k, v in res.donor_weights.items()}
    return df, weights, float(res.att)


def _cyclical(df):
    """The pre-period cyclical treated vector and donor matrix (authors' setup):
    treated detrended on the pre-window, donors on the full sample."""
    from mlsynth.utils.sbc_helpers.hamilton import fit_hamilton_filter

    wide = df.pivot(index="year", columns="country", values="gdp").sort_index()
    years = wide.index.tolist()
    T0 = sum(y < _TREAT_YEAR for y in years)
    treated_fit = fit_hamilton_filter(wide[_TREATED].to_numpy()[:T0], h=_H, p=_P)
    idx = np.where(~np.isnan(treated_fit.cycle_pre))[0]
    c1 = treated_fit.cycle_pre[idx]
    c0 = np.column_stack(
        [fit_hamilton_filter(wide[c].to_numpy(), h=_H, p=_P).cycle_pre[idx]
         for c in _DONORS])
    return treated_fit, c1, c0


def _detrend_max_abs_diff(df):
    """Max abs diff of mlsynth's treated Hamilton coefs+cycle vs the R lsq (the
    R values are captured; here we recompute the authors' lsq to compare)."""
    # The reference detrending is deterministic OLS; recompute it independently
    # (base numpy) as the oracle and compare mlsynth's Hamilton filter to it.
    wide = df.pivot(index="year", columns="country", values="gdp").sort_index()
    years = wide.index.tolist()
    T0 = sum(y < _TREAT_YEAR for y in years)
    z = wide[_TREATED].to_numpy()[:T0]
    tt = len(z)
    y = z[(_H + _P - 1):tt - 1 + 1]  # y = z[(h+p):tt] (1-indexed) -> 0-indexed below
    # Build the R lsq design: y = z[(h+p):tt]; X = embed(z[1:(tt-h)], p)
    y = z[_H + _P - 1: tt]
    zt = z[0: tt - _H]
    # embed(., p): rows [z[t], z[t-1]] for t = p..len(zt)
    X = np.column_stack([zt[_P - 1 - k: len(zt) - k] for k in range(_P)])
    n = min(len(y), X.shape[0])
    y, X = y[:n], X[:n]
    A = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef

    from mlsynth.utils.sbc_helpers.hamilton import fit_hamilton_filter
    ft = fit_hamilton_filter(z, h=_H, p=_P)
    cyc = ft.cycle_pre[~np.isnan(ft.cycle_pre)]
    return float(max(np.max(np.abs(ft.coefficients - coef)),
                     np.max(np.abs(cyc - resid))))


def run() -> dict:
    from mlsynth.utils.bilevel.simplex import simplex_lstsq

    df, w_ml, att_ml = _mlsynth_sbc()
    _ft, c1, c0 = _cyclical(df)

    w_cyc = simplex_lstsq(c0, c1)
    mls_sse = float(np.sum((c1 - c0 @ w_cyc) ** 2))
    ipop_w = np.array([_REF_WEIGHTS.get(d, 0.0) for d in _DONORS])
    ipop_sse_on_same = float(np.sum((c1 - c0 @ ipop_w) ** 2))

    return {
        "mls_att": att_ml,
        "att_negative": 1.0 if att_ml < 0 else 0.0,
        "mls_cyc_sse": mls_sse,
        # mlsynth reaches a weakly-lower (better) optimum than the authors' ipop
        # on the strictly-convex cyclical program.
        "mls_sse_le_ipop": 1.0 if mls_sse <= ipop_sse_on_same + 1.0 else 0.0,
        "ipop_sse_excess_frac": float((ipop_sse_on_same - mls_sse) / mls_sse),
        # detrending is the shared, exact SBC-specific machinery.
        "detrend_max_abs_diff": _detrend_max_abs_diff(df),
        # top cyclical donors mlsynth loads (Italy/Germany-dominant, as R).
        "italy_plus_germany": float(w_ml.get("Italy", 0.0) + w_ml.get("Germany", 0.0)),
        "n_donors": float(len(_DONORS)),
    }


def comparison() -> dict:
    """mlsynth SBC vs the authors' ``SBC_HK.R`` (live captured R), quantity by
    quantity.

    Lays the mlsynth SBC fit against the authors' own R method on the same Hong
    Kong handover panel: the post-handover ATT, the attained pre-period cyclical
    SSE (where mlsynth reaches the strictly-better optimum), and the SBC cyclical
    donor weights. The reference side is a live captured
    ``lsq``+``Synth::synth`` run in ``benchmarks/reference/sbc_hongkong/``, not
    transcribed. Returns ``{"rows": [...], "mlsynth_call": {...},
    "reference": {...}}``.
    """
    from mlsynth.utils.bilevel.simplex import simplex_lstsq

    df, w_ml, att_ml = _mlsynth_sbc()
    _ft, c1, c0 = _cyclical(df)
    mls_sse = float(np.sum((c1 - c0 @ simplex_lstsq(c0, c1)) ** 2))

    rows = [
        {"quantity": "ATT (post-handover)", "mlsynth": round(att_ml, 3),
         "reference": round(REF_ATT, 3)},
        {"quantity": "cyclical pre-SSE (lower=better)",
         "mlsynth": round(mls_sse, 1), "reference": round(REF_CYC_SSE, 1)},
    ]
    top = sorted(_REF_WEIGHTS.items(), key=lambda kv: -abs(kv[1]))[:4]
    for donor, w_ref in top:
        rows.append({"quantity": f"weight[{donor}]",
                     "mlsynth": round(float(w_ml.get(donor, 0.0)), 6),
                     "reference": round(float(w_ref), 6)})

    cfg = {"outcome": "gdp", "treat": "Handover", "unitid": "country",
           "time": "year", "estimator": "SBC", "h": _H, "p": _P,
           "weights_mode": "simplex"}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SBC (Hamilton detrend + simplex SC on cycle)",
                         "config": cfg},
        "reference": {"impl": "authors' SBC_HK.R (lsq detrend + trend_predict + "
                              "Synth::synth ipop), live run, captured",
                      "version": "jinxi-atlas/Synthetic-business-cycle-code "
                                 "(arXiv:2505.22388), reproduced on in-repo FRED data"},
    }


# Detrending is deterministic and shared: mlsynth's Hamilton filter reproduces
# the authors' lsq coefficients + cyclical residuals to ~5e-11. On the cyclical
# SC step the strictly-convex simplex program has a unique optimum; mlsynth's
# projected-gradient solver attains it (~1.648e7 SSE), strictly better than the
# authors' Synth::synth ipop (~1.749e7, +6%). Both give an Italy/Germany-dominant
# cyclical synthetic HK and a negative post-handover ATT. Targets are pinned from
# the live captured R run (benchmarks/reference/sbc_hongkong/) via
# reference_value/load_reference. mls_att is the mlsynth headline (it differs
# from the R ATT only through ipop's sub-optimal weights, so it is anchored with
# a band, not to the R value).
EXPECTED = {
    "mls_att": (-5110.78, 5.0),                 # mlsynth headline ATT (per-capita PPP USD)
    "att_negative": (1.0, 0.0),                 # negative post-handover effect
    "mls_cyc_sse": (16484515.7, 5000.0),        # mlsynth attains the verified optimum
    "mls_sse_le_ipop": (1.0, 0.0),              # <= authors' ipop on the same program
    "ipop_sse_excess_frac": (0.061, 0.02),      # ipop is ~6% worse
    "detrend_max_abs_diff": (0.0, 1e-6),        # shared exact detrending
    "italy_plus_germany": (0.770, 0.05),        # Italy+Germany carry the cyclical mass
    "n_donors": (11.0, 0.0),
}
