"""Path A benchmark: SCM-relaxation Brexit robustness check, treatment 2020Q1.

Reproduces the robustness analysis in the authors' replication repo
(``GDP_application_2020.ipynb`` of https://github.com/YapengZheng/Relaxed_SC),
which re-runs the Liao-Shi-Zheng (2026, arXiv:2508.01793) Brexit application with
the treatment moved from the 2016Q3 referendum (the main case, benchmarked in
``rescm_brexit``) to **2020Q1** -- the UK's formal EU exit / start of the
transition. The point of the robustness check is that the estimated GDP loss is
stable across the two treatment windows.

The assertion target is the L2 relaxation (the paper's recommended dense
estimator). On year-over-year quarterly real-GDP growth with the UK treated from
2020Q1 against the 57-country donor pool, ``RESCM(RELAX_L2)`` reproduces the
notebook's headline: a cumulative GDP loss of about 4.4% (the notebook reports an
average treatment effect of -4.42%, i.e. realized GDP ~4.4% below the no-Brexit
counterfactual over the post window), close to -- and confirming the robustness
of -- the 2016Q3 main result (~3.9%).

mlsynth selects the relaxation parameter ``tau`` with its own cross-validation
scheme, which differs from the authors', so this is a robustness/agreement check,
not a bit-for-bit reproduction: mlsynth's L2 lands at ~4.5% versus the notebook's
4.42%. The tolerance brackets both.

As in ``rescm_brexit``, the classic SC side is only sanity-checked qualitatively:
with ``J = 57`` donors and ``T0 < J`` the SC vertex is non-unique and its
post-period extrapolation is solver-dependent, which is exactly the pathology the
relaxation fixes. The empirical-likelihood / entropy relaxations are not pinned:
the authors' reference solves them with MOSEK (unavailable here) and -- compounded
by the different tau cross-validation -- mlsynth lands ~0.8pp from the notebook on
those two, whereas the L2 relaxation matches to ~0.1pp.

Provenance
----------
* Data: ``balanced_GDP_data.csv`` from the authors' repo (57 donors + UK,
  quarterly real GDP 2002Q4-2024Q2; CEIC). Fetched on demand; skips if absent.
* Method: ``RESCM(methods=["SC", "RELAX_L2"], standardize=False)`` -- matches the
  authors' ``scmrelax`` reference, which solves on the raw growth series.
* Headline: repo ``GDP_application_2020.ipynb`` -- L2 ATE -4.42% (treatment
  2020Q1), versus the 2016Q3 main result of ~3.9% (``rescm_brexit``).
"""
from __future__ import annotations

import warnings

import numpy as np

_URL = "https://raw.githubusercontent.com/YapengZheng/Relaxed_SC/main/balanced_GDP_data.csv"
_TREAT = "2020-03-31"   # 2020Q1 (the robustness-check treatment window)


def _load():
    import pandas as pd

    from benchmarks.compare import BenchmarkSkipped
    try:
        wide = pd.read_csv(_URL)
    except Exception as exc:  # network / repo unavailable
        raise BenchmarkSkipped(f"could not fetch Relaxed_SC data: {exc}")
    wide["Region"] = pd.to_datetime(wide["Region"])
    return wide.sort_values("Region").reset_index(drop=True)


def run() -> dict:
    import pandas as pd

    from mlsynth import RESCM

    wide = _load()
    countries = [c for c in wide.columns if c != "Region"]
    lvl = wide[countries].astype(float)
    z = lvl["United Kingdom"].to_numpy()                # UK level, all quarters
    treat = pd.Timestamp(_TREAT)

    # Year-over-year growth panel (removes trend + seasonality).
    g = (lvl / lvl.shift(4) - 1.0)
    g["date"] = wide["Region"]
    g = g.dropna().reset_index(drop=True)
    T0 = int((g["date"] < treat).sum())
    treat_widx = T0 + 4                                  # growth row T0 -> wide index T0+4

    rows = []
    for _, r in g.iterrows():
        for c in countries:
            rows.append({"unit": c, "time": r["date"], "y": r[c],
                         "treat": int(c == "United Kingdom" and r["date"] >= treat)})
    long = pd.DataFrame(rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = RESCM({"df": long, "outcome": "y", "treat": "treat",
                     "unitid": "unit", "time": "time",
                     "methods": ["SC", "RELAX_L2"], "standardize": False,
                     "n_taus": 30, "display_graphs": False}).fit()

    def cum_loss(cf_growth: np.ndarray) -> float:
        # Compound the predicted no-Brexit YoY growth into a level path; the loss
        # is the counterfactual level above the realized level, as a share of
        # realized GDP over the post window (the notebook's ATE, sign-flipped).
        yhat = z.astype(float).copy()
        for k in range(treat_widx, len(z)):
            yhat[k] = (1.0 + cf_growth[k - 4]) * yhat[k - 4]
        post = np.arange(treat_widx, len(z))
        return float((yhat[post] - z[post]).sum() / z[post].sum() * 100.0)

    sc, l2 = res.fits["SC"], res.fits["RELAX_L2"]
    w_sc, w_l2 = sc.donor_weights, l2.donor_weights

    return {
        "l2_cum_loss_pct": cum_loss(np.asarray(l2.counterfactual)),
        "l2_n_donors": float(len(w_l2)),
        "l2_germany": float(w_l2.get("Germany", 0.0)),
        "l2_france": float(w_l2.get("France", 0.0)),
        "l2_italy": float(w_l2.get("Italy", 0.0)),
        # Classic SC is sparse and drops the big-3 EU economies.
        "sc_big3_weight": float(w_sc.get("Germany", 0.0) + w_sc.get("France", 0.0)
                                + w_sc.get("Italy", 0.0)),
        "relax_denser_than_sc": float(len(w_l2) > 2 * len(w_sc)),
    }


# Pinned after a measured run; cross-checked against the repo notebook's L2 ATE
# of -4.42% (treatment 2020Q1). Tolerances follow rescm_brexit.
EXPECTED = {
    # mlsynth L2 4.55% vs the repo notebook's -4.42% ATE; the gap is mostly the
    # different tau cross-validation (mlsynth's vs the authors'), so the tolerance
    # is set to comfortably bracket both and absorb solver/version drift.
    "l2_cum_loss_pct": (4.55, 0.6),      # ~4.4-4.6%; robust to the 2016Q3 ~3.9%
    "l2_n_donors": (39.0, 12.0),         # dense relaxation
    "l2_germany": (0.044, 0.05),         # includes the major EU economies
    "l2_france": (0.040, 0.05),
    "l2_italy": (0.056, 0.05),
    "sc_big3_weight": (0.0, 0.001),      # SC drops the big-3 EU economies
    "relax_denser_than_sc": (1.0, 0.0),  # relaxation dense vs sparse SC
}
