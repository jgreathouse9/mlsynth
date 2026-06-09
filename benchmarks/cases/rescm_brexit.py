"""Path A benchmark: SCM-relaxation Brexit / UK real-GDP (Liao-Shi-Zheng 2026, Sec. 6).

Reproduces the empirical application: the economic cost of Brexit on UK real GDP,
estimated by the L2 SCM-relaxation against an OECD-style donor pool. The treated
unit is the United Kingdom, treatment starts 2016Q3 (referendum), the outcome is
year-over-year quarterly GDP growth, and the cumulative level effect is rebuilt
by compounding the predicted no-Brexit growth path.

Why the *relaxation* loss (not the classic SC loss) is the assertion target:
with ``J = 57`` donors and only ``T0 = 51`` pre-periods, the classic SC weight
vector is **non-unique** (different solvers pick different optimal vertices, all
with near-perfect pre-fit), so its post-period extrapolation -- and hence its
cumulative loss -- is solver-dependent. That non-uniqueness is precisely what the
paper's relaxation fixes: the L2 relaxation is unique and stable. So this case
asserts the well-posed relaxation result plus the paper's robust qualitative
weight contrast (sparse SC dropping the big-3 EU economies vs. dense relaxation
including them), and only sanity-checks the SC side qualitatively.

Provenance
----------
* Data: ``balanced_GDP_data.csv`` from the authors' application repo
  https://github.com/YapengZheng/Relaxed_SC (57 donors + UK, quarterly real GDP
  2002Q4-2024Q2; CEIC). Fetched on demand; the case skips if unavailable.
* Method: ``RESCM(methods=["SC","RELAX_L2"], standardize=False)`` -- the
  ``standardize=False`` path matches the authors' ``scmrelax`` reference, which
  solves on the raw growth series.
* Headline: Liao-Shi-Zheng (2026, arXiv 2508.01793) Sec. 6 -- L2 cumulative GDP
  loss 3.85% (treatment 2016Q3); the L2 relaxation is dense and weights the
  major EU economies (Italy/Germany/France/...), whereas classic SC is sparse,
  ~0.30 on the USA, and drops Germany/France/Italy.
"""
from __future__ import annotations

import warnings

import numpy as np

_URL = "https://raw.githubusercontent.com/YapengZheng/Relaxed_SC/main/balanced_GDP_data.csv"
_TREAT = "2016-09-30"   # 2016Q3


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
        # realized GDP over the post window.
        yhat = z.astype(float).copy()
        for k in range(treat_widx, len(z)):
            yhat[k] = (1.0 + cf_growth[k - 4]) * yhat[k - 4]
        post = np.arange(treat_widx, len(z))
        return float((yhat[post] - z[post]).sum() / z[post].sum() * 100.0)

    sc, l2 = res.fits["SC"], res.fits["RELAX_L2"]
    w_sc, w_l2 = sc.donor_weights, l2.donor_weights

    return {
        "l2_cum_loss_pct": cum_loss(np.asarray(l2.counterfactual)),
        # The relaxation is dense and includes the big-3 EU economies.
        "l2_n_donors": float(len(w_l2)),
        "l2_germany": float(w_l2.get("Germany", 0.0)),
        "l2_france": float(w_l2.get("France", 0.0)),
        "l2_italy": float(w_l2.get("Italy", 0.0)),
        "l2_usa": float(w_l2.get("United States", 0.0)),
        # Classic SC is sparse, USA-heavy, and drops the big-3 EU economies.
        "sc_usa": float(w_sc.get("United States", 0.0)),
        "sc_big3_weight": float(w_sc.get("Germany", 0.0) + w_sc.get("France", 0.0)
                                + w_sc.get("Italy", 0.0)),
        "relax_denser_than_sc": float(len(w_l2) > 2 * len(w_sc)),
    }


# Deterministic given the data + solver. The L2 relaxation is unique (unlike the
# non-unique J>T0 classic SC), so its loss is solver-stable; tolerance absorbs
# the tau-CV grid granularity (we use n_taus=30 vs the paper's finer CV) -- our
# 4.0% vs the paper's printed 3.85%.
EXPECTED = {
    "l2_cum_loss_pct": (3.9, 0.5),       # paper 3.85%; mlsynth ~4.0%
    "l2_n_donors": (34.0, 12.0),         # dense (paper: dozens of donors)
    "l2_germany": (0.056, 0.05),         # big-3 EU included (paper Fig. 4)
    "l2_france": (0.046, 0.05),
    "l2_italy": (0.069, 0.05),
    "l2_usa": (0.04, 0.05),              # USA small under the relaxation
    "sc_usa": (0.33, 0.12),              # classic SC ~0.30 on the USA
    "sc_big3_weight": (0.0, 0.001),      # classic SC drops Germany/France/Italy
    "relax_denser_than_sc": (1.0, 0.0),
}
