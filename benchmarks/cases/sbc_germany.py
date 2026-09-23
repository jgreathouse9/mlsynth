"""Path A benchmark: SBC German reunification (Shi, Xi & Xie 2025, Sec. 5.1).

Reproduces the Synthetic Business Cycle empirical illustration: on the 1990
German-reunification panel, SBC matches West Germany's *cyclical* component and
concentrates its weights on the donors whose short-run fluctuations track West
Germany's business cycle (Greece, the Netherlands, Italy) -- distinct from
classical level-matching SCM, which leans on trend donors (Austria, USA).

Provenance
----------
* Data: ``basedata/german_reunification.csv`` (16 OECD donors + West Germany,
  GDP per capita 1960-2003; effect from 1991) -- the authoritative Abadie panel
  (identical to ``basedata/repgermany.dta``).
* What the paper publishes for this application: Figures 2-4 and prose. Sec. 5.1
  prints no weights and no effect size, so there is no number here to match. Its
  checkable claims are qualitative -- the cyclical weights fall on the donors
  whose short-run fluctuations track West Germany's, and the effect is negative
  after a brief one-year boost -- and those are pinned exactly below. The
  numbers are mlsynth's own, cross-validated step by step against the authors'
  code in ``mlsynth/tests/test_sbc_reference.py``.

Verification
------------
The SBC method is cross-validated against the authors' own R code (Germany.R:
``lsq`` detrending, ``trend_predict`` forecast, ``Synth::synth``) run on the
authoritative ``repgermany.dta``, captured under
``benchmarks/reference/sbc_germany/`` and pinned per step in
``mlsynth/tests/test_sbc_reference.py``. Two findings the live replication
surfaced:

* mlsynth's Hamilton detrending and trend forecast reproduce every digit the
  authors' captured output records (the donor cycles are detrended on the full
  sample, the treated unit on the pre window -- exactly as the authors do).
* On the synthetic-control step the two diverge, and mlsynth is the more
  accurate one. The cyclical simplex least-squares is strictly convex and
  well-conditioned, so its optimum is unique; mlsynth's FISTA solve certifies
  itself to within 1.4e-6 of it and cvxpy's ECOS and CLARABEL land on the same
  point to 2.7e-6 in the weights, while the authors' ``Synth::synth`` (kernlab
  ``ipop``) converges to a point ~2.6% worse in SSE -- at any tolerance -- giving
  the Netherlands-dominant split and ATT ~-1006. mlsynth reaches the verified
  optimum: the Greece-dominant split, ATT ~-952.
* The authors' shipped wide CSV permutes its donor labels, and not in two
  columns but in fifteen of sixteen: only Italy sits under its own name. Its
  "Japan" and "Portugal" columns hold the Netherlands' and Greece's series, so
  the paper's cycle donors "Italy, Japan, Portugal" are, by the correct labels,
  Italy, the Netherlands and Greece -- the donor set mlsynth recovers. Its
  "Australia" and "Belgium" columns hold the USA's and Austria's, so the paper's
  trend donors are the two donors mlsynth's level-matching weights load.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"


def run() -> dict:
    from mlsynth import SBC

    d = pd.read_csv(_BASE / "german_reunification.csv")
    d["treat"] = ((d["country"] == "West Germany") & (d["year"] >= 1991)).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SBC({"df": d, "outcome": "gdp", "treat": "treat",
                   "unitid": "country", "time": "year",
                   "h": 4, "p": 2, "weights_mode": "simplex",
                   "display_graphs": False}).fit()

    w = res.weights_by_donor or {}
    top3 = sorted(w, key=lambda d: -w[d])[:3]
    return {
        "att": float(res.att),
        "greece_weight": float(w.get("Greece", 0.0)),
        "netherlands_weight": float(w.get("Netherlands", 0.0)),
        "italy_weight": float(w.get("Italy", 0.0)),
        # The paper's two checkable claims for this application.
        "cycle_donors_are_the_papers": float(
            set(top3) == {"Greece", "Netherlands", "Italy"}),
        "att_negative": float(res.att < 0.0),
    }


def _golden() -> dict:
    """The authors' per-step values, at the precision the capture records."""
    import numpy as np

    path = (Path(__file__).resolve().parents[1] / "reference" / "sbc_germany"
            / "golden_steps.txt")
    out: dict = {}
    for line in path.read_text().splitlines():
        if "\t" not in line:
            continue
        key, val = line.split("\t", 1)
        values = np.array([float(x) for x in val.split(",") if x.strip()])
        if key.startswith("hi:"):
            out[key[3:]] = values
        elif key not in out:
            out[key] = values
    return out


def comparison() -> dict:
    """mlsynth SBC against the authors' own Germany.R, quantity by quantity.

    Two kinds of row, and they answer different questions. The detrending and
    trend-forecast rows compare mlsynth against the authors' ``lsq`` and
    ``trend_predict``, captured per step at full precision in
    ``benchmarks/reference/sbc_germany/golden_steps.txt``: those agree to the
    least-squares kernels' own distance, 1.7e-14 of each series' scale. The
    weight, ATT and objective rows compare against what the authors' ipop solve
    produces, and there the two part company -- mlsynth attains a cyclical
    objective 2.6% lower on the identical strictly-convex program, so the gap in
    those rows is the reference solver's suboptimality. See
    ``docs/replications/sbc.rst``.
    """
    import numpy as np

    from benchmarks.reference import load_reference
    from mlsynth.utils.bilevel.simplex import simplex_lstsq
    from mlsynth.utils.sbc_helpers.hamilton import fit_hamilton_filter
    from mlsynth.utils.sbc_helpers.trend_forecast import forecast_treated_trend

    golden = _golden()
    d = pd.read_csv(_BASE / "german_reunification.csv")
    panel = d.pivot(index="year", columns="country", values="gdp")
    T0 = list(panel.index).index(1990) + 1
    fit = fit_hamilton_filter(panel["West Germany"].to_numpy()[:T0], h=4, p=2)
    cycle = fit.cycle_pre[~np.isnan(fit.cycle_pre)]
    forecast = forecast_treated_trend(y_target=panel["West Germany"].to_numpy(),
                                      treated_fit=fit, T0=T0, horizon=4)

    rows = []
    for i, name in enumerate(["intercept", "lag 1", "lag 2"]):
        rows.append({"quantity": f"Hamilton trend coef ({name})",
                     "mlsynth": float(fit.coefficients[i]),
                     "reference": float(golden["treated_trend_coef"][i])})
    for i, label in ((0, "first"), (-1, "last")):
        rows.append({"quantity": f"treated cycle ({label} pre-period)",
                     "mlsynth": float(cycle[i]),
                     "reference": float(golden["treated_cycle_pre"][i])})
    rows.append({"quantity": "donor cycle Greece (first)",
                 "mlsynth": float(fit_hamilton_filter(
                     panel["Greece"].to_numpy(), h=4, p=2).cycle_pre[
                         ~np.isnan(fit_hamilton_filter(
                             panel["Greece"].to_numpy(), h=4, p=2).cycle_pre)][0]),
                 "reference": float(golden["donor_cycle_full:Greece"][0])})
    for i, year in enumerate(range(1991, 1995)):
        rows.append({"quantity": f"trend forecast {year}",
                     "mlsynth": float(forecast[i]),
                     "reference": float(golden["trend_forecast"][i])})

    idx = np.where(~np.isnan(fit.cycle_pre))[0]
    donors = [c for c in panel.columns if c != "West Germany"]
    c0 = np.column_stack([fit_hamilton_filter(panel[c].to_numpy(), h=4, p=2)
                          .cycle_pre[idx] for c in donors])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w_cyc = simplex_lstsq(c0, fit.cycle_pre[idx])
    rows.append({"quantity": "cyclical pre-SSE (lower=better)",
                 "mlsynth": float(np.sum((fit.cycle_pre[idx] - c0 @ w_cyc) ** 2)),
                 "reference": float(golden["synth_loose_sse"][0])})

    ref = load_reference("sbc_germany")
    got = run()
    rows.append({"quantity": "ATT (1991-1994)", "mlsynth": got["att"],
                 "reference": float(ref["values"]["att"])})
    for donor, key in (("Greece", "greece_weight"),
                       ("Netherlands", "netherlands_weight"),
                       ("Italy", "italy_weight")):
        rows.append({"quantity": f"weight[{donor}]", "mlsynth": got[key],
                     "reference": float(ref["values"][key])})

    cfg = {"outcome": "gdp", "treat": "treat", "unitid": "country",
           "time": "year", "estimator": "SBC", "h": 4, "p": 2,
           "weights_mode": "simplex"}
    return {
        "rows": [{**r, "mlsynth": round(r["mlsynth"], 6),
                  "reference": round(r["reference"], 6)} for r in rows],
        "mlsynth_call": {"estimator": "SBC (Hamilton detrend + simplex SC on cycle)",
                         "config": cfg},
        "reference": {"impl": "authors' Germany.R (lsq detrend + trend_predict + "
                              "Synth::synth ipop), live run, captured",
                      "version": "jinxi-atlas/Synthetic-business-cycle-code "
                                 "(arXiv:2505.22388), run on the authoritative "
                                 "Abadie panel"},
    }


# The paper's claims are pinned exactly (tolerance 0). The four numbers are
# mlsynth's own output, and SBC is deterministic here -- repeated runs return
# them bit for bit -- so each is pinned as a value at 1e-6 relative. That width
# is six orders above the 3.5e-12 weight movement measured under 1e-12 relative
# jitter in the cyclical inputs (the scale of a floating-point difference between
# platforms), and three orders below the 1.2e-3 weight shift produced by a
# cyclical solver truncated to 200 iterations, which the previous 0.06 band could
# not see.
EXPECTED = {
    "att": (-952.2015762823139, 9.52e-4),
    "greece_weight": (0.43651006181125107, 4.4e-7),
    "netherlands_weight": (0.3652535087084823, 3.7e-7),
    "italy_weight": (0.1551787869337406, 1.6e-7),
    "cycle_donors_are_the_papers": (1.0, 0.0),
    "att_negative": (1.0, 0.0),
}
