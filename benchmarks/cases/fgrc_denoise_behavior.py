"""What the fGRC denoiser does to the donor matrix, across three panels.

``rpca_method="FGRC"`` reconstructs the donors from the subspace the fGRC
clustering step already fit, instead of deriving a second one with PCP.
``fgrc_keep`` chooses whether the disturbing block ``A2`` survives that
projection: ``"all"`` keeps it (rank ``c1 + c2``), ``"cluster"`` projects onto
``A1`` alone (rank ``c1``).

``"cluster"`` is the paper's own construction and it is the wrong default for
this use, which is what the case pins. Projecting ``A2`` out removes most of
the between-donor spread, and the weight step then has nothing left to combine:
on Basque, West Germany and Proposition 99 alike it puts the entire weight on a
single donor and the pre-period error is four to eleven times the
disturbance-retained fit.

Where the Basque account stops generalising
-------------------------------------------
The first diagnosis of this was Basque-only, where the projection drops the
treated unit outside the donor hull entirely -- above every donor in every
pre-period -- so no convex combination can reach it and the returned effect has
the wrong sign, +1.617 against about -0.70. Measured across three panels, that
out-of-hull condition is not what generalises:

===========  ===================================  ===========================
panel        treated above every donor, raw       ... after keep="cluster"
===========  ===================================  ===========================
Basque       0% of pre-periods                    100%
Germany      0%                                   33%
Prop 99      0%                                   0%
===========  ===================================  ===========================

On Proposition 99 the treated unit stays inside the hull throughout and the fit
still collapses onto one donor. So the mechanism the case pins is the spread
collapse, which holds on all three; leaving the hull is what turns it from a
much worse fit into a wrong-signed one, and that happens on some panels only.

What is deliberately not claimed
--------------------------------
That this denoiser is better than the alternatives. It is not, on any of the
three: cross-validated PCP reaches a lower pre-period error on every panel
(0.0842 against 0.1154, 83.3 against 128.5, 1.886 against 2.928). The
comparison is recorded so nobody has to rediscover it. What the method buys is
coherence -- one subspace estimated once, not two unrelated ones -- and
the measured cost of that is reported here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_PANELS = {
    "Basque": dict(file="basque_data.csv", unit="regionname", time="year",
                   outcome="gdpcap", treated="Basque Country (Pais Vasco)", t0=1975),
    "Germany": dict(file="german_reunification.csv", unit="country", time="year",
                    outcome="gdp", treated="West Germany", t0=1990),
    "Prop99": dict(file="smoking_data.csv", unit="state", time="year",
                   outcome="cigsale", treated="California", t0=1989),
}

#: Abadie-Gardeazabal's published Basque donor weights, as `masc_basque` pins
#: them from Kellogg-Mogstad-Pouliot-Torgovitsky: Cataluna 0.85, Madrid 0.15.
_PUBLISHED_BASQUE = {"Cataluna": 0.85, "Madrid (Comunidad De)": 0.15}


def _basedata(name: str):
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / "basedata" / name


def _long(spec):
    df = pd.read_csv(_basedata(spec["file"]))
    s = df[[spec["unit"], spec["time"], spec["outcome"]]].copy()
    s["treat"] = ((s[spec["unit"]] == spec["treated"])
                  & (s[spec["time"]] >= spec["t0"])).astype(int)
    return s


def _fit(spec, **kw):
    from mlsynth import CLUSTERSC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CLUSTERSC(dict(
            df=_long(spec), outcome=spec["outcome"], treat="treat",
            unitid=spec["unit"], time=spec["time"], method="rpca",
            cluster_method="fgrc", weight_objective="simplex",
            display_graphs=False, **kw)).fit()


def _donor_matrices(spec):
    """Raw, keep='all' and keep='cluster' pre-period donor matrices."""
    from mlsynth.utils.clustersc_helpers.rpca.fgrc import fgrc_lowrank, fgrc_subspace
    df = pd.read_csv(_basedata(spec["file"]))
    wide = df.pivot(index=spec["time"], columns=spec["unit"],
                    values=spec["outcome"]).dropna(axis=1)
    T0 = int((wide.index < spec["t0"]).sum())
    treated = wide[spec["treated"]].values[:T0].astype(float)
    raw = wide[[c for c in wide.columns if c != spec["treated"]]].values.T.astype(float)
    sub = fgrc_subspace(raw, c1=2, c2=1, k=2, n_knots=max(4, T0 // 2 - 2),
                        order=4, seed=0)
    return (treated, raw[:, :T0],
            fgrc_lowrank(sub, keep="all")[:, :T0],
            fgrc_lowrank(sub, keep="cluster")[:, :T0])


def _nonzero(result) -> dict:
    return {k: float(v) for k, v in (result.weights.donor_weights or {}).items()
            if abs(v) > 1e-3}


def run() -> dict:
    rmse_ratios, cluster_nonzero, spread_all, spread_cluster = [], [], [], []
    mean_preserved, outside_hull, beats_pcp = [], [], []

    for name, spec in _PANELS.items():
        keep_all = _fit(spec, rpca_method="FGRC")
        keep_c1 = _fit(spec, rpca_method="FGRC", fgrc_keep="cluster")
        pcp = _fit(spec, rpca_method="PCP", cv_lambda=True)

        rmse_ratios.append(keep_c1.fit_diagnostics.rmse_pre
                           / keep_all.fit_diagnostics.rmse_pre)
        cluster_nonzero.append(len(_nonzero(keep_c1)))
        beats_pcp.append(keep_all.fit_diagnostics.rmse_pre
                         < pcp.fit_diagnostics.rmse_pre)

        treated, raw, L_all, L_c1 = _donor_matrices(spec)
        spread_all.append(L_all.std(axis=0).mean() / raw.std(axis=0).mean())
        spread_cluster.append(L_c1.std(axis=0).mean() / raw.std(axis=0).mean())
        mean_preserved.append(
            abs(L_c1.mean() - raw.mean()) <= 1e-9 * abs(raw.mean())
            and abs(L_all.mean() - raw.mean()) <= 1e-9 * abs(raw.mean()))
        outside_hull.append(float((treated > L_c1.max(axis=0)).mean()) > 0.0)

    basque = _nonzero(_fit(_PANELS["Basque"], rpca_method="FGRC"))
    published_gap = max(
        abs(basque.get(donor, 0.0) - weight)
        for donor, weight in _PUBLISHED_BASQUE.items())

    return {
        "n_panels": float(len(_PANELS)),
        # the mechanism, and it holds on every panel
        "n_panels_cluster_collapses_to_one_donor": float(
            sum(n == 1 for n in cluster_nonzero)),
        "min_rmse_ratio_cluster_over_all": float(min(rmse_ratios)),
        "max_spread_retained_by_cluster": float(max(spread_cluster)),
        "min_spread_retained_by_all": float(min(spread_all)),
        "n_panels_mean_level_preserved": float(sum(mean_preserved)),
        # the part that does not generalise
        "n_panels_treated_leaves_hull": float(sum(outside_hull)),
        # against the published Basque solution, and against the alternative
        "basque_max_gap_from_published_weights": float(published_gap),
        "n_panels_fgrc_beats_cv_pcp_pre_fit": float(sum(beats_pcp)),
    }


# Deterministic: fGRC's multi-restart search is seeded from `random_state`, and
# repeat fits on all three panels return bit-identical reconstructions.
#
# The binding assertions are the mechanism, at zero tolerance where the quantity
# is a count: keep="cluster" collapses to exactly one donor on all three panels,
# the mean level survives on all three (which is what separates this from a
# reconstruction bug), and the pre-period error is at least three times worse.
# `n_panels_treated_leaves_hull` is 2 of 3 and banded to admit 1 or 3 -- it is
# recorded because the first diagnosis of this failure was Basque-only and read
# the hull condition as the mechanism, which three panels do not support.
# `n_panels_fgrc_beats_cv_pcp_pre_fit` is 0 and pinned at zero tolerance: the
# method does not beat cross-validated PCP on any panel, and a case that let
# that drift unnoticed would be advertising something untrue.
EXPECTED = {
    "n_panels": (3.0, 0.0),
    "n_panels_cluster_collapses_to_one_donor": (3.0, 0.0),       # binding
    "min_rmse_ratio_cluster_over_all": (4.18, 1.2),              # binding, floor ~3
    "max_spread_retained_by_cluster": (0.40, 0.15),
    "min_spread_retained_by_all": (0.985, 0.03),
    "n_panels_mean_level_preserved": (3.0, 0.0),                 # binding
    "n_panels_treated_leaves_hull": (2.0, 1.0),
    "basque_max_gap_from_published_weights": (0.0137, 0.05),
    "n_panels_fgrc_beats_cv_pcp_pre_fit": (0.0, 0.0),            # binding
}
