"""Lamba et al. (2023) tiger reserves: per-reserve SCM on a staggered panel.

Cross-validation (scenario 3, full reference package). Lamba, Teo, Sreekar,
Zeng, Carrasco & Koh (2023), "Climate co-benefits of tiger conservation",
*Nature Ecology & Evolution* 7:1104-1113, estimate avoided deforestation for
India's tiger reserves by fitting one synthetic control per reserve: the outcome
is cumulative forest loss in hectares over 2001-2020, the donor pool is the
untreated protected areas in the same tiger-conservation zone, and thirteen
predictors are averaged over 2001..treatment year (``AgeOfPA`` is taken at the
treatment year alone).

Why this panel earns a case
---------------------------

Not because the method is new -- it is ordinary Abadie synthetic control with
covariates, which ``vanillasc_prop99`` and ``mscmt_basque`` already cover -- but
because the *shape* is one no existing case exercises. Forty-five treated
reserves adopt across nine different years; each draws on a donor pool
restricted to its own conservation zone, so the pool differs by reserve; and the
pre-period is as short as six years against thirteen predictors. Prop 99 and
Basque are single-treated, single adoption date, one fixed donor pool, long
pre-period.

What is asserted, and what is only recorded
-------------------------------------------

The two are different in kind here, and conflating them would produce a case
that either fails for the wrong reason or passes while asserting nothing.

Asserted -- the seam. Both implementations reduce the panel to a 13 x n_units
predictor matrix, which is a deterministic function of the data (means over
stated windows). mlsynth reproduces ``tidysynth``'s matrix to 1e-9 across every
donor of every reserve. A disagreement there is a construction bug and nothing
else, so it is gated tightly. ``mlsynth/tests/test_lamba_tigers.py`` carries the
elementwise version.

Asserted -- mlsynth's own estimates. Pinned as a regression guard so a change in
the solver shows up, not as agreement with anything external.

Recorded, not gated -- the donor weights and effects against ``tidysynth``.
Given identical predictor matrices the two still diverge, because Abadie's
nested search for the predictor weights ``V`` is not identified and the two
optimisers land in different places. mlsynth reaches a *lower* pre-treatment
MSPE on ten of the eleven reserves the reference can fit (the exception is
Similipal-Hadagarh, 364.4 against 334.5), so gating agreement would hold mlsynth
to the worse optimum on most of the panel -- the same reasoning, and the same
resolution, as ``vanillasc_xval_references``. That count is pinned, so a move in
either direction has to be looked at rather than absorbed.

Two facts about the reference are worth stating plainly, because they bound what
this case can mean.

``tidysynth`` 0.2.0 does not honour ``quadopt``. The authors pass
``quadopt = "LowRankQP"``; ``R/synth.R:174`` calls ``kernlab::ipop``
unconditionally inside the ``V`` search, and the final solve at line 303 sits in
an ``if (quadopt == "ipop")`` with no ``else``, so that path assigns no weights
at all. The reference therefore passes ``"ipop"`` explicitly, which is what the
package actually does either way.

``ipop`` cannot fit four of the fifteen reserves, failing with a singular system
where the version the authors used fitted them. Those are recorded in
``failures.csv`` and counted below rather than hidden: a reference that silently
dropped them would overstate how much of the paper the current package can
reproduce.

The published numbers are NOT the target
----------------------------------------

The authors' script calls ``grab_signficance``, a misspelling ``tidysynth``
corrected on 2021-11-30, so their results predate a 2022-08-31 fix for
time-ordering problems that the commit message says distorted package output.
Under that older build their code reproduces Table 1 almost exactly -- all
eleven reserves, totalling 6556.7 ha against a published 6560. So the paper is
reproducible, on a package its own maintainer has since corrected. Pinning
mlsynth to those numbers would pin the defect. See
``docs/replications/lamba_tigers.rst``, which carries the version comparison.

Data ship as ``basedata/tiger_reserves.csv`` (the authors' ``TigerPAs.csv``
verbatim). Regenerate the bundle with
``python benchmarks/reference/generate.py lamba_tigers``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "tiger_reserves.csv")

#: The thirteen predictors, in the paper's order.
_COVS = ["loss", "population", "precipitation", "baselineForest", "AgeOfPA",
         "elevation", "slope", "aspect", "area", "distanceToCity", "ppp2000",
         "meanAGB", "roadLength"]

#: Reserves the current reference can fit, in descending published effect. The
#: four ``ipop`` drops (Valmiki, Biligiri Rangaswamy Temple, Sariska, Anamalai)
#: are deliberately absent -- see the module docstring.
_HEADLINE = "Nawegaon-Nagzira Tiger Reserve"
_ALSO = ["Similipal-Hadagarh WLS/Tiger Reserve", "Udanti WLS",
         "Pilibhit Tiger Reserve"]


def _raw() -> pd.DataFrame:
    d = pd.read_csv(os.path.abspath(_DATA))
    d.loc[d.Zone == "SB", "Zone"] = "CEEG"        # the authors' fold-in
    d["closest.city"] = d["closest.city"] / 60.0  # minutes -> hours
    return d


def _panel(raw: pd.DataFrame, name: str):
    """The paper's per-reserve panel: the reserve plus its same-zone donors."""
    t = raw[raw.name == name].iloc[0]
    ty = int(t["TR.year"])
    donors = raw[(raw["TR.year"] == 0) & (raw.Zone == t.Zone)]
    rows = []
    for _, r in pd.concat([raw[raw.name == name], donors]).iterrows():
        for y in range(2001, 2021):
            rows.append({
                "name": r["name"], "year": y, "loss": 100 * r[f"Loss.{y}"],
                "precipitation": r[f"Rainfall.{y - 1}"],
                "population": r[f"Population.{y - 1}"],
                "baselineForest": np.log(100 * r["Forest.2000"]),
                "elevation": r["elevation"], "slope": r["slope"],
                "aspect": r["aspect"], "area": np.log(r["PAArea"] * 1e-4),
                "distanceToCity": r["closest.city"], "ppp2000": r["ppp2000"],
                "roadLength": r["Road.length"] / 1000.0,
                "meanAGB": r["AGB.mean"], "AgeOfPA": r["agePA"]})
    p = pd.DataFrame(rows)
    p["treat"] = ((p.name == name) & (p.year >= ty)).astype(int)
    return p, ty, len(donors)


def _fit(panel: pd.DataFrame, ty: int):
    from mlsynth import VanillaSC

    wins = {c: (2001, ty) for c in _COVS}
    wins["AgeOfPA"] = (ty, ty)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({
            "df": panel, "outcome": "loss", "treat": "treat", "unitid": "name",
            "time": "year", "backend": "mscmt", "covariates": _COVS,
            "covariate_windows": wins, "fit_window": (2001, ty - 1),
            "seed": 42, "display_graphs": False, "inference": False}).fit()


def _pre_mspe(panel: pd.DataFrame, ty: int, name: str, weights: dict) -> float:
    Y = panel.pivot(index="year", columns="name", values="loss")
    donors = [c for c in Y.columns if c != name]
    pre = [y for y in Y.index if y < ty]
    w = np.array([weights.get(u, 0.0) for u in donors], dtype=float)
    return float(((Y.loc[pre, name] - Y.loc[pre, donors].values @ w) ** 2).mean())


def _averted(panel: pd.DataFrame, name: str, weights: dict) -> float:
    """Avoided forest loss in 2020: synthetic minus observed, in hectares.

    Deliberately not ``effects.att``. The ATT averages ``observed - synthetic``
    over every post-treatment year; the paper reports the terminal-year gap with
    the opposite sign. Substituting one for the other gives a plausible-looking
    number that is wrong in both magnitude and sign -- on the headline reserve,
    -1491 against +2825.
    """
    Y = panel.pivot(index="year", columns="name", values="loss")
    donors = [c for c in Y.columns if c != name]
    w = np.array([weights.get(u, 0.0) for u in donors], dtype=float)
    return float(Y.loc[2020, donors].values @ w - Y.loc[2020, name])


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def _key(name: str) -> str:
    return name.split()[0].lower().replace("-", "_")


def run() -> dict:
    raw = _raw()
    ref = load_reference("lamba_tigers")["values"]
    out: dict = {}

    for name in [_HEADLINE] + _ALSO:
        panel, ty, n_don = _panel(raw, name)
        w = {str(k): float(v) for k, v in _fit(panel, ty).weights.donor_weights.items()}
        k = _key(name)
        out[f"averted_{k}"] = _averted(panel, name, w)
        out[f"pre_mspe_{k}"] = _pre_mspe(panel, ty, name, w)
        if name == _HEADLINE:
            out["n_donors_headline"] = float(n_don)
            out["weight_sum_headline"] = float(sum(w.values()))
            out["n_negative_weights_headline"] = float(
                sum(1 for v in w.values() if v < -1e-9))
            # The solution is interior: every one of the 44 donors carries
            # weight, the smallest about 0.019. Worth pinning because a sparse
            # solution would mean the ridge or the solver had changed character,
            # and because it is the opposite of what a simplex QP usually gives.
            out["weight_min_headline"] = float(min(w.values()))

    # Does mlsynth reach a better pre-treatment fit than the reference? Measured
    # over every reserve the reference could fit, not just the four pinned
    # above, because this count is the stated reason for not gating the weights
    # against the reference -- so it should be the honest count.
    better = 0
    for name in ref_reserves(ref):
        panel, ty, _ = _panel(raw, name)
        w = {str(k): float(v) for k, v in _fit(panel, ty).weights.donor_weights.items()}
        better += int(_pre_mspe(panel, ty, name, w) < ref[f"pre_mspe_{_slug(name)}"])
    out["n_reserves_mlsynth_fits_better"] = float(better)
    out["n_reference_fitted"] = float(ref.get("n_fitted", float("nan")))
    out["n_reference_failed"] = float(ref.get("n_failed", float("nan")))
    return out


def ref_reserves(ref: dict) -> list:
    """Reserve names the reference bundle actually carries, read off its keys.

    Derived rather than hard-coded so that regenerating the bundle with a
    package that fits more (or fewer) reserves changes the comparison instead of
    silently checking a stale list.
    """
    names = list(_raw().name.unique())
    have = {k[len("pre_mspe_"):] for k in ref if k.startswith("pre_mspe_")}
    return [n for n in names if _slug(n) in have]


def comparison() -> dict:
    """mlsynth against ``tidysynth`` 0.2.0, reserve by reserve.

    Rows are ``{quantity, mlsynth, reference}``. The averted-loss rows are the
    ones that do *not* agree, and are shown precisely so the disagreement is
    visible in the benchmark output rather than buried; the pre-treatment MSPE
    rows underneath are the reason mlsynth's side is the one to trust.
    """
    r = run()
    ref = load_reference("lamba_tigers")["values"]
    rows = []
    for name in [_HEADLINE] + _ALSO:
        k, s = _key(name), _slug(name)
        rows.append({"quantity": f"averted_loss_ha[{k}]",
                     "mlsynth": r[f"averted_{k}"],
                     "reference": ref.get(f"averted_{s}", float("nan"))})
        rows.append({"quantity": f"pre_treatment_mspe[{k}]",
                     "mlsynth": r[f"pre_mspe_{k}"],
                     "reference": ref.get(f"pre_mspe_{s}", float("nan"))})
    rows.append({"quantity": "reserves_fitted_of_15",
                 "mlsynth": 15.0, "reference": r["n_reference_fitted"]})
    return {"rows": rows}


# Pins. The averted-loss values are mlsynth's own -- a regression guard, not
# agreement with anything external. Tolerances are set from the measured seed
# spread of the mscmt V search (Udanti moves ~21 ha across seeds; the others are
# stable to ~1e-5), so they survive a different BLAS while staying far tighter
# than the gap to the reference. The structural quantities are exact facts.
EXPECTED = {
    "n_donors_headline": (44.0, 0.0),          # Fig. 1A reports 44 for this zone
    "weight_sum_headline": (1.0, 1e-6),
    "n_negative_weights_headline": (0.0, 0.0),
    "weight_min_headline": (0.0189, 0.002),    # interior: no donor drops out
    "averted_nawegaon_nagzira": (2825.4, 1.0),
    "averted_similipal_hadagarh": (1612.9, 1.0),
    "averted_udanti": (1506.0, 25.0),          # seed-sensitive: 1495.6-1516.5
    "averted_pilibhit": (-339.9, 1.0),
    # mlsynth fits the pre-period better than tidysynth on ten of the eleven
    # reserves the reference could fit. The exception is Similipal-Hadagarh
    # (364.4 against 334.5). If this count moves in either direction the
    # decision not to gate the weights against the reference needs revisiting.
    "n_reserves_mlsynth_fits_better": (10.0, 0.0),
    # The current package fits 11 of the 15 reserves the paper reports; ipop
    # fails on the other four with a singular system. Pinned so a reference
    # regeneration that silently changed this is caught.
    "n_reference_fitted": (11.0, 0.0),
    "n_reference_failed": (4.0, 0.0),
}
