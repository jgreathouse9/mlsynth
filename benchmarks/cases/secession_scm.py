"""VanillaSC Path-A: lost-autonomy triggers & secessionism (Schulte et al. 2026).

Path A -- reproduce the paper's empirical finding on the authors' data. Schulte,
Scantamburlo and Ackren (2026), *"Lost autonomy triggers and the rise of
secessionism"* (European Political Science Review 18:175-193), use synthetic
control to argue that two "lost-autonomy triggers" -- the 2010 Spanish
Constitutional Court reform of Catalonia's autonomy statute, and the 1994
economic shock in the Faroe Islands -- caused the secessionist surges that
followed. For each treated region they build a synthetic control from the other
European autonomous regions and read the post-trigger gap in secessionist
sentiment (``av_sec1_all``).

The authors' replication runs the ``SyntheticControlMethods`` Python package
(``Synth``, ``pen="auto"``, ``n_optim=100``, covariate predictors) -- penalized
SC (Abadie-L'Hour) with a V-optimized predictor match. mlsynth's canonical
counterpart is :class:`~mlsynth.VanillaSC` with ``backend="outcome-only"`` (the
deterministic convex simplex fit) plus ``inference="placebo"`` (the in-space
RMSPE-ratio test the paper uses).

Because the reference is a different implementation (its own V-optimizer, auto
penalty, and random restarts), the agreement is *close*, not value-for-value:
mlsynth's synthetic tracks the authors' published series with correlation ~0.92
(Catalonia) / ~0.75 (Faroe), and both methods deliver the same finding -- a large
post-trigger secessionist surge in both regions (mlsynth ATT +28 / +24 vs the
paper's +24 / +27). mlsynth's outcome-only pre-treatment fit is a touch tighter
than the authors' penalized fit in both cases. This case pins the reproducible
(deterministic) mlsynth quantities and the correlation with the authors' output;
it is a Path-A finding replication, not a tight cross-validation, so it is not on
the value-for-value dashboard.

Provenance: Schulte, Scantamburlo & Ackren (2026) EPSR, DOI
10.1017/S175577392510026X; data and published synthetic series from the authors'
replication files (13 autonomous regions, 1975-2021).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_HERE = os.path.dirname(__file__)
_DATA = os.path.join(_HERE, "..", "..", "basedata", "secession_autonomy.csv")
_BUNDLE = os.path.join(_HERE, "..", "reference", "secession_scm")
_REF = load_reference("secession_scm")["values"]

# The two treated regions and their lost-autonomy trigger years.
_CASES = {"Catalonia": 2010, "Faroe Islands": 1994}
# mlsynth's canonical config: deterministic outcome-only SC + in-space placebo.
_MLSYNTH_KW = {
    "outcome": "av_sec1_all", "treat": "treat", "unitid": "region_name",
    "time": "year", "backend": "outcome-only", "inference": "placebo",
    "seed": 123, "display_graphs": False,
}


def _fit(region: str, trigger: int):
    """Fit VanillaSC (outcome-only) for one treated region, with the other 12
    autonomous regions as donors."""
    from mlsynth import VanillaSC

    df = pd.read_csv(os.path.abspath(_DATA))[["region_name", "year", "av_sec1_all"]].copy()
    df["treat"] = ((df.region_name == region) & (df.year >= trigger)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({"df": df, **_MLSYNTH_KW}).fit()


def _summary(region: str, trigger: int) -> dict:
    res = _fit(region, trigger)
    ts = res.time_series
    yr = np.asarray(ts.time_periods).ravel()
    obs = np.asarray(ts.observed_outcome, float).ravel()
    cf = np.asarray(ts.counterfactual_outcome, float).ravel()
    pre = yr < trigger
    # correlation of the mlsynth synthetic with the authors' published synthetic
    tag = "catalonia" if region == "Catalonia" else "faroe"
    paper = pd.read_csv(os.path.join(_BUNDLE, f"reference_synth_{tag}.csv"))
    ml = pd.Series(cf, index=yr).reindex(paper.year.values).to_numpy()
    corr = float(np.corrcoef(paper.synthetic.to_numpy(), ml)[0, 1])
    return {
        "att": float(res.att),
        "pre_rmse": float(np.sqrt(np.mean((obs[pre] - cf[pre]) ** 2))),
        "placebo_p": float(res.inference.p_value),
        "corr_paper": corr,
        "top_donor": max(res.weights.donor_weights.items(), key=lambda kv: kv[1])[0],
    }


def run() -> dict:
    cat = _summary("Catalonia", 2010)
    far = _summary("Faroe Islands", 1994)
    return {
        # the paper's headline: a large post-trigger secessionist surge, both regions
        "cat_att": cat["att"],
        "far_att": far["att"],
        "both_positive_surge": float(cat["att"] > 10.0 and far["att"] > 10.0),
        # deterministic pre-treatment fit
        "cat_pre_rmse": cat["pre_rmse"],
        "far_pre_rmse": far["pre_rmse"],
        # mlsynth's outcome-only fit is at least as tight as the authors' penalized fit
        "cat_pre_rmse_le_paper": float(cat["pre_rmse"] <= _REF["cat_paper_pre_rmse"] + 1e-6),
        "far_pre_rmse_le_paper": float(far["pre_rmse"] <= _REF["far_paper_pre_rmse"] + 1e-6),
        # tracks the authors' published synthetic series
        "cat_corr_paper": cat["corr_paper"],
        # Catalonia is the extreme case under the in-space RMSPE-ratio placebo
        "cat_placebo_p": cat["placebo_p"],
        "cat_top_south_tyrol": float(cat["top_donor"] == "South Tyrol"),
    }


# Path A (scenario: full repo). mlsynth's VanillaSC (outcome-only) reproduces
# Schulte et al. (2026)'s finding -- a large post-trigger secessionist surge in
# both Catalonia (ATT +28, trigger 2010) and the Faroe Islands (ATT +24, trigger
# 1994) -- tracking the authors' published penalized-SC synthetic (Catalonia
# corr 0.92) with a comparable-or-tighter pre-treatment fit. Deterministic convex
# solve => machine-tight tolerances on mlsynth's own quantities.
EXPECTED = {
    "cat_att": (27.9693, 0.5),
    "far_att": (23.8982, 0.5),
    "both_positive_surge": (1.0, 0.0),
    "cat_pre_rmse": (2.6640, 0.1),
    "far_pre_rmse": (4.9184, 0.1),
    "cat_pre_rmse_le_paper": (1.0, 0.0),
    "far_pre_rmse_le_paper": (1.0, 0.0),
    "cat_corr_paper": (0.9170, 0.03),
    "cat_placebo_p": (0.0769, 0.02),
    "cat_top_south_tyrol": (1.0, 0.0),
}


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(run(), indent=2))
