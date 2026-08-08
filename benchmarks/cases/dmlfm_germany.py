"""German reunification -- Pang, Liu & Xu (2022), DMLFM against pblasso 1.0.8.

Pang, X., Liu, L., & Xu, Y. (2022), *A Bayesian Alternative to Synthetic
Control for Comparative Case Studies*, Political Analysis 30(2):269-288,
`10.1017/pan.2021.22 <https://doi.org/10.1017/pan.2021.22>`_. Replication
package: Harvard Dataverse `10.7910/DVN/B6SWA1
<https://doi.org/10.7910/DVN/B6SWA1>`_.

Section 5.1 of the paper fits a dynamic multilevel latent factor model to the
Abadie, Diamond & Hainmueller (2015) panel -- 17 OECD countries, 1960 to 2003,
West Germany treated from 1990 -- and reports the result as Figures 5 and 6
with no printed point estimate. Path A is therefore unavailable, and this case
is a cross-validation against the estimator that produced those figures.

The reference is pblasso 1.0.8, which ships inside the replication package and
is not the same code as the public ``liulch/bpCausal`` repository: that is a
later rename whose argument list cannot express the paper's calls. Its ten-seed
output is staged at ``benchmarks/reference/dmlfm_germany``.

What is compared
----------------
The sampler is stochastic and its seed-to-seed spread is wide relative to the
effect -- the reference's own ten runs span -1639 to -1509. So the comparison
is on a mean across seeds, with the per-seed quantities checked only for the
range they occupy. Point-for-point equality is not available and demanding it
would reject a correct port; the first five reference seeds happened to give a
standard deviation of 18.0 and the next five gave 54.7.

Two properties are checked exactly instead, because they are deterministic.
The design objects the sampler consumes match the reference's own setup code
element-wise, and the covariate scaling matches R's ``sd``. That second one is
the seam that decided this port: ``blasso_default.R:88-91`` divides every
covariate by its pooled standard deviation, and skipping it lets a covariate
that is large in level -- here a mean GDP running to five figures -- absorb the
common structure the factor term should carry.

Factor loadings are compared only through the sorted spectrum of
``|omega_gamma|``. The sampler flips the sign of each loading scale and its
factor together every iteration (``blasso.cpp:454``), which leaves the fit
invariant, so nothing finer is identified.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd

from mlsynth import DMLFM

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REF = _ROOT / "benchmarks" / "reference" / "dmlfm_germany"

COVARIATES = ["pgdp", "trade", "inflation", "industry", "schooling", "invest"]
SEEDS = (1234, 1, 2, 3)
TREATED_INDEX = 7
FIRST_POST_YEAR = 1990

EXPECTED = {
    # the design the sampler consumes, against the reference's own setup code
    "design_matches_reference": (1.0, 0.0),
    # covariate scales against R's sd, the seam that decided the port
    "covariate_scales_max_rel_diff": (0.0, 1e-6),
    # the ATT, on a mean across seeds -- see the module docstring
    "att_abs_diff_from_gold": (0.0, 120.0),
    # pre-treatment fit: the reference tops out at 117 in absolute value
    "pre_gap_maxabs": (117.0, 30.0),
    # the shape of the post-treatment path
    "gap_1990_abs_diff": (0.0, 90.0),
    "gap_2003_abs_diff": (0.0, 700.0),
    # the counterfactual sits above the observed series from 1993 onward
    "crosses_zero_by_1994": (1.0, 0.0),
    # the leading loading scale, relative to the reference's 3247
    "leading_omega_ratio": (1.0, 0.5),
}


def _panel() -> pd.DataFrame:
    """The authors' covariate construction: unit means over all 44 years."""
    d = pd.read_stata(_ROOT / "basedata" / "repgermany.dta",
                      convert_categoricals=False).sort_values(["index", "year"])
    num = d.select_dtypes("number").columns
    d[num] = d[num].astype("float64")
    d["D"] = ((d["index"] == TREATED_INDEX) & (d.year >= FIRST_POST_YEAR)).astype(int)
    old = d.copy()
    for i in d["index"].unique():
        m = d["index"] == i
        s = old[old["index"] == i]
        d.loc[m, "pgdp"] = s.gdp.mean()
        d.loc[m, "trade"] = s.trade.mean()
        d.loc[m, "inflation"] = s.infrate.mean()
        d.loc[m, "industry"] = s.industry.mean()
        d.loc[m, "schooling"] = s.schooling.mean()
        d.loc[m, "invest"] = np.nanmean(
            s[["invest60", "invest70", "invest80"]].to_numpy())
    return d


def _fit(d: pd.DataFrame, seed: int):
    return DMLFM({"df": d, "outcome": "gdp", "unitid": "index", "time": "year",
                  "treat": "D", "covariates": COVARIATES, "re": "time", "r": 10,
                  "ar1": True, "niter": 25000, "burn": 5000, "seed": seed,
                  "display_graphs": False}).fit()


def run() -> dict:
    warnings.simplefilter("ignore")
    import json

    from mlsynth.utils.dmlfm_helpers.config import DMLFMConfig
    from mlsynth.utils.dmlfm_helpers.setup import prepare_dmlfm_inputs

    d = _panel()
    out: dict = {}

    # ---- the deterministic seams ----
    fixture = _ROOT / "mlsynth" / "tests" / "fixtures" / "dmlfm"
    design = json.loads((fixture / "german_design.json").read_text())
    cfg = DMLFMConfig(df=d, outcome="gdp", unitid="index", time="year", treat="D",
                      covariates=COVARIATES, re="time", r=10, niter=100, burn=10,
                      display_graphs=False)
    I = prepare_dmlfm_inputs(cfg)
    checks = [
        np.allclose(I.y, np.asarray(design["y"], float)),
        np.allclose(I.X, np.asarray(design["X"], float), atol=1e-9),
        np.allclose(I.A, np.asarray(design["A"], float), atol=1e-9),
        np.allclose(I.X_tr, np.asarray(design["X_tr"], float), atol=1e-9),
        np.allclose(I.A_tr, np.asarray(design["A_tr"], float), atol=1e-9),
        np.array_equal(I.unit_index + 1, np.asarray(design["id0"], int)),
        np.array_equal(I.time_index + 1, np.asarray(design["time"], int)),
        np.array_equal(I.unit_starts, np.asarray(design["IDbreak"], int)),
        np.array_equal(I.time_starts, np.asarray(design["TIMEbreak"], int)),
        np.array_equal(I.order_by_time + 1, np.asarray(design["id2time"], int)),
    ]
    out["design_matches_reference"] = float(all(checks))

    scales = json.loads((fixture / "covariate_scaling.json").read_text())
    got = d[COVARIATES].to_numpy(float).std(axis=0, ddof=1)
    ref = np.asarray(scales["scales"], float)
    out["covariate_scales_max_rel_diff"] = float(np.abs(got / ref - 1.0).max())

    # ---- the sampled quantities, on a mean across seeds ----
    gold = pd.read_csv(_REF / "gold_summary.csv")
    gold_year = (pd.read_csv(_REF / "gold_att_by_year.csv")
                 .groupby("year").estimated_ATT.mean())
    gold_spec = np.array([3247.0])

    atts, gaps, pre_max, leads = [], [], [], []
    for seed in SEEDS:
        res = _fit(d, seed)
        atts.append(float(res.effects.att))
        g = np.asarray(res.time_series.estimated_gap, float).ravel()
        gaps.append(g)
        pre_max.append(float(np.abs(g[:30]).max()))
        leads.append(float(res.additional_outputs["omega_gamma_spectrum"][0]))
    G = np.vstack(gaps)

    out["att_abs_diff_from_gold"] = float(abs(np.mean(atts) - gold.att.mean()))
    out["pre_gap_maxabs"] = float(np.mean(pre_max))
    out["gap_1990_abs_diff"] = float(abs(G[:, 30].mean() - gold_year.loc[1990]))
    out["gap_2003_abs_diff"] = float(abs(G[:, 43].mean() - gold_year.loc[2003]))
    out["crosses_zero_by_1994"] = float(G[:, 34].mean() < 0 < G[:, 30].mean())
    out["leading_omega_ratio"] = float(np.mean(leads) / gold_spec[0])
    return out
