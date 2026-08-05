"""EU ETS co-benefits -- Basaglia, Grunau & Drupp (2024), SDID.

Basaglia, P., Grunau, J., & Drupp, M. A. (2024), *The European Union Emissions
Trading System might yield large co-benefits from pollution reduction*, PNAS
121(28):e2319908121,
`10.1073/pnas.2319908121 <https://doi.org/10.1073/pnas.2319908121>`_.
Replication package: https://github.com/ccs282/EU_ETS_Co_Benefits

The paper's headline estimates come from a generalized synthetic control model.
Alongside it the authors run a synthetic difference-in-differences robustness
check in Stata, and that is what this case replicates: three air pollutants
(SO2, PM2.5, NOx) as ``log(emissions)``, EU-25 countries split into ETS-regulated
and unregulated sectors, treated from 2005.

Why the SDID half and not the GSCM half
----------------------------------------

The two halves run on different samples. The GSCM specifications use the full
panel, which is unbalanced -- Estonia, Latvia, Lithuania and Slovenia enter in
1995, Slovakia in 1992, Hungary in 1991, and the United Kingdom leaves after
2019. The authors' own comment records the consequence::

    tsset id year //Panel is unbalanced --> Need to make it strongly balanced
                  //to run SDID estimator (Arkhangelsky et al., 2021)

So the SDID do-file drops the six late-entering countries and caps the sample at
2019, giving 38 units x 30 years = 1,140 observations. That is the
``Observations 1140`` printed in their tables, and it is the sample reconstructed
below.

The covariate row rule, which is the whole story
------------------------------------------------

The authors call ``covariates(log_gdp log_gdp_2, projected)`` and cite Kranz
(2022). Both halves of that are true and they do not agree with each other.

Kranz's ``xsynthdid`` fits the covariate coefficients on every row where
treatment is not in force -- ``x.rows = as.integer(panel[[treatment]]) == 0`` in
``R/adjust_y.R``. That includes the treated units' own pre-2005 rows. mlsynth's
``covariates={'adjust': ...}`` implements exactly this and is cross-validated
against a live ``xsynthdid`` run at the seam.

Stata's ``sdid`` does something else. Its ``projected()`` selects
``Y[, 6 - NotYet] == 0``, and with the data matrix laid out as
``y, id, id, time, treat_post, treated, tyear, covariates...`` column 6 is the
*ever-treated unit* flag, not the treatment indicator. So the default fits beta
on never-treated units only; the undocumented-in-the-paper ``_not_yet`` option
is what switches it to column 5 and recovers Kranz's rule.

On this panel the difference is not cosmetic. Beta fit on the 19 never-treated
units uses 570 rows; Kranz's rule adds the 19 treated units' 15 pre-treatment
years, 855 rows. The two land in different places, and PM2.5 is where it shows:

    pollutant   Stata      beta on never-treated   beta on all untreated
    SO2        -0.20753          -0.20778                 -0.24492
    PM2.5      -0.32966          -0.32958                 -0.27433
    NOx        -0.12233          -0.12234                 -0.11560

Matching the reference's row rule reproduces all three published estimates to
3e-4. Kranz's rule is 0.007 to 0.055 away. mlsynth is faithful to Kranz and
Stata is not, so this is a disagreement between two reference implementations,
not a defect -- but it does mean mlsynth has no single config that reproduces
``sdid, projected`` directly.

What is therefore compared
--------------------------

``stata_rule_max_diff`` is the binding row. It fits beta under the reference's
own rule, hands the adjusted outcome to ``SDID`` with no covariates, and
compares against the Stata log. That isolates mlsynth's SDID core -- the unit
and time weight programs and the estimator -- from a covariate convention that
the two references do not share. The same separation is used by
``vanillasc_xval_references``, which fixes ``V`` so its comparison measures the
solver and not the predictor-weight search.

``kranz_rule_max_diff`` and ``adjust_backend_max_diff`` are recorded beside it,
so the size of the convention gap is pinned and a change in either direction
shows up.

Precision of the reference
--------------------------

The printed tables carry three decimals (SO2 -0.208, PM2.5 -0.330, NOx -0.122),
but ``Stata_SDID/logs/EUETS_SDID.log`` carries five, and those are the targets
used here. The bootstrap standard errors are not compared: 800 replications
under Stata's RNG cannot be reproduced from Python, and a seed-matched
comparison would be testing the random number stream.

The ``method(did)`` column of the same tables is not compared either. It is
Stata's plain two-way estimator on the same adjusted outcome, and mlsynth's SDID
does not expose a DiD mode, so there is nothing on this side to compare it to.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]

POLLUTANTS = ("so2", "pm25", "nox")

#: The do-file's balancing: drop the six late-entering countries, cap at 2019.
_DROPPED = ("Estonia", "Hungary", "Latvia", "Lithuania", "Slovakia", "Slovenia")
_LAST_YEAR = 2019
_COVS = ["log_gdp", "log_gdp_2"]

#: Stata_SDID/logs/EUETS_SDID.log, the synthetic DiD rows, at five decimals.
_STATA_SDID = {"so2": -0.20753, "pm25": -0.32966, "nox": -0.12233}
#: The same log's plain DiD rows, carried as context only (see the docstring).
_STATA_DID = {"so2": -0.34666, "pm25": -0.49103, "nox": -0.27287}

EXPECTED = {
    "n_pollutants": (3.0, 0.5),
    "n_units": (38.0, 0.5),
    "n_periods": (30.0, 0.5),
    "n_obs": (1140.0, 0.5),
    "panel_is_balanced": (1.0, 0.5),
    "n_treated_units": (19.0, 0.5),
    # the binding comparison: the reference's own covariate row rule
    "stata_rule_max_diff": (0.0, 0.002),
    "signs_match": (1.0, 0.5),
    # recorded: how far the two conventions sit apart
    "kranz_rule_max_diff": (0.0553, 0.01),
    "adjust_backend_max_diff": (0.0226, 0.01),
}


def _panel(pollutant: str) -> pd.DataFrame:
    """The do-file's balanced estimation sample for one pollutant."""
    d = pd.read_parquet(_ROOT / "basedata" / "euets_cobenefits.parquet")
    d = d[(d.pollutant == pollutant) & (d.year <= _LAST_YEAR)
          & (~d.country.isin(_DROPPED))]
    return d.reset_index(drop=True)


def _within(frame: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Two-way within transform. Exact in one pass on a balanced panel."""
    out = pd.DataFrame(index=frame.index)
    for c in cols:
        v = frame[c].astype(float)
        out[c] = (v - frame.groupby("unit_id")[c].transform("mean")
                  - frame.groupby("year")[c].transform("mean") + v.mean())
    return out


def _adjusted_outcome(d: pd.DataFrame, rule: str) -> pd.Series:
    """Kranz's step 1-3: fit beta on ``rule``'s rows, subtract ``X @ beta``.

    ``rule="stata"`` fits on never-treated units only, which is what
    ``sdid, projected`` does. ``rule="kranz"`` fits on every untreated row,
    which is ``xsynthdid``'s default and mlsynth's ``adjust``.
    """
    mask = (d.ets_coverage == "Unregulated") if rule == "stata" else (d.treat_post == 0)
    sub = d[mask]
    w = _within(sub, ["log_emissions"] + _COVS)
    beta = np.linalg.lstsq(w[_COVS].to_numpy(), w["log_emissions"].to_numpy(),
                           rcond=None)[0]
    return d["log_emissions"] - d[_COVS].to_numpy() @ beta


def _sdid(d: pd.DataFrame, outcome: str, covariates=None) -> float:
    from mlsynth import SDID

    cfg = {"df": d, "unitid": "unit_id", "time": "year", "outcome": outcome,
           "treat": "treat_post", "vce": "noinference", "display_graphs": False}
    if covariates is not None:
        cfg["covariates"] = covariates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(SDID(cfg).fit().effects.att)


def run() -> dict:
    d0 = _panel("so2")
    out: dict[str, float] = {
        "n_pollutants": float(len(POLLUTANTS)),
        "n_units": float(d0.unit_id.nunique()),
        "n_periods": float(d0.year.nunique()),
        "n_obs": float(len(d0)),
        "panel_is_balanced": float(len(d0) == d0.unit_id.nunique() * d0.year.nunique()),
        "n_treated_units": float(d0[d0.treat_post == 1].unit_id.nunique()),
    }

    stata_d, kranz_d, adjust_d, signs = [], [], [], []
    for p in POLLUTANTS:
        d = _panel(p)
        target = _STATA_SDID[p]

        for rule, bucket in (("stata", stata_d), ("kranz", kranz_d)):
            frame = d.assign(y_adj=_adjusted_outcome(d, rule))
            att = _sdid(frame, "y_adj")
            bucket.append(abs(att - target))
            if rule == "stata":
                signs.append(np.sign(att) == np.sign(target))

        # mlsynth's own covariate backend, which implements Kranz's rule.
        adjust_d.append(abs(_sdid(d, "log_emissions",
                                  covariates={"adjust": _COVS}) - target))

    out["stata_rule_max_diff"] = float(max(stata_d))
    out["kranz_rule_max_diff"] = float(max(kranz_d))
    out["adjust_backend_max_diff"] = float(max(adjust_d))
    out["signs_match"] = float(all(signs))
    return out
