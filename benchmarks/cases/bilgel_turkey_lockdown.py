"""Path A: Bilgel (2022), Covid-19 lockdowns and social distancing in Turkey, PPSCM.

Reproduces Table 3, column 1 of Bilgel, F. (2022), *Effects of Covid-19 lockdowns
on social distancing in Turkey*, Econometrics Journal 25(3):781-805,
`10.1093/ectj/utac016 <https://doi.org/10.1093/ectj/utac016>`_, from the author's
own replication package.

The specification is one line per outcome in the package's
``augsynth_replication_final.R``::

    set.seed(1234); multisynth(retail~lockdown, id, eday, nu=0.5,
                               data=augsynth_retail_t0c1)

which is what :class:`~mlsynth.PPSCM` is a port of, so nothing needed
reimplementing: the case reads the author's frames and fits them directly.

What this case pins
-------------------

The six published ATTs. Every row is a *distance* from the printed value, so it
reads as "how far apart are we" and cannot be re-fitted to whatever the
estimator currently returns.

The paper prints two decimals, so the published figure bounds the true one only
to within 0.005. The tolerances are that half-digit: mlsynth lands inside it for
all six, which is the tightest claim the published precision supports.

    retail       -25.0818  vs  -25.08
    grocery      -53.1004  vs  -53.10
    park         -33.4503  vs  -33.45
    transit      -16.7623  vs  -16.76
    workplace    -27.6114  vs  -27.61
    residential   12.0190  vs   12.02

Standard errors are checked but far more loosely, and the reason is not
tolerance-shopping. Column 1's errors are a wild bootstrap, so reproducing them
exactly would need R's RNG stream, not merely the same estimator. What is
checkable is that they land in the same place: the relative error runs 0.8 to
3.2 percent across the six, and the rows admit 10 percent. An inference routine
that broke would miss by far more than that.

Two structural rows guard the design itself. ``n_treated_residential`` is 24
against 31 everywhere else, which is the paper's own footnote a ("Only
residential mobility covers 24 provinces"), and ``n_post_periods`` is the 17
days Table 3 reports. Both would move if the panel were misread.

The vendored data
-----------------

``basedata/bilgel_turkey_lockdown.parquet`` is the package's six
``augsynth_dataset_<outcome>_t0c1.rdata`` frames concatenated, with the outcome
column renamed to ``mobility`` and tagged by ``outcome``. The frames are
otherwise untouched.

Selecting them needs care. The package's ``.rdata`` files are cumulative
workspace saves -- the residential file carries all six frames, the transit file
four -- so loading one and taking "the first object" silently returns a
different outcome's panel. The vendoring step selected each by name.

Treatment reversal is already handled upstream by the author. Sixteen provinces
lift lockdown before the window ends, and Table 3 records the action taken as
"post-lockdown concatenation", so the ``_t0c1`` frames reach the estimator in
absorbing form with a single adoption day.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"

# Table 3, column 1: ATT (SE). Wild-bootstrap standard errors.
_PUBLISHED = {
    "retail": (-25.08, 5.49),
    "grocery": (-53.10, 10.43),
    "park": (-33.45, 7.69),
    "transit": (-16.76, 3.90),
    "workplace": (-27.61, 6.37),
    "residential": (12.02, 2.04),
}

# Half the last printed digit: the paper reports two decimals.
_ATT_TOL = 0.005

EXPECTED = {
    "n_outcomes": (6.0, 0.5),
    "n_treated_retail": (31.0, 0.5),
    "n_treated_residential": (24.0, 0.5),
    "n_post_periods": (17.0, 0.5),
    "att_diff_retail": (0.0, _ATT_TOL),
    "att_diff_grocery": (0.0, _ATT_TOL),
    "att_diff_park": (0.0, _ATT_TOL),
    "att_diff_transit": (0.0, _ATT_TOL),
    "att_diff_workplace": (0.0, _ATT_TOL),
    "att_diff_residential": (0.0, _ATT_TOL),
    "att_diff_max": (0.0, _ATT_TOL),
    "n_se_reported": (6.0, 0.5),
    "se_rel_err_max": (0.0, 0.10),
    "signs_match": (1.0, 0.5),
}


def _panel(outcome: str) -> pd.DataFrame:
    df = pd.read_parquet(_BASE / "bilgel_turkey_lockdown.parquet")
    d = df[df.outcome == outcome].copy()
    return d.sort_values(["id", "eday"]).reset_index(drop=True)


def _fit(outcome: str):
    from mlsynth import PPSCM

    d = _panel(outcome)
    cfg = {
        "df": d,
        "unitid": "province",
        "time": "eday",
        "outcome": "mobility",
        "treat": "lockdown",
        "nu": 0.5,                          # the paper's partial pooling
        "inference_method": "bootstrap",    # column 1's wild bootstrap
        "seed": 1234,
        "display_graphs": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PPSCM(cfg).fit(), d


def run() -> dict:
    out: dict[str, float] = {}
    att_diffs, se_rel_errs, signs = [], [], []

    for name, (pub_att, pub_se) in _PUBLISHED.items():
        res, d = _fit(name)
        att = float(res.effects.att)
        out[f"att_diff_{name}"] = abs(att - pub_att)
        att_diffs.append(abs(att - pub_att))
        signs.append(np.sign(att) == np.sign(pub_att))

        se = getattr(res.effects, "att_std_err", None)
        if se is None:
            se = getattr(getattr(res, "inference", None), "standard_error", None)
        if se is not None and np.isfinite(float(se)):
            se_rel_errs.append(abs(float(se) - pub_se) / pub_se)

        if name == "retail":
            out["n_treated_retail"] = float(
                d.groupby("province").lockdown.max().sum())
            adopt = int(d[d.lockdown == 1].eday.min())
            out["n_post_periods"] = float(int(d.eday.max()) - adopt + 1)
        if name == "residential":
            out["n_treated_residential"] = float(
                d.groupby("province").lockdown.max().sum())

    out["n_outcomes"] = float(len(_PUBLISHED))
    out["att_diff_max"] = float(max(att_diffs))
    # Counted, so an inference routine that stops reporting a standard error
    # fails this case instead of passing it with an empty maximum.
    out["n_se_reported"] = float(len(se_rel_errs))
    out["se_rel_err_max"] = float(max(se_rel_errs)) if se_rel_errs else float("nan")
    out["signs_match"] = float(all(signs))
    return out
