"""SPILLSYNTH (Iterative SCM) Path-A: German reunification (Melnychuk 2024).

Reproduces the headline of Melnychuk (2024), *"Synthetic Controls with Spillover
Effects: A Comparative Study"* (arXiv 2405.01645): the Iterative ("waterfall")
SCM estimate of the 1990 German reunification's effect on West-German GDP per
capita, cleaning the spillover that Austria (the affected neighbour) carries.

Two arms run.

The first is mlsynth's own specification: Abadie's German predictor block
(``gdp`` / ``trade`` / ``infrate`` averaged over the ``time.predictors.prior``
window 1981-1990; special predictors ``industry`` [1981-1990], ``schooling``
[1980, 1985], ``invest80`` [1980]; SSR over 1960-1989) with the post-period-only
cleaning. The lagged-GDP predictor pulls Austria's weight in synthetic West
Germany to ``~0.43`` (the paper's 0.42), and the iterative estimate is more
negative than the naive SCM:

  ================  ===============  ===============
  Quantity          mlsynth          Melnychuk
  ================  ===============  ===============
  Iterative ATT     ~ -1813          ~ -1970
  naive SCM ATT     ~ -1333          ~ -1600 (Abadie)
  ================  ===============  ===============

The second arm is the reference's own configuration, so the two can be compared
like for like. It differs from the first in two ways, both read off the
authors' code, not the paper:

* the reference's German specification is ``predictors = c("trade","infrate")``
  -- Abadie's lagged-GDP predictor is absent from the script, though the paper
  reports Abadie's 0.42 weight for Austria. Dropping it is worth about -54 USD.
* ``runAllMethodsWithCov`` calls ``runIterativeSCMwithCov(dataprepDf, TRUE,
  TRUE)``, so ``replacePreTreatData`` is TRUE and the cleaned donor's whole
  series is replaced, not only its post-period. In covariate mode that is worth
  about +11 USD, because the switch acts on outcomes and leaves the predictor
  block alone. (Outcome-only it is worth ~320 USD; see
  ``spillsynth_iscm_xval``.)

In that configuration mlsynth returns ~ -1856 against the reference R's
-1869.33, and neither is the paper's -1970: the shipped code does not reproduce
its own headline.

The comparison to the reference cannot be made tight, and that is a property of
the reference. ``Synth::synth``'s ``V`` search on this specification selects a
different optimum under perturbations far below any measurement precision.
Multiplying every predictor by ``1 + eps * N(0,1)`` and refitting the reference:

  ==========  ===========  =============  ================
  eps         regular      inclusive      iterative (TRUE)
  ==========  ===========  =============  ================
  0           -1305.88     -1575.21       -1869.33
  1e-12       -1308.88     -1577.55       -1721.24
  1e-10       -1314.85     -1506.37       -1870.58
  1e-8        -1310.07     -1533.91       -1771.01
  1e-6        -1313.49     -1639.15       -1826.04
  ==========  ===========  =============  ================

A last-bit change moves the iterative estimate by 148 USD, and the movement is
not monotone in ``eps`` -- the signature of a discontinuous selection, not of
error propagation. The spread over the ladder is 149 USD, which is where the
``vs_reference_R`` tolerance comes from. It also explains a 108 USD difference
between two captures of the reference that differed only in reading
``repgermany.dta`` directly against round-tripping it through a CSV.

The instability belongs to the reference's solver, not to the specification.
mlsynth's ``mscmt`` bilevel search over the same ladder, same panel, same
configuration, moves by 0.08 USD and moves monotonically:

  ==========  ================  ===========
  eps         iterative (TRUE)  naive
  ==========  ================  ===========
  0           -1855.876         -1321.095
  1e-12       -1855.875         -1321.096
  1e-10       -1855.877         -1321.098
  1e-8        -1855.905         -1321.126
  1e-6        -1855.955         -1321.164
  ==========  ================  ===========

That asymmetry is why this case pins mlsynth's own cells to a few tens of USD
and the distance to the reference to 150.

Path A (the paper's empirical result) plus cross-validation against the
reference's live R, captured under
``benchmarks/reference/spillsynth_iscm_german/``. mlsynth's own cells are
deterministic (``mscmt`` seeded at 0) and pinned tightly as regression guards.

Runtime: ~5 s (four global ``mscmt`` V-optimisations; the previous
"2-3 min" in this docstring was never measured).
"""
from __future__ import annotations

import os
import warnings

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "repgermany.dta")

# Abadie's German spec: gdp / trade / infrate averaged over time.predictors.prior
# (1981-1990), plus the three special predictors. The lagged-GDP predictor is the
# one the paper carries and the Melnychuk replication file leaves out.
_COV = ["gdp", "trade", "infrate", "industry", "schooling", "invest80"]
_REF_COV = ["trade", "infrate", "industry", "schooling", "invest80"]
_WIN = {"gdp": (1981, 1990), "trade": (1981, 1990), "infrate": (1981, 1990),
        "industry": (1981, 1990), "schooling": (1980, 1985), "invest80": (1980, 1980)}


def _fit(covariates, replace_pre):
    import pandas as pd

    from mlsynth import SPILLSYNTH

    d = pd.read_stata(os.path.abspath(_DATA))
    # _COV may contain the outcome "gdp" (the lagged-GDP predictor); select each
    # column once so the panel doesn't carry a duplicate gdp column.
    cols = ["country", "year", "gdp"] + [c for c in dict.fromkeys(covariates)
                                         if c != "gdp"]
    d = d[cols].copy()
    d["treat"] = ((d["country"] == "West Germany") & (d["year"] >= 1990)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SPILLSYNTH({
            "df": d, "outcome": "gdp", "treat": "treat",
            "unitid": "country", "time": "year", "method": "iterative",
            "affected_units": ["Austria"], "covariates": list(covariates),
            "covariate_windows": {k: v for k, v in _WIN.items() if k in covariates},
            "bilevel_solver": "mscmt", "iterative_replace_pre": replace_pre,
            "display_graphs": False,
        }).fit()


def run() -> dict:
    from benchmarks.reference.clone_iscm import shipped_reference_german

    ref = shipped_reference_german()
    res = _fit(_COV, False)                       # mlsynth's specification
    ref_arm = _fit(_REF_COV, True)                # the reference's specification
    f = res.iterative
    return {
        "iterative_att": float(res.att),
        "naive_att": float(res.att_scm),
        "iterative_more_negative": float(res.att < res.att_scm),
        "austria_cleaned": float(f.cleaned_units == ["Austria"]),
        "n_clean": float(f.n_clean),
        "vs_paper_1970": abs(float(res.att) - (-1970.0)),
        # the reference's own configuration, both knobs matched
        "reference_spec_att": float(ref_arm.att),
        "vs_reference_R": abs(float(ref_arm.att)
                              - ref["cov_iterative_att_replace_pre_true"]),
        "reference_R_att": ref["cov_iterative_att_replace_pre_true"],
        "reference_R_vs_paper_1970": abs(
            ref["cov_iterative_att_replace_pre_true"] - (-1970.0)),
    }


# Deterministic on mlsynth's side (mscmt seeded at 0); the reference cells come
# from a committed capture. The ``*_att`` cells pin mlsynth's values as
# regression guards. ``vs_reference_R`` carries the tolerance the reference's own
# solver indeterminacy imposes -- 150 USD, the spread of Synth's estimate over a
# perturbation ladder from 0 to 1e-6 (see the docstring table). Pinning it
# tighter would report a precision the reference does not have.
EXPECTED = {
    "iterative_att": (-1813.0, 45.0),
    "naive_att": (-1333.0, 45.0),
    "iterative_more_negative": (1.0, 0.0),
    "austria_cleaned": (1.0, 0.0),
    "n_clean": (15.0, 0.0),
    "vs_paper_1970": (157.0, 130.0),       # |-1813 - (-1970)| ~ 157
    "reference_spec_att": (-1856.0, 60.0),
    "vs_reference_R": (13.0, 150.0),       # within the reference's own spread
    "reference_R_att": (-1869.3, 1.0),     # what the authors' code prints
    "reference_R_vs_paper_1970": (100.7, 1.0),  # the code misses its own headline
}
