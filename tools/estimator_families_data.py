"""Which family each estimator belongs to, and which carry no replication.

The coverage table on ``docs/replications.rst`` is generated from this module by
``tools/gen_coverage_table.py``; ``mlsynth/tests/test_replication_coverage.py``
fails if it and ``mlsynth.__all__`` disagree, or if the committed table is not
what the generator produces.

A new estimator lands red until it is given a family here. Which family a method
belongs to is a decision, and the generator does not guess it.

Plain Python and not TOML because ``tomllib`` arrived in 3.11 and the package
supports 3.10.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

#: Families, in the order their rows appear in the table: (key, title).
FAMILIES: List[Tuple[str, str]] = [
    ("canonical", "Canonical workhorses"),
    ("decomp", "Decomposition-first"),
    ("generalised", "Generalised estimand / treatment / unit"),
    ("hull", "Convex-hull relaxation"),
    ("highdim", "High-dimensional donors"),
    ("time", "Time-aware / factor models"),
    ("bayesian", "Bayesian"),
    ("staggered", "Staggered adoption"),
    ("spillover", "Spillover-aware (donor screening)"),
    ("missing", "Missing data"),
    ("endogeneity", "Identification under endogeneity"),
    ("compositional", "Compositional outcomes"),
    ("nocontrols", "No control units"),
    ("inference", "Honest inference on the ATT"),
    ("randomized", "Randomized assignment"),
    ("forecasting", "Prospective forecasting"),
    ("privacy", "Privacy-constrained release"),
    ("design", "Experimental design"),
]

#: Estimator -> family.
#:
#: The 36 assignments the previous hand-maintained table made are kept as it
#: made them. The rest follow the family sections of docs/replications.rst
#: where those place an estimator, and each method's own "When to use" section
#: where they do not.
ESTIMATORS: Dict[str, str] = {
    # The ones you reach for first.
    "VanillaSC": "canonical",
    "FDID": "canonical",
    "TSSC": "canonical",
    "MASC": "canonical",

    # Match on a cycle or spectral component, not raw outcomes.
    "HSC": "decomp",
    "SBC": "decomp",

    # A different estimand, treatment type, or unit scale.
    "SCMO": "generalised",
    "CTSC": "generalised",
    "DSC": "generalised",
    "SI": "generalised",
    "MicroSynth": "generalised",
    "SCTA": "generalised",
    "DTWSC": "generalised",
    "CSCM": "generalised",
    "DRSC": "generalised",
    "FSC": "generalised",
    "MEDSC": "generalised",
    "MOSC": "generalised",

    # Relaxing the convex-hull assumption.
    "NSC": "hull",
    "ISCM": "hull",
    "SRC": "hull",

    # Donor pools large relative to the pre-period.
    "CLUSTERSC": "highdim",
    "MLSC": "highdim",
    "PDA": "highdim",
    "CPDA": "highdim",
    "SL": "highdim",
    "RESCM": "highdim",
    "FSCM": "highdim",
    "SparseSC": "highdim",
    "BVSS": "highdim",
    "BEAST": "highdim",
    "SCD": "highdim",
    "DROSC": "highdim",
    "MSQRT": "highdim",
    "SCUL": "highdim",

    # State-space and factor models.
    "FMA": "time",
    "ATEL": "time",
    "TASC": "time",
    "CFM": "time",
    "CSCIPCA": "time",
    "LPCA": "time",

    # A posterior over the untreated outcome.
    "BFSC": "bayesian",
    "BPSCS": "bayesian",
    "BSCM": "bayesian",
    "CMBSTS": "bayesian",
    "DMLFM": "bayesian",
    "MTGP": "bayesian",
    "MVBBSC": "bayesian",

    # Many treated units, adopting together or at different times.
    "SDID": "staggered",
    "SpSyDiD": "staggered",
    "PPSCM": "staggered",
    "SSC": "staggered",
    "SequentialSDID": "staggered",
    "SPILLSYNTH": "staggered",
    "CAST": "staggered",
    "GSYNTH": "staggered",
    "ROLLDID": "staggered",
    "STACKEDSC": "staggered",

    # The treatment reaches the donors.
    "SPOTSYNTH": "spillover",
    "RRSC": "spillover",

    # Missing cells in the panel.
    "MCNNM": "missing",
    "SNN": "missing",
    "RMSI": "missing",

    # Treatment endogenous in a way SC cannot absorb.
    "SIV": "endogeneity",
    "PROXIMAL": "endogeneity",
    "DSCAR": "endogeneity",

    # Outcomes that are vectors of shares summing to a whole.
    "COMPSC": "compositional",
    "PROPSC": "compositional",

    # No untreated unit exists to borrow from.
    "SHC": "nocontrols",
    "GPITS": "nocontrols",

    # The counterfactual is not the hard part; the standard error is.
    "ESC": "inference",
    "ORTHSC": "inference",

    # Assignment is randomized, over few large units.
    "MUSC": "randomized",

    # Forecasting past the end of the panel, for a unit that has not adopted.
    "TWSF": "forecasting",

    # Releasing the counterfactual under a formal privacy guarantee.
    "DPSC": "privacy",

    # Choosing who to treat, before the experiment runs.
    "LEXSCM": "design",
    "MAREX": "design",
    "SYNDES": "design",
    "PANGEO": "design",
    "SPCD": "design",
    "GEOX": "design",
}

#: Estimators with no replication, and why. Everything not named here carries
#: one of the three paths in docs/replications.rst. The prose on that page says
#: ISCM is the only exception; the tests hold the two statements together.
UNVERIFIED: Dict[str, str] = {
    "ISCM": (
        "one-draw illustration only: the paper relies on a non-public panel "
        "and provides no Monte Carlo to reproduce"
    ),
}
