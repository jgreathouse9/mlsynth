"""Helper modules for the Synthetic Difference-in-Differences (SDID) estimator.

Implements:

    Arkhangelsky, D., Athey, S., Hirshberg, D., Imbens, G., & Wager, S. (2021).
    "Synthetic Difference-in-Differences." American Economic Review.

    Ciccia, D. (2024). "A Short Note on Event-Study Synthetic
    Difference-in-Differences Estimators." arXiv:2407.09565.

    Clarke, D., Pailanir, D., Athey, S., & Imbens, G. (2023). "Synthetic
    difference in differences estimation." arXiv preprint.

Layout follows the SCDI / SPCD / LEXSCM / TASC / MLSC helper packages.

    setup.py        : prepare_sdid_inputs() from one or two dataprep return shapes
    weights.py      : unit-weight QP, time-weight QP, and the zeta regularizer
    cohort.py       : per-cohort SDID estimator (Eq 2 / Eq 3 of Ciccia 2024)
    event_study.py  : pooled event-study aggregation (Eq 6, 7 of Ciccia 2024)
    inference.py    : placebo-based variance estimator + p-value helper
    plotter.py      : event-study chart wrapping resultutils.SDID_plot
    structures.py   : frozen dataclasses (SDIDInputs / SDIDCohort / SDIDEventStudy /
                      SDIDInference / SDIDResults)
"""
