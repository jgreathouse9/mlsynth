"""Helper modules for the Synthetic Business Cycle (SBC) estimator.

Implements:

    Shi, Z., Xi, J., & Xie, H. (2025). "A Synthetic Business Cycle Approach
    to Counterfactual Analysis with Nonstationary Macroeconomic Data."
    arXiv:2505.22388.

Layout:

    setup.py          : prepare_sbc_inputs (wraps dataprep, validates h+p)
    hamilton.py       : Eq. (2) -- Hamilton trend / cycle decomposition
    trend_forecast.py : Step 2  -- treated trend extrapolation
    synthetic.py      : Eq. (3) -- SCM on cycles
    orchestration.py  : end-to-end solve_sbc + summarize_effects
    plotter.py        : observed vs counterfactual
    structures.py     : frozen dataclasses
"""
