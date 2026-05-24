"""Helper modules for the Two-Step Synthetic Control (TSSC) estimator.

Implements:

    Li, K. T., & Shankar, V. (2023). "A Two-Step Synthetic Control Approach
    for Estimating Causal Effects of Marketing Events." Management Science.
    https://doi.org/10.1287/mnsc.2023.4878

TSSC fits the four SC-class variants (SC, MSCa, MSCb, MSCc) and, in a
formal first step, tests the SC pre-trends assumption via a subsampling
procedure to recommend the variant that best balances bias and
efficiency. The second step estimates the ATT with the recommended
variant.

Layout:

    structures.py       : TSSCInputs, TSSCVariantFit, TSSCRestrictionTest,
                          TSSCSelection, TSSCResults
    setup.py            : prepare_tssc_inputs (wraps dataprep)
    estimation.py       : fit_variant / fit_mscc_beta (constrained LS,
                          ATT, fit, bootstrap CI)
    selection.py        : select_method (Step-1 subsampling tests + tree)
    results_assembly.py : build_summary (standardized BaseEstimatorResults)
    plotter.py          : observed vs recommended-counterfactual plot
"""
