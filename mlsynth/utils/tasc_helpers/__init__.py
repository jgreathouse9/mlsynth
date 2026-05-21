"""Helper modules for the Time-Aware Synthetic Control (TASC) estimator.

Implements:

    Rho, S., Illick, C., Narasipura, S., Abadie, A., Hsu, D., & Misra, V. (2026).
    "Time-Aware Synthetic Control." arXiv:2601.03099.

Layout mirrors the SCDI / SPCD / LEXSCM helper packages.

    setup.py         : prepare_tasc_inputs() and initial parameter construction
    filtering.py     : Kalman filter (Alg 4) and infinite-variance filter (Alg 5)
    smoothing.py     : Rauch-Tung-Striebel smoother (Alg 6)
    mstep.py         : closed-form MLE M-step (Alg 7)
    em.py            : EM_pre loop (Alg 2)
    orchestration.py : full TASC procedure (Alg 3)
    inference.py     : counterfactual path and posterior confidence intervals
    plotter.py       : observed vs counterfactual trajectory with CI band
    structures.py    : frozen dataclasses tying it all together
"""
