"""Helper modules for the Sequential Synthetic Difference-in-Differences estimator.

Implements:

    Arkhangelsky, D., & Samkov, A. (2025). "Sequential Synthetic Difference
    in Differences." arXiv:2404.00164v2.

The Sequential SDiD estimator differs from the canonical SDiD of Arkhangelsky
et al. (2021) in five ways:

  * It operates on **cohort-level aggregates** ``Y_{a,t} = (1/n_a) Σ_{i:A_i=a} Y_{i,t}``
    rather than unit-level data.
  * Weights satisfy only a simplex sum constraint (no non-negativity).
  * The unit-weight penalty is ``η² Σ ω_j² / π_j`` (population-share-scaled),
    and the time-weight penalty is ``η² Σ λ_l²``.
  * Donors for cohort ``a`` are restricted to **later-adopting cohorts** ``j > a``.
  * Estimation cascades through ``(k, a)`` pairs; each ``τ̂_{a,k}`` imputes
    ``Y_{a, a+k} := Y_{a, a+k} - τ̂_{a,k}`` so subsequent steps see a panel
    where treated cells have been replaced with their estimated counterfactuals.

In the ``η → ∞`` limit the weights become ``ω_j ∝ π_j`` and ``λ_l = 1/(a+k-1)``,
which recovers the imputation-style sequential DiD discussed in Remark 2.2
(closely related to Borusyak et al. 2024).

Layout:

    structures.py   : frozen dataclasses for inputs, per-(a,k) effects,
                      pooled event-study, bootstrap inference, results
    setup.py        : aggregate the panel into cohort-level outcomes + shares
    weights.py      : the two unconstrained-sum QPs
    cohort.py       : per-(a,k) estimator with the imputation update
    algorithm.py    : the Algorithm 1 outer/inner loop
    inference.py    : Bayesian bootstrap (xi_i ~ Exp(1)) for SE / CI
    plotter.py      : event-study chart of tau_hat_k with bootstrap bands
"""
