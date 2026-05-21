"""Helper modules for the BVS-SS estimator.

Implements:

    Xu, Y., & Zhou, Q. (2025). "Bayesian Synthetic Control with a Soft
    Simplex Constraint." arXiv:2503.06454.

Layout:

    structures.py    : BVSSInputs, BVSSPosterior, BVSSInference, BVSSResults
    setup.py         : prepare_bvss_inputs (wraps dataprep + demeans)
    posterior.py     : V_{gamma,tau}, RSS, RSS2, AM, loglike (Eqs. (4)-(5))
    gibbs_pair.py    : (gamma_i, gamma_j, mu_i, mu_j) joint Gibbs update
                       (Lemmas S1, S2)
    mh.py            : tau Metropolis-Hastings step (Eq. (S1))
    sampler.py       : gibbs_BVS outer loop (Algorithm 1) + phi/tau update
    inference.py     : ATT mean, credible intervals, counterfactual paths
    plotter.py       : observed vs counterfactual + posterior CI band
"""
