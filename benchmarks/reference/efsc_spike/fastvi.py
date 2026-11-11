"""Drop-in replacement for ``efsc.vi.make_negative_elbo``.

The upstream objective evaluates ``log q`` through
``torch.distributions.MultivariateNormal(mean, torch.diag(var))``, which builds
and Cholesky-factorises a dense D x D matrix on every Adam step (D is about
1,000 at the paper's panel sizes). A diagonal Gaussian's log density is the sum
of univariate log densities, so the objective below is the same function of the
same random draws; ``check_fastvi.py`` confirms the two produce the same fit.
"""

import math

import torch


def make_negative_elbo(log_joint, dim, device, n_samples=2):
    def negative_elbo(params):
        d2 = len(params) // 2
        mean = params[:d2]
        log_s2 = params[d2:]
        sd = torch.sqrt(torch.exp(log_s2))
        z = torch.randn((n_samples, dim), device=device)
        theta_samples = mean[None, :] + sd[None, :] * z
        elbo = 0.0
        for s in range(n_samples):
            theta_s = theta_samples[s, :]
            log_q = -0.5 * torch.sum(
                math.log(2 * math.pi) + log_s2 + (theta_s - mean) ** 2 / torch.exp(log_s2)
            )
            elbo = elbo + log_joint(theta_s) - log_q
        return -elbo / n_samples

    return negative_elbo


def install():
    """Route ``efsc.fitting`` through the diagonal objective."""
    import efsc.fitting

    efsc.fitting.make_negative_elbo = make_negative_elbo
