"""Check the corrected gamma-Poisson Gibbs sampler against its own generative model.

Upstream's ``GAP.gibbs_sample`` has never run (four defects, listed in
``mosc_port``), so there is no reference output to match. Validate by
construction instead: draw a panel from the model at known ``H, Z``, sample, and
ask whether the posterior recovers the rate that generated it. The loadings
themselves are identified only up to permutation and scaling, so the rate matrix
``H @ Z`` is the estimable object and the one checked here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mosc_port import gap_gibbs, ppca_em


def main() -> dict:
    rng = np.random.default_rng(20260826)
    n_time, n_unit, latent_dim = 40, 25, 3

    H_true = rng.gamma(2.0, 2.0, size=(n_time, latent_dim))
    Z_true = rng.gamma(2.0, 2.0, size=(latent_dim, n_unit))
    rate_true = H_true @ Z_true
    data = rng.poisson(rate_true).astype(float)

    posterior = gap_gibbs(data, latent_dim=latent_dim, n_samples=400, warmup=400, seed=7)
    rate_hat = np.mean([posterior.rate(s) for s in range(posterior.n_samples)], axis=0)

    rel_err = float(np.mean(np.abs(rate_hat - rate_true) / rate_true))
    corr = float(np.corrcoef(rate_hat.ravel(), rate_true.ravel())[0, 1])

    # The sampler must beat the intercept-only fit that ignores all structure.
    baseline = float(np.mean(np.abs(data.mean() - rate_true) / rate_true))

    # A rank it cannot use should not help: K=1 must fit strictly worse.
    under = gap_gibbs(data, latent_dim=1, n_samples=200, warmup=200, seed=7)
    rate_under = np.mean([under.rate(s) for s in range(under.n_samples)], axis=0)
    rel_err_under = float(np.mean(np.abs(rate_under - rate_true) / rate_true))

    # Held-out cells must still be recovered when the sampler never sees them.
    mask = rng.random((n_time, n_unit)) > 0.10
    masked = gap_gibbs(data, latent_dim=latent_dim, mask=mask, n_samples=400, warmup=400, seed=7)
    rate_masked = np.mean([masked.rate(s) for s in range(masked.n_samples)], axis=0)
    heldout_err = float(np.mean(np.abs(rate_masked[~mask] - rate_true[~mask]) / rate_true[~mask]))

    # PPCA on a Gaussian panel, as a check that the EM arm is not broken.
    gauss = H_true @ Z_true + rng.normal(0, 5.0, size=(n_time, n_unit))
    ppca = ppca_em(gauss, latent_dim=latent_dim, n_samples=50, seed=7)
    ppca_rate = np.mean([ppca.rate(s) for s in range(ppca.n_samples)], axis=0)
    ppca_corr = float(np.corrcoef(ppca_rate.ravel(), rate_true.ravel())[0, 1])

    results = {
        "gap_rate_mean_relative_error": rel_err,
        "gap_rate_correlation": corr,
        "intercept_only_relative_error": baseline,
        "gap_underparameterised_K1_relative_error": rel_err_under,
        "gap_heldout_relative_error": heldout_err,
        "ppca_rate_correlation": ppca_corr,
    }

    checks = {
        "recovers the generating rate (rel err < 0.15)": rel_err < 0.15,
        "beats the intercept-only fit": rel_err < baseline,
        "K=1 fits strictly worse than the true rank": rel_err_under > rel_err,
        "recovers held-out cells (rel err < 0.25)": heldout_err < 0.25,
        "ppca EM recovers a Gaussian panel (corr > 0.99)": ppca_corr > 0.99,
    }
    results["checks"] = checks
    results["all_passed"] = all(checks.values())

    for name, value in results.items():
        if name not in {"checks", "all_passed"}:
            print(f"  {name:48s} {value: .4f}")
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    out = Path(__file__).parent / "gibbs_validation.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    main()
