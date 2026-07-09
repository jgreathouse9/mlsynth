"""EIV synthetic-control inference (Hirshberg 2021) Path-B coverage Monte Carlo.

Hirshberg (2021), *"Least Squares with Error in Variables"* (arXiv 2104.08931),
Corollary 3, proves the synthetic-control estimator is asymptotically normal with
an estimable variance as the number of pre-treatment periods grows, so
``tau_hat +/- z_{alpha/2} sigma_hat_tau`` has nominal coverage. The paper gives no
simulation; this case supplies one on its own low-rank-plus-noise DGP and checks
that mlsynth's shipped ``eiv_intervals`` delivers the promised behaviour.

DGP (the paper's model): an ``(T0+1) x J`` donor signal ``A`` of rank ``r``, true
simplex weights ``theta0``, treated signal ``b = A theta0``; add i.i.d. Gaussian
donor noise ``eps`` and treated noise ``nu``; the last period is the (untreated)
target with a known effect ``tau_true = 0``. Over Monte Carlo draws we fit simplex
weights and form the EIV interval, recording:

* coverage of the 95% interval near nominal (the ``sigma_hat = pre-residual SD``
  estimator that mlsynth uses -- consistent for the *full* per-period error scale
  ``sqrt(sigma_nu^2 + sigma_e^2 ||theta||^2)``);
* coverage rising toward nominal as ``T0`` grows (Hirshberg's ``T0 -> inf``);
* the point estimate is unbiased;
* Hirshberg's donor-only ``sigma_e ||theta||`` under-covers a real single unit
  (it drops the treated shock ``nu_e``), documenting why mlsynth uses the
  pre-residual scale instead.
"""
from __future__ import annotations

import numpy as np

_Z = 1.959963985


def _fit_simplex(Xpre, ypre):
    import cvxpy as cp

    J = Xpre.shape[1]
    th = cp.Variable(J)
    cp.Problem(cp.Minimize(cp.sum_squares(ypre - Xpre @ th)),
               [cp.sum(th) == 1, th >= 0]).solve(solver=cp.CLARABEL)
    return np.clip(np.asarray(th.value).ravel(), 0.0, None)


def _one(rng, n, p, r, sig, sige, tau_true):
    from mlsynth.utils.vanillasc_helpers.eiv import eiv_intervals

    U = rng.standard_normal((n + 1, r)); V = rng.standard_normal((p, r))
    A = U @ V.T / np.sqrt(r)
    th0 = rng.random(p); th0[rng.random(p) > 0.3] = 0.0
    if th0.sum() == 0:
        th0[0] = 1.0
    th0 = th0 / th0.sum()
    b = A @ th0
    X = A + sige * rng.standard_normal((n + 1, p))
    nu = sig * rng.standard_normal(n + 1)
    y_all = np.concatenate([b[:n] + nu[:n], [b[n] + nu[n] + tau_true]])
    Y0 = X
    W = _fit_simplex(X[:n], y_all[:n])
    ev = eiv_intervals(y_all, Y0, n, W, alpha=0.05)
    tau_hat = float(ev.tau[0])
    covered = ev.lower[0] <= tau_true <= ev.upper[0]
    # Hirshberg's donor-only sigma_tau = sigma_e * ||theta|| uses the *known*
    # donor-noise scale sige (his sigma_e), not the full prediction-error scale.
    sd_h = sige * ev.metadata["theta_l2"]
    cov_h = abs(tau_hat - tau_true) <= _Z * sd_h
    return covered, tau_hat, cov_h


def _coverage(n, nrep, seed, sig=0.3, sige=0.3, tau_true=0.0):
    rng = np.random.default_rng(seed)
    cov = 0; covh = 0; taus = []
    for _ in range(nrep):
        c, t, ch = _one(rng, n, 30, 3, sig, sige, tau_true)
        cov += c; covh += ch; taus.append(t)
    return cov / nrep, covh / nrep, float(np.mean(taus) - tau_true)


def run() -> dict:
    cov40, _, _ = _coverage(40, 600, seed=1)
    cov160, covh160, bias160 = _coverage(160, 600, seed=2)
    # Hirshberg's own scaling: treated noise p^{-1/2}-small -> his sigma_e||theta|| valid
    _, covh_regime, _ = _coverage(160, 600, seed=3, sig=0.3 / np.sqrt(30))
    return {
        "coverage_T0_40": cov40,
        "coverage_T0_160": cov160,
        "coverage_holds_as_T0_grows": float(cov160 >= cov40 - 0.02),
        "bias": bias160,
        # Hirshberg's donor-only sigma under-covers a real O(1)-noise single unit
        "hirshberg_donor_only_undercovers": float(covh160 < 0.85),
        # ... but recovers most of the coverage in his own p^{-1/2} scaling,
        # jumping well above its real-single-unit coverage
        "hirshberg_regime_recovers_coverage": float(covh_regime > 0.80
                                                    and covh_regime - covh160 > 0.30),
    }


# Path B. mlsynth's EIV interval (pre-residual-SD variance) covers near-nominal and
# rises toward 0.95 as T0 grows (Hirshberg's asymptotics); the point estimate is
# unbiased. Hirshberg's donor-only sigma_e||theta|| under-covers a real single unit
# (drops nu_e) but is ~nominal in his own scaling -- documenting mlsynth's choice.
EXPECTED = {
    "coverage_T0_40": (0.91, 0.05),
    "coverage_T0_160": (0.94, 0.05),
    "coverage_holds_as_T0_grows": (1.0, 0.0),
    "bias": (0.0, 0.03),
    "hirshberg_donor_only_undercovers": (1.0, 0.0),
    "hirshberg_regime_recovers_coverage": (1.0, 0.0),
}


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(run(), indent=2))
