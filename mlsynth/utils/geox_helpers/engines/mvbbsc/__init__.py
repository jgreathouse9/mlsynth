"""The Bayesian GEOX engine: MVBBSC in the scoring slot.

    Martinez, I. & Vives-i-Bastida, J. (2024). "Bayesian and Frequentist
    Inference for Synthetic Controls." arXiv:2206.01779.

A Dirichlet simplex on the donor weights with a HalfNormal scale, sampled by
NUTS. This is the arrangement a Bayesian geo product takes: GeoLift's design
loop with a Bayesian synthetic control doing the fitting, the estimator
occupying the slot augsynth occupies today.

One thing differs from a direct call to :class:`mlsynth.MVBBSC`, and it is a
consequence of what the scoring loop asks of an engine.

The ATT interval carries the pre-period autocorrelation. MVBBSC's own
counterfactual adds an iid shock, and the ATT averages a post window: under iid
the variance of that mean falls as ``sigma^2 / h``, under positive
autocorrelation it falls more slowly. On Meta's GeoLift panel, with every one of
its 40 locations taken as a placebo treated unit, the shipped form covers 65% of
a nominal 90% and this form covers 87.5%, against augsynth's conformal 89.7% at
1.7x the width. A per-period band looks acceptable either way; averaging is what
exposes the difference.

The null is the posterior, selected by ``inference="bayes"``. A conformal or
placebo null is refused here for the same reason the sdid engine refuses
conformal: the procedure would not be the one the interval came from.

Donor order needs no handling here. ``run_mvbbsc`` canonicalises its own
columns, so the invariance the engine property suite asserts -- permute the
donors, the weights permute with them -- is inherited, not re-imposed.

The other relations that suite asserts hold at ``fit_tolerance``, 5e-2, and the
number is measured. Rescaling or shifting a panel is exactly equivariant in
arithmetic; the model's standardization is not bit-exact in floating point, so
the arrays reaching the sampler differ by about 1e-14, and NUTS carries that
difference into the draws. Over six generated panels a rescale moved the
posterior weights by at most 5.7e-3 and a shift by 1.1e-2, against 1.2e-2 from
refitting the same panel at another seed: the transformation costs no more than
running the sampler again. 5e-2 is twice the worst of those. The frequentist
engines keep the 1e-6 default, and the donor-order relation stays there too,
since canonicalisation makes it exact.

``engine_kwargs`` carries ``n_warmup``, ``n_samples``, ``n_chains``,
``target_accept`` and ``autocorr``. The defaults cost roughly four seconds a
fit, which the scoring loop pays once per candidate, duration and backtest.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .....exceptions import MlsynthEstimationError
from ...engine import normal_p_value
from .. import Engine, EngineFit, placebo_detection_boundary, placebo_interval

N_WARMUP, N_SAMPLES, N_CHAINS, TARGET_ACCEPT = 600, 600, 2, 0.9


def _ar1(residuals: np.ndarray) -> float:
    """Lag-1 autocorrelation of the pre-period gap, clipped to a stable root."""
    r = np.asarray(residuals, dtype=float)
    r = r - r.mean()
    denom = float(r @ r)
    if denom <= 0:
        return 0.0
    return float(np.clip((r[1:] @ r[:-1]) / denom, -0.95, 0.95))


def fit_once(y, Y0, n_pre: int, start: int, end: int, n_tr: int,
             n_warmup: int = N_WARMUP, n_samples: int = N_SAMPLES,
             n_chains: int = N_CHAINS, target_accept: float = TARGET_ACCEPT,
             seed: int = 0, **_ignored: Any) -> EngineFit:
    """Sample the posterior on the pre-period and predict across the panel."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    try:
        from ....mvbbsc_helpers.model import run_mvbbsc
    except ImportError as exc:  # pragma: no cover - import shim
        raise MlsynthEstimationError(
            "the mvbbsc engine requires NumPyro (pip install 'mlsynth[bayes]')."
        ) from exc

    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)

    draws = run_mvbbsc(y, Y0, n_pre, n_warmup=n_warmup, n_samples=n_samples,
                       n_chains=n_chains, target_accept=target_accept,
                       seed=seed)

    # Rebuild the model's standardization to recover the noiseless mean. The
    # counterfactual it returns already carries an iid shock, which is the one
    # this engine replaces.
    loc = float(y[:n_pre].mean())
    scale = float(y[:n_pre].std(ddof=1)) or 1.0
    donor_loc = Y0[:n_pre].mean(axis=0)
    donor_scale = Y0[:n_pre].std(axis=0, ddof=1)
    donor_scale = np.where(donor_scale > 0, donor_scale, 1.0)
    standardized = (Y0 - donor_loc) / donor_scale
    weights = np.asarray(draws["weights"], dtype=float)
    mu = (weights @ standardized.T) * scale + loc            # (n_draws, T)
    sd = np.asarray(draws["sigma"], dtype=float) * scale
    counterfactual = mu.mean(axis=0)
    gap = y[:n_pre] - counterfactual[:n_pre]
    pre_rmspe = float(np.sqrt(np.mean(gap ** 2)))
    spread = float(np.std(y[:n_pre]))

    # Every candidate presents a different donor count, so each fit is a fresh
    # XLA trace; without this the compilation cache grows across a scoring run
    # until the process cannot allocate. The draws are already back in numpy.
    try:
        import jax

        jax.clear_caches()
    except Exception:  # pragma: no cover - backend detail
        pass

    return EngineFit(
        counterfactual=counterfactual,
        donor_weights=weights.mean(axis=0),
        pre_rmspe=pre_rmspe,
        scaled_l2=pre_rmspe / spread if spread > 0 else float("nan"),
        extras={"mu": mu, "sd": sd, "rho": _ar1(gap), "y": y, "seed": int(seed),
                "max_rhat": float(draws.get("max_rhat", float("nan"))),
                "n_divergent": int(draws.get("n_divergent", 0))},
    )


def att(fit: EngineFit, y, start: int, end: int) -> float:
    """Mean gap over the treatment window."""
    y = np.asarray(y, dtype=float).ravel()
    return float(np.mean(y[start:end + 1] - fit.counterfactual[start:end + 1]))


def att_posterior(fit: EngineFit, start: int, end: int, *,
                  autocorr: bool = True) -> np.ndarray:
    """Posterior predictive draws of the ATT over ``[start, end]``.

    With ``autocorr`` the shock is an AR(1) at the pre-period gap's own lag-1
    correlation; without it the shock is iid, which is what MVBBSC returns
    directly and what under-covers an averaged estimand.
    """
    y, mu, sd = fit.extras["y"], fit.extras["mu"], fit.extras["sd"]
    horizon = end - start + 1
    n_draws = mu.shape[0]
    rng = np.random.default_rng(fit.extras["seed"] + 11)
    if autocorr:
        rho = float(fit.extras["rho"])
        shock = np.empty((n_draws, horizon))
        shock[:, 0] = (rng.standard_normal(n_draws) * sd
                       / np.sqrt(max(1.0 - rho ** 2, 1e-6)))
        for t in range(1, horizon):
            shock[:, t] = rho * shock[:, t - 1] + rng.standard_normal(n_draws) * sd
    else:
        shock = rng.standard_normal((n_draws, horizon)) * sd[:, None]
    return np.mean(y[start:end + 1][None, :] - (mu[:, start:end + 1] + shock),
                   axis=1)


def point_inference(fit: EngineFit, y, Y0, n_pre: int, start: int, end: int, *,
                    alpha: float = 0.1, autocorr: bool = True,
                    **_ignored: Any) -> Tuple[float, Dict[str, Any]]:
    """One window's posterior tail probability and credible interval."""
    draws = att_posterior(fit, start, end, autocorr=autocorr)
    p = 2.0 * min(float(np.mean(draws <= 0.0)), float(np.mean(draws >= 0.0)))
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return min(p, 1.0), {
        "method": "bayes",
        "autocorr": bool(autocorr),
        "att": att(fit, y, start, end),
        "sigma": float(np.std(draws, ddof=1)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "rho": float(fit.extras["rho"]),
        "max_rhat": fit.extras["max_rhat"],
        "n_divergent": fit.extras["n_divergent"],
    }


def sweep_p_values(fit: EngineFit, y, Y0, n_pre: int, start: int, end: int,
                   effect_sizes: Iterable[float], *, alpha: float = 0.1,
                   autocorr: bool = True, inference: str = "bayes",
                   **_ignored: Any) -> Dict[str, Any]:
    """Test every effect size on one backtest.

    The posterior scale does not depend on the injected effect, so it is drawn
    once and reused across the grid; the fit uses pre-period data alone, so
    injecting into the post window moves the ATT by exactly
    ``effect_size * mean(treated post)`` and leaves the counterfactual alone.
    """
    y = np.asarray(y, dtype=float).ravel()
    draws = att_posterior(fit, start, end, autocorr=autocorr)
    tau0 = att(fit, y, start, end)
    sigma = float(np.std(draws, ddof=1))
    baseline = float(np.mean(y[start:end + 1]))
    taus = [tau0 + float(e) * baseline for e in effect_sizes]
    up, down = placebo_detection_boundary(tau0, baseline, sigma, alpha)
    return {"tau": taus, "p_value": [normal_p_value(t, sigma) for t in taus],
            "sigma": sigma, "tau0": tau0,
            "boundary_up": up, "boundary_down": down}


def detection_boundary(fit: EngineFit, y, Y0, n_pre: int, start: int, end: int,
                       *, alpha: float = 0.1, autocorr: bool = True,
                       **_ignored: Any):
    """Effects at which this backtest starts detecting, in each direction."""
    y = np.asarray(y, dtype=float).ravel()
    draws = att_posterior(fit, start, end, autocorr=autocorr)
    return placebo_detection_boundary(att(fit, y, start, end),
                                      float(np.mean(y[start:end + 1])),
                                      float(np.std(draws, ddof=1)), alpha)


ENGINE = Engine(name="mvbbsc", fit_once=fit_once, att=att,
                sweep_p_values=sweep_p_values, point_inference=point_inference,
                detection_boundary=detection_boundary, fit_tolerance=5e-2)

__all__ = ["ENGINE", "fit_once", "att", "att_posterior", "sweep_p_values",
           "point_inference", "detection_boundary"]
