"""Bayesian scoring engines for GEOX: BSCM and MVBBSC in the augsynth slot.

GEOX's engine seam (``mlsynth/utils/geox_helpers/engines/__init__.py``) asks for
five functions and owns nothing else -- candidate nomination, backtest windows,
effect injection, the power sweep, the MDE rule and the composite rank all sit
above it. That is what makes a Bayesian GeoLift a swap and not a rewrite, and it
is the shape Recast's product takes: GeoLift's design loop with a Bayesian
synthetic control doing the fitting.

Two estimators are wired here.

``bscm``
    Kim, Lee & Gupta (2020). Unconstrained donor weights under a horseshoe --
    no simplex at all -- drawn by a pure-numpy Gibbs sampler at ~0.28s a fit.

``mvbbsc``
    Martinez & Vives-i-Bastida (2024). A Dirichlet simplex with a HalfNormal
    scale, the closest thing in mlsynth to Recast's REBA (which adds a
    multiplier ``M`` to the same simplex). NUTS, ~4-12s a fit.

Both form the ATT interval from the posterior PREDICTIVE, and both carry the
pre-period AR(1) into the predictive shock. Neither step is optional, and
``calibration.py`` measures what each step buys:

    MVBBSC as shipped (iid shock)      65.0% coverage of a nominal 90%
    MVBBSC + the AR(1) step            87.5%
    augsynth (conformal)               89.7%

The failure the AR(1) step repairs is specific to an averaged estimand. Under
iid noise the variance of a 15-period mean falls as sigma^2/h; under positive
autocorrelation it falls more slowly. A per-period band looks fine either way,
so averaging is what exposes it.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from mlsynth.utils.bscm_helpers.sampler import gibbs_bscm
from mlsynth.utils.geox_helpers.engine import normal_p_value
from mlsynth.utils.geox_helpers.engines import (
    Engine,
    EngineFit,
    placebo_detection_boundary,
    placebo_interval,
)

BSCM_CHAINS, BSCM_ITER, BSCM_BURN = 4, 2000, 1000
MVB_WARMUP, MVB_SAMPLES, MVB_CHAINS = 600, 600, 2


def ar1(residuals: np.ndarray) -> float:
    """Lag-1 autocorrelation of the pre-period gap, clipped to a stationary root."""
    r = np.asarray(residuals, dtype=float)
    r = r - r.mean()
    denom = float(r @ r)
    if denom <= 0:
        return 0.0
    return float(np.clip((r[1:] @ r[:-1]) / denom, -0.95, 0.95))


def ar_shock(sd: np.ndarray, rho: float, horizon: int, rng) -> np.ndarray:
    """``(n_draws, horizon)`` predictive shock carrying the pre-period AR(1)."""
    d = sd.shape[0]
    e = np.empty((d, horizon))
    e[:, 0] = rng.standard_normal(d) * sd / np.sqrt(max(1.0 - rho ** 2, 1e-6))
    for t in range(1, horizon):
        e[:, t] = rho * e[:, t - 1] + rng.standard_normal(d) * sd
    return e


# --------------------------------------------------------------------------
# BSCM
# --------------------------------------------------------------------------
def bscm_fit_once(y, Y0, n_pre: int, start: int, end: int, n_tr: int,
                  prior: str = "horseshoe", seed: int = 0,
                  **_ignored: Any) -> EngineFit:
    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    draws = gibbs_bscm(y[:n_pre], Y0[:n_pre], Y0, prior=prior,
                       chains=BSCM_CHAINS, n_iter=BSCM_ITER, burn_in=BSCM_BURN,
                       rng=np.random.default_rng(seed))
    beta, beta0 = draws["beta"], draws["beta0"]
    sd = np.sqrt(np.asarray(draws["sigma2"], dtype=float))
    mu = (Y0 @ beta + beta0[None, :]).T                       # (draws, T)
    cf = mu.mean(axis=0)
    gap = y[:n_pre] - cf[:n_pre]
    pre_rmspe = float(np.sqrt(np.mean(gap ** 2)))
    scale = float(np.std(y[:n_pre]))
    return EngineFit(
        counterfactual=cf, donor_weights=beta.mean(axis=1),
        pre_rmspe=pre_rmspe,
        scaled_l2=pre_rmspe / scale if scale > 0 else float("nan"),
        extras={"mu": mu, "sd": sd, "rho": ar1(gap), "y": y, "seed": seed},
    )


# --------------------------------------------------------------------------
# MVBBSC
# --------------------------------------------------------------------------
def mvbbsc_fit_once(y, Y0, n_pre: int, start: int, end: int, n_tr: int,
                    seed: int = 0, **_ignored: Any) -> EngineFit:
    from mlsynth.utils.mvbbsc_helpers.model import run_mvbbsc

    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    res = run_mvbbsc(y, Y0, n_pre, n_warmup=MVB_WARMUP, n_samples=MVB_SAMPLES,
                     n_chains=MVB_CHAINS, target_accept=0.9, seed=seed)
    # Rebuild the model's standardization to recover the noiseless mean; the
    # counterfactual it returns already carries an iid shock, which is the one
    # this engine replaces.
    loc = float(y[:n_pre].mean())
    scale = float(y[:n_pre].std(ddof=1)) or 1.0
    mX = Y0[:n_pre].mean(axis=0)
    sX = Y0[:n_pre].std(axis=0, ddof=1)
    sX = np.where(sX > 0, sX, 1.0)
    mu = (res["weights"] @ ((Y0 - mX) / sX).T) * scale + loc   # (draws, T)
    sd = np.asarray(res["sigma"], dtype=float) * scale
    cf = mu.mean(axis=0)
    gap = y[:n_pre] - cf[:n_pre]
    pre_rmspe = float(np.sqrt(np.mean(gap ** 2)))
    ysd = float(np.std(y[:n_pre]))
    # Every candidate has a different donor count, so each fit is a fresh XLA
    # trace; without this the compilation cache grows until the process cannot
    # allocate. Results are already back in numpy.
    try:
        import jax

        jax.clear_caches()
    except Exception:  # pragma: no cover - backend detail
        pass
    return EngineFit(
        counterfactual=cf, donor_weights=res["weights"].mean(axis=0),
        pre_rmspe=pre_rmspe,
        scaled_l2=pre_rmspe / ysd if ysd > 0 else float("nan"),
        extras={"mu": mu, "sd": sd, "rho": ar1(gap), "y": y, "seed": seed,
                "max_rhat": res.get("max_rhat", float("nan")),
                "n_divergent": res.get("n_divergent", 0),
                "cf_shipped": res["counterfactual"]},
    )


# --------------------------------------------------------------------------
# shared readout
# --------------------------------------------------------------------------
def att(fit: EngineFit, y, start: int, end: int) -> float:
    y = np.asarray(y, dtype=float).ravel()
    return float(np.mean(y[start:end + 1] - fit.counterfactual[start:end + 1]))


def att_posterior(fit: EngineFit, start: int, end: int, *,
                  autocorr: bool = True) -> np.ndarray:
    """Posterior predictive draws of the ATT over ``[start, end]``."""
    y, mu, sd = fit.extras["y"], fit.extras["mu"], fit.extras["sd"]
    horizon = end - start + 1
    rng = np.random.default_rng(int(fit.extras["seed"]) + 11)
    if autocorr:
        e = ar_shock(sd, float(fit.extras["rho"]), horizon, rng)
    else:
        e = rng.standard_normal((mu.shape[0], horizon)) * sd[:, None]
    return np.mean(y[start:end + 1][None, :] - (mu[:, start:end + 1] + e), axis=1)


def point_inference(fit, y, Y0, n_pre, start, end, *, alpha: float = 0.1,
                    autocorr: bool = True, **_ignored: Any
                    ) -> Tuple[float, Dict[str, Any]]:
    d = att_posterior(fit, start, end, autocorr=autocorr)
    p = 2.0 * min(float(np.mean(d <= 0.0)), float(np.mean(d >= 0.0)))
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return min(p, 1.0), {
        "method": "posterior_predictive_ar" if autocorr else "posterior_predictive_iid",
        "att": att(fit, y, start, end), "sigma": float(np.std(d, ddof=1)),
        "ci_lower": float(lo), "ci_upper": float(hi),
        "rho": float(fit.extras["rho"]),
        "max_rhat": fit.extras.get("max_rhat", float("nan")),
    }


def sweep_p_values(fit, y, Y0, n_pre, start, end, effect_sizes: Iterable[float],
                   *, alpha: float = 0.1, **_ignored: Any) -> Dict[str, Any]:
    y = np.asarray(y, dtype=float).ravel()
    d = att_posterior(fit, start, end)
    tau0 = att(fit, y, start, end)
    sigma = float(np.std(d, ddof=1))
    baseline = float(np.mean(y[start:end + 1]))
    taus = [tau0 + float(e) * baseline for e in effect_sizes]
    up, down = placebo_detection_boundary(tau0, baseline, sigma, alpha)
    return {"tau": taus, "p_value": [normal_p_value(t, sigma) for t in taus],
            "sigma": sigma, "tau0": tau0, "boundary_up": up, "boundary_down": down}


def detection_boundary(fit, y, Y0, n_pre, start, end, *, alpha: float = 0.1,
                       **_ignored: Any):
    y = np.asarray(y, dtype=float).ravel()
    d = att_posterior(fit, start, end)
    return placebo_detection_boundary(att(fit, y, start, end),
                                      float(np.mean(y[start:end + 1])),
                                      float(np.std(d, ddof=1)), alpha)


BSCM = Engine(name="bscm", fit_once=bscm_fit_once, att=att,
              sweep_p_values=sweep_p_values, point_inference=point_inference,
              detection_boundary=detection_boundary)
MVBBSC = Engine(name="mvbbsc", fit_once=mvbbsc_fit_once, att=att,
                sweep_p_values=sweep_p_values, point_inference=point_inference,
                detection_boundary=detection_boundary)
REGISTRY = {"bscm": BSCM, "mvbbsc": MVBBSC}


def install() -> None:
    """Register the Bayesian engines everywhere GEOX resolves one."""
    import mlsynth.utils.geox_helpers.engines as eng
    import mlsynth.utils.geox_helpers.orchestration as orch
    import mlsynth.utils.geox_helpers.realize as realize
    import mlsynth.utils.geox_helpers.simulate as simulate

    base = eng.resolve_engine

    def resolve(name):
        return REGISTRY.get(name) or base(name)

    eng.resolve_engine = resolve
    simulate.resolve_engine = resolve
    orch.resolve_engine = resolve
    realize.resolve_engine = resolve
    eng.ENGINE_NAMES = frozenset(set(eng.ENGINE_NAMES) | set(REGISTRY))
