"""Monte Carlo harness validating SHC against the Chen-Yang-Yang (2024) DGP.

Runs :class:`mlsynth.SHC` on repeated draws of :func:`simulate_shc_panel`
and reports the paper's two performance measures:

* ``MSE_pre``  -- Eq. 31: mean squared matching error of the SHC
  reconstruction against the latent over the treated block's pre-segment.
* ``MSE_post(k)`` -- Eq. 38: mean squared prediction error of the SHC
  counterfactual against the *true* latent over the first ``k``
  post-intervention periods, for ``k`` in ``k_grid``.

The headline finding to reproduce: both measures are small, and
``MSE_post(k)`` grows with ``k`` (consistent with the bias bound of
Proposition 2 increasing in the horizon).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from .simulation import simulate_shc_panel


def _fit_one(df, m: int, use_augmented: bool) -> np.ndarray:
    """Fit SHC on one panel and return its counterfactual over the m+n window."""
    from mlsynth import SHC

    res = SHC({
        "df": df, "outcome": "y", "treat": "treated",
        "unitid": "unit", "time": "time", "m": m,
        "use_augmented": use_augmented, "display_graphs": False,
    }).fit()
    return np.asarray(res.counterfactual, dtype=float).ravel()


def monte_carlo_shc(
    *,
    n_reps: int = 50,
    m: int = 25,
    h: int = 4,
    n: int = 25,
    P: float = 10.0,
    sigma: float = 0.1,
    w_f: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    regular: bool = True,
    use_augmented: bool = False,
    k_grid: Sequence[int] = (1, 5, 10, 15, 25),
    seed: int = 0,
) -> Dict[str, Any]:
    """Replicate the SHC simulation study and aggregate over ``n_reps`` draws.

    Returns
    -------
    dict
        ``mse_pre`` (float, averaged over reps), ``mse_post`` (mapping
        ``k -> averaged MSE_post(k)``), ``n_reps`` (completed reps), and
        the configuration echoed back.
    """
    k_grid = [k for k in k_grid if k <= n]
    pre_errors = []
    post_errors = {k: [] for k in k_grid}

    completed = 0
    for r in range(n_reps):
        df, info = simulate_shc_panel(
            m=m, h=h, n=n, P=P, sigma=sigma, w_f=w_f,
            regular=regular, seed=seed + r,
        )
        cf = _fit_one(df, m=m, use_augmented=use_augmented)
        if cf.size != m + n:
            continue
        cf_pre, cf_post = cf[:m], cf[m:m + n]

        latent_pre = info["latent_pre_block"]
        latent_post = info["latent_post"]

        pre_errors.append(float(np.mean((cf_pre - latent_pre) ** 2)))
        for k in k_grid:
            post_errors[k].append(
                float(np.mean((cf_post[:k] - latent_post[:k]) ** 2))
            )
        completed += 1

    return {
        "mse_pre": float(np.mean(pre_errors)) if pre_errors else float("nan"),
        "mse_post": {k: (float(np.mean(v)) if v else float("nan"))
                     for k, v in post_errors.items()},
        "n_reps": completed,
        "config": {
            "m": m, "h": h, "n": n, "P": P, "sigma": sigma,
            "w_f": tuple(w_f), "regular": regular,
            "use_augmented": use_augmented,
        },
    }
