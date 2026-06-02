"""Path-B replication of the Cao, Lu & Wu (2026) simulation study (Section 3).

This reproduces the authors' *synthetic* (Monte-Carlo) study for the **SSC**
estimator -- a **Path B** replication (reproducing a paper's simulation-section
results, as opposed to a Path A empirical-data replication).

Design (paper Section 3)
------------------------
The data are generated from a factor model

    y_{i,t} = tau_{i,t} d_{i,t} + lambda_i' f_t + alpha_i + xi_t + c + eps_{i,t},

with ``r`` AR(1) common factors and a time effect (each ``g_t = 0.5 g_{t-1} +
N(0,1)``), loadings/unit effects ``U[-sqrt(3), sqrt(3)]``, intercept ``c = 5``,
``N(0,1)`` noise, and a dynamic effect ``tau_{i,t} = 1 + max(e_{i,t}, 0)`` where
``e_{i,t}`` is event time. The authors fix ``N = 33`` units (``30`` eventually
treated, staggered across an ``S = 7`` post-window), vary the number of factors
``r in {3, 6}`` and the pre-period length ``T in {15, 42, 157}``, and run 1,000
replications. Figure 1 reports the **event-time RMSE** of each method's ATT
estimates.

This module computes SSC's event-time RMSE -- the share of Figure 1 that
concerns our estimator -- through the public :meth:`mlsynth.SSC.fit` API:

    RMSE_e = sqrt( mean over replications of ( ATT^e_hat - (1 + e) )^2 ),

since the true event-time ATT at horizon ``e`` is ``1 + e`` under this DGP.
The ``PAPER`` preset is the authors' exact configuration; ``DEMO`` is a faster,
reduced-count version that reproduces the qualitative pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .simulation import simulate_ssc_panel


@dataclass(frozen=True)
class SSCSimConfig:
    """Parameters for the SSC Monte-Carlo study."""

    n_units: int = 33
    n_never: int = 3
    S: int = 7
    n_factors: int = 3
    base_effect: float = 1.0
    intercept: float = 5.0
    T0_grid: List[int] = field(default_factory=lambda: [15, 42, 157])
    n_reps: int = 1000


# The authors' exact design (Figure 1): r in {3, 6}, T in {15, 42, 157}.
PAPER = SSCSimConfig()

# Faster, reduced-count version that reproduces the qualitative pattern.
DEMO = SSCSimConfig(n_units=20, n_never=4, S=6, n_factors=2,
                    T0_grid=[42], n_reps=20)


def run_ssc_simulation(cfg: SSCSimConfig = DEMO, *, n_factors=None,
                       seed: int = 0, verbose: bool = True) -> Dict:
    """Run the SSC Monte-Carlo and return event-time RMSE per ``(r, T0)`` cell.

    For each replication the panel is simulated from the paper's factor DGP and
    the event-time ATT is estimated via :meth:`mlsynth.SSC.fit` (point estimates
    only; inference is off for speed). The RMSE of each event-time estimate
    against the truth ``1 + e`` is accumulated across replications.

    Parameters
    ----------
    cfg : SSCSimConfig
        Study configuration (preset ``PAPER`` or ``DEMO``).
    n_factors : int, optional
        Override the factor count ``r`` (default uses ``cfg.n_factors``).
    seed : int
        Base RNG seed.
    verbose : bool
        Print a small per-cell table.

    Returns
    -------
    dict
        ``{(r, T0): {event_time e: rmse_e}}``.
    """
    from ...estimators.ssc import SSC

    r = cfg.n_factors if n_factors is None else int(n_factors)
    out: Dict = {}
    for T0 in cfg.T0_grid:
        sq = {}      # event_time -> list of squared errors
        for rep in range(cfg.n_reps):
            df = simulate_ssc_panel(
                n_units=cfg.n_units, n_never=cfg.n_never, T0=T0, S=cfg.S,
                n_factors=r, base_effect=cfg.base_effect,
                intercept=cfg.intercept, seed=seed + rep,
            )
            res = SSC({"df": df, "outcome": "Y", "treat": "treated",
                       "unitid": "unit", "time": "time",
                       "inference": False, "display_graphs": False}).fit()
            for e, att in res.event_att.items():
                truth = cfg.base_effect + e
                sq.setdefault(e, []).append((att - truth) ** 2)
        rmse = {e: float(np.sqrt(np.mean(v))) for e, v in sorted(sq.items())}
        out[(r, T0)] = rmse
        if verbose:
            print(f"\n=== SSC event-time RMSE | r={r}, T0={T0}, "
                  f"reps={cfg.n_reps} ===")
            print(f"{'event time e':>12} {'RMSE':>8} {'true ATT':>9}")
            for e in sorted(rmse):
                print(f"{e:>12} {rmse[e]:>8.3f} {cfg.base_effect + e:>9.1f}")
    return out


if __name__ == "__main__":
    run_ssc_simulation(DEMO)
