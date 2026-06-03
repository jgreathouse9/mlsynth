"""Replications of Agarwal, Choi & Yuan (2026).

* **Path B** -- the paper's synthetic Monte-Carlo study (Section 5.1): the
  four-component DGP under the block-missing (MNAR) pattern. RMSI recovers the
  missing block at a lower error than the no-side-information baseline.
* **Path A** -- the empirical Proposition 99 / tobacco application (Section 5.2),
  run on the data shipped in ``basedata/P99data.csv``.

Both run through the public estimator surface / the documented helper functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .core import algorithm3

PROP99_URL = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
              "main/basedata/P99data.csv")


# ---------------------------------------------------------------------------
# Path B: the paper's synthetic four-component DGP (Section 5.1)
# ---------------------------------------------------------------------------

def _poly_features(C: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """K nonlinear functions ``f_k(c) = b0 + sum_d sum_{j=1}^4 b_{d,j} c_d^j``."""
    C = np.asarray(C, dtype=float)
    n, d = C.shape
    powers = np.concatenate([C ** j for j in range(1, 5)], axis=1)   # n x 4d
    B = rng.standard_normal((4 * d, K))
    return rng.standard_normal((1, K)) + powers @ B                  # n x K


def simulate_rmsi_dgp(N: int, T: int, *, alphas=(0.25, 0.25, 0.25, 0.25),
                      sigma: float = 0.5, seed: int = 0):
    """Generate ``(Y, M, X, Z)`` from the paper's four-component DGP (Section 5.1).

    Eight characteristics (four per margin) with the paper's distributions;
    ``M = sum_r alpha_r M_r`` with each ``M_r`` normalised to
    ``||M_r||_F = sqrt(2 N T)``; noise ``N(0, sigma^2)``.
    """
    rng = np.random.default_rng(seed)

    def chars(n):
        return np.column_stack([
            rng.uniform(-1, 1, n), rng.uniform(-0.5, 0.5, n),
            rng.normal(0, 0.2, n), rng.normal(0, 0.3, n)])

    X, Z = chars(N), chars(T)

    def norm(M):
        return M * (np.sqrt(2.0 * N * T) / np.linalg.norm(M))

    M1 = norm(_poly_features(X, 17, rng) @ _poly_features(Z, 17, rng).T)
    G2 = _poly_features(X, 3, rng)
    V1 = rng.normal(0, 1, (T, 3)) * np.sqrt([0.5, 1.0, 1.5])
    M2 = norm(G2 @ V1.T)
    Q2 = _poly_features(Z, 3, rng)
    W1 = rng.normal(0, 1, (N, 3)) * np.sqrt([0.5, 1.0, 1.5])
    M3 = norm(W1 @ Q2.T)
    W2 = rng.normal(0, 1, (N, 3)) * np.sqrt([0.5, 1.0, 1.5])
    V2 = rng.normal(0, 1.5, (T, 3))
    M4 = norm(W2 @ V2.T)

    a = alphas
    M = a[0] * M1 + a[1] * M2 + a[2] * M3 + a[3] * M4
    Y = M + rng.normal(0, sigma, (N, T))
    return Y, M, X, Z


@dataclass(frozen=True)
class RMSISimConfig:
    """Parameters for the RMSI synthetic Monte-Carlo (paper Section 5.1, MNAR)."""

    N: int = 400
    T: int = 400
    N0: int = 200          # number of control rows (wide block)
    T0: int = 200          # number of pre-treatment columns (tall block)
    J: int = 5
    sigma: float = 0.5
    alphas: tuple = (0.25, 0.25, 0.25, 0.25)
    n_reps: int = 100


PAPER = RMSISimConfig()
DEMO = RMSISimConfig(N=120, T=120, N0=60, T0=60, J=5, n_reps=10)


def run_rmsi_simulation(cfg: RMSISimConfig = DEMO, *, seed: int = 0,
                        verbose: bool = True) -> Dict:
    """Run the synthetic MNAR Monte-Carlo and return missing-block AMSE.

    Compares RMSI (with side information) to a no-side-information baseline
    (the same block estimator with empty covariates, i.e. a de-meaned low-rank
    completion) by the average mean squared error over the missing block.

    Returns
    -------
    dict
        ``{"rmsi": amse_side_info, "no_side_info": amse_baseline,
           "rel_improvement": ...}``.
    """
    rng = np.random.default_rng(seed)
    miss_rows = np.arange(cfg.N0, cfg.N)
    miss_cols = np.arange(cfg.T0, cfg.T)
    control = np.arange(cfg.N0)
    a_si, a_no = [], []
    Xz, Zz = np.zeros((cfg.N, 0)), np.zeros((cfg.T, 0))
    for rep in range(cfg.n_reps):
        Y, M, X, Z = simulate_rmsi_dgp(cfg.N, cfg.T, alphas=cfg.alphas,
                                       sigma=cfg.sigma, seed=int(rng.integers(1 << 31)))
        truth = M[np.ix_(miss_rows, miss_cols)]
        Msi, _ = algorithm3(Y, X, Z, control_idx=control, T0=cfg.T0, J=cfg.J)
        Mno, _ = algorithm3(Y, Xz, Zz, control_idx=control, T0=cfg.T0, J=cfg.J)
        a_si.append(np.mean((Msi[np.ix_(miss_rows, miss_cols)] - truth) ** 2))
        a_no.append(np.mean((Mno[np.ix_(miss_rows, miss_cols)] - truth) ** 2))
    si, no = float(np.mean(a_si)), float(np.mean(a_no))
    out = {"rmsi": si, "no_side_info": no, "rel_improvement": (no - si) / no}
    if verbose:
        print(f"RMSI synthetic MNAR (N={cfg.N}, T={cfg.T}, reps={cfg.n_reps}): "
              f"AMSE side-info={si:.4f}  no-side-info={no:.4f}  "
              f"improvement={out['rel_improvement']:+.1%}")
    return out


# ---------------------------------------------------------------------------
# Path A: the Proposition 99 / tobacco application (Section 5.2)
# ---------------------------------------------------------------------------

def replicate_prop99(data: Union[str, pd.DataFrame, None] = None, *,
                     rank: Optional[int] = 3, sieve_order: int = 2,
                     verbose: bool = True):
    """Estimate California's Proposition 99 effect with RMSI (Path A).

    Loads the Abadie et al. (2010) tobacco panel (``basedata/P99data.csv``),
    treats California from 1989, uses the state-level Abadie predictors
    (``lnincome``, ``beer``, ``age15to24``, ``retprice``) as unit covariates and
    the year-average retail price as the time covariate, and runs
    :class:`mlsynth.RMSI`.

    Parameters
    ----------
    data : str or pandas.DataFrame, optional
        Path/URL to the panel, or a DataFrame. Default downloads
        ``P99data.csv`` from ``basedata/``.
    rank : int, optional
        Factor rank (default 3; suits the single-treated-unit case -- the
        eigenvalue-ratio default tends to pick rank 1 here).
    sieve_order : int
        Polynomial sieve order.

    Returns
    -------
    mlsynth.utils.rmsi_helpers.structures.RMSIResults
    """
    from ...estimators.rmsi import RMSI

    if data is None:
        data = PROP99_URL
    d = pd.read_csv(data) if isinstance(data, str) else data.copy()
    covs = ["lnincome", "beer", "age15to24", "retprice"]
    for c in covs:
        d[c] = d.groupby("state")[c].transform(lambda s: s.fillna(s.mean()))
        d[c] = d[c].fillna(d[c].mean())
    d["treated"] = ((d["state"] == "California") & (d["year"] >= 1989)).astype(int)

    res = RMSI({"df": d, "outcome": "cigsale", "treat": "treated",
                "unitid": "state", "time": "year",
                "unit_covariates": covs, "time_covariates": ["retprice"],
                "sieve_order": sieve_order, "rank": rank,
                "display_graphs": False}).fit()
    if verbose:
        print(f"California Proposition 99 ATT (ADH ~ -19 to -20): "
              f"{res.att:+.2f} packs/capita  [rank={res.rank}, "
              f"1989={res.att_by_period[1989]:+.1f}, "
              f"2000={res.att_by_period[2000]:+.1f}]")
    return res


if __name__ == "__main__":  # pragma: no cover - manual CLI entry point
    run_rmsi_simulation(DEMO)
    replicate_prop99("../../../basedata/P99data.csv")
