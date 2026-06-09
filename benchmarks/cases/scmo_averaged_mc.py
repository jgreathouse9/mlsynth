"""Path B benchmark: averaged multi-outcome SCM (Sun-Ben-Michael-Feller 2025).

Reproduces the regime contrast that motivates the *averaged* SCMO. Under a
common factor shared across outcomes, averaging the standardized outcomes cancels
idiosyncratic noise, so the averaged (and concatenated) multi-outcome SC have
lower post-treatment bias than the separate single-outcome SC. Under purely
idiosyncratic factors, the outcomes share no signal, so averaging *mixes
unrelated series and hurts* -- the separate SC is best. This common-vs-
idiosyncratic adaptivity is exactly the contrast between the paper's Figure D.1
(common factor) and Figure D.2(b) (idiosyncratic).

The reported statistic per regime is the average absolute post-period gap
("bias") on outcome 1 over ``M`` null (``tau = 0``) draws, for the separate
(K=1), concatenated, and averaged schemes.

Provenance
----------
* DGP: :func:`mlsynth.utils.scmo_helpers.simulation.simulate_sun` -- the
  Appendix-D factor model of Sun-Ben-Michael-Feller (2025, REStat): N = 50,
  loadings evenly spaced on [1, 5], treated unit = second-largest loading,
  ``rho`` mixes the common (``rho = 1``) and idiosyncratic (``rho = 0``) factor
  structure.
* Headline: the paper reports its Monte Carlo as box plots (Figures D.1, D.2),
  not numeric cells, so this benchmark matches the **published geometry**:
  under the common factor the multi-outcome schemes beat separate and averaging
  helps; under the idiosyncratic factor averaging hurts. The pinned bias levels
  are mlsynth regression guards (M = 150, seed = 7; the paper used 1,000 draws).
"""
from __future__ import annotations

import warnings

import numpy as np

M = 150
SEED = 7
SCHEMES = ("separate", "concatenated", "averaged")
# (label, T0, K, rho)
REGIMES = (
    ("common_T10", 10, 10, 1.0),
    ("common_T40", 40, 10, 1.0),
    ("idio_T10", 10, 10, 0.0),
)


def _bias(T0: int, K: int, rho: float) -> dict:
    from mlsynth import SCMO
    from mlsynth.utils.scmo_helpers.simulation import simulate_sun, to_panel

    rng = np.random.default_rng(SEED)
    acc = {s: [] for s in SCHEMES}
    addout = [f"y{k}" for k in range(1, K)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(M):
            Ys, N, TT, tr = simulate_sun(rng, T0, K, rho=rho)
            df = to_panel(Ys, N, TT, tr)
            for s in SCHEMES:
                fit = SCMO({"df": df, "outcome": "y0", "treat": "treat",
                            "unitid": "unit", "time": "time", "schemes": [s],
                            "addout": addout, "display_graphs": False}).fit()._primary
                acc[s].append(abs(float(np.asarray(fit.gap)[-1])))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def run() -> dict:
    b = {label: _bias(T0, K, rho) for label, T0, K, rho in REGIMES}
    res: dict = {}
    for label in b:
        for s in SCHEMES:
            res[f"bias_{label}_{s}"] = b[label][s]
    # Headline geometry.
    res["multi_helps_common"] = float(
        b["common_T10"]["concatenated"] < b["common_T10"]["separate"]
        and b["common_T40"]["concatenated"] < b["common_T40"]["separate"])
    res["averaging_helps_common"] = float(
        b["common_T10"]["averaged"] < b["common_T10"]["separate"]
        and b["common_T40"]["averaged"] < b["common_T40"]["separate"])
    res["averaging_hurts_idio"] = float(
        b["idio_T10"]["averaged"] > b["idio_T10"]["separate"])
    return res


# Stochastic (M=150 vs the paper's 1,000). Bias SE ~ 0.7/sqrt(150) ~ 0.06, so the
# pinned bias levels carry +-0.10; the geometry booleans are exact.
EXPECTED = {
    "bias_common_T10_separate": (1.025, 0.10),
    "bias_common_T10_concatenated": (0.860, 0.10),
    "bias_common_T10_averaged": (0.906, 0.10),
    "bias_common_T40_separate": (0.913, 0.10),
    "bias_common_T40_concatenated": (0.824, 0.10),
    "bias_common_T40_averaged": (0.850, 0.10),
    "bias_idio_T10_separate": (0.997, 0.10),
    "bias_idio_T10_concatenated": (1.046, 0.10),
    "bias_idio_T10_averaged": (1.232, 0.10),
    "multi_helps_common": (1.0, 0.0),
    "averaging_helps_common": (1.0, 0.0),
    "averaging_hurts_idio": (1.0, 0.0),
}
