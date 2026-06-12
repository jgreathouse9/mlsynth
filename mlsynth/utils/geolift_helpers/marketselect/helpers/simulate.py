"""One pseudo-experiment simulation for GeoLift market-selection scoring.

The ``pvalueCalc`` equivalent: for a single lookback placement, fit the augsynth
model **once** (CV-ing the ridge penalty once), then sweep the effect sizes,
injecting each onto the post block and computing a conformal p-value at the
*fixed* penalty -- exactly augsynth's behaviour, minus GeoLift's redundant
re-cross-validation. The fit-once point estimates (``scaled_l2``, ``pre_rmspe``)
are identical across effect sizes; only the injected effect (hence the p-value
and ATT) varies.
"""

from typing import List, Optional

import numpy as np

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue

from .windows import lookback_pre_periods, lookback_treatment_window
from .fit import fit_augsynth_once


def inject_effect(treated, start: int, end: int, es: float) -> np.ndarray:
    """Scale the post block ``[start, end]`` of ``treated`` by ``(1 + es)``.

    GeoLift's multiplicative injection (``Y[D==1] *= 1 + es``). Returns a new
    array; the input is never mutated.

    Raises
    ------
    MlsynthConfigError
        If ``[start, end]`` is not a valid in-bounds, non-empty window.
    """
    t = np.asarray(treated, dtype=float).ravel()
    n = t.shape[0]
    if not (0 <= start <= end < n):
        raise MlsynthConfigError(
            f"invalid post window [{start}, {end}] for a series of length {n}."
        )
    out = t.copy()
    out[start : end + 1] = out[start : end + 1] * (1.0 + float(es))
    return out


def simulate_lookback(
    treated,
    donors,
    n_periods: int,
    duration: int,
    sim: int,
    effect_sizes,
    *,
    augment: Optional[str] = "ridge",
    q: float = 1.0,
    ns: int = 1000,
    seed: int = 0,
) -> List[dict]:
    """Simulate one lookback placement across a grid of effect sizes.

    Fits the SCM once on the placement's pre-period, then for each effect size
    injects it on the post block and computes the conformal p-value at the
    fixed (CV-once) ridge penalty.

    Parameters
    ----------
    treated : array-like, shape (n_periods,)
        Aggregated treated series over the full panel.
    donors : array-like, shape (n_periods, J)
        Donor pool over the full panel.
    n_periods, duration, sim : int
        Panel length, pseudo-treatment duration, and lookback placement index.
    effect_sizes : iterable of float
        Effect sizes to sweep.
    augment : {"ridge", None}, default "ridge"
        Point-fit estimator (see :func:`fit_augsynth_once`).
    q, ns, seed
        Conformal-inference settings (augsynth defaults ``1``, ``1000``, ``0``).

    Returns
    -------
    list of dict
        One row per effect size with ``sim``, ``duration``, ``effect_size``,
        ``p_value``, ``att``, ``scaled_l2``, ``pre_rmspe``.

    Raises
    ------
    MlsynthConfigError
        If the placement runs off the start of the panel, or ``treated`` /
        ``donors`` do not have ``n_periods`` rows.
    """
    n_pre = lookback_pre_periods(n_periods, duration, sim)
    start, end = lookback_treatment_window(n_periods, duration, sim)

    treated_arr = np.asarray(treated, dtype=float).ravel()
    donors_arr = np.asarray(donors, dtype=float)
    if treated_arr.shape[0] != n_periods or donors_arr.shape[0] != n_periods:
        raise MlsynthConfigError(
            f"treated and donors must both have n_periods={n_periods} rows; got "
            f"{treated_arr.shape[0]} and {donors_arr.shape[0]}."
        )

    # Fit once: CV the penalty here; reuse it for every conformal refit below.
    fit = fit_augsynth_once(treated_arr[:n_pre], donors_arr[:n_pre], augment=augment)
    counterfactual = fit.predict(donors_arr)

    rows: List[dict] = []
    for es in effect_sizes:
        treated_injected = inject_effect(treated_arr, start, end, es)
        p_value = conformal_pvalue(
            treated_injected, donors_arr, n_pre,
            lambda_=fit.lambda_, q=q, ns=ns, seed=seed,
        )
        att = float(np.mean((treated_injected - counterfactual)[start : end + 1]))
        rows.append(
            {
                "sim": sim,
                "duration": duration,
                "effect_size": float(es),
                "p_value": float(p_value),
                "att": att,
                "scaled_l2": fit.scaled_l2,
                "pre_rmspe": fit.pre_rmspe,
            }
        )
    return rows
