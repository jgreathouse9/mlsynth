"""Data-generating processes for partially-pooled SCM, after Ben-Michael, Feller &
Rothstein (2022) Section 6.

Three designs, all under a sharp null so the true ATT -- and the true cumulative
effect -- are exactly zero, which is what makes them usable as a coverage oracle: any
interval that excludes zero has missed.

The designs are chosen to bracket the estimator rather than to flatter it:

* ``"twfe"`` -- two-way fixed effects, the case partially-pooled SCM is unbiased for
  (their Theorem 2), so it is the design where inference should be near nominal;
* ``"factor"`` -- a linear factor model, where the synthetic control cannot fit
  exactly and the shared structure is what makes the wild bootstrap conservative;
* ``"ar"`` -- an autoregressive process with heterogeneous coefficients, the least
  favourable of the three.

Treatment is not assigned at random. Units are swept through the offered adoption
times and adopt with probability ``expit(theta0 + theta1 * u_i)``, where ``u_i`` is
the unit's own level (and, for the factor design, its loadings). That is the paper's
"key component": donors differ systematically from the treated, which is the whole
reason a synthetic control is needed.

References
----------
Ben-Michael, E., Feller, A., & Rothstein, J. (2022). Synthetic controls with staggered
adoption. Journal of the Royal Statistical Society: Series B, 84(2), 351-381.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError

#: Designs :func:`simulate_bfr_panel` understands.
DESIGNS = ("twfe", "factor", "ar")


@dataclass(frozen=True)
class BFRPanel:
    """One simulated panel under a sharp null.

    Attributes
    ----------
    Y : np.ndarray
        ``(n_units, n_periods)`` observed outcomes. Under the sharp null this is also
        the untreated potential outcome, so ``Y`` and ``Y_untreated`` are equal by
        construction -- kept as separate names so a benchmark reads as what it means.
    Y_untreated : np.ndarray
        The untreated potential outcome.
    trt : np.ndarray
        ``(n_units,)`` adoption times; ``+inf`` for never-treated, the convention
        :func:`~mlsynth.utils.ppscm_helpers.engine.run_multisynth` expects.
    true_att, true_cumulative : float
        Both ``0.0``: the null is sharp.
    design : str
        Which design generated the panel.
    """

    Y: np.ndarray
    Y_untreated: np.ndarray
    trt: np.ndarray
    true_att: float
    true_cumulative: float
    design: str


def _expit(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise MlsynthConfigError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def _assign_adoption(
    score: np.ndarray, adoption_times: Sequence[int], theta0: float, theta1: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sweep the adoption times, treating still-untreated units with prob expit(.).

    Treatment absorbs: a unit that adopts is not offered a later time.
    """
    n = score.size
    trt = np.full(n, np.inf)
    p = _expit(theta0 + theta1 * score)
    for g in adoption_times:
        free = ~np.isfinite(trt)
        if not free.any():
            break
        draw = rng.random(n) < p
        trt[free & draw] = float(g)
    return trt


def simulate_bfr_panel(
    *,
    design: str = "twfe",
    n_units: int = 49,
    n_periods: int = 39,
    n_factors: int = 2,
    adoption_times: Optional[Sequence[int]] = None,
    theta0: float = -2.7,
    theta1: float = -1.0,
    sigma_unit: float = 1.0,
    sigma_eps: float = 0.5,
    sigma_time: float = 0.5,
    ar_coefs: Tuple[float, float, float] = (0.6, 0.2, 0.1),
    ar_coef_sd: float = 0.08,
    seed: int = 0,
) -> BFRPanel:
    """Simulate one panel under a sharp null.

    Parameters
    ----------
    design : {"twfe", "factor", "ar"}
        Which data-generating process to use.
    n_units, n_periods : int
        Panel dimensions (the paper's application shape is 49 x 39).
    n_factors : int
        Number of latent factors; used by ``design="factor"``.
    adoption_times : sequence of int, optional
        Times units may adopt at, swept in order. Defaults to a spread across the
        middle of the panel, leaving pre-periods before the first and post-periods
        after the last.
    theta0, theta1 : float
        Selection intercept and slope on the unit's level. ``theta1 = 0`` gives
        random assignment, which is useful as a control in tests.
    sigma_unit, sigma_eps, sigma_time : float
        Standard deviations of the unit effects, idiosyncratic errors and time
        effects. In the paper these come from fitting the model to the application
        panel; here they are inputs so a caller can calibrate them from real data
        rather than inherit magic numbers.
    ar_coefs, ar_coef_sd : tuple of float, float
        Mean AR(3) coefficients and the spread across units. The paper simulates at
        eight times the fitted spread to stress the estimator.
    seed : int
        Seed; the same seed reproduces the panel exactly.

    Raises
    ------
    MlsynthConfigError
        On an unknown design, a degenerate dimension, or an adoption time that does
        not fall inside the panel.
    """
    if design not in DESIGNS:
        raise MlsynthConfigError(
            f"design must be one of {DESIGNS}; got {design!r}."
        )
    n_units = _positive_int(n_units, "n_units")
    n_periods = _positive_int(n_periods, "n_periods")
    n_factors = _positive_int(n_factors, "n_factors")

    if adoption_times is None:
        lo, hi = max(1, n_periods // 3), max(2, (2 * n_periods) // 3)
        adoption_times = tuple(int(t) for t in np.unique(np.linspace(lo, hi, 4).astype(int)))
    adoption_times = tuple(int(t) for t in adoption_times)
    for g in adoption_times:
        if not 1 <= g < n_periods:
            raise MlsynthConfigError(
                f"adoption time {g} does not fall inside a panel of {n_periods} "
                "periods; it must leave at least one pre-period and one post-period."
            )

    rng = np.random.default_rng(seed)
    unit = rng.normal(scale=sigma_unit, size=n_units)
    time = rng.normal(scale=sigma_time, size=n_periods)
    eps = rng.normal(scale=sigma_eps, size=(n_units, n_periods))

    if design == "twfe":
        Y = unit[:, None] + time[None, :] + eps
        score = unit

    elif design == "factor":
        loadings = rng.normal(size=(n_units, n_factors))
        factors = rng.normal(size=(n_factors, n_periods))
        Y = unit[:, None] + time[None, :] + loadings @ factors + eps
        score = unit + loadings.sum(axis=1)

    else:  # "ar"
        rho = np.asarray(ar_coefs, dtype=float)[None, :] + rng.normal(
            scale=ar_coef_sd, size=(n_units, 3))
        Y = np.zeros((n_units, n_periods))
        Y[:, :3] = unit[:, None] + eps[:, :3]
        for t in range(3, n_periods):
            Y[:, t] = (rho[:, 0] * Y[:, t - 1] + rho[:, 1] * Y[:, t - 2]
                       + rho[:, 2] * Y[:, t - 3] + eps[:, t])
        score = Y[:, :3].mean(axis=1)

    trt = _assign_adoption(score, adoption_times, theta0, theta1, rng)

    # Sharp null: nothing is added to the treated units, so the observed panel is the
    # untreated potential outcome and every true effect is exactly zero.
    return BFRPanel(
        Y=Y, Y_untreated=Y, trt=trt,
        true_att=0.0, true_cumulative=0.0, design=design,
    )
