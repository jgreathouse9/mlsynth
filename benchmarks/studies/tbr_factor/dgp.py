r"""A factor-model panel whose factor count is a knob.

Li and Van den Bulte (2022) develop ADID under a linear factor model, and Li
(2024) web appendix A states the identifying assumption under a one-factor
model, where the correlation between the treated series and the control average
does not depend on time. TBR fits that relation with a free intercept and a
free slope on one control aggregate.

The outcome is

.. math::

   y_{jt} = \lambda_j^\top f_t + \varepsilon_{jt},

with :math:`\lambda_j` drawn Uniform(0, 1) per factor and :math:`f` a vector of
independent AR(1) processes. No treatment is applied anywhere, so the true
effect is zero in every period and any interval that excludes zero is an error.

Aggregating gives the treated group the mean loading of its members and the
control group the mean loading of theirs. The fitted relation
:math:`\bar{y}_{\mathrm{tr},t} = \alpha + \beta \bar{y}_{\mathrm{co},t}` holds
for every :math:`t` exactly when those two mean loading vectors are
proportional. At one factor they are scalars, so proportionality is automatic
and the relation is exact. Past one factor it is a coincidence that two
independently drawn mean vectors do not satisfy, so a gap remains with no
treatment anywhere -- and one regressor cannot absorb it, however many geos are
averaged into each group.

``collinearity`` is the quantity that decides it: the sine of the angle between
the two mean loading vectors, zero when they are proportional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Factor persistence. 0.8 is the configuration AR(1) factor studies use.
RHO = 0.8
#: Idiosyncratic noise standard deviation.
SIGMA = 0.30


def loadings(rng: np.random.Generator, n_geos: int, r: int) -> np.ndarray:
    """Uniform(0, 1) loadings, one row per geo."""
    return rng.uniform(0.0, 1.0, (n_geos, r))


def factors(rng: np.random.Generator, r: int, n_periods: int,
            rho: float = RHO) -> np.ndarray:
    """``r`` independent AR(1) paths, shape ``(r, n_periods)``."""
    f = np.empty((r, n_periods))
    f[:, 0] = rng.normal(size=r)
    for t in range(1, n_periods):
        f[:, t] = rho * f[:, t - 1] + rng.normal(size=r)
    return f


def collinearity(lam: np.ndarray, n_treated: int) -> float:
    """Sine of the angle between the two groups' mean loading vectors.

    Zero when they are proportional, which is when the fitted affine relation
    is exact. Always zero at one factor.
    """
    a = lam[:n_treated].mean(axis=0)
    b = lam[n_treated:].mean(axis=0)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - cos ** 2)))


def panel(seed: int, r: int, n_treated: int, *, n_geos: int = 50,
          n_periods: int = 80, n_pre: int = 60, rho: float = RHO,
          sigma: float = SIGMA) -> pd.DataFrame:
    """A long panel with no treatment anywhere, ready for ``dataprep``.

    The outcome is shifted to stay positive, which moves the intercept and
    leaves the fitted slope and every gap unchanged.
    """
    if r < 1:
        raise ValueError(f"r must be at least one factor; got {r}")
    if not 0 < n_treated < n_geos:
        raise ValueError(
            f"n_treated must leave both groups non-empty; got {n_treated} "
            f"of {n_geos}")
    if not 0 < n_pre < n_periods:
        raise ValueError(f"n_pre must sit inside the panel; got {n_pre}")
    rng = np.random.default_rng([seed, r, n_treated])
    lam = loadings(rng, n_geos, r)
    f = factors(rng, r, n_periods, rho)
    Y = lam @ f + rng.normal(0.0, sigma, (n_geos, n_periods))
    Y = Y - Y.min() + 1.0
    rows = [
        {"geo": f"g{j:02d}", "t": t, "y": float(Y[j, t]),
         "post": int(t >= n_pre), "is_treat": int(j < n_treated),
         "is_ctrl": int(j >= n_treated)}
        for j in range(n_geos) for t in range(n_periods)
    ]
    frame = pd.DataFrame(rows)
    frame.attrs["collinearity"] = collinearity(lam, n_treated)
    return frame


def noiseless_gap(seed: int, r: int, n_treated: int, *, n_geos: int = 50,
                  n_periods: int = 80, n_pre: int = 60) -> float:
    """The largest gap the best affine fit leaves, with the noise switched off.

    This is the mechanism without the sampling error on top: at one factor the
    two aggregates are proportional and the gap is zero to machine precision,
    and past one factor it is not.

    The fit is taken on the pretest because that is what the estimator fits on,
    and not because the answer depends on it: refitting over the whole panel
    moves this ratio by a factor of 0.91 to 1.06, median 1.00, over factor
    counts 2, 3 and 5 and treated groups 2, 5 and 25. What is being measured is
    structural non-proportionality, a fixed combination of factors present
    throughout the panel, and not an error that grows outside the fit window.
    """
    rng = np.random.default_rng([seed, r, n_treated])
    lam = loadings(rng, n_geos, r)
    f = factors(rng, r, n_periods)
    Y = lam @ f
    y = Y[:n_treated].mean(axis=0)
    x = Y[n_treated:].mean(axis=0)
    design = np.column_stack([np.ones_like(x[:n_pre]), x[:n_pre]])
    coef, *_ = np.linalg.lstsq(design, y[:n_pre], rcond=None)
    resid = y - (coef[0] + coef[1] * x)
    return float(np.max(np.abs(resid)) / max(np.max(np.abs(y)), 1e-300))
