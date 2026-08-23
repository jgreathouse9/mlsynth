"""A synthetic weekly geo-experiment panel.

Marketing geo tests share a recognisable shape: a few hundred metro markets
observed weekly for two or three years, one dominant national factor carrying
seasonality and demand, market sizes spread over two orders of magnitude, and a
small persistent remainder once that factor is removed. The panel below is drawn
to behave that way. Nothing in it is measured: every constant is a design
parameter, markets are integers, weeks are integers, and no observation, label,
date or magnitude from any real panel appears here or is recoverable from it.

The parameters and what each controls:

============  ======  ====================================================
constant      value   what it sets
============  ======  ====================================================
LEVEL_SD      1.2     spread of market log-levels; markets differ in size
                      by roughly two orders of magnitude end to end
FACTOR_AR     0.40    week-to-week persistence of the national factor
FACTOR_SEASON 52      the annual cycle laid over it
SEASON_AMP    0.6     how much of the factor the cycle accounts for
SNR           3.7     factor sd over idiosyncratic sd, typical market
IDIO_SD       ~0.06   idiosyncratic sd in logs, with a decile spread
IDIO_AR       ~0.20   idiosyncratic persistence, with a decile spread
BLOCK_RHO     0.09    correlation shared inside a regional block
BLOCK_SIZES           an uneven partition of the markets into regions
============  ======  ====================================================

The panel is returned in logs, which is the scale a multiplicative outcome with
this much size dispersion is modelled on.
"""
import numpy as np

N_MARKETS = 210
N_WEEKS = 130
BLOCK_SIZES = (10, 14, 16, 18, 20, 22, 24, 26, 28, 32)   # sums to 210
LEVEL_SD = 1.2
FACTOR_AR = 0.40
FACTOR_SEASON = 52
SEASON_AMP = 0.6
SNR = 3.7
IDIO_SD = (0.04, 0.06, 0.09)      # p10, median, p90
IDIO_AR = (0.02, 0.20, 0.42)      # p10, median, p90
BLOCK_RHO = 0.09

# Share of US TV homes held by each of the thirty largest designated market
# areas, largest first, from published Nielsen market data. Public geography,
# carrying no market name, code or coordinate -- only the size profile, which is
# what a geo test's cross-section is shaped by. The thirty together hold 53.8 per
# cent of US TV homes and span 7.4x from largest to smallest, a far tighter
# spread than the full two hundred, because a geo test runs on large markets.
TOP30_POPULATION_SHARE = (
    6.468, 4.917, 3.052, 2.583, 2.267, 2.191, 2.073, 2.066, 2.038, 1.941,
    1.617, 1.593, 1.587, 1.582, 1.513, 1.420, 1.372, 1.301, 1.273, 1.215,
    1.089, 1.035, 1.021, 1.008, 0.995, 0.954, 0.950, 0.942, 0.889, 0.873,
)


def _spread(rng, n, p10, med, p90):
    """n positive draws with roughly the given decile spread."""
    sigma = (np.log(p90) - np.log(p10)) / (2 * 1.2816)
    return np.exp(rng.normal(np.log(med), sigma, size=n))


def make_panel(rng, n_markets=N_MARKETS, n_weeks=N_WEEKS, size_profile=None):
    """``(n_weeks, n_markets)`` log-outcome panel, and each market's block index.

    ``size_profile`` gives the relative size of each market, largest first --
    :data:`TOP30_POPULATION_SHARE` for a real one. Given it, market log-levels are
    its logs and ``n_markets`` comes from its length; otherwise levels are drawn
    with spread :data:`LEVEL_SD`.
    """
    if size_profile is not None:
        profile = np.asarray(size_profile, dtype=float)
        if profile.ndim != 1 or profile.size < 1 or not np.all(profile > 0):
            raise ValueError(
                "size_profile must be a non-empty 1-D sequence of positive sizes; "
                f"got {size_profile!r}.")
        n_markets = profile.size
    if int(n_markets) < 1:
        raise ValueError(f"n_markets must be at least 1; got {n_markets!r}.")
    if int(n_weeks) < 2:
        raise ValueError(
            f"n_weeks must be at least 2 for a series to have a lag; got {n_weeks!r}.")
    n_markets, n_weeks = int(n_markets), int(n_weeks)
    sizes = np.array(BLOCK_SIZES, dtype=float)
    sizes = np.round(sizes / sizes.sum() * n_markets).astype(int)
    sizes = sizes[sizes > 0] if (sizes > 0).any() else np.array([n_markets])
    sizes[-1] += n_markets - sizes.sum()
    if sizes[-1] < 1:                       # rounding overshot; fold the tail in
        sizes = np.array([n_markets]) if sizes.size == 1 else np.append(
            sizes[:-2], sizes[-2] + sizes[-1])
    block = np.repeat(np.arange(len(sizes)), sizes)

    f = np.zeros(n_weeks)
    e = rng.normal(0.0, 1.0, size=n_weeks)
    for t in range(1, n_weeks):
        f[t] = FACTOR_AR * f[t - 1] + e[t]
    f = f / f.std() + SEASON_AMP * np.sin(2 * np.pi * np.arange(n_weeks) / FACTOR_SEASON)
    f = (f - f.mean()) / f.std()

    idio_sd = _spread(rng, n_markets, *IDIO_SD)
    idio_ar = np.clip(rng.normal(IDIO_AR[1], (IDIO_AR[2] - IDIO_AR[0]) / 2.56,
                                 size=n_markets), -0.6, 0.9)
    loadings = SNR * idio_sd * rng.normal(1.0, 0.25, size=n_markets)

    block_shock = rng.normal(0.0, 1.0, size=(n_weeks, len(sizes)))
    own = rng.normal(0.0, 1.0, size=(n_weeks, n_markets))
    innov = (np.sqrt(BLOCK_RHO) * block_shock[:, block]
             + np.sqrt(1.0 - BLOCK_RHO) * own)
    eps = np.zeros((n_weeks, n_markets))
    eps[0] = innov[0]
    for t in range(1, n_weeks):
        eps[t] = idio_ar * eps[t - 1] + innov[t]
    eps = eps / eps.std(axis=0, keepdims=True) * idio_sd

    if size_profile is None:
        level = rng.normal(0.0, LEVEL_SD, size=n_markets)
    else:
        level = np.log(profile / profile.mean())
    # eps and the factor term are each centred over the weeks, so the panel's
    # market means are the levels exactly -- a profile is reproduced, not blurred.
    series = np.outer(f, loadings) + eps
    return level + series - series.mean(axis=0, keepdims=True), block
