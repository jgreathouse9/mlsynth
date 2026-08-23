r"""Whether PPSCM's cumulative conformal band delivers the level it claims.

A geo lift test is the setting a cumulative band is built for: a few dozen metro
markets observed weekly, a campaign switched on in some of them at a known week,
and one question -- how much total sales did it buy, and how sure are we. PPSCM
answers the second half with a per-unit conformal band on the cumulative effect
(:func:`mlsynth.utils.ppscm_helpers.inference.cumulative_conformal_per_unit`),
calibrated on non-overlapping pre-period windows.

This case measures whether that band covers at the rate it advertises, and it has
an analytic target to measure against, so it needs no reference implementation.

Two quantities are exactly predictable before anything is fitted.

The window count. Origins start at ``max(10, min_train_frac * T0)`` and step by
the horizon, so

.. code-block:: text

    m = len(range(max(10, int(T0 * frac)), T0 - H + 1, H))

At ``T0 = 120`` weeks that is ``m = 10`` windows for an eight-week campaign and
``m = 21`` for a four-week one -- a closed form in the design, carrying no
Monte Carlo error.

The coverage. :func:`mlsynth.utils.conformal.split_conformal_quantile` takes the
``ceil((m + 1)(1 - alpha))``-th of ``m`` absolute scores, so under exchangeability
the band covers at exactly that rank over ``m + 1``: 10/11 = 0.9091 for a 90% band
at ``m = 10``, and 21/22 = 0.9545 for a 95% band at ``m = 21``. When the rank
exceeds ``m`` the order statistic does not exist and the band is infinite, which
is why ten windows cannot buy a 95% band at any width.

So the case pins three things: the window counts, which levels come back finite,
and the distance between realised coverage and the exchangeable prediction, whose
target is zero. The last is the real assertion -- it says the scores PPSCM builds
by refitting at rolling origins behave like an exchangeable sample, which is the
claim that could fail and the one no closed form can settle.

The panel is synthetic and generated here. Its cross-section is shaped by the
published share of US TV homes held by each of the thirty largest market areas
(:data:`benchmarks.studies.cumulative_calibration.synthetic_geo_panel.TOP30_POPULATION_SHARE`),
so market sizes span the 7.4x range a real top-thirty geo test does; its time
series carries one dominant national factor with trend and seasonality, mildly
persistent idiosyncratic errors, and a small correlation shared inside a region.
The generator holds design parameters only -- no observation, label, date or
magnitude from any real sales panel is in it.

Consequences for a user planning a geo test: at two and a third years of weekly
history, a four-week campaign supports a 95% band and an eight-week campaign does
not, and no amount of extra markets changes that -- the binding quantity is
non-overlapping windows in the pre-period, which only more history supplies.
"""
from __future__ import annotations

import math

_T0 = 120                  # campaign starts at week 120
_N_TREATED = 10            # ten of the thirty markets get the campaign
_MIN_TRAIN_FRAC = 0.3
_NU = 0.5                  # partial pooling held fixed, so the case measures the band
_ARMS = ((8, 60, (0.05, 0.10)), (4, 40, (0.05, 0.10)))   # horizon, reps, levels


def _windows(t0: int, horizon: int, frac: float = _MIN_TRAIN_FRAC) -> int:
    """Non-overlapping calibration windows the design yields. A closed form."""
    return len(range(max(10, int(t0 * frac)), t0 - horizon + 1, horizon))


def _predicted(m: int, alpha: float) -> float:
    """Coverage an exchangeable sample of ``m`` scores gives, or ``inf``."""
    k = math.ceil((m + 1) * (1.0 - alpha))
    return float("inf") if k > m else k / (m + 1)


def _replicate(seed: int, horizon: int, alphas):
    """One geo test: fit, calibrate at rolling origins, band every treated market.

    The planted effect is zero, so a band covers exactly when it contains zero.
    """
    import sys
    from pathlib import Path

    import numpy as np

    study = Path(__file__).resolve().parents[1] / "studies" / "cumulative_calibration"
    if str(study) not in sys.path:
        sys.path.insert(0, str(study))
    import synthetic_geo_panel as sgp

    from mlsynth.utils.conformal import cumulative_conformal_interval
    from mlsynth.utils.ppscm_helpers.engine import run_multisynth
    from mlsynth.utils.ppscm_helpers.inference import rolling_pooled_block_sums

    rng = np.random.default_rng(seed)
    values, _ = sgp.make_panel(rng, n_weeks=_T0 + horizon,
                               size_profile=sgp.TOP30_POPULATION_SHARE)
    n_markets = values.shape[1]
    adoption = np.full(n_markets, np.inf)
    adoption[rng.choice(n_markets, _N_TREATED, replace=False)] = _T0
    panel = values.T.copy()

    fit = run_multisynth(panel, adoption, _T0, horizon, _T0, fixedeff=True,
                         time_cohort=False, nu=_NU, lam=0.0, solver=None)
    point = [float(np.sum(np.asarray(fit["tau_rel"][k], dtype=float)[:horizon]))
             for k in range(len(fit["groups"]))]
    scores = rolling_pooled_block_sums(
        panel, adoption, d=_T0, n_leads=horizon, n_lags=_T0, fixedeff=True,
        time_cohort=False, nu_used=_NU, lam=0.0, solver=None, horizon=horizon,
        min_train_frac=_MIN_TRAIN_FRAC)

    covered = {a: [] for a in alphas}
    finite = {a: [0, 0] for a in alphas}
    m = 0
    for k, s in enumerate(scores):
        m = int(s.size)
        for alpha in alphas:
            band = cumulative_conformal_interval(point[k], s, alpha=alpha,
                                                 horizon=horizon)
            finite[alpha][1] += 1
            if np.isfinite(band.lower):
                finite[alpha][0] += 1
                covered[alpha].append(float(band.lower <= 0.0 <= band.upper))
    return covered, finite, m


def run() -> dict:
    import numpy as np

    out = {}
    for horizon, reps, alphas in _ARMS:
        per_rep = {a: [] for a in alphas}
        finite = {a: [0, 0] for a in alphas}
        observed_m = 0
        for r in range(reps):
            covered, fin, observed_m = _replicate(2000 + r, horizon, alphas)
            for a in alphas:
                finite[a][0] += fin[a][0]
                finite[a][1] += fin[a][1]
                if covered[a]:
                    per_rep[a].append(float(np.mean(covered[a])))
        out[f"windows_h{horizon}"] = float(observed_m)
        # The closed form and the loop must agree exactly; a drift in either is
        # a change in the design, not noise.
        out[f"windows_h{horizon}_vs_formula"] = float(
            observed_m - _windows(_T0, horizon))
        for a in alphas:
            tag = f"h{horizon}_a{int(a * 100):02d}"
            out[f"finite_share_{tag}"] = finite[a][0] / finite[a][1]
            if not per_rep[a]:
                continue
            coverage = float(np.mean(per_rep[a]))
            out[f"coverage_{tag}"] = coverage
            out[f"coverage_gap_{tag}"] = coverage - _predicted(observed_m, a)
    return out


# Window counts and finite shares are exact consequences of the design, pinned at
# zero tolerance. The coverage gaps target zero -- the exchangeable prediction --
# and each tolerance is about four replication-clustered standard errors at these
# replication counts, so a real loss of exchangeability trips them and resampling
# noise does not:
#
#     h = 8, alpha = 0.10   coverage 0.9017   se 0.0120   4 se 0.048   predicted 0.9091
#     h = 4, alpha = 0.05   coverage 0.9675   se 0.0083   4 se 0.033   predicted 0.9545
#     h = 4, alpha = 0.10   coverage 0.9125   se 0.0144   4 se 0.058   predicted 0.9091
#
# Coverage itself is recorded alongside its gap so a failure says which way it
# moved. All three land inside two standard errors of the prediction.
#
# The same measurement at 211 markets instead of 30 agrees at both horizons, so
# none of this is an artefact of the smaller cross-section:
#
#                        30 markets   211 markets   predicted
#     h = 8, alpha=0.10      0.9017        0.910       0.9091
#     h = 4, alpha=0.05      0.9675        0.956       0.9545
#     h = 4, alpha=0.10      0.9125        0.914       0.9091
#
# The 211-market figures come from 49 replications at h = 8 and 18 at h = 4,
# under the study arm in benchmarks/studies/cumulative_calibration/.
EXPECTED = {
    "windows_h8": (10.0, 0.0),
    "windows_h8_vs_formula": (0.0, 0.0),
    "finite_share_h8_a05": (0.0, 0.0),
    "finite_share_h8_a10": (1.0, 0.0),
    "coverage_h8_a10": (0.9017, 0.05),
    "coverage_gap_h8_a10": (0.0, 0.05),
    "windows_h4": (21.0, 0.0),
    "windows_h4_vs_formula": (0.0, 0.0),
    "finite_share_h4_a05": (1.0, 0.0),
    "finite_share_h4_a10": (1.0, 0.0),
    "coverage_h4_a05": (0.9675, 0.04),
    "coverage_gap_h4_a05": (0.0, 0.04),
    "coverage_h4_a10": (0.9125, 0.06),
    "coverage_gap_h4_a10": (0.0, 0.06),
}
