"""PPSCM's cumulative band, and the replicate paths it is built from.

Both the jackknife and the wild bootstrap fit the estimator many times and get a
whole per-horizon path back from each fit. Until now PPSCM collapsed those paths
to one standard error per horizon and dropped the rest, which throws away the
only thing a cumulative band needs: how the horizons move together. A caller
wanting a band for the running total then has to rebuild the replicates from the
primitives, refitting everything a second time to recover what the first pass
already computed.

So the paths are kept, and the band is built from them here.

Two claims are pinned. The band is simultaneous, so reading the whole cumulative
path -- "the total is positive by week six and stays there" -- is covered at
``1 - alpha``, not at whatever a pointwise band degrades to across horizons. And
its width follows the correlation the replicates actually show, growing like
``sqrt(L)`` when the period errors are independent and like ``L`` when they move
as one, instead of assuming either.

Levels: smoke, unit invariants, edge, failure.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.ppscm_helpers.inference import cumulative_supt_band


def _panel(seed=0, n_donors=10, n_treated=3, pre=18, post=6, effect=0.0, noise=0.4):
    rng = np.random.default_rng(seed)
    T = pre + post
    factors = rng.standard_normal((T, 2))
    load_d = rng.standard_normal((n_donors, 2)) * 0.5
    load_t = load_d.mean(axis=0)
    rec = []
    for j in range(n_treated):
        s = factors @ (load_t + 0.1 * rng.standard_normal(2))
        s = s + rng.standard_normal(T) * noise
        s[pre:] += effect
        rec += [{"unit": f"t_{j}", "year": 2000 + t, "y": float(s[t]),
                 "tr": int(t >= pre)} for t in range(T)]
    for d in range(n_donors):
        s = factors @ load_d[d] + rng.standard_normal(T) * noise
        rec += [{"unit": f"d_{d}", "year": 2000 + t, "y": float(s[t]), "tr": 0}
                for t in range(T)]
    return pd.DataFrame(rec)


def _fit(**kw):
    cfg = dict(df=_panel(), outcome="y", treat="tr", unitid="unit", time="year",
               display_graphs=False, run_inference=True, alpha=0.10)
    cfg.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PPSCM(cfg).fit()


def _paths(n=40, H=6, rho=0.0, seed=0):
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(n, 1))
    idio = rng.normal(size=(n, H))
    return np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def test_the_band_has_one_entry_per_horizon():
    pt = np.array([1.0, 0.5, -0.2, 0.8, 0.1, 0.3])
    band = cumulative_supt_band(pt, _paths(), alpha=0.10)
    for arr in (band.horizons, band.point, band.lower, band.upper, band.se):
        assert arr.shape == pt.shape
    assert band.horizons[0] == 1 and band.horizons[-1] == pt.size


def test_the_point_path_is_the_running_total_of_the_effects():
    pt = np.array([1.0, 2.0, -0.5])
    band = cumulative_supt_band(pt, _paths(H=3), alpha=0.10)
    assert band.point == pytest.approx(np.array([1.0, 3.0, 2.5]))


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------

def test_the_band_is_symmetric_about_the_point_path():
    pt = np.array([0.4, -0.2, 0.9, 0.1])
    band = cumulative_supt_band(pt, _paths(H=4), alpha=0.10)
    assert (band.upper - band.point) == pytest.approx(band.point - band.lower)


def test_the_band_is_wider_than_the_pointwise_one_it_replaces():
    """The correction is the whole point; without it the path is under-covered."""
    from scipy.stats import norm

    pt = np.zeros(8)
    band = cumulative_supt_band(pt, _paths(H=8, rho=0.0, seed=3), alpha=0.10)
    pointwise = norm.ppf(0.95) * band.se
    assert np.all(band.upper - band.point > pointwise)
    assert band.critical_value > norm.ppf(0.95)


def test_independent_period_errors_widen_the_band_like_sqrt_L():
    """The reason the band is built from paths instead of assembled from endpoints."""
    pt = np.zeros(9)
    band = cumulative_supt_band(pt, _paths(n=4000, H=9, rho=0.0, seed=2), alpha=0.10)
    assert band.se / band.se[0] == pytest.approx(np.sqrt(np.arange(1, 10)), rel=0.15)


def test_perfectly_correlated_period_errors_widen_it_like_L():
    rng = np.random.default_rng(11)
    paths = np.repeat(rng.normal(size=(4000, 1)), 9, axis=1)
    band = cumulative_supt_band(np.zeros(9), paths, alpha=0.10)
    assert band.se / band.se[0] == pytest.approx(np.arange(1, 10), rel=0.02)


def test_a_smaller_alpha_never_narrows_the_band():
    pt = np.zeros(6)
    wide = cumulative_supt_band(pt, _paths(rho=0.3), alpha=0.01)
    narrow = cumulative_supt_band(pt, _paths(rho=0.3), alpha=0.10)
    assert np.all(wide.upper >= narrow.upper - 1e-12)


def test_the_jackknife_inflation_is_applied_by_default():
    """Delete-one replicates differ by O(1/m), so they need the inflation.

    Without it the band would be narrower by roughly ``(m-1)/sqrt(m)``, which at
    a few dozen units is a factor of several -- an interval that looks precise
    and covers nothing.
    """
    paths = _paths(n=30, H=4, seed=1)
    jack = cumulative_supt_band(np.zeros(4), paths, alpha=0.10, jackknife=True)
    boot = cumulative_supt_band(np.zeros(4), paths, alpha=0.10, jackknife=False)
    assert np.all(jack.se > boot.se)


def test_missing_replicates_are_dropped_not_counted():
    """A leave-one-out refit that failed is absent, and must not read as zero."""
    paths = _paths(n=30, H=5, seed=4)
    holed = paths.copy()
    holed[3] = np.nan
    a = cumulative_supt_band(np.zeros(5), paths[np.arange(30) != 3], alpha=0.10)
    b = cumulative_supt_band(np.zeros(5), holed, alpha=0.10)
    assert b.se == pytest.approx(a.se)
    assert b.n_replicates == a.n_replicates == 29


# ---------------------------------------------------------------------------
# Wiring on the estimator
# ---------------------------------------------------------------------------

def test_the_replicate_paths_are_kept_rather_than_discarded():
    """The cross-horizon structure survives the fit, so no second pass is needed."""
    res = _fit(inference_method="jackknife")
    paths = res.inference_detail.replicate_paths
    assert paths is not None
    assert paths.ndim == 2
    assert paths.shape[1] == res.event_study.tau.size


def test_the_cumulative_band_is_off_unless_asked_for():
    res = _fit()
    assert res.inference_detail.cumulative is None


def test_asking_for_it_attaches_a_band_covering_every_horizon():
    res = _fit(cumulative_band=True, inference_method="jackknife")
    band = res.inference_detail.cumulative
    assert band is not None
    assert band.point.size == res.event_study.tau.size
    assert band.point == pytest.approx(np.cumsum(res.event_study.tau))
    assert np.all(band.lower <= band.point) and np.all(band.point <= band.upper)


def test_the_band_covers_a_true_zero_at_about_its_nominal_rate():
    """Coverage over draws, not on one draw.

    A 90% band misses one panel in ten by construction, so asserting coverage on
    a single fit is asserting that a fair coin lands heads. Over 20 null panels
    the measured rate is 0.90 at the total and 0.875 simultaneously across all
    horizons (0.925 and 0.875 at M=40). The floor below sits about four binomial
    standard errors under nominal -- ``sqrt(.9*.1/20) = 0.067`` -- so it cannot
    flake, and it still catches the failure that matters: dropping the
    delete-one inflation shrinks the band by a factor of several and coverage
    collapses toward zero.
    """
    M = 20
    hits = 0
    for seed in range(M):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = PPSCM(dict(
                df=_panel(seed=seed), outcome="y", treat="tr", unitid="unit",
                time="year", display_graphs=False, run_inference=True,
                alpha=0.10, cumulative_band=True, inference_method="jackknife",
            )).fit()
        band = res.inference_detail.cumulative
        hits += int(band.lower[-1] <= 0.0 <= band.upper[-1])
    assert hits / M >= 0.65


def test_the_bootstrap_path_also_yields_a_band():
    """Both ensembles produce paths, so both can carry a cumulative band."""
    res = _fit(cumulative_band=True, inference_method="bootstrap", n_boot=60)
    assert res.inference_detail.cumulative is not None
    assert res.inference_detail.cumulative.method == "bootstrap"


def test_the_band_records_which_ensemble_it_came_from():
    """A jackknife band and a bootstrap band are not interchangeable numbers."""
    res = _fit(cumulative_band=True, inference_method="jackknife")
    assert res.inference_detail.cumulative.method == "jackknife"


# ---------------------------------------------------------------------------
# Edge / failure
# ---------------------------------------------------------------------------

def test_requesting_the_band_without_inference_is_reported():
    with pytest.raises(MlsynthConfigError, match="run_inference"):
        _fit(cumulative_band=True, run_inference=False)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, "0.1", None])
def test_an_invalid_alpha_is_rejected(bad):
    with pytest.raises(MlsynthConfigError, match="alpha"):
        cumulative_supt_band(np.zeros(4), _paths(H=4), alpha=bad)


def test_a_path_matrix_of_the_wrong_width_is_reported():
    with pytest.raises(Exception):
        cumulative_supt_band(np.zeros(5), _paths(H=4), alpha=0.10)


def test_fewer_than_two_replicates_is_reported():
    with pytest.raises(Exception):
        cumulative_supt_band(np.zeros(4), _paths(n=1, H=4), alpha=0.10)
