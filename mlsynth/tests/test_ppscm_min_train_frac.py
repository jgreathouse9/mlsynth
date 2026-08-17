"""Where the cumulative band's calibration starts, and why a caller needs to say.

The per-unit cumulative band calibrates on non-overlapping windows rolled across
the pre-period. The roll does not start at period zero: the fit at each origin
has to be trained on the periods before it, so the first origin sits at
``max(MIN_TRAIN_PERIODS, min_train_frac * T0)``. That fraction was fixed at 0.3
and unreachable from the config, which makes it a hidden choice in two ways that
matter on a real panel.

Windows against training length. Every period spent on the warm-up is a period
not available for calibration, so raising the fraction buys better-trained fits
and costs windows -- and windows are what the level costs: a ``1-alpha`` band
needs at least ``ceil(1/alpha) - 1`` of them or it is honestly infinite. The two
pull against each other and only the caller knows their panel.

Donors against training length. The concrete case: a panel with 120 pre-periods
and 60 donors starts calibrating at period 36, so the early fits have fewer
training periods than donors. Those fits can interpolate their training window,
and their out-of-sample errors are not the errors of the fit being calibrated.
Split conformal takes an upper order statistic of the scores, so inflated early
scores widen the band. Pushing the start past the donor count fixes it, at the
price of windows -- which is exactly the trade the caller has to be able to make.

Levels: smoke, unit invariants, edge, failure.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.conformal import MIN_TRAIN_PERIODS


def _expected_windows(pre, horizon, frac):
    """The origins the roll visits, computed independently of the implementation."""
    start = max(MIN_TRAIN_PERIODS, int(pre * frac))
    return len(range(start, pre - horizon + 1, horizon))


def _panel(seed=0, n_donors=8, n_treated=2, pre=40, post=6, effect=0.0, noise=0.4):
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
               display_graphs=False, run_inference=False, conformal_horizon=3,
               alpha=0.2)
    cfg.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PPSCM(cfg).fit()


def _windows(res):
    counts = {u.cumulative_windows for u in res.per_unit.values()}
    assert len(counts) == 1, f"units disagree on the window count: {counts}"
    return counts.pop()


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def test_the_field_exists_and_defaults_to_the_previous_fixed_value():
    """0.3 was the hardcoded value, so the default has to reproduce it exactly."""
    from mlsynth.utils.ppscm_helpers.config import PPSCMConfig

    assert PPSCMConfig.model_fields["conformal_min_train_frac"].default == 0.3


def test_the_default_matches_passing_it_explicitly():
    """A caller who sets 0.3 gets what a caller who sets nothing gets."""
    implicit = _fit()
    explicit = _fit(conformal_min_train_frac=0.3)
    for key, unit in implicit.per_unit.items():
        other = explicit.per_unit[key]
        assert unit.cumulative_windows == other.cumulative_windows
        assert unit.cumulative_lower == pytest.approx(other.cumulative_lower)
        assert unit.cumulative_upper == pytest.approx(other.cumulative_upper)


# ---------------------------------------------------------------------------
# Unit: the field does the arithmetic it claims
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frac", [0.3, 0.4, 0.5])
def test_the_window_count_follows_the_documented_formula(frac):
    """``floor``-style arithmetic on the origins, checked against the roll itself."""
    res = _fit(conformal_min_train_frac=frac)
    assert _windows(res) == _expected_windows(pre=40, horizon=3, frac=frac)


def test_raising_the_fraction_spends_windows():
    """The trade the caller is being handed, asserted as a monotone one."""
    counts = [_windows(_fit(conformal_min_train_frac=f)) for f in (0.3, 0.5, 0.7)]
    assert counts[0] > counts[1] > counts[2]


def test_lowering_the_fraction_cannot_start_before_the_minimum_training_length():
    """``MIN_TRAIN_PERIODS`` floors the roll however small the fraction goes.

    A fit needs periods to be a fit. Without the floor a fraction near zero would
    calibrate on origins trained on one or two periods, whose out-of-sample error
    says nothing about the estimator being calibrated.
    """
    tiny = _windows(_fit(conformal_min_train_frac=0.01))
    assert tiny == _expected_windows(pre=40, horizon=3, frac=0.01)
    assert tiny == len(range(MIN_TRAIN_PERIODS, 40 - 3 + 1, 3))


def test_the_band_still_covers_a_known_null_after_the_fraction_moves():
    """Changing where calibration starts must not cost validity, only width."""
    for frac in (0.3, 0.5):
        res = _fit(conformal_min_train_frac=frac)
        for unit in res.per_unit.values():
            assert unit.cumulative_lower <= unit.cumulative_effect
            assert unit.cumulative_effect <= unit.cumulative_upper


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

def test_a_fraction_large_enough_to_starve_the_calibration_gives_infinite_bands():
    """Reported as infinite, not as a narrow band that does not cover.

    Split conformal at ``1-alpha`` needs ``ceil(1/alpha) - 1`` windows. Below
    that the order statistic does not exist and the honest answer is unbounded.
    """
    res = _fit(conformal_min_train_frac=0.9, alpha=0.05)
    for unit in res.per_unit.values():
        assert not np.isfinite(unit.cumulative_lower)
        assert not np.isfinite(unit.cumulative_upper)


def test_the_field_is_inert_when_no_horizon_is_requested():
    """It configures the cumulative band, so it must not perturb anything else."""
    base = _fit(conformal_horizon=None)
    moved = _fit(conformal_horizon=None, conformal_min_train_frac=0.6)
    for key, unit in base.per_unit.items():
        other = moved.per_unit[key]
        assert unit.att == pytest.approx(other.att)
        assert unit.cumulative_windows is None and other.cumulative_windows is None


def test_the_window_count_is_reported_so_the_choice_is_visible():
    """A band at 6 windows and one at 12 are different products.

    The interval alone does not say which, so the count travels with it.
    """
    few = _fit(conformal_min_train_frac=0.6)
    many = _fit(conformal_min_train_frac=0.3)
    assert _windows(few) < _windows(many)
    assert all(u.cumulative_windows is not None for u in few.per_unit.values())


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------

def test_a_numeric_string_is_coerced_rather_than_refused():
    """Pydantic's lax coercion applies, and the coerced value is the one used.

    Pinned because it is the one input that looks like it should fail and does
    not; a caller reading a fraction out of a YAML config gets a string, and
    the alternative would be a confusing rejection of a perfectly good number.
    """
    coerced = _fit(conformal_min_train_frac="0.5")
    assert _windows(coerced) == _windows(_fit(conformal_min_train_frac=0.5))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, 2, True, "abc", None])
def test_a_fraction_outside_the_open_unit_interval_is_rejected_at_config_time(bad):
    """The fraction is a proportion of the pre-period, so 0 and 1 are both wrong.

    Zero would place the roll at the minimum training length regardless of panel
    length, which is the floor's job and not the fraction's; one would leave no
    pre-period to calibrate on at all.
    """
    with pytest.raises((MlsynthConfigError, ValueError)):
        _fit(conformal_min_train_frac=bad)


def test_the_helper_rejects_the_same_values():
    """The validator on the config is not the only guard; the helper is callable."""
    from mlsynth.utils.ppscm_helpers.inference import cumulative_conformal_per_unit

    Xy = np.random.default_rng(0).normal(size=(6, 20))
    trt = np.full(6, np.inf)
    trt[:2] = 16
    with pytest.raises(MlsynthConfigError, match="min_train_frac"):
        cumulative_conformal_per_unit(
            Xy, trt, d=16, n_leads=4, n_lags=16, fixedeff=True, time_cohort=False,
            nu_used=0.5, lam=0.0, solver=None, alpha=0.1, horizon=2,
            min_train_frac=1.5,
        )
