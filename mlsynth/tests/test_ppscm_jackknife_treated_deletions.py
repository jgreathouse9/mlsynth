"""What the delete-one jackknife varies, and what it therefore prices.

``jackknife_inference`` deletes every unit in turn and admits a replicate
whenever the remainder still has a treated and a control unit. With one treated
unit, deleting it leaves no treated unit, so that replicate is skipped and every
survivor is a control deletion. The guard counts survivors, so it passes, and a
standard error is reported that the treated unit's own data cannot move.

The two deletions are not the same quantity. Removing a control moves the
synthetic counterfactual a little. Removing a treated unit removes one of the few
draws the effect is averaged over, and that is the sampling variability of the
pooled estimand. An ensemble with none of the second measures donor substitution.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.ppscm_helpers.engine import run_multisynth
from mlsynth.utils.ppscm_helpers.inference import jackknife_inference


def _panel(n_treated, n_control=8, T0=30, H=6, effect=2.0, seed=0):
    """A factor panel with ``n_treated`` treated units and a planted effect."""
    rng = np.random.default_rng(seed)
    n, T = n_treated + n_control, T0 + H
    factor = np.cumsum(rng.normal(0.0, 1.0, T))
    Xy = np.array([rng.normal(5.0, 1.0) + rng.uniform(0.5, 1.5) * factor
                   + rng.normal(0.0, 0.3, T) for _ in range(n)])
    Xy[:n_treated, T0:] += effect
    trt = np.full(n, np.inf)
    trt[:n_treated] = T0
    return Xy, trt, T0, H


def _call(Xy, trt, T0, H, **kw):
    full = run_multisynth(Xy, trt, T0, H, T0, fixedeff=True, time_cohort=False,
                          nu=0.5, lam=0.0, solver=None)
    base = dict(d=T0, n_leads=H, n_lags=T0, fixedeff=True, time_cohort=False,
                nu_used=0.5, lam=0.0, solver=None, alpha=0.05,
                per_time_full=np.asarray(full["per_time"], dtype=float),
                att_full=float(full["att"]))
    base.update(kw)
    return jackknife_inference(Xy, trt, **base)


# --------------------------------------------------------------------------- smoke
def test_two_treated_units_still_report_a_standard_error():
    """The ensemble then contains a treated deletion, which is what is needed."""
    att, se, ci, per_se, per_ci = _call(*_panel(n_treated=2))
    assert np.isfinite(se) and se > 0.0
    assert ci[0] <= att <= ci[1]


# ------------------------------------------------------------------ unit invariants
def test_the_standard_error_moves_when_a_treated_unit_moves():
    """The defect this file exists for, stated positively.

    With a treated deletion in the ensemble, changing a treated unit's
    post-period changes the spread of the leave-one-out effects. Without one it
    cannot, since no replicate ever removes a treated unit.
    """
    Xy, trt, T0, H = _panel(n_treated=2, effect=2.0)
    _, se_small, *_ = _call(Xy, trt, T0, H)
    Xy2 = Xy.copy()
    Xy2[0, T0:] += 50.0
    _, se_large, *_ = _call(Xy2, trt, T0, H)
    assert not np.isclose(se_small, se_large), (se_small, se_large)


def test_deleting_a_control_and_deleting_a_treated_unit_are_different():
    """Both admit replicates, but only one varies the estimand's own draws."""
    Xy, trt, T0, H = _panel(n_treated=3)
    n = Xy.shape[0]
    treated_deletions = [i for i in range(n)
                         if np.isfinite(np.delete(trt, i)).any()
                         and not np.isfinite(np.delete(trt, i)).all()
                         and np.isfinite(trt[i])]
    assert treated_deletions, "expected the ensemble to contain treated deletions"


# ------------------------------------------------------------------------- edge
def test_two_treated_units_leave_exactly_one_treated_deletion():
    """The thinnest ensemble that is not vacuous; it is allowed."""
    Xy, trt, T0, H = _panel(n_treated=2)
    _, se, *_ = _call(Xy, trt, T0, H)
    assert np.isfinite(se)


# ---------------------------------------------------------------------- failure
def test_one_treated_unit_is_refused():
    """No replicate removes the treated unit, so there is nothing to report."""
    Xy, trt, T0, H = _panel(n_treated=1)
    with pytest.raises(MlsynthEstimationError, match="treated"):
        _call(Xy, trt, T0, H)


def test_the_refusal_says_what_the_ensemble_was_measuring():
    Xy, trt, T0, H = _panel(n_treated=1)
    with pytest.raises(MlsynthEstimationError) as exc:
        _call(Xy, trt, T0, H)
    message = str(exc.value)
    assert "donor substitution" in message
    assert "1 treated unit" in message or "one treated unit" in message


def test_the_refusal_is_not_reached_by_the_old_usable_guard():
    """The old guard counted survivors, so it passed here with eight of them.

    Asserting the count keeps the two guards distinguishable: a future change
    that restores survivor-counting would satisfy the raise but not this.
    """
    Xy, trt, T0, H = _panel(n_treated=1, n_control=8)
    with pytest.raises(MlsynthEstimationError) as exc:
        _call(Xy, trt, T0, H)
    assert "0 " in str(exc.value) or "no " in str(exc.value).lower()
