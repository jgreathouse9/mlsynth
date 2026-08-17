"""Whether the prediction-interval bootstrap may assume independent errors.

Jiang, Li, Shen and Zhou's Algorithm 2.1 resamples the pre-period prediction
error with a *dependent* wild bootstrap: Bartlett-correlated multipliers whose
dependence range is a bandwidth in ``T0``, so a persistent error is resampled as
a persistent error. Their Remark 2.2 gives the cheaper alternative -- ordinary
i.i.d. standard-normal multipliers -- and states its condition, that the errors
are independent to begin with.

``pda_prediction_intervals`` has always taken ``dependent`` and defaulted it to
``True``. What it did not have was a way for a caller to say otherwise:
``PDAConfig`` never passed the argument, so Remark 2.2 was unreachable from the
estimator. This suite pins the field that exposes it and, more to the point, pins
the difference it makes -- a flag that changes nothing is worse than no flag,
because it invites a caller to believe they chose something.

The difference is a directional claim, not a number, and it is a claim about the
average panel rather than every panel. The flag reaches only the pre-period
multipliers, so what it moves is the estimation-error variance; on any single
draw that shift is comparable to the bootstrap's own Monte Carlo noise. Measured
over eight panels the dependent scheme gave the wider band on five to eight of
them, mean ratio 1.015 on white noise and 1.043 at rho = 0.9. So the tests below
assert the mean over panels and the ordering between persistence levels -- the
level at which the claim is actually true. An earlier draft of this suite
asserted it per panel and failed on three seeds of six, which is how the scope of
the claim was established.

Levels: smoke, unit invariants, property, edge, failure.
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth import PDA
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.inferutils import pda_prediction_intervals


def _panel(n_donors=8, T0=30, T1=6, effect=1.0, seed=0, rho=0.0):
    """Factor panel with one treated unit adopting at ``T0``.

    ``rho`` is the AR(1) coefficient of the idiosyncratic errors -- the thing the
    dependent wild bootstrap exists to carry and the i.i.d. scheme discards.
    """
    rng = np.random.default_rng(seed)
    T = T0 + T1
    f = rng.standard_normal((T, 2))
    load_d = rng.standard_normal((n_donors, 2)) * 0.5
    e = rng.standard_normal((n_donors + 1, T)) * 0.3
    if rho:
        for t in range(1, T):
            e[:, t] = rho * e[:, t - 1] + np.sqrt(1 - rho ** 2) * e[:, t]
    rec = []
    y = f @ load_d.mean(axis=0) + e[0]
    y[T0:] += effect
    rec += [{"unit": "treated", "t": t, "y": float(y[t]), "d": int(t >= T0)}
            for t in range(T)]
    for j in range(n_donors):
        s = f @ load_d[j] + e[j + 1]
        rec += [{"unit": f"d{j}", "t": t, "y": float(s[t]), "d": 0} for t in range(T)]
    return pd.DataFrame(rec)


def _fit(df=None, **kw):
    cfg = dict(df=_panel() if df is None else df, outcome="y", treat="d",
               unitid="unit", time="t", method="LASSO", display_graphs=False,
               prediction_intervals=True, pi_n_boot=300, pi_seed=0)
    cfg.update(kw)
    return PDA(cfg).fit()


def _pi(res, method="lasso"):
    fit = res.fits[method] if isinstance(res.fits, dict) else res.fits[0]
    return fit.prediction_intervals


# --------------------------------------------------------------------------- #
# smoke                                                                        #
# --------------------------------------------------------------------------- #
def test_both_settings_produce_a_usable_fit():
    for dep in (True, False):
        pi = _pi(_fit(pi_dependent=dep))
        assert np.isfinite(np.asarray(pi["effect"]["eq_lower"])).all()
        assert np.isfinite(np.asarray(pi["effect"]["eq_upper"])).all()


# --------------------------------------------------------------------------- #
# unit invariants                                                              #
# --------------------------------------------------------------------------- #
def test_the_paper_s_algorithm_is_the_default():
    """Algorithm 2.1 is the method; Remark 2.2 is the caller's exception to it."""
    from mlsynth.utils.pda_helpers.config import PDAConfig
    assert PDAConfig.model_fields["pi_dependent"].default is True
    assert _pi(_fit())["dependent"] is True


def test_the_choice_is_recorded_on_the_result():
    """A band built under Remark 2.2 and one built under Algorithm 2.1 are not
    interchangeable numbers, so the result has to say which it is."""
    assert _pi(_fit(pi_dependent=False))["dependent"] is False
    assert _pi(_fit(pi_dependent=True))["dependent"] is True


def test_the_flag_actually_reaches_the_bootstrap():
    """Guards the plumbing itself: the field existed to be passed, and a config
    field that is silently dropped is the failure this test is here to catch."""
    a = _pi(_fit(df=_panel(rho=0.9, seed=1), pi_dependent=True))
    b = _pi(_fit(df=_panel(rho=0.9, seed=1), pi_dependent=False))
    assert not np.allclose(a["effect"]["eq_lower"], b["effect"]["eq_lower"])


def test_it_leaves_the_point_estimates_alone():
    """It is a resampling choice. The fit it resamples must not move."""
    base = _fit(pi_dependent=True)
    other = _fit(pi_dependent=False)
    fb = base.fits["lasso"] if isinstance(base.fits, dict) else base.fits[0]
    fo = other.fits["lasso"] if isinstance(other.fits, dict) else other.fits[0]
    assert fo.att == pytest.approx(fb.att)
    np.testing.assert_allclose(fo.counterfactual, fb.counterfactual)
    np.testing.assert_allclose(fo.beta, fb.beta)


# --------------------------------------------------------------------------- #
# property: the two agree on white noise and separate as errors persist        #
# --------------------------------------------------------------------------- #
def _mean_width_ratio(rho, n_seeds=6, n_boot=250):
    """Mean of ``se_dependent / se_iid`` at the last cumulative horizon.

    Averaged over panels because the per-panel ordering is not guaranteed: the
    flag changes only the pre-period multipliers, so it moves the estimation-error
    variance, and on a single draw that shift is comparable to the bootstrap's own
    Monte Carlo noise. Measured over eight panels the dependent scheme was the
    wider one on five to eight of them depending on persistence, with a mean
    ratio of 1.015 (white noise) and 1.043 (rho = 0.9). The mean is the level at
    which the claim is true, so the mean is what is asserted.
    """
    r = []
    for seed in range(n_seeds):
        kw = dict(df=_panel(rho=rho, seed=seed), cumulative_band=True, pi_n_boot=n_boot)
        a = _fit(pi_dependent=True, **kw)
        b = _fit(pi_dependent=False, **kw)
        fa = a.fits["lasso"] if isinstance(a.fits, dict) else a.fits[0]
        fb = b.fits["lasso"] if isinstance(b.fits, dict) else b.fits[0]
        r.append(float(np.asarray(fa.cumulative_band.se)[-1]
                       / np.asarray(fb.cumulative_band.se)[-1]))
    return float(np.mean(r))


def test_the_dependent_bootstrap_is_the_wider_choice_on_average():
    """Remark 2.2 is the cheaper option, and cheaper here means narrower.

    Correlated multipliers carry variance that i.i.d. ones discard, so dropping
    to the i.i.d. scheme can only lose width, never gain it. Asserted as a mean
    over panels -- see :func:`_mean_width_ratio` for why not per panel.
    """
    assert _mean_width_ratio(rho=0.9) > 1.0


def test_persistence_is_what_makes_the_choice_matter():
    """The directional claim behind the default.

    On white noise the two schemes are resampling the same object and the
    correction is near nothing. As the errors persist, the i.i.d. scheme keeps
    drawing them as if they did not, and the gap widens. If this ordering ever
    reversed, the default would be indefensible.
    """
    assert _mean_width_ratio(rho=0.9) > _mean_width_ratio(rho=0.0)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_the_band_stays_valid_at_every_persistence(rho):
    """Whatever the errors do, the band must still bracket its point and be
    finite -- the flag changes the width, never the well-formedness."""
    for dep in (True, False):
        res = _fit(df=_panel(rho=rho, seed=7), cumulative_band=True,
                   pi_dependent=dep, pi_n_boot=300)
        fit = res.fits["lasso"] if isinstance(res.fits, dict) else res.fits[0]
        b = fit.cumulative_band
        assert np.isfinite(np.asarray(b.se)).all()
        assert (np.asarray(b.lower) <= np.asarray(b.point)).all()
        assert (np.asarray(b.point) <= np.asarray(b.upper)).all()


# --------------------------------------------------------------------------- #
# edge                                                                         #
# --------------------------------------------------------------------------- #
def test_a_single_post_period_still_honours_the_flag():
    """With one horizon there is nothing to accumulate, but the pre-period
    resampling still differs, so the interval still moves."""
    df = _panel(T1=1, rho=0.9, seed=2)
    a = _pi(_fit(df=df, pi_dependent=True))
    b = _pi(_fit(df=df, pi_dependent=False))
    assert a["dependent"] is True and b["dependent"] is False
    assert np.asarray(a["effect"]["eq_lower"]).shape == (1,)


def test_it_is_honoured_by_every_variant_not_just_lasso():
    res = _fit(methods=["l2", "LASSO", "fs"], pi_dependent=False, pi_n_boot=200)
    for name, fit in res.fits.items():
        assert fit.prediction_intervals["dependent"] is False


# --------------------------------------------------------------------------- #
# failure -- reported, not swallowed                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["yes", 1.5, None, "dependent"])
def test_a_non_boolean_is_refused_at_config_time(bad):
    with pytest.raises(MlsynthConfigError):
        PDA(dict(df=_panel(), outcome="y", treat="d", unitid="unit", time="t",
                 method="LASSO", prediction_intervals=True, pi_dependent=bad,
                 display_graphs=False))


def test_the_helper_still_defaults_to_the_paper_s_algorithm():
    """The estimator-agnostic entry point is reused by other estimators, so its
    own default must not drift when the config field is added above it."""
    import inspect
    sig = inspect.signature(pda_prediction_intervals)
    assert sig.parameters["dependent"].default is True
