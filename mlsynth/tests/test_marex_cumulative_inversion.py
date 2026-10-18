"""MAREX's band for the total effect by test inversion.

``cumulative_method="resample"``, the default, calibrates a total on the
blank-period effects alone. Any band built only from errors observed before the
post period is limited by how many horizon-length windows the blank stretch
holds. ``cumulative_method="inversion"`` reads the post outcomes as well. Under a
constant-effect null it imputes the post block of the synthetic treated series,
refits a synthetic control for it on every period from the units the design left
untreated, and compares the signed post sum of the residuals with its cyclic
shifts over the blank and post periods (Chernozhukov, Wuthrich and Zhu 2021,
Remark 1). The fitting periods stay out of the reference because the design was
chosen on them, and the residuals there are smaller.

These tests pin the wiring: which series is tested, which units are the donors,
where the reference starts, which statistic is used, and which point the band
reports. The test itself is
:func:`~mlsynth.utils.conformal.cumulative_conformal_by_inversion`'s, which has
its own suite.

Levels: smoke, unit invariants, edge, failure.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.conformal import (CumulativeConformalBand,
                                     CumulativeInversionBand,
                                     cumulative_conformal_by_inversion)
from mlsynth.utils.marex_helpers.inference import compute_inference


def _panel(J=10, T=48, seed=0, clusters=None):
    """A two-factor panel; ``clusters`` maps unit position to a cluster label."""
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, (T, 2))
    lam = rng.normal(0, 1, (J, 2))
    Y = lam @ F.T + 0.4 * rng.standard_normal((J, T))
    rows = [{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t])}
            for j in range(J) for t in range(T)]
    if clusters is not None:
        for r in rows:
            r["grp"] = clusters[int(r["unit"][1:])]
    return pd.DataFrame(rows)


# Tb + T_post = 26 periods in the reference, so the smallest attainable p-value
# is 1/26 and a 5% test can reject.
_CFG = dict(outcome="y", unitid="unit", time="time", T0=40, m_eq=2,
            inference=True, blank_periods=18, T_post=8, backend="exact")
_TE = _CFG["T0"] - _CFG["blank_periods"]


def _fit(df=None, **kw):
    cfg = dict(df=_panel() if df is None else df, **_CFG)
    cfg.update(kw)
    return MAREX(cfg).fit()


def _inversion(df=None, **kw):
    return _fit(df, cumulative_band=True, cumulative_method="inversion", **kw)


def _library(y, controls, pre, te, alpha=0.05):
    """The band the wiring should produce, called on the library directly."""
    return cumulative_conformal_by_inversion(
        np.asarray(y, float), np.asarray(controls, float).T, pre, alpha=alpha,
        refit="sc", conformal_type="block", statistic="signed",
        reference_start=te)


def _contrast(Tb=16, T_post=8, TE=24, J=5, seed=1, effect=1.5):
    """A treated series inside its donors' hull, and a control series."""
    rng = np.random.default_rng(seed)
    T = TE + Tb + T_post
    F = rng.normal(0, 1, (T, 2))
    lam = rng.uniform(0.2, 1.0, (J, 2))
    controls = lam @ F.T + 0.3 * rng.standard_normal((J, T))
    y_t = lam.mean(axis=0) @ F.T + 0.3 * rng.standard_normal(T)
    y_t[TE + Tb:] += effect
    return y_t, controls.mean(axis=0), controls, TE + Tb, TE, Tb


def _direct(Tb=16, T_post=8, TE=24, J=5, seed=1, effect=1.5, alpha=0.05,
            **kw):
    y_t, y_c, controls, T0, te, tb = _contrast(Tb, T_post, TE, J, seed, effect)
    kw.setdefault("cumulative_band", True)
    kw.setdefault("cumulative_method", "inversion")
    kw.setdefault("control_outcomes", controls)
    return compute_inference(y_t, y_c, T0, te, tb, alpha=alpha,
                             random_state=0, **kw)


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_it_produces_a_finite_band_by_inversion():
    band = _inversion().globres.inference.cumulative_band
    assert isinstance(band, CumulativeInversionBand)
    assert np.isfinite(band.lower) and np.isfinite(band.upper)
    assert band.lower < band.upper
    assert not band.is_empty and not band.is_unbounded


def test_every_cluster_gets_a_band_by_inversion():
    res = _inversion()
    for c in res.clusters.values():
        assert isinstance(c.inference.cumulative_band, CumulativeInversionBand)


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_the_default_method_is_the_resampled_band():
    """No existing caller sees a different band."""
    default = _fit(cumulative_band=True, cumulative_n_sim=400)
    named = _fit(cumulative_band=True, cumulative_n_sim=400,
                 cumulative_method="resample")
    a = default.globres.inference.cumulative_band
    b = named.globres.inference.cumulative_band
    assert isinstance(a, CumulativeConformalBand)
    assert (a.lower, a.upper, a.half_width) == (b.lower, b.upper, b.half_width)


def test_the_method_alone_adds_no_band():
    """``cumulative_method`` chooses a construction; ``cumulative_band`` asks."""
    assert _fit(cumulative_method="inversion").globres.inference \
        .cumulative_band is None


def test_it_uses_the_signed_statistic_the_sc_refit_and_cyclic_shifts():
    band = _inversion().globres.inference.cumulative_band
    assert band.statistic == "signed"
    assert band.refit == "sc"
    assert band.conformal_type == "block"


def test_the_reference_starts_at_the_blank_window():
    """The fitting periods, where the design was chosen, stay out of it."""
    band = _inversion().globres.inference.cumulative_band
    assert band.reference_start == _TE
    assert band.n_reference == _CFG["blank_periods"] + _CFG["T_post"]


def test_the_point_is_marex_own_total():
    """The same number the resampled band reports, so the two read alike."""
    inf = _inversion().globres.inference
    assert inf.cumulative_band.point == pytest.approx(
        float(inf.treated_effects.sum()), rel=1e-12)
    assert inf.cumulative_band.horizon == inf.treated_effects.size


def test_the_pointwise_outputs_are_untouched():
    plain = _fit().globres.inference
    inv = _inversion().globres.inference
    np.testing.assert_allclose(plain.ci, inv.ci, equal_nan=True)
    assert plain.global_p_value == inv.global_p_value
    np.testing.assert_array_equal(plain.per_period_pvals, inv.per_period_pvals)


def test_the_global_band_tests_the_treated_synthetic_on_the_untreated_units():
    res = _inversion()
    g = res.globres
    controls = g.Y_full[np.asarray(g.treated_weights_agg) <= 1e-8]
    want = _library(g.synthetic_treated, controls, _CFG["T0"], _TE)
    got = g.inference.cumulative_band
    assert (got.lower, got.upper) == (want.lower, want.upper)


def test_a_cluster_band_draws_its_donors_from_its_own_cluster():
    """A cluster's control is built from its members; so is the refit's."""
    J = 12
    labels = ["a"] * 6 + ["b"] * 6
    df = _panel(J=J, clusters=labels)
    res = _inversion(df, cluster="grp", m_eq=1)
    Y = res.globres.Y_full
    pivot = df.pivot(index="unit", columns="time", values="y").to_numpy()
    np.testing.assert_array_equal(Y, pivot)          # rows in unit order
    row = {u: i for i, u in enumerate(sorted(df["unit"].unique()))}
    untreated = np.asarray(res.globres.treated_weights_agg) <= 1e-8
    assert len(res.clusters) == 2
    for c in res.clusters.values():
        idx = np.array([row[u] for u in c.members])
        assert {labels[i] for i in idx} in ({"a"}, {"b"})
        own = idx[np.asarray(c.treated_weights)[idx] <= 1e-8]
        want = _library(c.synthetic_treated, Y[own], _CFG["T0"], _TE)
        got = c.inference.cumulative_band
        assert (got.lower, got.upper) == (want.lower, want.upper)
        # and the pool of every untreated unit gives a different band, so the
        # assertion above has the power to tell the two apart
        wide = _library(c.synthetic_treated, Y[untreated], _CFG["T0"], _TE)
        assert (wide.lower, wide.upper) != (want.lower, want.upper)


def test_compute_inference_is_the_library_band_with_marex_point():
    y_t, y_c, controls, T0, te, tb = _contrast()
    inf = _direct()
    want = _library(y_t, controls, T0, te)
    band = inf.cumulative_band
    assert (band.lower, band.upper) == (want.lower, want.upper)
    assert band.point == pytest.approx(float((y_t - y_c)[T0:].sum()), rel=1e-12)
    assert band.point != want.point


def test_a_post_shift_moves_the_band_by_the_horizon_times_the_shift():
    """The test reads the post outcomes: shifting them shifts the band."""
    y_t, y_c, controls, T0, te, tb = _contrast()
    H = y_t.size - T0
    base = compute_inference(y_t, y_c, T0, te, tb, cumulative_band=True,
                             cumulative_method="inversion",
                             control_outcomes=controls).cumulative_band
    y_s = y_t.copy()
    y_s[T0:] += 2.0
    moved = compute_inference(y_s, y_c, T0, te, tb, cumulative_band=True,
                              cumulative_method="inversion",
                              control_outcomes=controls).cumulative_band
    assert moved.point == pytest.approx(base.point + 2.0 * H, rel=1e-12)
    assert moved.lower == pytest.approx(base.lower + 2.0 * H, rel=1e-6)
    assert moved.upper == pytest.approx(base.upper + 2.0 * H, rel=1e-6)


def test_it_records_the_level():
    assert _direct(alpha=0.1).cumulative_band.alpha == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_a_reference_too_short_to_reject_warns_and_is_unbounded():
    """At 14 periods the smallest p-value is 1/14 > 0.05: nothing is rejected."""
    with pytest.warns(UserWarning, match="cannot reject"):
        band = _direct(Tb=8, T_post=6, alpha=0.05).cumulative_band
    assert band.is_unbounded
    assert band.lower == -np.inf and band.upper == np.inf


def test_a_reference_of_exactly_one_over_alpha_can_reject():
    """At 20 periods the smallest p-value is 0.05, which a 5% test rejects.

    Twelve blank periods are below the pointwise band's floor, so that band
    warns; the inversion's own warning is the one that must stay silent.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        band = _direct(Tb=12, T_post=8, alpha=0.05).cumulative_band
    assert not any("cannot reject" in str(w.message) for w in rec)
    assert np.isfinite(band.lower) and np.isfinite(band.upper)


def test_one_short_of_one_over_alpha_warns():
    with pytest.warns(UserWarning, match="cannot reject"):
        _direct(Tb=11, T_post=8, alpha=0.05)


def test_the_warning_names_the_reference_and_the_level():
    with pytest.warns(UserWarning) as rec:
        _direct(Tb=8, T_post=6, alpha=0.05)
    msg = next(str(w.message) for w in rec if "cannot reject" in str(w.message))
    assert "14" in msg and "0.05" in msg


def test_a_single_donor_still_gives_a_band():
    y_t, y_c, controls, T0, te, tb = _contrast(J=1)
    band = compute_inference(y_t, y_c, T0, te, tb, cumulative_band=True,
                             cumulative_method="inversion",
                             control_outcomes=controls).cumulative_band
    want = _library(y_t, controls, T0, te)
    assert (band.lower, band.upper) == (want.lower, want.upper)


def test_no_post_period_gives_no_band():
    y_t, y_c, controls, T0, te, tb = _contrast()
    inf = compute_inference(y_t[:T0], y_c[:T0], T0, te, tb,
                            cumulative_band=True, cumulative_method="inversion",
                            control_outcomes=controls[:, :T0])
    assert inf.cumulative_band is None


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
def test_an_unknown_method_is_refused_by_name():
    with pytest.raises(MlsynthConfigError,
                       match="cumulative_method must be.*resample.*inversion"):
        _fit(cumulative_band=True, cumulative_method="bootstrap")


@pytest.mark.parametrize("bad", [1, None, True, ["inversion"]])
def test_a_non_string_method_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="cumulative_method must be"):
        _fit(cumulative_band=True, cumulative_method=bad)


def test_compute_inference_refuses_an_unknown_method():
    with pytest.raises(MlsynthConfigError, match="cumulative_method must be"):
        _direct(cumulative_method="bogus")


def test_inversion_without_donor_outcomes_is_refused():
    with pytest.raises(MlsynthConfigError, match="control_outcomes"):
        _direct(control_outcomes=None)


def test_donor_outcomes_over_other_periods_are_refused():
    y_t, y_c, controls, T0, te, tb = _contrast()
    with pytest.raises(MlsynthDataError, match="same periods"):
        compute_inference(y_t, y_c, T0, te, tb, cumulative_band=True,
                          cumulative_method="inversion",
                          control_outcomes=controls[:, 1:])


def test_the_resampled_band_ignores_donor_outcomes():
    """The donors are read only by the inversion."""
    y_t, y_c, controls, T0, te, tb = _contrast()
    a = compute_inference(y_t, y_c, T0, te, tb, cumulative_band=True,
                          cumulative_n_sim=300).cumulative_band
    b = compute_inference(y_t, y_c, T0, te, tb, cumulative_band=True,
                          cumulative_n_sim=300,
                          control_outcomes=controls).cumulative_band
    assert isinstance(b, CumulativeConformalBand)
    assert (a.lower, a.upper) == (b.lower, b.upper)
