"""BFR's simulation designs for partially-pooled SCM.

Ben-Michael, Feller & Rothstein (JRSS-B 2022) Section 6 evaluate PPSCM on three
data-generating processes calibrated to their application panel, under a sharp null so
the true ATT -- and the true cumulative effect -- are exactly zero. This pins the
generator: shapes, the structure each design is supposed to have, the sharp null, the
selection model, and reproducibility.

The designs are not interchangeable. Two-way fixed effects is the case PPSCM is
unbiased for; the factor model and the AR process are where inexact fit bites, and
where the paper finds the wild bootstrap turns conservative. A generator that quietly
produced the same panel for all three would make the benchmark meaningless, so the
structural tests below assert what distinguishes them.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError


def _sim(**kw):
    from mlsynth.utils.ppscm_helpers.simulation import simulate_bfr_panel
    base = dict(design="twfe", n_units=30, n_periods=25, seed=0)
    base.update(kw)
    return simulate_bfr_panel(**base)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("design", ["twfe", "factor", "ar"])
def test_smoke_every_design_returns_a_finite_panel(design):
    s = _sim(design=design)
    assert s.Y.shape == (30, 25)
    assert np.isfinite(s.Y).all()
    assert s.trt.shape == (30,)
    assert s.design == design


# ---------------------------------------------------------------------------
# The sharp null
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("design", ["twfe", "factor", "ar"])
def test_the_null_is_sharp_so_the_true_effect_is_exactly_zero(design):
    """Every reported truth in the benchmark rests on this."""
    s = _sim(design=design)
    assert s.true_att == 0.0
    assert s.true_cumulative == 0.0


@pytest.mark.parametrize("design", ["twfe", "factor", "ar"])
def test_treated_units_are_not_given_a_different_outcome_path(design):
    """Under a sharp null the observed panel IS the untreated potential outcome."""
    s = _sim(design=design)
    np.testing.assert_array_equal(s.Y, s.Y_untreated)


# ---------------------------------------------------------------------------
# Structure -- what distinguishes the three designs
# ---------------------------------------------------------------------------

def test_twfe_leaves_no_factor_structure_behind_its_fixed_effects():
    """Sweeping unit and time means should leave essentially unstructured noise."""
    s = _sim(design="twfe", n_units=60, n_periods=60, seed=1)
    R = (s.Y - s.Y.mean(axis=1, keepdims=True)
         - s.Y.mean(axis=0, keepdims=True) + s.Y.mean())
    sv = np.linalg.svd(R, compute_uv=False)
    # no single direction should dominate the residual once the two-way part is out
    assert sv[0] ** 2 / (sv ** 2).sum() < 0.15


def test_factor_design_leaves_low_rank_structure_behind_its_fixed_effects():
    """The factor design must differ from two-way FE in exactly this way."""
    s = _sim(design="factor", n_units=60, n_periods=60, seed=1, n_factors=2)
    R = (s.Y - s.Y.mean(axis=1, keepdims=True)
         - s.Y.mean(axis=0, keepdims=True) + s.Y.mean())
    sv = np.linalg.svd(R, compute_uv=False)
    assert sv[0] ** 2 / (sv ** 2).sum() > 0.30


def test_ar_design_is_serially_correlated():
    s = _sim(design="ar", n_units=60, n_periods=80, seed=1)
    d = s.Y - s.Y.mean(axis=1, keepdims=True)
    lag1 = np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in d])
    assert lag1 > 0.25


# ---------------------------------------------------------------------------
# Selection into treatment
# ---------------------------------------------------------------------------

def test_adoption_times_come_from_the_offered_set_and_absorb():
    s = _sim(design="twfe", adoption_times=(8, 12, 16))
    adopted = s.trt[np.isfinite(s.trt)]
    assert set(np.unique(adopted)).issubset({8.0, 12.0, 16.0})
    # never-treated are +inf, not a sentinel that could be mistaken for a period
    assert np.isinf(s.trt[~np.isfinite(s.trt)]).all()


def test_some_but_not_all_units_adopt():
    s = _sim(design="twfe", n_units=60, seed=3)
    n_treated = int(np.isfinite(s.trt).sum())
    assert 0 < n_treated < 60


def test_selection_is_on_unit_effects_not_uniform():
    """BFR's point: treatment is not assigned at random, so donors differ."""
    s = _sim(design="twfe", n_units=200, seed=5, theta1=-2.0)
    treated = np.isfinite(s.trt)
    assert treated.any() and (~treated).any()
    pre_level = s.Y[:, :6].mean(axis=1)
    # a non-zero selection slope must separate the groups' levels
    assert abs(pre_level[treated].mean() - pre_level[~treated].mean()) > 0.1


def test_zero_selection_slope_gives_no_separation():
    s = _sim(design="twfe", n_units=200, seed=5, theta1=0.0)
    treated = np.isfinite(s.trt)
    pre_level = s.Y[:, :6].mean(axis=1)
    assert abs(pre_level[treated].mean() - pre_level[~treated].mean()) < 0.35


# ---------------------------------------------------------------------------
# Reproducibility -- a benchmark asserting cells needs this
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("design", ["twfe", "factor", "ar"])
def test_the_same_seed_gives_the_same_panel(design):
    a, b = _sim(design=design, seed=7), _sim(design=design, seed=7)
    np.testing.assert_array_equal(a.Y, b.Y)
    np.testing.assert_array_equal(a.trt, b.trt)


def test_different_seeds_give_different_panels():
    a, b = _sim(seed=1), _sim(seed=2)
    assert not np.array_equal(a.Y, b.Y)


# ---------------------------------------------------------------------------
# Failure -- translated errors
# ---------------------------------------------------------------------------

def test_unknown_design_raises_config_error():
    with pytest.raises(MlsynthConfigError):
        _sim(design="two-way-fixed-effects")


@pytest.mark.parametrize("kw", [
    {"n_units": 0}, {"n_periods": 0}, {"n_units": -5}, {"n_factors": 0},
])
def test_degenerate_dimensions_raise_config_error(kw):
    with pytest.raises(MlsynthConfigError):
        _sim(**kw)


def test_adoption_time_outside_the_panel_raises_config_error():
    with pytest.raises(MlsynthConfigError):
        _sim(n_periods=20, adoption_times=(25,))


def test_selection_stops_once_every_unit_has_adopted():
    """A saturating selection probability must not keep sweeping later times."""
    s = _sim(design="twfe", n_units=20, seed=11, theta0=40.0, theta1=0.0,
             adoption_times=(6, 10, 14))
    assert np.isfinite(s.trt).all()          # everyone adopted
    assert set(np.unique(s.trt)) == {6.0}    # all at the first offered time
