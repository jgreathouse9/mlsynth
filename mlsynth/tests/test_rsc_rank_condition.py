"""The rank condition behind synthetic control's extrapolation (RSC Thm 6).

Amjad, Shah & Shen (2018) assume the treated unit's signal is a linear
combination of the donors' signals *in the pre-intervention window*,
``M_1^- = (M^-)^T beta*``. Whether that relation still holds after the
intervention -- which is the entire basis for using it to forecast a
counterfactual -- is Theorem 6: it does, provided
``rank(M^-) == rank(M)``. The authors observe that the point "has been amiss
in the literature, potentially implicitly believed or assumed" since Abadie
and Gardeazabal (2003).

These test the generator that makes both sides of that condition
constructible: a factor panel whose factors are all active throughout
(rank preserved), and one where a factor lies dormant until the
intervention (rank deficient), so a relation fitted on the pre-period is
under-determined and need not extrapolate.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.clustersc_helpers.simulation import (
    RSCPanel,
    simulate_rank_shift_panel,
)


def _pre_solution(panel: RSCPanel) -> np.ndarray:
    """Some beta satisfying the pre-period relation, via least squares.

    Theorem 6 is about whether *a* pre-period solution extrapolates, so the
    tests take the one an estimator would land on.
    """
    M, T0 = panel.means, panel.T0
    donors_pre = M[1:, :T0].T                      # (T0, N-1)
    return np.linalg.lstsq(donors_pre, M[0, :T0], rcond=None)[0]


# --------------------------------------------------------------------------
# Smoke and structure
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dormant", [False, True])
def test_smoke(dormant):
    p = simulate_rank_shift_panel(dormant_factor=dormant, N=12, T=60, T0=40,
                                  n_factors=3, noise=0.5, seed=0)
    assert isinstance(p, RSCPanel)
    assert p.means.shape == (12, 60)
    assert p.observed.shape == (12, 60)
    assert p.T0 == 40
    assert np.all(np.isfinite(p.means))


def test_observed_is_the_mean_plus_noise():
    p = simulate_rank_shift_panel(dormant_factor=False, N=12, T=400, T0=300,
                                  n_factors=3, noise=0.7, seed=1)
    resid = p.observed - p.means
    assert resid.std() == pytest.approx(0.7, rel=0.05)
    assert abs(resid.mean()) < 0.05


def test_noise_zero_gives_the_mean_exactly():
    p = simulate_rank_shift_panel(dormant_factor=False, N=10, T=50, T0=30,
                                  n_factors=3, noise=0.0, seed=2)
    assert np.array_equal(p.observed, p.means)


# --------------------------------------------------------------------------
# The rank condition itself
# --------------------------------------------------------------------------

def test_all_factors_active_preserves_the_rank():
    p = simulate_rank_shift_panel(dormant_factor=False, N=12, T=60, T0=40,
                                  n_factors=3, noise=0.0, seed=3)
    assert np.linalg.matrix_rank(p.means[:, :p.T0]) == 3
    assert np.linalg.matrix_rank(p.means) == 3


def test_a_dormant_factor_drops_the_pre_period_rank():
    p = simulate_rank_shift_panel(dormant_factor=True, N=12, T=60, T0=40,
                                  n_factors=3, noise=0.0, seed=3)
    assert np.linalg.matrix_rank(p.means[:, :p.T0]) == 2
    assert np.linalg.matrix_rank(p.means) == 3


@pytest.mark.parametrize("dormant", [False, True])
def test_the_pre_period_relation_holds_by_construction(dormant):
    """Equation (6) is satisfied either way -- that is the premise, not the
    conclusion. Theorem 6 asks what happens *after* the intervention."""
    p = simulate_rank_shift_panel(dormant_factor=dormant, N=12, T=60, T0=40,
                                  n_factors=3, noise=0.0, seed=4)
    beta = _pre_solution(p)
    fitted = p.means[1:, :p.T0].T @ beta
    assert np.abs(fitted - p.means[0, :p.T0]).max() < 1e-8


def test_rank_preserved_means_the_relation_extrapolates():
    """Theorem 6's conclusion, on the side where its hypothesis holds."""
    p = simulate_rank_shift_panel(dormant_factor=False, N=12, T=60, T0=40,
                                  n_factors=3, noise=0.0, seed=5)
    beta = _pre_solution(p)
    post = p.means[1:, p.T0:].T @ beta
    assert np.abs(post - p.means[0, p.T0:]).max() < 1e-8


def test_rank_deficient_means_it_need_not():
    """And on the side where the hypothesis fails.

    The dormant factor leaves the pre-period system under-determined, so a
    pre-period solution is free in the direction the factor will occupy.
    The generator picks such a solution deliberately, which is what an
    estimator fitting only the pre-period can also do.
    """
    p = simulate_rank_shift_panel(dormant_factor=True, N=12, T=60, T0=40,
                                  n_factors=3, noise=0.0, seed=5)
    beta = _pre_solution(p)
    post = p.means[1:, p.T0:].T @ beta
    assert np.abs(post - p.means[0, p.T0:]).max() > 0.1


def test_dormant_factor_is_zero_before_and_active_after():
    p = simulate_rank_shift_panel(dormant_factor=True, N=12, T=60, T0=40,
                                  n_factors=3, noise=0.0, seed=6)
    assert np.abs(p.factors[:p.T0, -1]).max() == 0.0
    assert np.abs(p.factors[p.T0:, -1]).std() > 0.1


def test_treated_loading_lies_in_the_donor_span():
    """The treated unit is a genuine combination of the donors' signals, so
    any failure below is the rank condition and not a missing relation."""
    p = simulate_rank_shift_panel(dormant_factor=True, N=12, T=60, T0=40,
                                  n_factors=3, noise=0.0, seed=7)
    resid = np.linalg.lstsq(p.loadings[1:].T, p.loadings[0], rcond=None)[1]
    assert resid.size == 0 or float(resid.sum()) < 1e-16


# --------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(N=2, T=60, T0=40, n_factors=3),       # fewer donors than factors
    dict(N=12, T=60, T0=2, n_factors=3),       # pre-period shorter than rank
    dict(N=12, T=40, T0=40, n_factors=3),      # no post-period
    dict(N=12, T=60, T0=40, n_factors=1),      # a dormant factor needs r >= 2
])
def test_degenerate_shapes_raise(kwargs):
    with pytest.raises(ValueError):
        simulate_rank_shift_panel(dormant_factor=True, noise=0.0, seed=0,
                                  **kwargs)


def test_negative_noise_raises():
    with pytest.raises(ValueError, match="noise"):
        simulate_rank_shift_panel(dormant_factor=False, N=12, T=60, T0=40,
                                  n_factors=3, noise=-1.0, seed=0)
