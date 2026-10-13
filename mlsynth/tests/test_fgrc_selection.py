"""Cluster-count diagnostics for fGRC: the Gap statistic and the paper's rule.

Yamamoto & Hwang (2017) Algorithm 1 selects the number of clusters by a
self-consistency check, not by maximising a criterion. For fixed
(L_C, L_D, K) the method is fitted, the Gap statistic (Tibshirani, Walther &
Hastie 2001) is computed on the resulting component scores over a wider grid
of k, and K is accepted only when

    argmax_k Gap(k | L_C, L_D, K) = K.

Three layers are tested separately because they fail independently: the Gap
statistic itself, on data whose answer is known by construction; the rule on
top of it, on the planted design of the paper's Section 5; and the rule on the
null of that same design, with the cluster separation removed.

The third layer is the one a root-cause analysis added, and it is the reason
the rule ships as a diagnostic and not as a configuration option. The
acceptance is not independent of the fit it checks, so a non-empty confident
set is not evidence of structure; and a Gap maximum at an end of the grid is a
property of the grid. The module under test reports both, and declines when
nothing is left. See its docstring for the measurements.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import (MlsynthConfigError, MlsynthDataError,
                                MlsynthEstimationError)


# --------------------------------------------------------------------------
# Layer 1: the Gap statistic, against answers known by construction
# --------------------------------------------------------------------------
def test_gap_recovers_three_separated_blobs():
    from mlsynth.utils.clustersc_helpers.rpca.selection import gap_statistic
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [12.0, 0.0], [6.0, 12.0]])
    F = np.vstack([c + rng.standard_normal((40, 2)) for c in centres])
    gaps, _ = gap_statistic(F, range(1, 7), n_ref=25, seed=0)
    assert int(np.argmax(gaps)) + 1 == 3


def test_gap_reports_one_cluster_when_there_is_no_structure():
    """The case that matters for using this as a diagnostic: a single Gaussian
    blob has no clusters, and the statistic has to say so instead of splitting it."""
    from mlsynth.utils.clustersc_helpers.rpca.selection import gap_statistic
    rng = np.random.default_rng(1)
    F = rng.standard_normal((120, 2))
    gaps, _ = gap_statistic(F, range(1, 7), n_ref=25, seed=0)
    assert int(np.argmax(gaps)) + 1 == 1


def test_gap_is_deterministic_under_a_fixed_seed():
    from mlsynth.utils.clustersc_helpers.rpca.selection import gap_statistic
    rng = np.random.default_rng(2)
    F = rng.standard_normal((60, 2))
    a, _ = gap_statistic(F, range(1, 5), n_ref=10, seed=7)
    b, _ = gap_statistic(F, range(1, 5), n_ref=10, seed=7)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_gap_rejects_a_k_above_the_sample_size():
    from mlsynth.utils.clustersc_helpers.rpca.selection import gap_statistic
    with pytest.raises(MlsynthConfigError):
        gap_statistic(np.random.default_rng(0).standard_normal((5, 2)),
                      range(1, 9), n_ref=5, seed=0)


def test_gap_rejects_a_degenerate_score_matrix():
    from mlsynth.utils.clustersc_helpers.rpca.selection import gap_statistic
    with pytest.raises(MlsynthDataError):
        gap_statistic(np.zeros((30, 2)), range(1, 4), n_ref=5, seed=0)


# --------------------------------------------------------------------------
# Layer 2: the consistency rule, on the paper's Section 5 planted design
# --------------------------------------------------------------------------
def _planted(n_per=50, T=40, lc=2, sep=7.0, noise=0.35, seed=0):
    """Section 5: three clusters in a two-dimensional subspace, as curves.

    The paper's structure lives in coefficient space -- F is N x L_C with a
    three-group mean structure and identity covariance, and the observed
    functions follow from a basis expansion of it. fGRC basis-expands whatever
    it is handed, so the fixture generates the curves, not the
    coefficients: each unit is a smooth combination of two basis functions
    with cluster-structured loadings, plus a disturbing direction independent
    of the grouping and observation noise.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, T)
    phi = np.vstack([np.sin(2 * np.pi * t), np.cos(np.pi * t)])       # lc x T
    disturb = np.sin(4 * np.pi * t + 0.7)
    centres = np.array([[0.0, 0.0], [sep, 0.0], [sep / 2, sep]])
    F = np.vstack([c + rng.standard_normal((n_per, lc)) for c in centres])
    E = rng.standard_normal((F.shape[0], 1)) * 3.0                    # independent of clusters
    X = F @ phi + E @ disturb[None, :] + rng.standard_normal((F.shape[0], T)) * noise
    truth = np.repeat([1, 2, 3], n_per)
    return X, truth


def test_selector_recovers_the_planted_number_of_clusters():
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    X, truth = _planted()
    sel = select_fgrc_k(X, k_candidates=(2, 3, 4), c1=2, c2=0,
                        n_ref=15, seed=0, n_random=6, nstart=6)
    assert sel.selected_k == 3
    assert 3 in sel.confident


def test_selector_reports_the_full_diagnostic():
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    X, _ = _planted()
    sel = select_fgrc_k(X, k_candidates=(2, 3), c1=2, c2=0,
                        n_ref=10, seed=0, n_random=4, nstart=4)
    assert set(sel.gaps) == {2, 3}                     # per-candidate Gap curves
    assert all(len(v) == len(sel.k_eval) for v in sel.gaps.values())
    assert set(sel.one_se) == {2, 3}                   # the 1-SE reading of each
    assert all(k in sel.k_eval for k in sel.one_se.values())
    assert isinstance(sel.confident, tuple)
    assert isinstance(sel.boundary, tuple)


def test_a_verdict_is_never_a_candidate_whose_curve_peaked_at_an_edge():
    """The invariant the Basque panel violated: a candidate ranked inside a Gap
    curve that peaks at an end of the evaluation grid is ranked against the
    grid, not against the data, so it can never be the answer."""
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    rng = np.random.default_rng(5)
    X = rng.standard_normal((60, 10))
    sel = select_fgrc_k(X, k_candidates=(2, 3), c1=2, c2=0,
                        n_ref=10, seed=0, n_random=4, nstart=4)
    assert sel.selected_k is None or sel.selected_k not in sel.boundary
    assert not set(sel.confident) & set(sel.boundary)
    if sel.selected_k is None:
        assert sel.confident == () and sel.relaxation_level == 0


# --------------------------------------------------------------------------
# Layer 3: what the rule does when there is nothing to find
# --------------------------------------------------------------------------
def _null(n_per=6, seed=0):
    """The planted design with the cluster separation removed: smooth curves
    from the same generator, sharing a disturbing direction, in one group."""
    X, _ = _planted(n_per=n_per, sep=0.0, seed=seed)
    return X


@pytest.mark.parametrize("seed", [200, 201, 202])
def test_the_rule_abstains_when_every_curve_peaks_at_an_edge(seed):
    """On a structureless panel every candidate's Gap curve peaks at k = 1, so
    no ranking inside any of them is evidence. The rule must decline instead of
    relaxing to the t-th rank and returning the largest candidate."""
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    sel = select_fgrc_k(_null(seed=seed), k_candidates=(2, 3, 4), c1=2, c2=0,
                        n_ref=15, seed=0, n_random=10, nstart=10)
    assert sel.boundary == (2, 3, 4)
    assert sel.selected_k is None
    assert sel.confident == ()
    assert sel.relaxation_level == 0
    assert sel.gaps and sel.gap_se                     # the evidence is still reported


@pytest.mark.parametrize("seed", [200, 201, 202])
def test_the_one_se_reading_does_not_manufacture_structure(seed):
    """Tibshirani's own safeguard on the same curves. Where the argmax rule
    calls a structureless panel confident, the 1-SE reading says one cluster."""
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    for c2 in (0, 1):
        sel = select_fgrc_k(_null(seed=seed), k_candidates=(2, 3, 4), c1=2, c2=c2,
                            n_ref=15, seed=0, n_random=10, nstart=10)
        assert set(sel.one_se.values()) == {1}


def test_the_one_se_reading_finds_the_planted_clusters():
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    X, _ = _planted(n_per=10)
    sel = select_fgrc_k(X, k_candidates=(2, 3, 4), c1=2, c2=1,
                        n_ref=15, seed=0, n_random=10, nstart=10)
    assert sel.one_se[3] == 3
    assert sel.one_se[4] == 3


def test_the_default_grid_is_pinned_to_the_candidate_set():
    """The grid is the instrument's aperture, and the root-cause analysis found
    the evidence the rule reads off moves with it: on the Basque panel the Gap
    maximum tracked the largest k in the grid out to 16 clusters on 17 units.
    Two past the largest candidate is enough to reject one and no more."""
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    X, _ = _planted(n_per=10)
    sel = select_fgrc_k(X, k_candidates=(2, 3, 4), c1=2, c2=0,
                        n_ref=5, seed=0, n_random=2, nstart=2)
    assert sel.k_eval == (1, 2, 3, 4, 5, 6)
    sel = select_fgrc_k(X, k_candidates=(2, 3), c1=2, c2=0,
                        n_ref=5, seed=0, n_random=2, nstart=2)
    assert sel.k_eval == (1, 2, 3, 4, 5)


def test_the_grid_must_extend_past_the_candidates():
    """The check asks whether a candidate is the argmax of its own curve. A grid
    that stops at the largest candidate cannot answer it, because the largest
    candidate can then only win at the edge."""
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    X, _ = _planted(n_per=10)
    with pytest.raises(MlsynthConfigError):
        select_fgrc_k(X, k_candidates=(2, 3, 4), c1=2, c2=0, k_eval=range(1, 5),
                      n_ref=5, seed=0, n_random=2, nstart=2)


def test_a_panel_too_small_for_the_grid_is_refused_not_truncated():
    """Trimming k_eval to fit the panel used to leave max(k_eval) at the largest
    candidate, which silently removed the check's power: on five units of noise
    the rule reported two candidates as confident."""
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    X = np.random.default_rng(3).standard_normal((5, 30))
    with pytest.raises(MlsynthConfigError):
        select_fgrc_k(X, k_candidates=(2, 3, 4), c1=2, c2=0,
                      n_ref=5, seed=0, n_random=2, nstart=2)


@pytest.mark.parametrize("kw", [
    dict(k_candidates=()),
    dict(k_candidates=(1,)),
    dict(k_candidates=(2,), c1=0),
])
def test_selector_rejects_invalid_configuration(kw):
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    X, _ = _planted(n_per=10)
    base = dict(c1=2, c2=0, n_ref=5, seed=0, n_random=2, nstart=2)
    with pytest.raises(MlsynthConfigError):
        select_fgrc_k(X, **{**base, **kw})


# --------------------------------------------------------------------------
# Layer 4: every way the caller can be wrong, and how it is told
# --------------------------------------------------------------------------
@pytest.mark.parametrize("args", [
    (np.zeros(8), range(1, 4)),                        # not a matrix
    (np.zeros((1, 2)), range(1, 4)),                   # one unit
    (np.zeros((30, 2)) + 1.0, []),                     # no k to evaluate
    (np.zeros((30, 2)) + 1.0, [0, 1]),                 # k below one
])
def test_gap_rejects_malformed_arguments(args):
    from mlsynth.utils.clustersc_helpers.rpca.selection import gap_statistic
    with pytest.raises(MlsynthConfigError):
        gap_statistic(args[0], args[1], n_ref=3, seed=0)


def test_gap_rejects_a_score_matrix_with_holes_in_it():
    from mlsynth.utils.clustersc_helpers.rpca.selection import gap_statistic
    F = np.random.default_rng(0).standard_normal((30, 2))
    F[7, 1] = np.nan
    with pytest.raises(MlsynthDataError):
        gap_statistic(F, range(1, 4), n_ref=3, seed=0)


def test_one_se_falls_through_to_the_largest_k_when_no_k_qualifies():
    """A Gap curve that keeps climbing by more than a standard error has no
    smallest adequate k, and the reading is then the end of the grid -- which
    is the boundary condition the caller is told about separately."""
    from mlsynth.utils.clustersc_helpers.rpca.selection import one_se_k
    gap = np.array([0.0, 1.0, 2.0, 3.0])
    se = np.zeros(4)
    assert one_se_k(gap, se, [1, 2, 3, 4]) == 4


@pytest.mark.parametrize("kw", [
    dict(trajectories=np.zeros(8)),                    # not a panel
    dict(c2=-1),                                       # negative disturbing dimension
    dict(c1=30, n_knots=4),                            # wider than the B-spline basis
])
def test_selector_rejects_more_invalid_configuration(kw):
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    X = kw.pop("trajectories", None)
    if X is None:
        X = np.random.default_rng(0).standard_normal((12, 30))
    base = dict(c1=2, c2=0, n_ref=3, seed=0, n_random=2, nstart=2)
    with pytest.raises(MlsynthConfigError):
        select_fgrc_k(X, k_candidates=(2, 3), **{**base, **kw})


def test_a_panel_no_candidate_can_be_fitted_on_says_so():
    """Identical trajectories centre to a zero basis matrix, so every restart
    of every candidate hits a degenerate partition. The failure is reported;
    it is not a silent fall-through to some default k."""
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    X = np.tile(np.linspace(0.0, 1.0, 30), (12, 1))
    with pytest.raises(MlsynthEstimationError):
        select_fgrc_k(X, k_candidates=(2, 3), c1=2, c2=0,
                      n_ref=3, seed=0, n_random=2, nstart=2)
