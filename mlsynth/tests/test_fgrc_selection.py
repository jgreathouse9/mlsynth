"""Model selection for fGRC: the Gap-statistic confidence rule.

Yamamoto & Hwang (2017) Algorithm 1 selects the number of clusters by a
self-consistency check, not by maximising a criterion. For fixed
(L_C, L_D, K) the method is fitted, the Gap statistic (Tibshirani, Walther &
Hastie 2001) is computed on the resulting component scores over a wider grid
of k, and the combination counts as confident only when

    argmax_k Gap(k | L_C, L_D, K) = K,

that is, the subspace fitted under the assumption of K clusters independently
looks like it holds K clusters. Among confident combinations the largest Gap
wins; if none is confident the rule relaxes to the t-th largest argmax.

Two layers are tested separately because they can fail independently: the Gap
statistic itself, on data whose answer is known by construction, and the
consistency rule on top of it, on the planted design of the paper's Section 5.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


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
    blob has no clusters, and the statistic has to say so rather than split it."""
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
    it is handed, so the fixture generates the curves rather than the
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
    assert sel.relaxation_level >= 1
    assert isinstance(sel.confident, tuple)


def test_selector_relaxes_when_nothing_is_confident():
    """With no cluster structure no candidate should be confident at t=1, and
    the rule must fall through to a higher t instead of failing."""
    from mlsynth.utils.clustersc_helpers.rpca.selection import select_fgrc_k
    rng = np.random.default_rng(5)
    X = rng.standard_normal((60, 10))
    sel = select_fgrc_k(X, k_candidates=(2, 3), c1=2, c2=0,
                        n_ref=10, seed=0, n_random=4, nstart=4)
    assert sel.selected_k in (2, 3)
    if not sel.confident:
        assert sel.relaxation_level > 1


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
