"""A ridge-augmented design carries a certificate the guard need not factorise.

``gram_reduction_is_safe`` asks whether ``sv_min / sv_max > tol`` on the design
the solve actually sees, and answers it with a full singular spectrum. Where
that design is ridge augmented -- ``[X_c; sqrt(r) I]``, which is what SDID's
unit-weight program hands it -- the answer is often settled without factorising
anything. The Gram is ``X_c' X_c + r I``, so

    lambda_min >= r                      (``X_c' X_c`` is positive semidefinite)
    lambda_max <= ||X_c||_F^2 + r        (the trace bounds the largest eigenvalue)

and therefore

    sv_min / sv_max >= sqrt(r / (||X_c||_F^2 + r)).

One Frobenius norm, no factorisation. The bound is one-sided in the safe
direction: clearing it proves the guard's answer is ``True``, and failing to
clear it proves nothing, so the SVD still runs. Decisions are therefore
identical by construction, not by tuning -- which is the acceptance bar for
touching a correctness guard at all.

On the placebo family of a 101x120 panel the bound is 7.21e-2 against a true
ratio of 7.43e-2, tight to three percent, and both clear the 1e-8 tolerance by
six orders of magnitude.

The time-weight program carries no ridge, so ``lambda_min >= 0`` is vacuous and
it keeps paying for its SVD. That is recorded here rather than papered over.
"""

import numpy as np
import pytest

from mlsynth.utils.bilevel.minnorm import (
    gram_reduction_is_safe,
    ridged_gram_reduction_is_safe,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _augment(design_c, ridge):
    """The matrix the shipped guard is handed today."""
    if ridge <= 0.0:
        return design_c
    J = design_c.shape[1]
    return np.vstack([design_c, np.sqrt(ridge) * np.eye(J)])


def _shipped(design_c, ridge, tol=1e-8):
    return gram_reduction_is_safe(_augment(design_c, ridge), tol)


def _true_ratio(design_c, ridge):
    aug = _augment(design_c, ridge)
    if aug.shape[0] < aug.shape[1] or min(aug.shape) == 0:
        return None
    sv = np.linalg.svd(aug, compute_uv=False)
    return float(sv[-1] / sv[0]) if sv[0] > 0 else 0.0


def _bound(design_c, ridge):
    if ridge <= 0.0:
        return 0.0
    return float(np.sqrt(ridge / (float(np.sum(design_c ** 2)) + ridge)))


def _centred(rng, m, J, scale=1.0):
    X = rng.standard_normal((m, J)) * scale
    return X - X.mean(axis=0)


class _SvdSpy:
    """Counts the SVDs the guard performs."""

    def __init__(self, monkeypatch):
        self.n = 0
        real = np.linalg.svd

        def counting(*args, **kwargs):
            self.n += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(np.linalg, "svd", counting)


# --------------------------------------------------------------------------- #
# 1. smoke
# --------------------------------------------------------------------------- #
class TestSmoke:
    def test_returns_a_bool(self):
        rng = np.random.default_rng(0)
        out = ridged_gram_reduction_is_safe(_centred(rng, 30, 12), 1.0)
        assert isinstance(out, bool)

    def test_a_well_conditioned_ridged_design_is_safe(self):
        rng = np.random.default_rng(1)
        assert ridged_gram_reduction_is_safe(_centred(rng, 40, 15), 0.5) is True


# --------------------------------------------------------------------------- #
# 2. the acceptance bar -- identical decisions, never merely close
# --------------------------------------------------------------------------- #
class TestDecisionsAreIdentical:
    @pytest.mark.parametrize("seed", range(6))
    def test_agrees_with_the_shipped_guard_over_a_fuzz(self, seed):
        rng = np.random.default_rng(seed)
        for _ in range(120):
            m = int(rng.integers(2, 60))
            J = int(rng.integers(1, 60))
            design = _centred(rng, m, J, scale=rng.uniform(1e-4, 1e4))
            if rng.random() < 0.3:                       # collinear columns
                k = max(1, J // 3)
                design[:, :k] = design[:, k:2 * k][:, :k] if 2 * k <= J \
                    else design[:, :k]
            if rng.random() < 0.2:                       # a near-null column
                design[:, 0] *= 1e-12
            ridge = float(rng.choice([0.0, 1e-16, 1e-12, 1e-6, 1e-2, 1.0, 1e3]))
            assert (ridge, m, J, ridged_gram_reduction_is_safe(design, ridge)) == \
                   (ridge, m, J, _shipped(design, ridge))

    @pytest.mark.parametrize("ridge", [1e-16, 1e-13, 1e-10, 1e-8, 1e-6])
    def test_agrees_where_the_bound_cannot_certify(self, ridge):
        """A ridge too small to certify must fall through to the SVD and give
        the SVD's answer -- including when that answer is ``False``.

        The design is scaled so ``||X||_F^2`` exceeds ``ridge / tol^2`` for every
        ridge listed, which is what puts the bound below the tolerance.
        """
        rng = np.random.default_rng(7)
        design = _centred(rng, 50, 40, scale=1e4)
        assert _bound(design, ridge) <= 1e-8, "fixture must not be certifiable"
        assert ridged_gram_reduction_is_safe(design, ridge) == _shipped(design, ridge)

    def test_agrees_on_a_rank_deficient_design(self):
        rng = np.random.default_rng(8)
        core = _centred(rng, 30, 10)
        design = np.hstack([core, core])                 # every column duplicated
        for ridge in (0.0, 1e-14, 1e-6, 1.0):
            assert ridged_gram_reduction_is_safe(design, ridge) == \
                   _shipped(design, ridge)


# --------------------------------------------------------------------------- #
# 3. the bound is one-sided -- it can only certify what is true
# --------------------------------------------------------------------------- #
class TestTheBoundIsSound:
    @pytest.mark.parametrize("seed", range(4))
    def test_bound_never_exceeds_the_true_ratio(self, seed):
        rng = np.random.default_rng(100 + seed)
        for _ in range(120):
            m = int(rng.integers(2, 60))
            J = int(rng.integers(1, 60))
            design = _centred(rng, m, J, scale=rng.uniform(1e-3, 1e3))
            ridge = float(rng.uniform(1e-14, 1e3))
            true = _true_ratio(design, ridge)
            assert true is not None
            assert _bound(design, ridge) <= true + 1e-15

    def test_certifying_implies_the_guard_says_yes(self):
        """The only direction that matters: whenever the bound clears the
        tolerance, the shipped guard's answer is ``True``."""
        rng = np.random.default_rng(11)
        certified = 0
        for _ in range(300):
            m = int(rng.integers(2, 50))
            J = int(rng.integers(1, 50))
            design = _centred(rng, m, J, scale=rng.uniform(1e-3, 1e3))
            ridge = float(rng.uniform(1e-12, 1e3))
            if _bound(design, ridge) > 1e-8:
                certified += 1
                assert _shipped(design, ridge) is True
        assert certified > 50, "fixture should certify often enough to mean something"


# --------------------------------------------------------------------------- #
# 4. the work it removes
# --------------------------------------------------------------------------- #
class TestWorkRemoved:
    def test_no_svd_when_the_certificate_fires(self, monkeypatch):
        rng = np.random.default_rng(12)
        design = _centred(rng, 96, 99)
        ridge = 225.0
        assert _bound(design, ridge) > 1e-8, "fixture must be certifiable"
        spy = _SvdSpy(monkeypatch)
        assert ridged_gram_reduction_is_safe(design, ridge) is True
        assert spy.n == 0

    def test_svd_still_runs_when_it_cannot_certify(self, monkeypatch):
        rng = np.random.default_rng(13)
        design = _centred(rng, 50, 40, scale=1e3)
        spy = _SvdSpy(monkeypatch)
        ridged_gram_reduction_is_safe(design, 1e-16)
        assert spy.n == 1

    def test_no_augmented_matrix_is_built_on_the_fast_path(self, monkeypatch):
        """The shipped path allocates an ``(m + J, J)`` copy per call. The
        certificate reads the design in place."""
        rng = np.random.default_rng(14)
        design = _centred(rng, 96, 99)
        calls = {"vstack": 0, "eye": 0}
        real_vstack, real_eye = np.vstack, np.eye
        monkeypatch.setattr(np, "vstack",
                            lambda *a, **k: (calls.__setitem__("vstack", calls["vstack"] + 1),
                                             real_vstack(*a, **k))[1])
        monkeypatch.setattr(np, "eye",
                            lambda *a, **k: (calls.__setitem__("eye", calls["eye"] + 1),
                                             real_eye(*a, **k))[1])
        ridged_gram_reduction_is_safe(design, 225.0)
        assert calls == {"vstack": 0, "eye": 0}


# --------------------------------------------------------------------------- #
# 5. edges
# --------------------------------------------------------------------------- #
class TestEdges:
    def test_zero_ridge_is_the_shipped_guard_exactly(self):
        rng = np.random.default_rng(15)
        for m, J in [(40, 12), (8, 39), (5, 5), (30, 30)]:
            design = _centred(rng, m, J)
            assert ridged_gram_reduction_is_safe(design, 0.0) == \
                   gram_reduction_is_safe(design)

    def test_single_column(self):
        rng = np.random.default_rng(16)
        design = _centred(rng, 20, 1)
        for ridge in (0.0, 1.0):
            assert ridged_gram_reduction_is_safe(design, ridge) == \
                   _shipped(design, ridge)

    def test_wide_design_with_ridge_is_still_decidable(self):
        """More columns than rows: the un-augmented design is rank deficient,
        and the ridge is the only thing separating the columns -- exactly the
        case the guard exists for."""
        rng = np.random.default_rng(17)
        design = _centred(rng, 10, 40)
        assert ridged_gram_reduction_is_safe(design, 1.0) == _shipped(design, 1.0)
        assert ridged_gram_reduction_is_safe(design, 1e-18) == \
               _shipped(design, 1e-18)

    def test_all_zero_design(self):
        design = np.zeros((12, 5))
        assert ridged_gram_reduction_is_safe(design, 1.0) is True   # r I is perfect
        assert ridged_gram_reduction_is_safe(design, 0.0) is False

    def test_extreme_scales(self):
        rng = np.random.default_rng(18)
        for scale in (1e-8, 1e8):
            design = _centred(rng, 30, 20, scale=scale)
            for ridge in (1e-6, 1.0, 1e6):
                assert ridged_gram_reduction_is_safe(design, ridge) == \
                       _shipped(design, ridge)

    def test_non_finite_design_is_not_certified(self):
        design = np.full((10, 4), np.nan)
        assert ridged_gram_reduction_is_safe(design, 1.0) is False

    def test_empty_design(self):
        assert ridged_gram_reduction_is_safe(np.zeros((0, 3)), 1.0) is False
        assert ridged_gram_reduction_is_safe(np.zeros((5, 0)), 1.0) is False


# --------------------------------------------------------------------------- #
# 6. failure -- a bad ridge is reported, not absorbed
# --------------------------------------------------------------------------- #
class TestFailures:
    @pytest.mark.parametrize("bad", [-1.0, -1e-12])
    def test_negative_ridge_raises(self, bad):
        rng = np.random.default_rng(19)
        with pytest.raises(ValueError, match="ridge"):
            ridged_gram_reduction_is_safe(_centred(rng, 20, 5), bad)

    def test_non_finite_ridge_raises(self):
        rng = np.random.default_rng(20)
        for bad in (np.nan, np.inf):
            with pytest.raises(ValueError, match="ridge"):
                ridged_gram_reduction_is_safe(_centred(rng, 20, 5), bad)

    def test_non_2d_design_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            ridged_gram_reduction_is_safe(np.zeros(10), 1.0)
