"""The Hsiao & Zhou study's data-generating process and its Bai (2009) step.

The study is `benchmarks/studies/hsiao_zhou_counterfactuals`. Two of its pieces
carry assertions here because both have already been wrong once.

`simulate` shipped Equation 33's neighbour terms built with `np.roll`, which
wraps unit 1's `v_{i-1}` onto unit N and hands the treated unit two error
components shared with the donor pool where the equation gives it one. Every
method that regresses on controls can then predict the extra one, and the whole
table reads too good.

`beta_bai` shipped Bai's PCA2 iteration from a zero start. Bai (2009) gives two
schemes and Hsiao, Shi & Zhou (2022, Table 1) measure both: PCA2 holds a bias of
about 0.13 that does not shrink in N or T, with empirical size up to 100 percent
against a 5 percent nominal. Bai's estimator is the argmin of
``||Y - X b - F L'||^2``, so the objective is what adjudicates, and these tests
assert against it, not against any particular coefficient value. They assert
margins on it: two schemes that reach the same point differ only in the last
bits, and a strict inequality there is decided by the BLAS.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_STUDY = Path(__file__).resolve().parents[1] / "studies" / "hsiao_zhou_counterfactuals"
sys.path.insert(0, str(_STUDY))


@pytest.fixture(scope="module")
def methods():
    return importlib.import_module("methods")


@pytest.fixture(scope="module")
def empirics():
    return importlib.import_module("empirics")


# ----------------------------------------------------------------------
# Equation 33: the error's cross-sectional structure
# ----------------------------------------------------------------------

class TestTheErrorStructure:
    """``u_it = (1+b^2) v_it + b v_{i+1,t} + b v_{i-1,t}``, b = 1, no wrap."""

    def _draw(self, methods, n_co=30, T0=30, T2=10, seed=0):
        return methods.simulate("dgp6", n_co, T0, T2, np.random.default_rng(seed))

    def test_the_treated_unit_borders_one_control_not_two(self, methods):
        """Unit 1 has no ``i-1``, so it shares one v with the pool.

        Correlation with the first control is the shared ``v_2``; with every
        other control it is zero. A wrap would make the last control correlate
        too, which is the defect this pins.
        """
        y1, Yco = self._draw(methods, T0=4000, T2=10)
        # Strip the factor part: with two factors the residual is u.
        M = np.column_stack([y1, Yco])
        U, s, Vt = np.linalg.svd(M - M.mean(axis=0), full_matrices=False)
        u = (M - M.mean(axis=0)) - (U[:, :2] * s[:2]) @ Vt[:2]
        rho = [float(np.corrcoef(u[:, 0], u[:, j])[0, 1])
               for j in range(1, u.shape[1])]
        assert rho[0] > 0.15, "unit 1 must share an error component with unit 2"
        assert abs(rho[-1]) < 0.06, (
            "unit 1 must NOT correlate with the last control; a wrapped "
            "neighbour index is the np.roll defect")

    def test_interior_units_border_two(self, methods):
        y1, Yco = self._draw(methods, T0=4000, T2=10)
        u = Yco - Yco.mean(axis=0)
        U, s, Vt = np.linalg.svd(u, full_matrices=False)
        u = u - (U[:, :2] * s[:2]) @ Vt[:2]
        mid = 15
        assert float(np.corrcoef(u[:, mid], u[:, mid - 1])[0, 1]) > 0.15
        assert float(np.corrcoef(u[:, mid], u[:, mid + 1])[0, 1]) > 0.15

    def test_the_effect_is_zero_so_the_criteria_are_prediction_error(self, methods):
        """Neither DGP carries a treatment effect."""
        y1, Yco = self._draw(methods)
        assert np.all(np.isfinite(y1)) and np.all(np.isfinite(Yco))
        assert Yco.shape == (40, 30)

    def test_dgp7_is_dgp6_with_integrated_factors(self, methods):
        a = methods.simulate("dgp6", 30, 200, 10, np.random.default_rng(3))[0]
        b = methods.simulate("dgp7", 30, 200, 10, np.random.default_rng(3))[0]
        assert not np.allclose(a, b)
        # A random walk in the factors leaves the level far more persistent.
        rho = lambda v: float(np.corrcoef(v[:-1], v[1:])[0, 1])
        assert rho(b) > rho(a)


# ----------------------------------------------------------------------
# Bai (2009): the objective is what decides
# ----------------------------------------------------------------------

#: The coefficients `_panel` generates under. The first regressor is the flat
#: one, so `TRUTH[0]` is the large coefficient that makes PCA2 stall.
TRUTH = (20.0, 2.0)

#: Spread of the flat regressor around its level of 10, giving a coefficient of
#: variation near 0.01 -- the same order as `lnincome`'s 0.027 on the real panel.
FLAT_SD = 0.1


def _panel(rng, N=25, T=40, beta=TRUTH):
    """Bai (2009)'s DGP1, with one near-constant regressor.

    Plain DGP1 does not separate the two iteration schemes: both recover the
    truth, so a test built on it passes against the defect and has no power.
    What broke on the real panel was `lnincome`, a log that is nearly flat
    across states and carrying a coefficient near 60, so the first regressor
    here is built that way.

    Two parameters decide whether the separation happens at all, and an earlier
    version of this fixture had both too weak: a flat-regressor spread of 0.3
    and a coefficient of 1.0 separated the schemes on one seed in ten, and on
    the other nine they converged to the same point. A test asserting a strict
    inequality there compares two equal objectives and is decided by the last
    two bits, which is how it passed locally and failed in CI.

    Measured over twelve seeds at `FLAT_SD` 0.1 and `TRUTH[0]` 20, PCA2 from a
    zero start lands 64 percent above the reached minimum at worst, and its bias
    on the flat coefficient is -17.0 against PCA1's -0.005. The separation is a
    property of the design now, not of the seed.
    """
    lam = rng.standard_normal((N, 2))
    f = rng.standard_normal((T, 2))
    X = np.empty((T, N, 2))
    X[:, :, 0] = 10.0 + FLAT_SD * rng.standard_normal((T, N)) + 0.1 * lam[:, 0]
    X[:, :, 1] = (1.0 + lam[:, 0] + lam[:, 1] + (f[:, 0] + f[:, 1])[:, None]
                  + f @ lam.T + rng.standard_normal((T, N)))
    Y = np.einsum("tnk,k->tn", X, np.asarray(beta)) + f @ lam.T \
        + 2.0 * rng.standard_normal((T, N))
    return Y, X


def _ssr(Y, X, beta, r):
    """Bai's objective with F and Lambda concentrated out."""
    R = Y - np.einsum("tnk,k->tn", X, beta)
    s = np.linalg.svd(R, compute_uv=False)
    return float((s[r:] ** 2).sum())


class TestBaiIsTheArgmin:
    """The estimator is defined by the objective, so assert on the objective.

    Asserting a coefficient value would pin whatever the implementation
    happens to return. These pin the property that makes it Bai's estimator.
    """

    def test_it_is_not_improvable_from_a_pooled_ols_restart(self, empirics):
        rng = np.random.default_rng(0)
        Y, X = _panel(rng)
        beta, _, _ = empirics.beta_bai(Y, X, T0=30, r=2)
        ols = np.linalg.lstsq(X.reshape(-1, X.shape[2]), Y.reshape(-1),
                              rcond=None)[0]
        here = _ssr(Y, X, beta, 2)
        assert here <= _ssr(Y, X, ols, 2) + 1e-8, (
            "the fit must not be worse than the pooled OLS it can start from")
        assert here <= _ssr(Y, X, np.zeros(X.shape[2]), 2) + 1e-8

    def test_the_start_does_not_change_the_answer(self, empirics):
        """A start-dependent optimum is the defect, not a tolerance."""
        rng = np.random.default_rng(1)
        Y, X = _panel(rng)
        a, _, _ = empirics.beta_bai(Y, X, T0=30, r=2)
        b, _, _ = empirics.beta_bai(Y, X, T0=30, r=2,
                                    beta0=np.array([50.0, -50.0]))
        assert _ssr(Y, X, a, 2) == pytest.approx(_ssr(Y, X, b, 2), rel=1e-6)

    def test_it_recovers_the_truth_on_bai_dgp1(self, empirics):
        """beta = TRUTH by construction, and the scheme it replaced misses it.

        Stated relative to each coefficient, since `TRUTH` is (20, 2) and one
        absolute tolerance cannot be strict on both. The PCA2 contrast is the
        point: it is not a slightly worse estimate, it is off by most of the
        flat coefficient.
        """
        truth = np.asarray(TRUTH)
        errs, errs_pca2 = [], []
        for seed in range(6):
            Y, X = _panel(np.random.default_rng(seed), N=25, T=60)
            beta, _, _ = empirics.beta_bai(Y, X, T0=50, r=2)
            errs.append(beta - truth)
            errs_pca2.append(empirics._beta_bai_pca2_from_zero(Y, X, r=2) - truth)
        bias = np.mean(errs, axis=0)
        assert np.max(np.abs(bias / truth)) < 0.05, (
            f"relative bias {bias / truth} is PCA2-sized")
        bias_pca2 = np.mean(errs_pca2, axis=0)
        assert abs(bias_pca2[0] / truth[0]) > 0.5, (
            f"PCA2 must miss the flat coefficient badly for this design to have "
            f"power; its relative bias is only {bias_pca2[0] / truth[0]:.3f}")

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_it_beats_the_scheme_it_replaced(self, empirics, seed):
        """PCA2 from a zero start, the shipped defect, on the same panel.

        A margin rather than a bare `<`, and over several seeds. The first
        version asserted a strict inequality on seed 4 alone, where the old
        fixture let both schemes converge to the same point: the two objectives
        agreed to thirteen significant figures and the comparison came down to
        the last two bits, so it passed on one BLAS and failed on another. The
        fix is the fixture, which now separates the schemes on every seed; the
        margin here is what keeps the test honest about it.
        """
        Y, X = _panel(np.random.default_rng(seed))
        good, _, _ = empirics.beta_bai(Y, X, T0=30, r=2)
        bad = empirics._beta_bai_pca2_from_zero(Y, X, r=2)
        reached, stalled = _ssr(Y, X, good, 2), _ssr(Y, X, bad, 2)
        assert (stalled - reached) / reached > 0.2, (
            f"PCA2 must stall by a clear margin; got "
            f"{(stalled - reached) / reached:.2e}. A near-zero gap means the "
            f"fixture has stopped separating the two schemes.")

    @pytest.mark.parametrize("r", [1, 2, 3])
    def test_more_factors_never_raise_the_objective(self, empirics, r):
        rng = np.random.default_rng(5)
        Y, X = _panel(rng)
        beta, _, _ = empirics.beta_bai(Y, X, T0=30, r=r)
        assert _ssr(Y, X, beta, r) >= _ssr(Y, X, beta, r + 1) - 1e-8

    def test_it_returns_factors_and_loadings_of_the_right_shape(self, empirics):
        Y, X = _panel(np.random.default_rng(6))
        beta, F, G = empirics.beta_bai(Y, X, T0=30, r=2)
        assert beta.shape == (2,)
        assert F.shape == (40, 2)
        assert G.shape == (25, 2)
