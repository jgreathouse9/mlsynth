"""NSC's dimensionless penalties mean what Tian (2023) means by them.

``a`` and ``b`` enter NSC as dimensionless numbers in ``[0, 1]``, and what they
buy is fixed by a spectrum: ``b`` indexes the eigenvalues of the donor Gram and
``a`` those of the same Gram once ``b`` has been added to its diagonal. Which
Gram, exactly, is the whole question. Tian's ``NSC.R`` poses the program on the
stacked design ``rbind(ZJ, -ZJ)`` -- the split-variable encoding of the L1 term
-- and takes eigenvalues of ``ZJstar ZJstar'``, a ``(2J, 2J)`` matrix. mlsynth
encodes the same L1 term with auxiliary variables ``u >= |w|`` and took them
from ``ZJ ZJ'``, ``(J, J)``.

Both encodings have the same feasible set and the same optimum for a *given*
``(a, b)``, which is why the QP was never the problem. They do not have the same
spectrum, and the spectrum is what defines ``(a, b)``. Reading Table 2 and
passing ``a=0.3, b=0.7`` therefore solved a different program: on Proposition 99
that is an ATT of -19.13 against the author's -20.59 (#463).

The scaling tests start red and are the assertion the analysis found missing:
they pin the map from ``(a*, b*)`` to raw multipliers directly, against a
transcription of ``NSC.R``, instead of inferring it from an ATT with a wide
tolerance around it. They are checked across ranks on purpose. On Proposition 99
the error is exactly a factor of two in both multipliers, so a fix that doubled
them would pass there and stay wrong: the doubling is a coincidence of the
low-rank regime (19 of 38), and once the rank approaches ``J`` the ``a``
multiplier is out by 1.36 to 1.63 instead.

The QP tests pass before the change and must pass after it. Nothing about the
solver moves; only what it is asked to solve.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from mlsynth.utils.nsc_helpers.optimization import (
    design_eigenvalues,
    scale_a,
    scale_b,
    solve_nsc_weights,
)

_REF_DIR = (pathlib.Path(__file__).resolve().parents[2]
            / "benchmarks" / "reference" / "nsc_prop99")
_BASEDATA = pathlib.Path(__file__).resolve().parents[2] / "basedata"


# --------------------------------------------------------------------------- #
# the oracle: NSC.R's scaling, transcribed
# --------------------------------------------------------------------------- #
def nsc_r_scaling(ZJ: np.ndarray, a_star: float, b_star: float):
    """``fn_weights``'s penalty scaling from ``NSC.R``, line for line.

    R, with ``Negativity = TRUE``::

        ZJstar = rbind(ZJ,-ZJ)
        Dmat0  = ZJstar %*% t(ZJstar)
        eg0    = eigen(Dmat0)$values; eg0 = eg0[eg0>10^-7]
        if (b>0) {b = sort(eg0)[ceiling(b*length(eg0))]*b}
        Dmat   = Dmat0 + (b+10^-8)*diag(2*J)
        eg1    = eigen(Dmat)$values;  eg1 = eg1[eg1>10^-7]
        if (a>0) {a = sort(eg1)[ceiling(a*length(eg1))]*a}

    ``sort`` is ascending and R indexes from one.
    """
    J = ZJ.shape[0]
    ZJstar = np.vstack([ZJ, -ZJ])
    Dmat0 = ZJstar @ ZJstar.T
    eg0 = np.linalg.eigvalsh(0.5 * (Dmat0 + Dmat0.T))
    eg0 = np.sort(eg0[eg0 > 1e-7])
    b = 0.0
    if b_star > 0 and eg0.size:
        b = float(eg0[int(np.ceil(b_star * eg0.size)) - 1] * b_star)
    Dmat = Dmat0 + (b + 1e-8) * np.eye(2 * J)
    eg1 = np.linalg.eigvalsh(0.5 * (Dmat + Dmat.T))
    eg1 = np.sort(eg1[eg1 > 1e-7])
    a = 0.0
    if a_star > 0 and eg1.size:
        a = float(eg1[int(np.ceil(a_star * eg1.size)) - 1] * a_star)
    return a, b


def donor_design(J, p, seed=0, rank=None):
    """A standardised donor block, as ``NSC.R`` standardises ``Z``."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((J, p))
    if rank is not None:
        U = rng.standard_normal((J, rank))
        V = rng.standard_normal((rank, p))
        Z = U @ V
    Z = Z - Z.mean(axis=0)
    sd = Z.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return Z / sd


def treated_row(Z, seed=0, noise=0.1):
    """A treated unit shaped like one of the donors.

    Standardisation is across units, so a treated row is on the donors' scale
    and near their span -- not an independent draw. (Standardising a single row
    by its own column SDs divides by zero, which is how this helper first went
    wrong: the QP was handed NaN and raised, correctly.)
    """
    rng = np.random.default_rng(seed)
    k = max(1, min(3, Z.shape[0]))
    return Z[:k].mean(axis=0) + noise * rng.standard_normal(Z.shape[1])


# shapes chosen to straddle the rank regimes, since the size of the error
# depends on the rank and not only on J
SHAPES = [
    ("prop99-like", 38, 19),
    ("wide-pool-short-pre", 100, 12),
    ("square", 60, 60),
    ("more-pre-than-donors", 20, 50),
    ("near-full-rank", 38, 40),
    ("tiny", 10, 3),
    ("two-donors", 2, 5),
]
STARS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]


# =========================================================================== #
# the scaling contract -- these start red
# =========================================================================== #
class TestScalingMatchesTheReference:
    @pytest.mark.parametrize("name,J,p", SHAPES)
    @pytest.mark.parametrize("b_star", STARS)
    def test_b_matches(self, name, J, p, b_star):
        Z = donor_design(J, p, seed=hash(name) % 1000)
        _, b_ref = nsc_r_scaling(Z, 0.3, b_star)
        b_got = scale_b(b_star, design_eigenvalues(Z))
        assert b_got == pytest.approx(b_ref, rel=1e-9, abs=1e-12), (
            f"{name}: b*={b_star} -> {b_got}, reference {b_ref}")

    @pytest.mark.parametrize("name,J,p", SHAPES)
    @pytest.mark.parametrize("a_star", STARS)
    def test_a_matches(self, name, J, p, a_star):
        Z = donor_design(J, p, seed=hash(name) % 1000)
        a_ref, b_ref = nsc_r_scaling(Z, a_star, 0.7)
        b_got = scale_b(0.7, design_eigenvalues(Z))
        a_got = scale_a(a_star, Z, b_got)
        assert a_got == pytest.approx(a_ref, rel=1e-9, abs=1e-12), (
            f"{name}: a*={a_star} -> {a_got}, reference {a_ref}")

    @pytest.mark.parametrize("name,J,p", SHAPES)
    def test_rank_deficient_designs_agree_too(self, name, J, p):
        """The regime where the error is not a constant.

        On a low-rank design both spectra put the selected element on the ridge
        value and the old scaling was out by exactly two; as the rank approaches
        ``J`` they select structurally different elements and it is out by less.
        A fix that doubles the multipliers passes the first and fails this.
        """
        r = max(1, min(J, p) // 3)
        Z = donor_design(J, p, seed=7, rank=r)
        for a_star, b_star in ((0.3, 0.7), (0.5, 0.5), (0.9, 0.2)):
            a_ref, b_ref = nsc_r_scaling(Z, a_star, b_star)
            b_got = scale_b(b_star, design_eigenvalues(Z))
            a_got = scale_a(a_star, Z, b_got)
            assert b_got == pytest.approx(b_ref, rel=1e-9, abs=1e-12)
            assert a_got == pytest.approx(a_ref, rel=1e-9, abs=1e-12)


class TestScalingBoundaries:
    def test_zero_stars_give_no_penalty(self):
        Z = donor_design(20, 8)
        assert scale_b(0.0, design_eigenvalues(Z)) == 0.0
        assert scale_a(0.0, Z, 1.0) == 0.0

    def test_negative_stars_give_no_penalty(self):
        Z = donor_design(20, 8)
        assert scale_b(-0.5, design_eigenvalues(Z)) == 0.0
        assert scale_a(-0.5, Z, 1.0) == 0.0

    def test_star_of_one_takes_the_largest_eigenvalue(self):
        Z = donor_design(20, 8)
        a_ref, b_ref = nsc_r_scaling(Z, 1.0, 1.0)
        b_got = scale_b(1.0, design_eigenvalues(Z))
        assert b_got == pytest.approx(b_ref, rel=1e-9)
        assert scale_a(1.0, Z, b_got) == pytest.approx(a_ref, rel=1e-9)

    def test_an_empty_donor_block_is_unpenalised(self):
        assert scale_b(0.5, np.asarray([])) == 0.0
        assert design_eigenvalues(np.zeros((0, 4))).size == 0
        assert scale_b(0.5, design_eigenvalues(np.zeros((0, 4)))) == 0.0

    def test_no_donors_leaves_a_unpenalised(self):
        """No donor pool is no spectrum to index, and no penalty to scale."""
        assert scale_a(0.5, np.zeros((0, 4)), 1.0) == 0.0

    def test_a_zero_design_is_unpenalised(self):
        Z = np.zeros((6, 4))
        assert scale_b(0.5, design_eigenvalues(Z)) == 0.0
        assert scale_a(0.5, Z, 0.0) == 0.0

    def test_scaling_is_monotone_in_the_star(self):
        """More penalty asked for is never less penalty delivered."""
        Z = donor_design(30, 15)
        eig = design_eigenvalues(Z)
        vals = [scale_b(s, eig) for s in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        assert all(x <= y + 1e-12 for x, y in zip(vals, vals[1:])), vals


# =========================================================================== #
# what the solver does with it -- green before and after
# =========================================================================== #
def scaled(Z, a_star, b_star):
    """The penalties as the estimator forms them, on the design's own scale.

    The solver is only ever called with eigenvalue-scaled multipliers. Handing
    it an arbitrary raw pair leaves the Hessian off the design's scale and it
    raises ``MlsynthEstimationError`` -- correctly, since a QP it cannot solve
    is not a weight vector -- so the tests below scale first, as the caller
    does.
    """
    b = scale_b(b_star, design_eigenvalues(Z))
    return scale_a(a_star, Z, b), b


class TestQpUnchanged:
    """The QP solves what it is handed; only what it is handed moves."""

    @pytest.mark.parametrize("name,J,p", [s for s in SHAPES if s[1] > 2])
    def test_weights_are_feasible_and_sum_to_one(self, name, J, p):
        Z = donor_design(J, p, seed=3)
        Z1 = treated_row(Z, seed=99)
        a, b = scaled(Z, 0.3, 0.7)
        w = solve_nsc_weights(Z1, Z, a, b)
        assert w.shape == (J,)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_negative_weights_are_permitted(self):
        """NSC drops non-negativity; that is the method, not a bug."""
        Z = donor_design(12, 6, seed=5)
        Z1 = -3.0 * Z[0] + 2.0 * Z[1]
        a, b = scaled(Z, 0.0, 0.05)
        w = solve_nsc_weights(Z1, Z, a, b)
        assert (w < -1e-6).any()

    def test_a_given_penalty_pair_is_deterministic(self):
        Z = donor_design(15, 7, seed=8)
        Z1 = treated_row(Z, seed=4)
        a, b = scaled(Z, 0.3, 0.7)
        np.testing.assert_allclose(solve_nsc_weights(Z1, Z, a, b),
                                   solve_nsc_weights(Z1, Z, a, b), atol=0)

    def test_a_larger_l2_shrinks_the_spread_of_the_weights(self):
        Z = donor_design(20, 9, seed=2)
        Z1 = treated_row(Z, seed=11)
        eig = design_eigenvalues(Z)
        spreads = [solve_nsc_weights(Z1, Z, 0.0, scale_b(s, eig)).std()
                   for s in (0.05, 0.5, 1.0)]
        assert spreads[0] > spreads[-1], spreads


# =========================================================================== #
# Proposition 99 against the author's captured run
# =========================================================================== #
def _prop99_blocks():
    import pandas as pd
    d = pd.read_csv(_BASEDATA / "smoking_data.csv").sort_values(["state", "year"])
    states = list(dict.fromkeys(d["state"]))
    years = sorted(d["year"].unique())
    Y = d["cigsale"].to_numpy().reshape(len(states), len(years))
    treat_year = int(d.loc[d["Proposition 99"] == 1, "year"].min())
    T0 = years.index(treat_year)
    ti = states.index(d.loc[d["Proposition 99"] == 1, "state"].iloc[0])
    Z = Y[:, :T0]
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
    return states, years, Y, T0, ti, Z


@pytest.mark.skipif(not (_REF_DIR / "reference.json").is_file(),
                    reason="captured NSC.R reference not present")
class TestProp99:
    def test_weights_match_the_authors_run(self):
        states, years, Y, T0, ti, Z = _prop99_blocks()
        ref = json.loads((_REF_DIR / "reference.json").read_text())
        rw = ref.get("weights") or ref["values"]["weights"]
        donors = [s for s in states if s != states[ti]]
        ref_w = np.array([rw[s] for s in donors])

        ZJ = np.delete(Z, ti, axis=0)
        b = scale_b(0.7, design_eigenvalues(ZJ))
        a = scale_a(0.3, ZJ, b)
        w = solve_nsc_weights(Z[ti], ZJ, a, b)

        # The reference is rounded to three decimals by NSC.R's own return, so
        # 5e-4 per weight is the floor a faithful port can reach.
        assert np.max(np.abs(w - ref_w)) < 1e-3
        assert np.corrcoef(w, ref_w)[0, 1] > 0.9999

    def test_effect_path_matches_the_authors_run(self):
        states, years, Y, T0, ti, Z = _prop99_blocks()
        ref = json.loads((_REF_DIR / "reference.json").read_text())
        vals = ref.get("values", ref)
        ZJ = np.delete(Z, ti, axis=0)
        b = scale_b(0.7, design_eigenvalues(ZJ))
        a = scale_a(0.3, ZJ, b)
        w = solve_nsc_weights(Z[ti], ZJ, a, b)
        ite = Y[ti] - w @ np.delete(Y, ti, axis=0)
        for yr in (1990, 1995, 2000):
            got = float(ite[years.index(yr)])
            exp = float(vals[f"ite_{yr}"])
            assert abs(got - exp) < 0.02, f"{yr}: {got} vs {exp}"
        assert abs(float(ite[T0:].mean()) - float(vals["att_mean_post"])) < 0.02
