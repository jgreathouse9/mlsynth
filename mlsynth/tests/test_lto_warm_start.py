"""Work bound for the LTO refined placebo test, and the parity that licenses it.

The LTO test solves ``3 * C(J, 2)`` simplex QPs over donor pools that differ by
two columns out of ``J``. Solved cold, each one rebuilds an active set the
neighbouring pool already found: on Prop 99 that is 30.5 pivots per solve to
reach a support of 5.7 donors. Seeding each solve from a base fit collapses the
pivot count, and because a seed only chooses where the active set starts, the
weights it certifies are the same ones.

That last clause is the whole safety argument, so it is the property asserted
here: the seeded path and the cold path agree on the p-value, the loss count and
the pair count, exactly. The speed claim is asserted as a machine-independent
proxy -- the number of LAPACK least-squares calls the solver issues -- not
wall-clock, which flakes in CI.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.utils.bilevel import BilevelSCM
from mlsynth.utils.vanillasc_helpers.lto import _seed_from_base, lto_placebo_test


def _panel(seed: int, T: int, J: int, pre: int, rank: int = 3):
    """A factor-model panel with a treated unit carrying a post-period jump."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, rank))
    L = rng.normal(size=(rank, J + 1))
    Y = F @ L + 0.3 * rng.normal(size=(T, J + 1))
    y = Y[:, 0].copy()
    y[pre:] += 3.0
    return y, Y[:, 1:].copy()


def _run(y, Y0, pre, *, warm_start):
    return lto_placebo_test(
        BilevelSCM(), y, Y0, pre, alpha=0.05, warm_start=warm_start
    )


# --- rung 3: the invariant a seed must not break -------------------------------

@settings(deadline=None, max_examples=25)
@given(
    seed=st.integers(min_value=0, max_value=500),
    J=st.integers(min_value=3, max_value=12),
    pre=st.integers(min_value=4, max_value=14),
    post=st.integers(min_value=2, max_value=8),
)
def test_warm_start_leaves_every_reported_quantity_unchanged(seed, J, pre, post):
    """A seed picks where the active set starts, not where it lands."""
    y, Y0 = _panel(seed, pre + post, J, pre)
    cold = _run(y, Y0, pre, warm_start=False)
    warm = _run(y, Y0, pre, warm_start=True)
    for key in ("p_value", "treated_losses", "n_pairs", "p_powered", "reject"):
        assert warm[key] == cold[key], f"{key} moved: {warm[key]} != {cold[key]}"


@pytest.mark.parametrize("J,pre,post", [(3, 5, 3), (8, 6, 4), (15, 10, 6)])
def test_warm_start_parity_on_wide_and_narrow_pools(J, pre, post):
    """Parity holds where the pool outruns the pre-window (rank-deficient Gram,
    so the optimum is a face and a solver may land anywhere on it)."""
    y, Y0 = _panel(11 + J, pre + post, J, pre)
    assert _run(y, Y0, pre, warm_start=True) == _run(y, Y0, pre, warm_start=False)


# --- rung 2: the work the seed is there to remove ------------------------------

def test_warm_start_collapses_solver_work(monkeypatch):
    """Machine-independent proxy: LAPACK least-squares calls issued by the
    active set. Each pivot costs one, so this counts the pivots the seed saves."""
    from mlsynth.utils.bilevel import active_set

    calls = {"n": 0}
    inner = active_set._gelsy_lstsq

    def counting(M, b):
        calls["n"] += 1
        return inner(M, b)

    monkeypatch.setattr(active_set, "_gelsy_lstsq", counting)

    y, Y0 = _panel(3, 30, 20, 18)
    calls["n"] = 0
    _run(y, Y0, 18, warm_start=False)
    cold = calls["n"]
    calls["n"] = 0
    _run(y, Y0, 18, warm_start=True)
    warm = calls["n"]

    solves = 3 * (20 * 19 // 2)
    assert warm * 3 < cold, f"seeded path did not collapse work: {warm} vs {cold}"
    # Absolute bound, not just a ratio. A seed that is merely feasible -- the
    # treated unit's base carried onto a donor's subproblem, say -- still beats
    # the uniform point by enough to pass a ratio test, while costing about
    # twice the pivots of the right seed (4.2 vs 2.2 per solve here).
    assert warm < 3.0 * solves, f"{warm / solves:.1f} calls/solve; the seed is wrong"


def test_every_subproblem_leaves_out_both_donors_of_the_pair(monkeypatch):
    """The design the pair loop actually builds, pinned by donor count.

    Leaving out one donor instead of two is the ordinary placebo test wearing
    LTO's name, and it is invisible to any test that only compares the seeded
    path against the cold one -- it moves both alike.
    """
    from mlsynth.utils.bilevel import engine as engine_mod

    widths: list[int] = []
    inner = engine_mod.BilevelSCM.fit

    def recording(self, y_pre, Y0_pre, **kw):
        widths.append(Y0_pre.shape[1])
        return inner(self, y_pre, Y0_pre, **kw)

    monkeypatch.setattr(engine_mod.BilevelSCM, "fit", recording)

    J, pre = 7, 12
    y, Y0 = _panel(4, 20, J, pre)

    widths.clear()
    res = _run(y, Y0, pre, warm_start=False)
    n_pairs = J * (J - 1) // 2
    assert res["n_pairs"] == n_pairs
    assert widths == [J - 2] * (3 * n_pairs)

    widths.clear()
    _run(y, Y0, pre, warm_start=True)
    # One treated base on the full pool, one per donor on the other J-1, then
    # the pair loop. The seeding cost is O(J) against the loop's O(J^2).
    assert widths[0] == J
    assert widths[1:1 + J] == [J - 1] * J
    assert widths[1 + J:] == [J - 2] * (3 * n_pairs)


# --- rung 4: the contract on the seed itself -----------------------------------

def test_seed_from_base_restricts_and_renormalises():
    base = np.array([0.5, 0.25, 0.25, 0.0])
    seed = _seed_from_base(base, np.array([1, 2, 3]))
    assert seed == pytest.approx([0.5, 0.5, 0.0])
    assert seed.sum() == pytest.approx(1.0)


def test_seed_from_base_declines_when_the_pool_carries_no_mass():
    """All the base weight sat on the two donors this pair removes. There is no
    feasible point to carry over, so the solve goes in cold."""
    base = np.array([0.6, 0.4, 0.0, 0.0])
    assert _seed_from_base(base, np.array([2, 3])) is None


def test_seed_from_base_declines_on_a_non_finite_base():
    base = np.array([np.nan, 1.0, 0.0])
    assert _seed_from_base(base, np.array([0, 1])) is None


def test_warm_start_survives_a_failing_base_fit(monkeypatch):
    """A base fit that raises must degrade to the cold path, not to an error."""
    import mlsynth.utils.vanillasc_helpers.lto as lto_mod

    y, Y0 = _panel(5, 20, 6, 12)
    expected = _run(y, Y0, 12, warm_start=False)

    def boom(*a, **k):
        raise RuntimeError("base fit failed")

    monkeypatch.setattr(lto_mod, "_base_seeds", boom)
    assert _run(y, Y0, 12, warm_start=True) == expected


def test_covariate_path_is_unchanged_by_the_seed():
    """With predictors the engine takes a backend the seed does not reach; the
    answer must be identical either way."""
    y, Y0 = _panel(9, 22, 7, 14)
    rng = np.random.default_rng(9)
    X0 = rng.normal(size=(2, Y0.shape[1]))
    X1 = rng.normal(size=2)
    kw = dict(X1=X1, X0=X0, alpha=0.05)
    cold = lto_placebo_test(BilevelSCM(), y, Y0, 14, warm_start=False, **kw)
    warm = lto_placebo_test(BilevelSCM(), y, Y0, 14, warm_start=True, **kw)
    assert warm == cold


# --- pre-existing gaps in the same file, closed while it is open ---------------

def test_too_few_donors_is_refused():
    """Two donors cannot be left out of two and leave a pool behind."""
    y, Y0 = _panel(1, 12, 2, 8)
    with pytest.raises(ValueError, match="at least 3 donor units"):
        _run(y, Y0, 8, warm_start=True)


def test_max_pairs_subsamples_deterministically_and_says_so():
    """The cap is a deterministic draw, so two runs at one seed agree."""
    y, Y0 = _panel(6, 24, 10, 15)
    full = _run(y, Y0, 15, warm_start=True)
    assert full["n_pairs"] == 45 and not full["subsampled"]

    kw = dict(alpha=0.05, max_pairs=12, seed=3)
    capped = lto_placebo_test(BilevelSCM(), y, Y0, 15, **kw)
    assert capped["subsampled"] and capped["n_pairs"] == 12
    assert capped == lto_placebo_test(BilevelSCM(), y, Y0, 15, **kw)


def test_type_i_rate_function_clamps_outside_its_valid_alpha_range():
    """`f(N, alpha)` takes a square root of a quantity decreasing in alpha, which
    goes negative past the range the Type-I bound is defined on. It clamps there,
    so the rate saturates instead of returning nan."""
    from mlsynth.utils.vanillasc_helpers.lto import lto_f

    saturated = (3.0 - 3.0 / 40) / 2.0
    assert lto_f(40, 0.74) < saturated             # still inside the range
    for alpha in (0.75, 0.9, 1.0):
        assert lto_f(40, alpha) == pytest.approx(saturated)
