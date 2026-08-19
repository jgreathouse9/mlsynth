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

    assert warm * 3 < cold, f"seeded path did not collapse work: {warm} vs {cold}"


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
