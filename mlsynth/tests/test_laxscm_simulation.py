"""The Liao-Shi-Zheng (2026) Section-5 latent-group DGP.

Two things are checked here. The design contract -- that ``w_star`` really is
equal within each group, which is the premise every weight-recovery claim about
this DGP rests on -- and the backward compatibility of the four-tuple entry
point, which callers outside this repository may hold.
"""

import numpy as np
import pytest

from mlsynth.utils.laxscm_helpers.simulation import (
    RelaxationDesign,
    simulate_relaxation_groups,
    simulate_relaxation_groups_design,
    to_panel,
)

_J, _T0, _T1 = 12, 20, 10


def _design(seed=0, **kw):
    kw.setdefault("K", 3)
    return simulate_relaxation_groups_design(
        np.random.default_rng(seed), _J, _T0, T1=_T1, **kw
    )


# --------------------------------------------------------------- smoke


def test_design_runs_and_returns_finite_arrays_of_the_stated_shape():
    d = _design()
    assert isinstance(d, RelaxationDesign)
    T = _T0 + _T1
    assert d.Yc.shape == (_J, T)
    assert d.y0.shape == (T,)
    assert d.oracle_cf.shape == (T,)
    assert d.w_star.shape == (_J,)
    assert d.groups.shape == (_J,)
    assert d.group_weights.shape == (3,)
    assert d.T0 == _T0
    for a in (d.Yc, d.y0, d.oracle_cf, d.w_star, d.group_weights):
        assert np.all(np.isfinite(a))


# ------------------------------------------------- the design contract


def test_oracle_weights_are_equal_within_every_group():
    """The premise of the paper's mechanism claim, and of this DGP's oracle."""
    d = _design(K=4)
    for k in range(4):
        members = d.w_star[d.groups == k]
        assert members.size > 0
        assert np.allclose(members, members[0], rtol=0, atol=0)


def test_oracle_weights_are_a_convex_combination():
    d = _design(K=4)
    assert np.all(d.w_star >= 0.0)
    assert d.w_star.sum() == pytest.approx(1.0)


def test_each_group_carries_its_own_weight():
    """``w_star`` is ``group_weights`` spread equally over each group."""
    d = _design(K=4)
    for k in range(4):
        assert d.w_star[d.groups == k].sum() == pytest.approx(d.group_weights[k])


def test_the_first_group_is_given_zero_weight_when_there_is_more_than_one():
    """Section 5.1 puts the treated unit outside group 0's span."""
    d = _design(K=3)
    assert d.group_weights[0] == 0.0
    assert np.all(d.w_star[d.groups == 0] == 0.0)


def test_a_single_group_takes_all_the_weight():
    d = _design(K=1)
    assert d.group_weights.shape == (1,)
    assert d.group_weights[0] == pytest.approx(1.0)
    assert np.allclose(d.w_star, 1.0 / _J)


def test_oracle_counterfactual_is_the_oracle_weights_applied_to_the_donors():
    d = _design()
    assert np.allclose(d.oracle_cf, d.w_star @ d.Yc, rtol=0, atol=0)


def test_group_labels_cover_every_donor_and_only_the_declared_groups():
    d = _design(K=5)
    assert set(np.unique(d.groups)) == set(range(5))
    assert d.groups.shape == (_J,)


def test_approximate_structure_breaks_the_within_group_equality_of_loadings():
    """``approximate=True`` is Section 5.2: the groups hold only approximately.

    The oracle weights stay group-equal by construction -- it is the loadings
    that are perturbed -- so the panel, not ``w_star``, is what moves.
    """
    exact = _design(seed=7, approximate=False)
    approx = _design(seed=7, approximate=True)
    assert not np.allclose(exact.Yc, approx.Yc)
    for k in range(3):
        members = approx.w_star[approx.groups == k]
        assert np.allclose(members, members[0], rtol=0, atol=0)


# ----------------------------------------------- backward compatibility


def test_the_four_tuple_entry_point_still_returns_four_things():
    out = simulate_relaxation_groups(np.random.default_rng(3), _J, _T0, T1=_T1, K=3)
    assert isinstance(out, tuple)
    assert len(out) == 4


def test_the_four_tuple_draws_are_identical_to_the_design_draws():
    """The accessor must not consume the generator differently."""
    Yc, y0, oracle_cf, T0 = simulate_relaxation_groups(
        np.random.default_rng(11), _J, _T0, T1=_T1, K=3
    )
    d = _design(seed=11, K=3)
    assert np.array_equal(Yc, d.Yc)
    assert np.array_equal(y0, d.y0)
    assert np.array_equal(oracle_cf, d.oracle_cf)
    assert T0 == d.T0


def test_repeat_draws_from_the_same_seed_agree_bit_for_bit():
    a, b = _design(seed=5), _design(seed=5)
    assert np.array_equal(a.Yc, b.Yc)
    assert np.array_equal(a.w_star, b.w_star)


# --------------------------------------------------------- edge cases


def test_defaults_derive_the_factor_count_from_the_pre_period_length():
    """``r = floor(log T0)``, ``K = r`` when neither is given."""
    d = simulate_relaxation_groups_design(
        np.random.default_rng(0), _J, _T0, T1=_T1
    )
    r = max(1, int(np.floor(np.log(_T0))))
    assert d.group_weights.shape == (r,)


def test_one_pre_period_still_draws_a_panel():
    d = simulate_relaxation_groups_design(
        np.random.default_rng(0), _J, 1, T1=_T1, K=2, r=1
    )
    assert d.Yc.shape == (_J, 1 + _T1)
    assert d.T0 == 1


def test_one_donor_puts_all_the_weight_on_it():
    d = simulate_relaxation_groups_design(
        np.random.default_rng(0), 1, _T0, T1=_T1, K=1
    )
    assert d.w_star.shape == (1,)
    assert d.w_star[0] == pytest.approx(1.0)


def test_groups_are_balanced_to_within_one_member():
    d = _design(K=5)
    sizes = np.bincount(d.groups, minlength=5)
    assert sizes.max() - sizes.min() <= 1


# ------------------------------------------------------------ failures


def test_more_groups_than_donors_is_refused():
    """An empty group's weight is dropped from ``w_star``.

    ``groups = arange(J) % K`` leaves groups ``J..K-1`` with no members, so the
    oracle weights would sum to less than one and ``oracle_cf`` would not be a
    convex combination of the donors -- a design that looks drawn and is not.
    """
    with pytest.raises(ValueError, match="groups"):
        simulate_relaxation_groups_design(
            np.random.default_rng(0), 3, _T0, T1=_T1, K=5
        )


def test_the_four_tuple_entry_point_refuses_it_too():
    with pytest.raises(ValueError, match="groups"):
        simulate_relaxation_groups(np.random.default_rng(0), 3, _T0, T1=_T1, K=5)


# ------------------------------------------------------------ to_panel


def test_to_panel_is_long_with_one_treated_unit_and_the_declared_shape():
    d = _design()
    panel = to_panel(d.Yc, d.y0, d.T0)
    T = _T0 + _T1
    assert len(panel) == (_J + 1) * T
    assert set(panel.columns) == {"unit", "time", "y", "treat"}
    assert panel["unit"].nunique() == _J + 1
    assert panel.loc[panel["treat"] == 1, "unit"].nunique() == 1
    assert (panel["treat"] == 1).sum() == _T1
