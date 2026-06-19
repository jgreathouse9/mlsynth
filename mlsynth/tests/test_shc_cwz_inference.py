"""Tests for the exact Chernozhukov-Wuthrich-Zhu (2021) conformal test in SHC.

The function under test is
:func:`mlsynth.utils.shc_helpers.inference.cwz_conformal_test`, the *exact*
permutation conformal inference of CWZ (2021) -- as distinct from the
Chen-Yang-Yang (2024, footnote 21) with-replacement residual bootstrap already
covered in ``test_shc_inference.py``.

Key facts pinned here (CWZ 2021, Definitions 1-2, Figure 2):

* test statistic ``S_q(u) = ( (1/sqrt(n)) * sum_{post} |u_t|^q )^{1/q}`` with
  ``q = 1`` the default;
* the reference distribution comes from *permutations* of the full residual
  vector ``u = [pre, post]`` (length ``T = T0 + n``), evaluated on the trailing
  ``n`` positions -- NOT a bootstrap from the pre-period pool;
* ``scheme="moving_block"`` enumerates the ``T`` cyclic shifts ``Pi_->`` (exact
  under stationary weak dependence); the identity shift is included so the
  attainable p-values are multiples of ``1/T``;
* ``scheme="iid"`` samples random permutations ``Pi_all`` (exact under
  exchangeability), always including the identity so ``p >= 1/|Pi|``.

Assertions are structural / hand-computable, not magic numbers from output.
"""

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.shc_helpers.inference import cwz_conformal_test


# --------------------------------------------------------------------------
# success path -- structure
# --------------------------------------------------------------------------
def test_keys_and_ranges():
    rng = np.random.default_rng(0)
    pre = rng.standard_normal(40)
    post = rng.standard_normal(5)
    res = cwz_conformal_test(pre, post, scheme="moving_block")
    for k in ("test_statistic", "p_value", "critical_values", "reject",
              "null_distribution", "num_permutations", "scheme", "q", "levels"):
        assert k in res
    assert 0.0 <= res["p_value"] <= 1.0
    assert np.isfinite(res["test_statistic"])
    assert res["scheme"] == "moving_block"
    assert set(res["critical_values"]) == set(res["reject"]) == {0.01, 0.05, 0.10}


def test_statistic_is_cwz_s1():
    # S_1 = (1/sqrt(n)) * sum |post|.
    pre = np.zeros(6)
    post = np.array([3.0, -4.0])
    res = cwz_conformal_test(pre, post, q=1.0, scheme="moving_block")
    assert res["test_statistic"] == pytest.approx((3.0 + 4.0) / np.sqrt(2))


def test_statistic_general_q():
    pre = np.zeros(6)
    post = np.array([3.0, -4.0])
    res = cwz_conformal_test(pre, post, q=2.0, scheme="moving_block")
    expected = (np.sqrt((9.0 + 16.0) / np.sqrt(2)))  # (sum|.|^2 / sqrt(n))^{1/2}
    assert res["test_statistic"] == pytest.approx(expected)


# --------------------------------------------------------------------------
# moving block -- exact, hand-computable
# --------------------------------------------------------------------------
def test_moving_block_enumerates_T_shifts():
    pre = np.zeros(4)
    post = np.array([1.0])
    res = cwz_conformal_test(pre, post, scheme="moving_block")
    # T = T0 + n = 4 + 1 = 5 cyclic shifts.
    assert res["num_permutations"] == 5
    assert res["null_distribution"].shape == (5,)


def test_moving_block_single_post_pvalue_is_one_over_T():
    # u = [0,0,0,0,10]; the trailing-1 block over the 5 cyclic shifts takes each
    # element once, so only the identity hits |10|. p = 1/5.
    pre = np.zeros(4)
    post = np.array([10.0])
    res = cwz_conformal_test(pre, post, scheme="moving_block")
    assert res["p_value"] == pytest.approx(0.2)
    assert res["test_statistic"] == pytest.approx(10.0)


def test_moving_block_two_post_pvalue_hand_value():
    # u = [0,0,0,6,8], n = 2. Cyclic trailing-2 blocks:
    #   [6,8]->14, [8,0]->8, [0,0]->0, [0,0]->0, [0,6]->6  (all /sqrt(2)).
    # Observed 14/sqrt(2) is the unique max -> p = 1/5.
    pre = np.zeros(3)
    post = np.array([6.0, 8.0])
    res = cwz_conformal_test(pre, post, scheme="moving_block")
    assert res["p_value"] == pytest.approx(0.2)
    assert res["test_statistic"] == pytest.approx(14.0 / np.sqrt(2))


def test_moving_block_identity_is_included():
    # The observed statistic must appear in the null (j=0 shift), so the
    # smallest attainable p-value is exactly 1/T.
    pre = np.zeros(9)
    post = np.array([100.0])
    res = cwz_conformal_test(pre, post, scheme="moving_block")
    assert res["p_value"] == pytest.approx(1.0 / 10.0)


def test_moving_block_is_deterministic_without_seed_dependence():
    # Moving block enumerates a fixed set; two calls match regardless of seed.
    pre = np.linspace(-1, 1, 30)
    post = np.array([0.3, -0.2, 0.5])
    a = cwz_conformal_test(pre, post, scheme="moving_block", random_state=1)
    b = cwz_conformal_test(pre, post, scheme="moving_block", random_state=999)
    assert a["p_value"] == b["p_value"]
    assert np.allclose(a["null_distribution"], b["null_distribution"])


# --------------------------------------------------------------------------
# iid permutations
# --------------------------------------------------------------------------
def test_iid_includes_identity_and_count():
    rng = np.random.default_rng(2)
    pre = rng.standard_normal(50)
    post = rng.standard_normal(4)
    res = cwz_conformal_test(pre, post, scheme="iid", num_permutations=200,
                             random_state=0)
    assert res["num_permutations"] == 200
    assert res["null_distribution"].shape == (200,)
    # identity is in the set, so p >= 1/|Pi|.
    assert res["p_value"] >= 1.0 / 200 - 1e-12


def test_iid_determinism():
    rng = np.random.default_rng(3)
    pre = rng.standard_normal(60)
    post = rng.standard_normal(5)
    a = cwz_conformal_test(pre, post, scheme="iid", num_permutations=300, random_state=7)
    b = cwz_conformal_test(pre, post, scheme="iid", num_permutations=300, random_state=7)
    assert a["p_value"] == b["p_value"]
    assert np.allclose(a["null_distribution"], b["null_distribution"])


def test_iid_default_num_permutations():
    rng = np.random.default_rng(4)
    pre = rng.standard_normal(40)
    post = rng.standard_normal(3)
    res = cwz_conformal_test(pre, post, scheme="iid")
    assert res["num_permutations"] >= 1000  # sensible default


# --------------------------------------------------------------------------
# power / behaviour
# --------------------------------------------------------------------------
def test_large_effect_rejects_small_effect_does_not():
    rng = np.random.default_rng(5)
    pre = rng.standard_normal(60)
    big = np.array([8.0, 9.0, 7.5])           # clear outliers vs N(0,1) pre
    small = rng.standard_normal(3) * 0.5      # typical of the pre pool
    p_big = cwz_conformal_test(pre, big, scheme="moving_block")["p_value"]
    p_small = cwz_conformal_test(pre, small, scheme="moving_block")["p_value"]
    assert p_big < p_small
    assert p_big <= 0.05
    assert p_small > 0.10


def test_pvalue_monotone_in_effect_size():
    rng = np.random.default_rng(6)
    pre = rng.standard_normal(80)
    ps = [cwz_conformal_test(pre, np.full(3, c), scheme="moving_block")["p_value"]
          for c in (0.0, 1.0, 3.0, 6.0)]
    assert ps[0] >= ps[1] >= ps[2] >= ps[3]


def test_reject_consistent_with_critical_values():
    rng = np.random.default_rng(7)
    pre = rng.standard_normal(50)
    post = np.array([5.0, 6.0])
    res = cwz_conformal_test(pre, post, scheme="moving_block")
    for lvl, cv in res["critical_values"].items():
        assert res["reject"][lvl] == (res["test_statistic"] > cv)


# --------------------------------------------------------------------------
# error paths
# --------------------------------------------------------------------------
def test_empty_pre_raises():
    with pytest.raises(MlsynthDataError):
        cwz_conformal_test(np.array([]), np.array([1.0]))


def test_empty_post_raises():
    with pytest.raises(MlsynthDataError):
        cwz_conformal_test(np.array([1.0, 2.0]), np.array([]))


@pytest.mark.parametrize("bad_q", [0.0, -1.0])
def test_nonpositive_q_raises(bad_q):
    with pytest.raises(MlsynthConfigError):
        cwz_conformal_test(np.zeros(5), np.array([1.0]), q=bad_q)


def test_bad_scheme_raises():
    with pytest.raises(MlsynthConfigError):
        cwz_conformal_test(np.zeros(5), np.array([1.0]), scheme="bogus")


def test_bad_num_permutations_raises():
    with pytest.raises(MlsynthConfigError):
        cwz_conformal_test(np.zeros(5), np.array([1.0]), scheme="iid",
                           num_permutations=1)


def test_accepts_list_and_2d_inputs():
    res = cwz_conformal_test([0.0, 0.0, 0.0], [[1.0], [2.0]], scheme="moving_block")
    assert np.isfinite(res["test_statistic"])
    assert res["num_permutations"] == 5
