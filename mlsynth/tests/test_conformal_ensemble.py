"""Bootstrap leave-one-out errors: the ensemble route to a calibration series.

A cumulative band is calibrated on out-of-sample errors, and the usual way to get
them is to withhold periods -- a blank window, or a rolling origin that refits and
scores the next stretch. Both spend pre-periods to buy errors.

An ensemble buys them differently. Fit many members, each on a bootstrap sample of
the training periods, and score period ``t`` using only the members whose sample
excluded it. Every training period then carries an error that the members scoring
it never saw, at one fit per member instead of one per origin. This is the
construction from Xu and Xie's EnbPI, with the model left to the caller: a
``weight_fn`` is handed the resampled period indices and returns donor weights, so
a simplex control, a ridge-augmented one and a lasso all reach the same series.

What the ensemble does not do is make the training periods out of sample for
anything decided before it runs. If a design chose its donor pool or its treated
set having seen those periods, the errors there are out of sample for the weights
alone, and the caller has to say whether that is enough.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.bilevel.simplex import simplex_lstsq
from mlsynth.utils.conformal import EnsembleErrors, bootstrap_loo_errors


def _panel(n_donors=5, n_periods=40, seed=0):
    """A factor panel whose treated aggregate is a donor mixture plus noise."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.standard_normal(n_periods))
    donors = np.array([1.0 + 0.3 * j + (0.8 + 0.1 * j) * f
                       + 0.2 * rng.standard_normal(n_periods)
                       for j in range(n_donors)])
    w = rng.dirichlet(np.ones(n_donors))
    target = w @ donors + 0.1 * rng.standard_normal(n_periods)
    return target, donors


def _simplex_fit(target, donors):
    """The weight_fn every test uses unless it needs something contrived."""
    return lambda idx: simplex_lstsq(donors[:, idx].T, target[idx])


def _run(target=None, donors=None, n_train=30, n_members=40, seed=0, **kw):
    if target is None:
        target, donors = _panel()
    return bootstrap_loo_errors(
        target, donors, n_train, _simplex_fit(target, donors),
        n_members=n_members, rng=np.random.default_rng(seed), **kw)


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_it_produces_an_error_series_and_a_counterfactual():
    target, donors = _panel()
    out = _run(target, donors, n_train=30)
    assert isinstance(out, EnsembleErrors)
    assert out.errors.shape == (30,)
    assert out.counterfactual.shape == (target.size - 30,)
    assert np.isfinite(out.errors).all()
    assert np.isfinite(out.counterfactual).all()
    assert out.n_members == 40 and out.n_dropped == 0


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_a_period_is_scored_only_by_members_that_did_not_see_it():
    """The whole point, pinned by replaying the draw.

    One member sees roughly 1 - 1/e of the periods, so at ``n_members=1`` most
    periods have no member that excluded them. Which ones is not a matter of
    opinion: the same seed reproduces the member's bootstrap sample, and the
    scored periods must be exactly its complement.
    """
    target, donors = _panel()
    n_train = 30
    out = _run(target, donors, n_train=n_train, n_members=1, seed=7)
    seen = np.unique(np.random.default_rng(7).integers(0, n_train, size=n_train))
    scored = np.flatnonzero(np.isfinite(out.errors))
    assert np.array_equal(scored, np.setdiff1d(np.arange(n_train), seen))
    assert out.n_dropped == seen.size


def test_a_period_no_member_excluded_is_reported_missing_not_guessed():
    target, donors = _panel()
    out = _run(target, donors, n_train=30, n_members=1, seed=7)
    assert out.n_dropped > 0
    assert np.isnan(out.errors[~np.isfinite(out.errors)]).all()


def test_more_members_leave_fewer_periods_unscored():
    """A period is dropped only when every member sampled it, which happens with
    probability about ``0.63 ** n_members``."""
    target, donors = _panel()
    counts = [_run(target, donors, n_train=30, n_members=b, seed=3).n_dropped
              for b in (1, 3, 10, 40)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > 0 and counts[-1] == 0


def test_a_donor_that_reproduces_the_target_leaves_no_error():
    """A simplex fit can put all its weight on a donor equal to the target, and
    the leave-one-out machinery must not manufacture error where there is none."""
    _, donors = _panel()
    target = donors[2].copy()
    out = _run(target, donors, n_train=30)
    assert np.allclose(out.errors, 0.0, atol=1e-6)
    assert np.allclose(out.counterfactual, target[30:], atol=1e-6)


def test_a_constant_offset_comes_back_as_that_offset():
    """Sign convention: the error is target minus prediction, not the reverse.

    The fit is pinned to one donor instead of being solved, because a simplex fit
    could not reproduce ``donors[2] + 4`` anyway -- that series is outside the
    donors' convex hull, and the solver would trade level against slope to get
    near it. Fixing the weights leaves only the convention under test.
    """
    target, donors = _panel()
    one_hot = np.zeros(donors.shape[0])
    one_hot[2] = 1.0
    shifted = donors[2] + 4.0
    out = bootstrap_loo_errors(shifted, donors, 30, lambda idx: one_hot,
                               n_members=10, rng=np.random.default_rng(0))
    assert np.allclose(out.errors, 4.0)
    assert np.allclose(out.counterfactual, donors[2, 30:])


def test_the_counterfactual_aggregates_every_member():
    """Post periods are not leave-one-out: no member saw them, so all of them
    count. Replaying the draw pins that against a hand-built aggregate."""
    target, donors = _panel()
    n_train, n_members = 30, 12
    out = _run(target, donors, n_train=n_train, n_members=n_members, seed=5)
    rng = np.random.default_rng(5)
    preds = []
    for _ in range(n_members):
        idx = rng.integers(0, n_train, size=n_train)
        preds.append(simplex_lstsq(donors[:, idx].T, target[idx]) @ donors[:, n_train:])
    assert np.allclose(out.counterfactual, np.median(np.array(preds), axis=0))


def test_the_two_aggregations_differ_and_both_are_offered():
    target, donors = _panel(seed=4)
    med = _run(target, donors, aggregate="median")
    mean = _run(target, donors, aggregate="mean")
    assert not np.allclose(med.errors, mean.errors)
    assert np.isfinite(mean.errors).all()


def test_the_same_generator_state_gives_the_same_series():
    target, donors = _panel()
    a = _run(target, donors, seed=11)
    b = _run(target, donors, seed=11)
    assert np.array_equal(a.errors, b.errors)
    assert np.array_equal(a.counterfactual, b.counterfactual)


def test_different_generator_states_give_different_series():
    target, donors = _panel()
    a = _run(target, donors, seed=11)
    b = _run(target, donors, seed=12)
    assert not np.allclose(a.errors, b.errors)


def test_the_fit_sees_only_training_periods_and_one_sample_per_member():
    """A member must never be handed a post period, and there must be exactly as
    many fits as members -- an ensemble that refits more often costs more than it
    claims to."""
    target, donors = _panel()
    calls = []

    def spy(idx):
        calls.append(np.asarray(idx))
        return simplex_lstsq(donors[:, idx].T, target[idx])

    bootstrap_loo_errors(target, donors, 30, spy, n_members=9,
                         rng=np.random.default_rng(0))
    assert len(calls) == 9
    for idx in calls:
        assert idx.size == 30
        assert idx.min() >= 0 and idx.max() < 30


def test_errors_are_not_the_in_sample_residuals():
    """An in-sample residual is smaller than an out-of-sample one; if the
    exclusion were dropped the two would coincide."""
    target, donors = _panel(n_donors=12, seed=2)
    out = _run(target, donors, n_train=30, n_members=60)
    w_full = simplex_lstsq(donors[:, :30].T, target[:30])
    in_sample = target[:30] - w_full @ donors[:, :30]
    assert np.std(out.errors) > np.std(in_sample)


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_training_on_the_whole_series_leaves_no_counterfactual():
    target, donors = _panel(n_periods=30)
    out = _run(target, donors, n_train=30)
    assert out.counterfactual.shape == (0,)
    assert out.errors.shape == (30,)


def test_a_single_donor_is_the_counterfactual():
    target, donors = _panel(n_donors=1)
    out = _run(target, donors, n_train=30)
    assert np.allclose(out.errors, target[:30] - donors[0, :30])


def test_a_single_training_period_cannot_be_left_out():
    """A bootstrap sample of one period from one period is that period, every
    time. There is no member that excluded it, so nothing is scored and the
    result says so instead of returning an in-sample residual."""
    target, donors = _panel()
    out = _run(target, donors, n_train=1, n_members=20)
    assert out.n_dropped == 1
    assert np.isnan(out.errors).all()


def test_two_training_periods_still_score_something():
    target, donors = _panel()
    out = _run(target, donors, n_train=2, n_members=50)
    assert np.isfinite(out.errors).any()


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
def test_a_donor_block_of_the_wrong_length_is_refused():
    target, donors = _panel(n_periods=40)
    with pytest.raises(MlsynthDataError, match="periods"):
        bootstrap_loo_errors(target, donors[:, :20], 15, _simplex_fit(target, donors))


@pytest.mark.parametrize("bad", [0, -1, 41])
def test_a_training_length_outside_the_series_is_refused(bad):
    target, donors = _panel(n_periods=40)
    with pytest.raises((MlsynthConfigError, MlsynthDataError), match="n_train"):
        bootstrap_loo_errors(target, donors, bad, _simplex_fit(target, donors))


@pytest.mark.parametrize("bad", [0, -3, 2.5, "40", True, None])
def test_a_bad_member_count_is_refused(bad):
    target, donors = _panel()
    with pytest.raises(MlsynthConfigError, match="n_members"):
        bootstrap_loo_errors(target, donors, 30, _simplex_fit(target, donors),
                             n_members=bad)


@pytest.mark.parametrize("bad", ["mode", "MEDIAN", "", None])
def test_an_unknown_aggregation_is_refused(bad):
    target, donors = _panel()
    with pytest.raises(MlsynthConfigError, match="aggregate"):
        bootstrap_loo_errors(target, donors, 30, _simplex_fit(target, donors),
                             aggregate=bad)


def test_a_non_callable_fit_is_refused():
    target, donors = _panel()
    with pytest.raises(MlsynthConfigError, match="weight_fn"):
        bootstrap_loo_errors(target, donors, 30, "simplex")


@pytest.mark.parametrize("spoil", ["nan", "inf"])
def test_a_non_finite_target_is_refused(spoil):
    target, donors = _panel()
    target = target.copy()
    target[3] = np.nan if spoil == "nan" else np.inf
    with pytest.raises(MlsynthDataError, match="finite"):
        bootstrap_loo_errors(target, donors, 30, _simplex_fit(target, donors))


def test_a_non_finite_donor_block_is_refused():
    target, donors = _panel()
    donors = donors.copy()
    donors[1, 5] = np.nan
    with pytest.raises(MlsynthDataError, match="finite"):
        bootstrap_loo_errors(target, donors, 30, _simplex_fit(target, donors))


def test_a_one_dimensional_donor_block_is_refused():
    target, donors = _panel()
    with pytest.raises(MlsynthDataError, match="two-dimensional"):
        bootstrap_loo_errors(target, donors[0], 30, _simplex_fit(target, donors))


def test_weights_of_the_wrong_length_are_refused():
    """A caller's fit that returns the wrong shape is a bug in the caller, and
    the failure names it instead of broadcasting into a wrong answer."""
    target, donors = _panel()
    with pytest.raises(MlsynthDataError, match="weight_fn"):
        bootstrap_loo_errors(target, donors, 30, lambda idx: np.ones(3) / 3.0)


def test_non_finite_weights_are_refused():
    target, donors = _panel()
    bad = np.full(donors.shape[0], np.nan)
    with pytest.raises(MlsynthDataError, match="finite"):
        bootstrap_loo_errors(target, donors, 30, lambda idx: bad)


def test_a_two_dimensional_target_is_refused():
    """A caller who passes the donor block twice, or forgets to aggregate the
    treated units, gets told which argument is wrong."""
    target, donors = _panel()
    with pytest.raises(MlsynthDataError, match="one-dimensional"):
        bootstrap_loo_errors(np.tile(target, (2, 1)), donors, 30,
                             _simplex_fit(target, donors))


@pytest.mark.parametrize("bad", [2.5, "30", True, None])
def test_a_non_integer_training_length_is_refused(bad):
    target, donors = _panel()
    with pytest.raises(MlsynthConfigError, match="n_train"):
        bootstrap_loo_errors(target, donors, bad, _simplex_fit(target, donors))
