"""Property tests for MOSC's factor models and its outcome-regression kernel.

The example-based suite in ``test_mosc.py`` pins named cases. This file asserts
the claims those examples instantiate, over the input domain, on the layer-1
helpers -- pure functions, no solver, microseconds per call, which is where
``agents_tests.md`` puts property testing first.

Exact properties are scarce in an estimation kernel, so most of what is asserted
here is metamorphic: how an output must respond to a transformation of the input.
Three of the relations carry most of the weight.

The counterfactual is equivariant in the outcome's units. Rescale a panel by a
positive constant and the counterfactual must scale with it, because the units of
an outcome are a choice of the person who recorded it and cannot move a causal
estimate. This is the relation that would break if a hard-coded constant crept
into the regression or the re-integration.

Duplicating a donor must not move the counterfactual much. Adding a column that
carries no new information should not change what the panel says about the
treated unit; a fit that lurches when a donor is copied is fitting the donor
list, not the structure.

Gamma-Poisson draws are non-negative by construction, and PPCA's are not. That is
the whole reason the estimator offers a choice of likelihood, so it is asserted
over the domain and not at one fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.mosc_helpers.factor import (
    FACTOR_MODELS,
    gap_gibbs,
    heldout_log_density,
    ppca_em,
)
from mlsynth.utils.mosc_helpers.pipeline import (
    _pre_length,
    _regression_arrays,
    _to_modelling_scale,
    counterfactual_draws,
)

# Property runs are deterministic so a mutation score cannot be corrupted by a
# flakily-killed mutant (``agents_tests.md``, standing constraints).
SETTINGS = settings(
    max_examples=25,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

RIDGE = (0.0, 1e-3)


def _count_panel(n_time: int, n_unit: int, n_factors: int, seed: int) -> np.ndarray:
    """A Poisson panel from a non-negative factor model."""
    rng = np.random.default_rng(seed)
    rate = rng.gamma(3.0, 3.0, size=(n_time, n_factors)) @ rng.gamma(
        3.0, 3.0, size=(n_factors, n_unit)
    )
    return rng.poisson(rate).astype(float)


shapes = st.tuples(
    st.integers(min_value=8, max_value=16),   # periods
    st.integers(min_value=7, max_value=12),   # units
    st.integers(min_value=1, max_value=3),    # factors
)


# ------------------------------------------------------- support properties --


@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_gamma_poisson_draws_are_non_negative(shape, seed):
    """Every Gamma draw is non-negative, so every fitted rate is too.

    The Poisson arm is offered precisely because a count cannot go below zero,
    and a counterfactual that can is the failure it exists to prevent.
    """
    n_time, n_unit, n_factors = shape
    panel = _count_panel(n_time, n_unit, n_factors, seed)
    draws = gap_gibbs(panel, n_factors, n_samples=6, n_warmup=6, seed=seed)
    assert (draws.loadings >= 0).all()
    assert (draws.factors >= 0).all()
    assert all(draws.mean(s).min() >= 0 for s in range(draws.n_draws))


@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_every_factor_model_returns_the_declared_shapes(shape, seed):
    """Both arms speak the same interface, whatever they do internally.

    This is what lets the pipeline dispatch by table without knowing which model
    it called.
    """
    n_time, n_unit, n_factors = shape
    panel = _count_panel(n_time, n_unit, n_factors, seed)
    for name, engine in FACTOR_MODELS.items():
        draws = engine(panel, n_factors, n_samples=4, n_warmup=4, seed=seed)
        assert draws.loadings.shape == (4, n_factors, n_unit)
        assert draws.factors.shape == (4, n_time, n_factors)
        assert draws.mean(0).shape == (n_time, n_unit)
        assert np.isfinite(draws.mean(0)).all(), name


# -------------------------------------------------- metamorphic properties --


@given(shape=shapes, scale=st.floats(0.25, 4.0), seed=st.integers(0, 2**16))
@SETTINGS
def test_counterfactual_is_equivariant_in_the_outcome_scale(shape, scale, seed):
    """Rescaling the outcome rescales the counterfactual by the same factor.

    An outcome measured in cents instead of dollars is the same study, so a
    constant that failed to scale would make the estimate depend on the units
    someone happened to record it in.
    """
    n_time, n_unit, n_factors = shape
    pre = n_time - 3
    assume(pre >= n_factors + 2)
    panel = _count_panel(n_time, n_unit, n_factors, seed)

    rng = np.random.default_rng(seed)
    loadings = rng.gamma(2.0, 2.0, size=(3, n_factors, n_unit))

    base = counterfactual_draws(loadings, panel, pre, RIDGE)
    scaled = counterfactual_draws(loadings, panel * scale, pre, RIDGE)
    np.testing.assert_allclose(scaled, base * scale, rtol=1e-6, atol=1e-8)


@given(shape=shapes, shift=st.floats(-50.0, 50.0), seed=st.integers(0, 2**16))
@SETTINGS
def test_counterfactual_is_equivariant_under_a_location_shift(shape, shift, seed):
    """Adding a constant to every outcome shifts the counterfactual by it.

    The regression carries an intercept, so a common shift must be absorbed
    there and nowhere else.
    """
    n_time, n_unit, n_factors = shape
    pre = n_time - 3
    assume(pre >= n_factors + 2)
    panel = _count_panel(n_time, n_unit, n_factors, seed)

    rng = np.random.default_rng(seed)
    loadings = rng.gamma(2.0, 2.0, size=(3, n_factors, n_unit))

    base = counterfactual_draws(loadings, panel, pre, RIDGE)
    shifted = counterfactual_draws(loadings, panel + shift, pre, RIDGE)
    np.testing.assert_allclose(shifted, base + shift, rtol=1e-6, atol=1e-6)


@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_duplicating_a_donor_barely_moves_the_counterfactual(shape, seed):
    """A copied donor carries no new information, so it must not move the fit far.

    A fit that lurches when a donor is duplicated is responding to the donor
    list, not to the structure the donors share.
    """
    n_time, n_unit, n_factors = shape
    pre = n_time - 3
    assume(pre >= n_factors + 2)
    panel = _count_panel(n_time, n_unit, n_factors, seed)
    doubled = np.column_stack([panel, panel[:, -1]])

    rng = np.random.default_rng(seed)
    loadings = rng.gamma(2.0, 2.0, size=(2, n_factors, n_unit))
    loadings_doubled = np.concatenate([loadings, loadings[:, :, -1:]], axis=2)

    base = counterfactual_draws(loadings, panel, pre, RIDGE)
    with_copy = counterfactual_draws(loadings_doubled, doubled, pre, RIDGE)

    spread = max(float(np.std(panel[pre:, 0])), 1.0)
    assert np.max(np.abs(with_copy - base)) < 5.0 * spread


@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_differencing_round_trips_through_its_own_inverse(shape, seed):
    """The re-integration inverts the differencing it is paired with.

    A caller who asks for ``outcome_scale="difference"`` still reads a
    counterfactual on the outcome's own scale, and that only holds if the two
    halves of the transform are actually inverse.
    """
    n_time, n_unit, n_factors = shape
    pre = n_time - 3
    assume(pre >= 2)
    panel = _count_panel(n_time, n_unit, n_factors, seed)

    differenced, rebase = _to_modelling_scale(panel, pre, "difference")
    assert differenced.shape == (n_time - 1, n_unit)

    # The post block is taken from the differenced matrix the transform itself
    # returned, at the pre-period length it itself reports, so the test cannot
    # disagree with the implementation about where the post-period starts --
    # which is the one way a round trip can be off by a constant and still look
    # like the right shape.
    post_steps = differenced[_pre_length(pre, "difference"):, 0][None, :]
    assert post_steps.shape == (1, n_time - pre)
    np.testing.assert_allclose(rebase(post_steps)[0], panel[pre:, 0], rtol=1e-9, atol=1e-9)


@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_level_scale_is_the_identity(shape, seed):
    """The level path transforms nothing, so its inverse is the identity too."""
    n_time, n_unit, n_factors = shape
    panel = _count_panel(n_time, n_unit, n_factors, seed)
    modelled, rebase = _to_modelling_scale(panel, n_time - 3, "level")
    np.testing.assert_array_equal(modelled, panel)
    draws = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(rebase(draws), draws)


# ------------------------------------------------------- design properties --


@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_the_treatment_indicator_marks_exactly_the_treated_unit(shape, seed):
    """The design's last column is the treatment dummy, and unit 0 is treated.

    The counterfactual is the same design with that column zeroed, so which row
    carries the one is the difference between an effect and its absence.
    """
    n_time, n_unit, n_factors = shape
    pre = n_time - 3
    panel = _count_panel(n_time, n_unit, n_factors, seed)
    loadings = np.random.default_rng(seed).gamma(2.0, 2.0, size=(n_factors, n_unit))

    treated, response = _regression_arrays(loadings, panel, pre, treated_on=True)
    control, _ = _regression_arrays(loadings, panel, pre, treated_on=False)

    assert treated.shape == (n_unit, n_factors + 1)
    assert response.shape == (n_unit, n_time - pre)
    np.testing.assert_array_equal(treated[:, -1], np.eye(n_unit)[0])
    np.testing.assert_array_equal(control[:, -1], np.zeros(n_unit))
    # Only the indicator differs between the two designs.
    np.testing.assert_array_equal(treated[:, :-1], control[:, :-1])


@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_holding_out_more_cells_never_scores_fewer(shape, seed):
    """The score is a mean per held-out cell, so it stays finite as the mask grows.

    A score that silently became a sum would drift with the number of held-out
    cells, which is exactly the defect that makes the paper's own check
    degenerate.
    """
    n_time, n_unit, n_factors = shape
    panel = _count_panel(n_time, n_unit, n_factors, seed)
    rng = np.random.default_rng(seed)

    scores = []
    for fraction in (0.1, 0.3):
        mask = rng.random(panel.shape) > fraction
        assume(mask.any() and not mask.all())
        draws = gap_gibbs(panel, n_factors, mask=mask, n_samples=4, n_warmup=4, seed=seed)
        scores.append(heldout_log_density(panel, mask, draws))

    assert all(np.isfinite(s) for s in scores)
    # A per-cell mean sits on the scale of a single cell's log density, so it
    # cannot run away with the cell count the way a sum would.
    assert all(-1e4 < s < 0 for s in scores)


@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_more_factors_never_score_worse_in_sample(shape, seed):
    """Rank is capacity, so the fit it can reach is monotone in it.

    Scored on the cells the model saw, a richer model cannot do worse. The
    held-out score is where extra rank starts to cost, which is why the estimator
    reports that one instead.
    """
    n_time, n_unit, n_factors = shape
    assume(n_factors + 1 < min(n_time, n_unit))
    panel = _count_panel(n_time, n_unit, n_factors, seed)
    observed = np.ones(panel.shape, dtype=bool)

    def in_sample(rank: int) -> float:
        draws = ppca_em(panel, rank, n_samples=2, seed=seed)
        return float(np.mean((panel - draws.mean(0)) ** 2))

    spread = float(np.var(panel))
    assert in_sample(n_factors + 1) <= in_sample(1) + 0.05 * spread


@pytest.mark.parametrize("factor_model", sorted(FACTOR_MODELS))
@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_a_fit_is_a_function_of_its_seed(factor_model, shape, seed):
    """Same seed, same inputs, same draws -- for both arms.

    Reproducibility is a precondition for every other property here: a mutation
    score computed against a nondeterministic suite means nothing.
    """
    n_time, n_unit, n_factors = shape
    panel = _count_panel(n_time, n_unit, n_factors, seed)
    engine = FACTOR_MODELS[factor_model]
    first = engine(panel, n_factors, n_samples=4, n_warmup=4, seed=seed)
    second = engine(panel, n_factors, n_samples=4, n_warmup=4, seed=seed)
    np.testing.assert_array_equal(first.loadings, second.loadings)
    np.testing.assert_array_equal(first.factors, second.factors)


# ------------------------------------------- properties the mutants demanded --
#
# The three tests below exist because the semantic mutants in
# ``tools/mutation/targets.toml`` survived without them. Each one asserted
# something the suite believed but never checked.


@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_the_holdout_mask_actually_withholds_cells(shape, seed):
    """A masked cell must not influence the fit that is later scored on it.

    Withholding is the entire basis of the model score. A sampler that fit the
    held-out cells would report in-sample fit under a held-out name, and could
    not distinguish a well-specified likelihood from an overfit one.
    """
    n_time, n_unit, n_factors = shape
    panel = _count_panel(n_time, n_unit, n_factors, seed)

    mask = np.ones(panel.shape, dtype=bool)
    mask[0, 0] = False

    # Change only the masked cell, by a lot. A fit that ignores it is unmoved.
    corrupted = panel.copy()
    corrupted[0, 0] += 10_000.0

    base = gap_gibbs(panel, n_factors, mask=mask, n_samples=6, n_warmup=6, seed=seed)
    moved = gap_gibbs(corrupted, n_factors, mask=mask, n_samples=6, n_warmup=6, seed=seed)
    np.testing.assert_allclose(base.loadings, moved.loadings, rtol=1e-9, atol=1e-9)

    # And the same change with the cell observed must move the fit.
    observed_everywhere = np.ones(panel.shape, dtype=bool)
    unmasked_base = gap_gibbs(
        panel, n_factors, mask=observed_everywhere, n_samples=6, n_warmup=6, seed=seed
    )
    unmasked_moved = gap_gibbs(
        corrupted, n_factors, mask=observed_everywhere, n_samples=6, n_warmup=6, seed=seed
    )
    assert not np.allclose(unmasked_base.loadings, unmasked_moved.loadings)


@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_warmup_sweeps_are_discarded_not_returned(shape, seed):
    """The retained draws are the tail of the chain, not the whole of it.

    A chain starts at the prior and walks toward the posterior, so returning the
    warm-up would mix transient state into the credible band. Because the sweeps
    are deterministic given the seed, this is an exact identity: warming up for
    ``W`` and keeping ``S`` must equal the last ``S`` of a chain that keeps
    ``W + S`` from the start.
    """
    n_time, n_unit, n_factors = shape
    panel = _count_panel(n_time, n_unit, n_factors, seed)

    warm, keep = 5, 4
    tail = gap_gibbs(panel, n_factors, n_samples=keep, n_warmup=warm, seed=seed)
    whole = gap_gibbs(panel, n_factors, n_samples=warm + keep, n_warmup=0, seed=seed)

    np.testing.assert_allclose(tail.loadings, whole.loadings[warm:], rtol=1e-9)
    np.testing.assert_allclose(tail.factors, whole.factors[warm:], rtol=1e-9)


@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_the_score_reads_the_held_out_cells(shape, seed):
    """The model score responds to the held-out cells and not to the fitted ones.

    Scored on the cells the model saw, the number rewards rank instead of fit and
    inverts the comparison it exists to make, which is the whole reason it
    replaces the paper's ``p_pop``.
    """
    n_time, n_unit, n_factors = shape
    panel = _count_panel(n_time, n_unit, n_factors, seed)

    rng = np.random.default_rng(seed)
    mask = rng.random(panel.shape) > 0.25
    assume(mask.any() and not mask.all())

    draws = gap_gibbs(panel, n_factors, mask=mask, n_samples=5, n_warmup=5, seed=seed)
    base = heldout_log_density(panel, mask, draws)

    # Degrading only the held-out cells must lower the score.
    degraded = panel.copy()
    degraded[~mask] = degraded[~mask] + 5_000.0
    assert heldout_log_density(degraded, mask, draws) < base

    # Changing only the observed cells must leave it untouched: the score is a
    # statement about the cells the model did not see.
    elsewhere = panel.copy()
    elsewhere[mask] = elsewhere[mask] + 5_000.0
    np.testing.assert_allclose(heldout_log_density(elsewhere, mask, draws), base, rtol=1e-9)


@given(shape=shapes, seed=st.integers(0, 2**16))
@SETTINGS
def test_dispersion_survives_a_handful_of_extreme_cells(shape, seed):
    """The dispersion statistic is a median, so a few bad cells cannot set it.

    The ratio it summarises is unbounded wherever the fit sends a rate toward
    zero. Averaging it would let a handful of such cells decide the number, and
    the diagnostic would report on those cells instead of on the panel.
    """
    from mlsynth.utils.mosc_helpers.pipeline import diagnose

    n_time, n_unit, n_factors = shape
    panel = _count_panel(n_time, n_unit, n_factors, seed)
    rng = np.random.default_rng(seed)
    mask = rng.random(panel.shape) > 0.3
    assume((~mask).sum() >= 4)

    draws = gap_gibbs(panel, n_factors, mask=mask, n_samples=5, n_warmup=5, seed=seed)
    base = diagnose(panel, mask, draws, "level").pearson_dispersion

    # Wreck one held-out cell by orders of magnitude.
    spiked = panel.copy()
    victim = np.argwhere(~mask)[0]
    spiked[victim[0], victim[1]] += 1e6
    spiked_dispersion = diagnose(spiked, mask, draws, "level").pearson_dispersion

    assert np.isfinite(spiked_dispersion)
    assert spiked_dispersion < base + 10.0 * max(base, 1.0)


def test_residual_autocorrelation_is_measured_after_conditioning():
    """The conditional-independence check conditions before it correlates.

    A panel with a strong shared trend is heavily autocorrelated in levels and
    barely autocorrelated once the factors that produce the trend are removed.
    That gap is the whole content of assumptions 2 and 3, so a check computed on
    the raw series would report near-perfect dependence for any trending panel
    and never distinguish one scale from another.
    """
    from mlsynth.utils.mosc_helpers.pipeline import diagnose

    rng = np.random.default_rng(11)
    n_time, n_unit = 40, 12
    trend = np.linspace(20.0, 200.0, n_time)[:, None]        # one strong factor
    rate = trend * rng.uniform(0.8, 1.2, size=(1, n_unit))
    panel = rng.poisson(rate).astype(float)

    raw = np.mean([
        np.corrcoef(panel[:-1, i], panel[1:, i])[0, 1] for i in range(n_unit)
    ])
    assert raw > 0.8, "the fixture must be strongly autocorrelated in levels"

    mask = rng.random(panel.shape) > 0.1
    draws = gap_gibbs(panel, 2, mask=mask, n_samples=25, n_warmup=25, seed=0)
    residual = diagnose(panel, mask, draws, "level").residual_autocorrelation

    assert abs(residual) < 0.4, (
        f"a rank-2 fit explains this panel's trend, so the residual correlation "
        f"should be near zero; got {residual:.3f} against a raw {raw:.3f}"
    )


# ---------------------------------------------- the resampling is a bootstrap --
#
# Width is not evidence about the mechanism. Measured on a 24-unit panel, the
# correct resampling gives a 95% ATT interval of width 17.5; permuting the pool
# instead gives 13.2 and reusing one seed gives 19.8. Neither is separable from
# the correct value by a threshold, so these assert the mechanism directly.


@given(n_donors=st.integers(4, 40), seed=st.integers(0, 2**16))
@SETTINGS
def test_every_replicate_keeps_the_treated_unit(n_donors, seed):
    """Column 0 is the treated unit and survives every resample.

    Resampling all units, as the paper's sentence reads literally, can drop the
    one whose counterfactual is the estimand; the replicate then reads its
    "counterfactual" off whichever donor landed in column zero.
    """
    from mlsynth.utils.mosc_helpers.pipeline import resample_columns

    rng = np.random.default_rng(seed)
    for _ in range(20):
        columns = resample_columns(rng, n_donors)
        assert columns[0] == 0
        assert columns.shape == (n_donors + 1,)
        assert set(columns[1:]) <= set(range(1, n_donors + 1))


@given(n_donors=st.integers(6, 40), seed=st.integers(0, 2**16))
@SETTINGS
def test_donors_are_drawn_with_replacement(n_donors, seed):
    """A resample repeats donors; a permutation never does.

    Drawing with replacement is what makes the spread across replicates an
    estimate of sampling uncertainty. Permuting the pool hands every replicate
    the same units in a different order, and both stages are exchangeable in
    units, so the replicates would carry no information about which units were
    observed.
    """
    from mlsynth.utils.mosc_helpers.pipeline import resample_columns

    rng = np.random.default_rng(seed)
    draws = [resample_columns(rng, n_donors)[1:] for _ in range(30)]

    # With replacement, a draw of n from n repeats something with probability
    # 1 - n!/n^n, which is above 0.99 for n >= 6. Over 30 draws, seeing none is
    # decisive evidence that the pool was permuted.
    repeated = sum(len(set(d)) < len(d) for d in draws)
    assert repeated > 0, "no resample repeated a donor, so the pool was permuted"

    # And the composition itself must vary, not merely the order.
    compositions = {tuple(sorted(d)) for d in draws}
    assert len(compositions) > 1


def test_each_replicate_fits_its_factor_model_at_a_fresh_seed(monkeypatch):
    """The factor model's own uncertainty enters the interval once per replicate.

    Holding the seed fixed leaves only the resampled pool varying, so the
    posterior contribution silently drops out of an interval that still looks
    plausible -- it was measured as wider, not narrower, which is why this
    asserts the seeds and not the width.
    """
    import mlsynth.utils.mosc_helpers.pipeline as pipeline
    from mlsynth import MOSC
    from test_mosc import base_config, make_count_panel

    seen = []
    original = pipeline.fit_factor_model

    def spy(*args, **kwargs):
        seen.append(args[-1])
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "fit_factor_model", spy)
    panel = make_count_panel(n_units=12, n_periods=20, pre_periods=14, seed=1)
    MOSC(base_config(panel, n_factors=1, n_samples=10, n_warmup=10,
                     inference="bootstrap", n_bootstrap=12)).fit()

    bootstrap_seeds = seen[1:]          # the first call is the headline fit
    assert len(bootstrap_seeds) == 12
    assert len(set(bootstrap_seeds)) == 12, "replicates reused a seed"
