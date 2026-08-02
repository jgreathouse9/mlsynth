"""Placebo inference for STACKEDSC: sampled placebo averages.

With more than one treated unit the permutation distribution is not a set of
placebo *units* but a set of placebo *averages*. Wiltshire's procedure, which
``allsynth`` automates, reassigns treatment to each donor in each treated unit's
pool, then builds a sampled average by drawing exactly one placebo path per
treated unit and averaging under the same ``gamma_i`` used for the estimate. The
number of such averages is the product of the pool sizes, so it is sampled
rather than enumerated -- 39 donors and 566 treated counties is 39 ** 566.

Three things drive the tests below.

The placebo paths inherit the estimator's own normalisation. A pseudo-treated
donor is indexed to 100 at the same cohort base period as everything else, and
its pool weights sum to one, so its gap at ``e = -1`` is exactly zero just as the
real one is. A placebo fitted on some other clock fails that check immediately.

The donor pool for a placebo is a choice. ``allsynth`` documents "i and the
remaining untreated units collectively constituting the donor pool for each j"
-- under the permutation the treated unit is a control. That reading reproduces
the published numbers, and it makes the placebo path a property of the pair
``(i, j)`` rather than of the cohort: 566 x 39 solves instead of 6 x 39.

The two p-values answer different questions. The RMSPE-ranked one is a rank
within the sampled distribution and needs no distributional assumption; the
variance one reads the sampled spread as a standard error and assumes normality.
Both are reported because ``allsynth`` reports both.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pd = pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
def _staggered(n_treated=4, n_donors=6, T=12, cohorts=(6, 8), effect=0.0,
               seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2)).cumsum(axis=0)
    rows = []
    for j in range(n_donors):
        load = rng.normal(size=2)
        base = 100.0 + 20.0 * rng.random()
        for t in range(T):
            rows.append({"unit": f"d{j}", "t": t, "adopt": np.nan,
                         "y": float(base + F[t] @ load + rng.normal() * 0.05),
                         "w": 1.0})
    for i in range(n_treated):
        g = cohorts[i % len(cohorts)]
        load = rng.normal(size=2)
        base = 100.0 + 20.0 * rng.random()
        for t in range(T):
            y = base + F[t] @ load + rng.normal() * 0.05
            if t >= g:
                y *= (1.0 + effect)
            rows.append({"unit": f"x{i}", "t": t, "adopt": float(g),
                         "y": float(y), "w": float(1 + i)})
    d = pd.DataFrame(rows)
    d["treat"] = ((~d.adopt.isna()) & (d.t >= d.adopt)).astype(int)
    return d


def _fit(df=None, **over):
    from mlsynth import STACKEDSC

    cfg = {"df": _staggered() if df is None else df, "outcome": "y",
           "treat": "treat", "unitid": "unit", "time": "t",
           "display_graphs": False}
    cfg.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return STACKEDSC(cfg).fit()


def _infer(df=None, **over):
    over.setdefault("inference", "placebo")
    over.setdefault("n_placebo_samples", 60)
    over.setdefault("seed", 11)
    return _fit(df=df, **over)


# ---------------------------------------------------------------------------
class TestSmoke:
    def test_it_populates_the_standard_inference_block(self):
        res = _infer()
        inf = res.inference
        assert inf.method is not None and "placebo" in inf.method
        assert np.isfinite(inf.standard_error)
        assert np.isfinite(inf.p_value)
        assert np.isfinite(inf.ci_lower) and np.isfinite(inf.ci_upper)
        assert inf.confidence_level == pytest.approx(0.95)

    def test_it_populates_the_typed_placebo_block(self):
        res = _infer()
        pb = res.placebo
        assert pb is not None
        assert pb.placebo_averages.shape == (60, len(pb.horizons))
        assert np.isfinite(pb.placebo_averages).all()

    def test_the_detail_field_points_at_the_same_block(self):
        res = _infer()
        assert res.inference.details is res.placebo


# ---------------------------------------------------------------------------
class TestItIsOffUnlessAskedFor:
    """``allsynth`` requires ``pvalues()`` and warns that it "will greatly
    extend the run-time". The cost is a product of treated units and donors, so
    it is opt-in here for the same reason."""

    def test_none_is_the_default(self):
        from mlsynth.utils.stackedsc_helpers.config import STACKEDSCConfig

        assert STACKEDSCConfig.model_fields["inference"].default == "none"

    def test_nothing_is_computed_when_it_is_off(self):
        res = _fit()
        assert res.placebo is None
        assert res.inference.p_value is None
        assert res.inference.standard_error is None


# ---------------------------------------------------------------------------
class TestThePlacebosShareTheEstimatorsClock:
    def test_every_placebo_average_is_zero_at_the_base_period(self):
        """A pseudo-treated donor is indexed to 100 at the same cohort base
        period, and its pool weights sum to one, so its gap at ``e = -1`` is
        identically zero -- and therefore so is any weighted average of such
        paths. A placebo fitted on some other clock breaks this at once."""
        res = _infer()
        e = np.asarray(res.placebo.horizons, int)
        P = np.asarray(res.placebo.placebo_averages, float)
        assert np.abs(P[:, e == -1]).max() < 1e-8

    def test_the_horizons_match_the_event_study(self):
        res = _infer()
        np.testing.assert_array_equal(
            np.asarray(res.placebo.horizons, int),
            np.asarray(res.event_study.horizons, int))


# ---------------------------------------------------------------------------
class TestTheSampling:
    def test_it_returns_the_requested_number_of_averages(self):
        res = _infer(n_placebo_samples=45)
        assert res.placebo.n_samples == 45
        assert res.placebo.placebo_averages.shape[0] == 45

    def test_the_same_seed_gives_the_same_distribution(self):
        a, b = _infer(seed=7), _infer(seed=7)
        np.testing.assert_allclose(a.placebo.placebo_averages,
                                   b.placebo.placebo_averages)

    def test_a_different_seed_gives_a_different_draw(self):
        a, b = _infer(seed=7), _infer(seed=8)
        assert not np.allclose(a.placebo.placebo_averages,
                               b.placebo.placebo_averages)

    def test_the_aggregation_weights_are_the_estimators_own(self):
        """Putting all the aggregation mass on one treated unit makes each
        sampled average exactly one of that unit's placebo paths -- the average
        is a weighted mean, so this is exact rather than approximate."""
        df = _staggered()
        df["only"] = (df.unit == "x2").astype(float)
        res = _infer(df=df, agg_weights="only")
        paths = np.asarray(res.placebo.paths["x2"], float)
        for row in np.asarray(res.placebo.placebo_averages, float):
            assert np.abs(paths - row).max(axis=1).min() < 1e-8


# ---------------------------------------------------------------------------
class TestTheDonorPoolChoice:
    def test_the_permutation_pool_is_the_default(self):
        from mlsynth.utils.stackedsc_helpers.config import STACKEDSCConfig

        assert (STACKEDSCConfig.model_fields["placebo_donor_pool"].default
                == "permutation")

    def test_the_permutation_pool_fits_one_path_per_treated_donor_pair(self):
        res = _infer()
        assert res.placebo.n_fits == 4 * 6

    def test_dropping_the_treated_unit_makes_the_path_a_cohort_property(self):
        """Without the treated unit in the pool, every unit in a cohort faces
        the same design and the same targets, so one solve per (cohort, donor)
        serves the whole cohort."""
        res = _infer(placebo_donor_pool="donors-only")
        assert res.placebo.n_fits == 2 * 6
        a = np.asarray(res.placebo.paths["x0"], float)
        b = np.asarray(res.placebo.paths["x2"], float)   # same cohort
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_the_two_pools_do_not_give_the_same_answer(self):
        """Including the treated unit changes each placebo's design matrix, and
        under the alternative its post-treatment path carries the very effect
        being tested."""
        a = _infer().placebo.se
        b = _infer(placebo_donor_pool="donors-only").placebo.se
        assert not np.allclose(a, b, atol=1e-8)

    def test_the_choice_is_recorded(self):
        for pool in ("permutation", "donors-only"):
            res = _infer(placebo_donor_pool=pool)
            assert res.placebo.donor_pool == pool
            assert res.method_details.parameters_used[
                "placebo_donor_pool"] == pool


# ---------------------------------------------------------------------------
class TestTheTwoPValues:
    def test_the_rmspe_p_values_are_reported_for_every_post_horizon(self):
        res = _infer()
        e = np.asarray(res.placebo.horizons, int)
        post = sorted(int(x) for x in e[e >= 0])
        assert sorted(res.placebo.rmspe_p) == post
        assert all(0.0 <= p <= 1.0 for p in res.placebo.rmspe_p.values())

    def test_the_rmspe_statistic_is_a_post_over_pre_ratio(self):
        """``RMSPE_E`` is the mean squared post gap through ``E`` over the mean
        squared pre gap, so it is recomputable from the reported event path."""
        res = _infer()
        e = np.asarray(res.placebo.horizons, int)
        tau = np.asarray(res.event_study.tau, float)
        pre = float(np.mean(tau[e < 0] ** 2))
        E = int(e.max())
        want = float(np.mean(tau[(e >= 0) & (e <= E)] ** 2)) / pre
        assert res.placebo.rmspe_actual[E] == pytest.approx(want, rel=1e-9)

    def test_a_placebo_with_a_perfect_pre_fit_counts_rather_than_vanishing(self):
        """Its ratio has a zero denominator. Taking the limit -- unbounded when
        it has a post-treatment gap -- keeps it in the numerator; discarding
        the row would shrink ``p_E``, because the denominator is ``S + 1``
        either way."""
        from mlsynth.utils.stackedsc_helpers.inference import _rmspe

        grid = np.array([-2, -1, 0, 1])
        tau = np.array([1.0, 0.0, 4.0, 4.0])
        placebo = np.array([
            [0.0, 0.0, 9.0, 9.0],     # perfect pre-fit, large post gap
            [3.0, 0.0, 0.1, 0.1],     # poor pre-fit, small post gap
            [0.0, 0.0, 0.0, 0.0],     # fits everywhere: ratio is 0, not inf
        ])
        actual, p = _rmspe(grid, tau, placebo)
        assert actual[1] == pytest.approx(16.0 / 0.5)
        assert p[1] == pytest.approx(1.0 / 4.0)

    def test_the_variance_interval_is_symmetric_around_the_estimate(self):
        res = _infer()
        e = np.asarray(res.placebo.horizons, int)
        tau = np.asarray(res.event_study.tau, float)
        lo = np.asarray(res.placebo.ci_lower, float)
        hi = np.asarray(res.placebo.ci_upper, float)
        np.testing.assert_allclose(0.5 * (lo + hi), tau, atol=1e-9)
        assert (hi >= lo).all()

    def test_the_headline_interval_brackets_the_att(self):
        res = _infer()
        assert res.inference.ci_lower <= res.effects.att <= res.inference.ci_upper

    def test_a_narrower_level_gives_a_wider_interval(self):
        wide = _infer(alpha=0.01).inference
        narrow = _infer(alpha=0.10).inference
        assert (wide.ci_upper - wide.ci_lower) > (narrow.ci_upper
                                                  - narrow.ci_lower)

    def test_a_planted_effect_is_harder_to_call_noise(self):
        """Not a claim about size or power -- only that the machinery moves in
        the right direction when a large effect is present."""
        null = _infer(df=_staggered(effect=0.0, seed=5))
        real = _infer(df=_staggered(effect=0.50, seed=5))
        assert real.inference.p_value < null.inference.p_value


# ---------------------------------------------------------------------------
class TestItComposesWithTheEstimatorsOptions:
    def test_the_estimate_is_untouched_by_asking_for_inference(self):
        plain, with_p = _fit(), _infer()
        assert plain.effects.att == pytest.approx(with_p.effects.att, rel=1e-12)

    def test_the_placebos_are_bias_corrected_when_the_estimate_is(self):
        df = _staggered()
        df["x1"] = df.groupby("unit").y.transform("mean")
        plain = _infer(df=df, covariates=["x1"])
        corrected = _infer(df=df, covariates=["x1"], bias_correct=True)
        assert not np.allclose(plain.placebo.se, corrected.placebo.se,
                               atol=1e-10)

    def test_a_donor_predicate_restricts_the_placebo_pool_too(self):
        """If donor ``d0`` may not serve treated unit ``x0``, it must not appear
        as a placebo for ``x0`` either -- the permutation is over that unit's
        own pool."""
        res = _infer(donor_predicate=lambda i, d: not (i == "x0"
                                                       and d == "d0"))
        assert np.asarray(res.placebo.paths["x0"]).shape[0] == 5
        assert np.asarray(res.placebo.paths["x1"]).shape[0] == 6


# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_one_treated_unit_still_gets_a_distribution(self):
        res = _infer(df=_staggered(n_treated=1, cohorts=(7,)))
        assert res.placebo.placebo_averages.shape[0] == 60
        assert np.isfinite(res.inference.standard_error)

    def test_two_donors_is_the_smallest_workable_pool(self):
        res = _infer(df=_staggered(n_donors=2))
        assert res.placebo.n_fits == 4 * 2


# ---------------------------------------------------------------------------
class TestThePlot:
    def test_it_draws_the_interval_when_the_distribution_exists(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from mlsynth.utils.stackedsc_helpers.plotter import plot_stackedsc

        out = tmp_path / "es.png"
        plot_stackedsc(_infer(), save=str(out))
        assert out.exists()

    def test_it_still_draws_without_one(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from mlsynth.utils.stackedsc_helpers.plotter import plot_stackedsc

        out = tmp_path / "plain.png"
        plot_stackedsc(_fit(), save=str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
class TestFailuresAreReported:
    def test_a_single_donor_cannot_form_a_placebo(self):
        """Reassigning the only donor leaves nothing to fit it against, so the
        design admits no placebo distribution at all and says so instead of
        returning an empty one."""
        from mlsynth.exceptions import MlsynthDataError

        with pytest.raises((MlsynthDataError, ValueError), match="placebo"):
            _infer(df=_staggered(n_donors=1))

    def test_too_few_samples_is_a_configuration_error(self):
        """``allsynth`` requires ``sampleavgs()`` to be at least 30, and the
        floor is carried over rather than left to the caller."""
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises((MlsynthConfigError, ValueError)):
            _infer(n_placebo_samples=5)

    def test_an_unknown_inference_mode_is_rejected(self):
        with pytest.raises(Exception):
            _fit(inference="bootstrap")

    def test_a_pre_window_of_one_period_cannot_rank_by_rmspe(self):
        """The RMSPE denominator is the mean squared pre-treatment gap, and at
        ``e = -1`` that gap is identically zero under the normalisation. With no
        other pre-period the ratio is 0/0, which is reported as undefined rather
        than as a p-value."""
        res = _infer(n_lags=1)
        assert all(np.isnan(v) for v in res.placebo.rmspe_p.values())
        assert np.isfinite(res.inference.standard_error)
