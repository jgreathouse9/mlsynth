"""``WeightsResults.is_empty`` separates three states that look alike.

An estimator's weights container can be in one of three conditions, and a
caller needs to tell them apart:

* it has weights, on one of the four faces;
* it has weights that fit no face -- one mapping per cohort, factor matrices
  per treated unit -- and names where they are in ``weights_at``;
* it has no weights, and says so.

Only the fourth state is a defect: silence, where every face is ``None`` and
nothing points anywhere, so a caller reaching for ``donor_weights`` cannot tell
whether the estimator has none or simply did not say (#475).

The two distinctions below are the ones that were got wrong while building
this, and each was got wrong in a way that a conformance sweep reported as
good news:

* ``{}`` is a statement and ``None`` is not. The factor-model estimators set
  ``donor_weights={}`` to mean "checked, this method has none". A truthiness
  test reads that as silence and fails six correct estimators.
* ``summary_stats`` is not a statement. It carries prose -- "partially-pooled
  SC donor weights (per cohort)" -- that no caller can resolve to a location.
  Counting it lets exactly the estimators #475 was about pass unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.config_models import WeightsResults


class TestSilenceIsTheOnlyFailure:
    def test_a_bare_container_is_empty(self):
        assert WeightsResults().is_empty is True

    @pytest.mark.parametrize("kwargs", [
        {"donor_weights": {"a": 1.0}},
        {"time_weights": {0: 0.5}},
        {"unit_weights": np.zeros(3)},
        {"weights_at": ["donor_weights_by_cohort"]},
    ])
    def test_any_face_or_a_pointer_is_not_empty(self, kwargs):
        assert WeightsResults(**kwargs).is_empty is False


class TestAnExplicitNoneIsAStatement:
    """``{}`` means checked-and-none; ``None`` means nothing was said."""

    @pytest.mark.parametrize("kwargs", [
        {"donor_weights": {}},
        {"time_weights": {}},
        {"unit_weights": np.array([])},
        {"weights_at": []},
    ])
    def test_an_empty_face_is_not_silence(self, kwargs):
        assert WeightsResults(**kwargs).is_empty is False

    def test_the_distinction_is_none_and_not_truthiness(self):
        """The pair that a truthiness test collapses.

        Six factor-model estimators report ``donor_weights={}``. Reading that
        as silence fails all of them while changing nothing about the ones the
        contract is aimed at.
        """
        assert WeightsResults(donor_weights={}).is_empty is False
        assert WeightsResults(donor_weights=None).is_empty is True


class TestProseIsNotALocation:
    def test_summary_stats_alone_does_not_satisfy_the_contract(self):
        """The permissive reading that let #475's own estimators pass.

        ``summary_stats`` holds cardinality and a constraint label. A sentence
        saying the weights are per cohort is not a way to reach them, so it
        cannot stand in for a face or a pointer.
        """
        w = WeightsResults(summary_stats={
            "constraint": "partially-pooled SC donor weights (per cohort)"})
        assert w.is_empty is True

    def test_summary_stats_alongside_a_pointer_is_fine(self):
        w = WeightsResults(weights_at=["donor_weights_by_cohort"],
                           summary_stats={"constraint": "per cohort"})
        assert w.is_empty is False


class TestTheFaceStillWorks:
    def test_weight_vector_is_unaffected(self):
        w = WeightsResults(donor_weights={"a": 0.25, "b": 0.75})
        assert np.allclose(w.weight_vector, [0.25, 0.75])

    def test_weight_vector_of_an_explicit_none_is_an_empty_array(self):
        assert WeightsResults(donor_weights={}).weight_vector.size == 0

    def test_weight_vector_of_silence_is_none(self):
        assert WeightsResults().weight_vector is None
