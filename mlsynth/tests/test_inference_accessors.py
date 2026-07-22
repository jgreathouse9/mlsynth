"""Convenience accessors on the standardized :class:`InferenceResults`.

The estimator docs advertise short inference accessors -- ``res.inference.se``
(docs/ppscm.rst) and ``res.inference.ci`` (docs/microsynth.rst) -- but the model
only stored ``standard_error`` / ``ci_lower`` / ``ci_upper``, so those documented
examples raised ``AttributeError``. These pin the read-only ``se`` and ``ci``
views that back the documented surface.
"""
from __future__ import annotations

import pytest

from mlsynth.config_models import InferenceResults


# -- se: alias of standard_error -------------------------------------------------

def test_se_aliases_standard_error():
    inf = InferenceResults(standard_error=1.2345)
    assert inf.se == pytest.approx(1.2345)
    assert inf.se == inf.standard_error


def test_se_none_when_standard_error_unset():
    # No SE reported -> se is None, never an AttributeError (edge case).
    assert InferenceResults().se is None


def test_se_is_a_view_not_a_stored_field():
    # `se` must be computed from standard_error, never a separate field that
    # could drift out of sync.
    assert "se" not in InferenceResults.model_fields
    assert InferenceResults(standard_error=0.5).se == 0.5


# -- ci: (lower, upper) view -----------------------------------------------------

def test_ci_is_lower_upper_tuple():
    inf = InferenceResults(ci_lower=-2.0, ci_upper=3.5)
    assert inf.ci == (-2.0, 3.5)
    # subscriptable exactly as docs/microsynth.rst uses it: res.inference.ci[0/1]
    assert inf.ci[0] == -2.0
    assert inf.ci[1] == 3.5


def test_ci_tuple_when_bounds_unset():
    assert InferenceResults().ci == (None, None)


# -- the documented ppscm.rst usage now resolves ---------------------------------

def test_documented_ppscm_line_formats():
    # docs/ppscm.rst: f"Average ATT : {res.att:.3f}  (SE {res.inference.se:.3f})"
    inf = InferenceResults(standard_error=0.4210)
    assert f"SE {inf.se:.3f}" == "SE 0.421"
