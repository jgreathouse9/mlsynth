"""The test-suite memo (mlsynth/tests/_memo.py): a fit is computed once for each
distinct content of its arguments, and never shared across different content."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.tests._memo import _key, memo


def _counter():
    calls = []

    @memo
    def f(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    return f, calls


def _panel(v=1.0):
    return pd.DataFrame({"unit": ["a", "a", "b", "b"], "time": [1, 2, 1, 2], "y": [v, 2.0, 3.0, 4.0]})


# ---------------------------------------------------------------- smoke


def test_smoke_a_repeated_call_is_computed_once():
    f, calls = _counter()
    a = f(_panel(), alpha=0.1)
    b = f(_panel(), alpha=0.1)
    assert a is b and len(calls) == 1


# ---------------------------------------------------------------- unit


def test_a_different_value_is_a_different_key():
    f, calls = _counter()
    assert f(_panel(1.0)) is not f(_panel(1.5))
    assert len(calls) == 2


def test_a_different_keyword_is_a_different_key():
    f, calls = _counter()
    f(_panel(), alpha=0.1); f(_panel(), alpha=0.2); f(_panel(), beta=0.1)
    assert len(calls) == 3


def test_a_different_column_name_or_index_is_a_different_key():
    f, calls = _counter()
    p = _panel()
    f(p); f(p.rename(columns={"y": "z"})); f(p.set_index([pd.Index([9, 8, 7, 6])]))
    assert len(calls) == 3


def test_a_different_dtype_is_a_different_key():
    f, calls = _counter()
    p = _panel()
    f(p); f(p.astype({"time": "float64"}))
    assert len(calls) == 2


def test_dict_order_does_not_matter():
    f, calls = _counter()
    f({"a": 1, "b": _panel()}); f({"b": _panel(), "a": 1})
    assert len(calls) == 1


def test_arrays_key_by_content_shape_and_dtype():
    f, calls = _counter()
    x = np.arange(6.0)
    f(x); f(x.copy()); f(x.reshape(2, 3)); f(x.astype(np.float32))
    assert len(calls) == 3


def test_an_exception_is_not_cached():
    calls = []

    @memo
    def boom(x):
        calls.append(x)
        raise ValueError("refused")

    for _ in range(2):
        with pytest.raises(ValueError, match="refused"):
            boom(1)
    assert len(calls) == 2


# ---------------------------------------------------------------- edge


def test_no_arguments_and_empty_frames():
    f, calls = _counter()
    f(); f()
    f(pd.DataFrame()); f(pd.DataFrame())
    f(np.empty(0)); f(np.empty(0))
    assert len(calls) == 3


def test_a_series_and_nested_containers():
    f, calls = _counter()
    s = pd.Series([1.0, 2.0], name="y")
    f([s, (1, {2})]); f([s.copy(), (1, {2})]); f([s.rename("z"), (1, {2})])
    assert len(calls) == 2


# ---------------------------------------------------------------- properties


_frames = st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=1, max_size=8).map(
    lambda v: pd.DataFrame({"y": v, "t": range(len(v))}))


@settings(max_examples=60, deadline=None)
@given(a=_frames, b=_frames)
def test_property_keys_agree_exactly_when_the_frames_are_equal(a, b):
    """Equal content gives equal keys; any difference in values or length gives a
    different key."""
    same = a.shape == b.shape and a.equals(b)
    assert (_key(a) == _key(b)) == same


@settings(max_examples=60, deadline=None)
@given(v=st.lists(st.integers(-5, 5), max_size=6), extra=st.dictionaries(st.text(max_size=3), st.integers(), max_size=3))
def test_property_one_computation_per_distinct_argument(v, extra):
    f, calls = _counter()
    for _ in range(3):
        f(np.asarray(v, dtype=float), **extra)
    assert len(calls) == 1
