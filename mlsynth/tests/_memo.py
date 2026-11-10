"""Memoise a deterministic test helper by the content of its arguments.

Many test modules fit the same estimator on the same panel in several tests: the
fit is a deterministic function of its inputs, so each test that reads it computes
the same object again. ``memo`` computes it once per process and hands every later
caller the same object. Keys are built from the arguments' content, never their
identity, so a test that builds a different panel gets its own fit:

* a DataFrame or Series by its pandas row hashes, index, columns and dtypes;
* an ndarray by its dtype, shape and bytes;
* dicts, lists, tuples and sets by their elements, recursively;
* anything else by ``repr``.

A call that raises is not cached, so a failure test that expects an exception
sees it on every call. A cached result is shared: tests read it and never mutate
it. CI runs with ``--dist loadgroup`` and modules that share fits carry an
``xdist_group`` mark, so their tests run on one worker and share its cache."""
from __future__ import annotations

import functools
import hashlib

import numpy as np


def _key(obj):
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a dependency of the library
        pd = None
    if pd is not None and isinstance(obj, (pd.DataFrame, pd.Series)):
        h = hashlib.sha1(pd.util.hash_pandas_object(obj, index=True).values.tobytes()).hexdigest()
        if isinstance(obj, pd.DataFrame):
            cols, dtypes = tuple(map(str, obj.columns)), tuple(map(str, obj.dtypes))
        else:
            cols, dtypes = (str(obj.name),), (str(obj.dtype),)
        return ("pandas", type(obj).__name__, h, cols, dtypes)
    if isinstance(obj, np.ndarray):
        return ("ndarray", str(obj.dtype), obj.shape, hashlib.sha1(np.ascontiguousarray(obj).tobytes()).hexdigest())
    if isinstance(obj, dict):
        return ("dict", tuple(sorted((repr(k), _key(v)) for k, v in obj.items())))
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, tuple(_key(v) for v in obj))
    if isinstance(obj, (set, frozenset)):
        return ("set", tuple(sorted(repr(_key(v)) for v in obj)))
    return ("repr", repr(obj))


def memo(fn):
    """``fn`` with its results cached by the content of its arguments."""
    cache = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        k = (_key(args), _key(kwargs))
        if k not in cache:
            cache[k] = fn(*args, **kwargs)
        return cache[k]

    wrapper.cache = cache
    return wrapper
