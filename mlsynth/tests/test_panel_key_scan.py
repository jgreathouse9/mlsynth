"""The panel's key structure is scanned once per panel, not once per pivot.

``dataprep`` reads a long panel by pivoting it, and every pivot begins by
establishing the same thing: which unit and which period each row belongs to.
``_fast_pivot`` factorizes both key columns, builds a linear cell index, counts
it to rule out duplicate keys, and sorts both axes -- and all of that depends
on ``(df, index, columns)`` alone, never on ``values``. A single-treated
``dataprep`` calls it twice, once for the treatment indicator and once for the
outcome, so the second call recomputes the first's answer exactly. The cohort
branch is worse: it rebuilds the whole treatment pivot that the pre-branch code
already built.

Two kinds of test live here, red for different reasons.

The work assertions start red. They count key-column ``factorize`` calls and
``_wide_pivot`` calls per ``dataprep`` and hold them to one scan per panel and
one pivot per distinct ``values`` column. That is the contract rung 4 of the
analysis in #459 found nobody had written down: ``dataprep``'s contract names
its outputs and says nothing about how many passes over the panel may be spent
producing them, so a second pivot conformed and cost nothing at review.
Counting is deterministic, which is why the assertion is a count and not a
wall-clock threshold.

The equivalence tests pass before the change and must pass after it. They are
the whole acceptance bar, since 51 modules read what ``dataprep`` returns. They
check three layers, and the third is easy to lose: the values, the failures and
which one fires, and the memory layout -- ``_wide_pivot`` verifies that its fast
path reproduces ``df.pivot``'s C-contiguous ``.to_numpy()``, because a
value-equal but F-contiguous array changes BLAS reduction order and moves
float-sensitive downstream code at ~1e-16.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils import datautils as DU
from mlsynth.utils.datautils import balance, dataprep


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def make_panel(n_units=6, n_periods=10, n_treated=1, onset=7, seed=0,
               unit="id", time="time", outcome="y", treat="d",
               covariates=None):
    """A balanced long panel with ``n_treated`` units adopting at ``onset``."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(1, n_periods + 1):
            treated = int(u < n_treated and t >= onset)
            rows.append((u, t, float(rng.normal()) + 2.0 * treated, treated))
    df = pd.DataFrame(rows, columns=[unit, time, outcome, treat])
    if covariates:
        for k, name in enumerate(covariates):
            df[name] = rng.normal(size=len(df)) + k
    return df


def make_staggered(n_units=9, n_periods=12, onsets=(7, 9), seed=1):
    """A panel whose treated units adopt at different times (cohort mode)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        onset = onsets[u] if u < len(onsets) else None
        for t in range(1, n_periods + 1):
            treated = int(onset is not None and t >= onset)
            rows.append((u, t, float(rng.normal()) + 2.0 * treated, treated))
    return pd.DataFrame(rows, columns=["id", "time", "y", "d"])


ARGS = ("id", "time", "y", "d")


# --------------------------------------------------------------------------- #
# counters
# --------------------------------------------------------------------------- #
class KeyScanCounter:
    """Counts ``pd.factorize`` calls on the panel's key columns."""

    def __init__(self, monkeypatch, *key_columns):
        self.key_columns = set(key_columns)
        self.calls: list[str] = []
        real = pd.factorize

        def traced(values, *a, **kw):
            name = getattr(values, "name", None)
            if name in self.key_columns:
                self.calls.append(name)
            return real(values, *a, **kw)

        monkeypatch.setattr(pd, "factorize", traced)
        monkeypatch.setattr(DU.pd, "factorize", traced)

    @property
    def total(self) -> int:
        return len(self.calls)

    def per_column(self, column: str) -> int:
        return self.calls.count(column)


class PivotCounter:
    """Records the ``(index, columns, values)`` of every ``_wide_pivot`` call."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple] = []
        real = DU._wide_pivot

        def traced(df, index, columns, values, **kw):
            self.calls.append((index, columns, values))
            return real(df, index=index, columns=columns, values=values, **kw)

        monkeypatch.setattr(DU, "_wide_pivot", traced)

    @property
    def total(self) -> int:
        return len(self.calls)

    @property
    def distinct(self) -> int:
        return len(set(self.calls))


# =========================================================================== #
# work assertions -- these start red
# =========================================================================== #
class TestScanCount:
    """One scan per panel, whatever the branch."""

    def test_single_treated_dataprep_scans_each_key_column_once(self, monkeypatch):
        df = make_panel()
        counter = KeyScanCounter(monkeypatch, "id", "time")
        dataprep(df, *ARGS)
        assert counter.per_column("id") == 1
        assert counter.per_column("time") == 1

    def test_cohort_dataprep_scans_each_key_column_once(self, monkeypatch):
        df = make_staggered()
        counter = KeyScanCounter(monkeypatch, "id", "time")
        out = dataprep(df, *ARGS)
        assert "cohorts" in out, "fixture must exercise the cohort branch"
        assert counter.per_column("id") == 1
        assert counter.per_column("time") == 1

    def test_dataprep_with_covariates_scans_each_key_column_once(self, monkeypatch):
        df = make_panel(covariates=["x1", "x2"])
        counter = KeyScanCounter(monkeypatch, "id", "time")
        dataprep(df, *ARGS, covariates=["x1", "x2"])
        assert counter.per_column("id") == 1
        assert counter.per_column("time") == 1

    def test_supplied_keys_remove_the_scan_entirely(self, monkeypatch):
        df = make_panel()
        keys = balance(df, "id", "time")
        counter = KeyScanCounter(monkeypatch, "id", "time")
        dataprep(df, *ARGS, keys=keys)
        assert counter.total == 0

    def test_a_fit_shaped_call_pair_scans_once(self, monkeypatch):
        """``balance`` then ``dataprep`` -- the pattern at nine call sites."""
        df = make_panel()
        counter = KeyScanCounter(monkeypatch, "id", "time")
        keys = balance(df, "id", "time")
        dataprep(df, *ARGS, keys=keys)
        assert counter.per_column("id") == 1
        assert counter.per_column("time") == 1


class TestPivotCount:
    """A pivot per distinct ``values`` column, and no more."""

    def test_single_treated_pivots_twice(self, monkeypatch):
        df = make_panel()
        counter = PivotCounter(monkeypatch)
        dataprep(df, *ARGS)
        assert counter.total == 2
        assert counter.distinct == 2

    def test_cohort_does_not_pivot_the_treatment_indicator_twice(self, monkeypatch):
        df = make_staggered()
        counter = PivotCounter(monkeypatch)
        out = dataprep(df, *ARGS)
        assert "cohorts" in out, "fixture must exercise the cohort branch"
        assert counter.total == counter.distinct, (
            f"a (index, columns, values) signature was pivoted twice: "
            f"{counter.calls}")
        assert counter.total == 2


# =========================================================================== #
# equivalence -- green before and after
# =========================================================================== #
def _compare(a, b, path="prep"):
    """Assert two ``dataprep`` payloads agree, recursing into containers."""
    assert type(a) is type(b), f"{path}: {type(a)} vs {type(b)}"
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys differ"
        for k in a:
            _compare(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=path)
        assert (a.to_numpy().flags["C_CONTIGUOUS"]
                == b.to_numpy().flags["C_CONTIGUOUS"]), f"{path}: layout differs"
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, check_exact=True, obj=path)
    elif isinstance(a, pd.Index):
        pd.testing.assert_index_equal(a, b, exact=True, obj=path)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype, f"{path}: dtype {a.dtype} vs {b.dtype}"
        assert a.shape == b.shape, f"{path}: shape {a.shape} vs {b.shape}"
        np.testing.assert_array_equal(a, b, err_msg=path)
        assert a.flags["C_CONTIGUOUS"] == b.flags["C_CONTIGUOUS"], f"{path}: layout"
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length"
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, f"{path}[{i}]")
    else:
        same = a == b
        if hasattr(same, "__len__") or isinstance(same, np.ndarray):
            assert np.array_equal(np.asarray(a), np.asarray(b)), f"{path}"
        else:
            assert same or (a != a and b != b), f"{path}: {a!r} vs {b!r}"


PANELS = {
    "single-treated": lambda: make_panel(),
    "cohort": lambda: make_staggered(),
    "one-donor": lambda: make_panel(n_units=2, n_periods=8, onset=6),
    "single-post-period": lambda: make_panel(n_periods=6, onset=6),
    "wide-short": lambda: make_panel(n_units=25, n_periods=4, onset=3),
    "tall-narrow": lambda: make_panel(n_units=3, n_periods=60, onset=41),
    "string-units": lambda: make_panel().assign(
        id=lambda d: "u" + d["id"].astype(str)),
    "unsorted-rows": lambda: make_panel().sample(frac=1.0, random_state=7),
    "float-outcome-nan-free": lambda: make_panel(seed=3),
}


class TestOutputUnchanged:
    """The payload, to the byte, with and without a supplied scan."""

    @pytest.mark.parametrize("name", sorted(PANELS))
    def test_supplied_keys_give_the_same_payload(self, name):
        df = PANELS[name]()
        without = dataprep(df, *ARGS)
        keys = balance(df, "id", "time")
        with_keys = dataprep(df, *ARGS, keys=keys)
        _compare(without, with_keys)

    @pytest.mark.parametrize("name", sorted(PANELS))
    def test_pivots_still_match_pandas(self, name):
        df = PANELS[name]()
        keys = balance(df, "id", "time")
        for values in ("y", "d"):
            reference = df.pivot(index="time", columns="id", values=values)
            shared = DU._wide_pivot(df, index="time", columns="id",
                                    values=values, keys=keys)
            pd.testing.assert_frame_equal(shared, reference, check_exact=True)
            assert shared.to_numpy().flags["C_CONTIGUOUS"]

    def test_covariate_payload_unchanged(self):
        df = make_panel(covariates=["x1", "x2"])
        without = dataprep(df, *ARGS, covariates=["x1", "x2"])
        keys = balance(df, "id", "time")
        with_keys = dataprep(df, *ARGS, covariates=["x1", "x2"], keys=keys)
        _compare(without, with_keys)

    def test_cohort_treatment_matrix_is_the_pre_branch_one(self):
        """The frame the cohort branch used to rebuild equals the one built
        before the branch, so reusing it cannot move a cohort onset."""
        df = make_staggered()
        first = DU._wide_pivot(df, index="time", columns="id", values="d")
        second = DU._wide_pivot(df, index="time", columns="id", values="d")
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        out = dataprep(df, *ARGS)
        assert "cohorts" in out
        assert sorted(out["cohorts"]) == sorted(
            {7, 9}), "cohort onsets must be unchanged"


# =========================================================================== #
# the balance contract
# =========================================================================== #
class TestBalanceStillDecides:
    """A new return value may not move which error fires, or when."""

    def test_returns_a_scan_of_the_panel(self):
        df = make_panel()
        keys = balance(df, "id", "time")
        assert keys is not None
        expected_units = pd.factorize(df["id"], sort=False)[0]
        expected_times = pd.factorize(df["time"], sort=False)[0]
        np.testing.assert_array_equal(keys.unit_codes, expected_units)
        np.testing.assert_array_equal(keys.time_codes, expected_times)

    def test_duplicate_rows_raise(self):
        df = pd.concat([make_panel(), make_panel().head(1)], ignore_index=True)
        with pytest.raises(MlsynthDataError, match="Duplicate observations"):
            balance(df, "id", "time")

    def test_missing_period_raises_not_strongly_balanced(self):
        df = make_panel().drop(index=3).reset_index(drop=True)
        with pytest.raises(MlsynthDataError, match="not strongly balanced"):
            balance(df, "id", "time")

    def test_missing_key_raises_before_any_wide_frame(self):
        df = make_panel()
        df.loc[2, "time"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing values found"):
            balance(df, "id", "time")

    def test_dataprep_alone_still_refuses_a_missing_key(self):
        df = make_panel()
        df.loc[2, "id"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing values found"):
            dataprep(df, *ARGS)

    def test_ignoring_the_return_value_still_works(self):
        """The 38 call sites that discard it must be unaffected."""
        df = make_panel()
        assert balance(df, "id", "time") is not None
        balance(df, "id", "time")          # discarded, as they do
        dataprep(df, *ARGS)                # and the fit proceeds


# =========================================================================== #
# the keys guard
# =========================================================================== #
class TestKeysGuard:
    """A scan may only be reused for the panel and columns it was taken from."""

    def test_keys_from_another_frame_are_refused(self):
        df = make_panel()
        other = make_panel(seed=99)
        keys = balance(other, "id", "time")
        with pytest.raises(MlsynthDataError, match="different panel|not for"):
            dataprep(df, *ARGS, keys=keys)

    def test_keys_for_other_columns_are_refused(self):
        df = make_panel().rename(columns={"id": "unit"})
        df["id"] = df["unit"]
        keys = balance(df, "unit", "time")
        with pytest.raises(MlsynthDataError, match="different panel|not for"):
            dataprep(df, *ARGS, keys=keys)

    def test_keys_taken_from_a_frame_that_then_changed_are_refused(self):
        df = make_panel()
        keys = balance(df, "id", "time")
        shorter = df.drop(index=0)
        with pytest.raises(MlsynthDataError, match="different panel|not for"):
            dataprep(shorter, *ARGS, keys=keys)

    def test_none_keys_are_the_default(self):
        df = make_panel()
        _compare(dataprep(df, *ARGS), dataprep(df, *ARGS, keys=None))


# =========================================================================== #
# edges the fast path declines
# =========================================================================== #
class TestFastPathDeclines:
    """A shared scan may not change when pandas takes over."""

    def test_sparse_grid_falls_back_and_still_matches_pandas(self):
        # one row per unit: the grid is n_units x n_units, far sparser than 2n
        rows = [(u, u, float(u), 0) for u in range(40)]
        df = pd.DataFrame(rows, columns=["id", "time", "y", "d"])
        assert DU._fast_pivot(df, "time", "id", "y") is None
        pd.testing.assert_frame_equal(
            DU._wide_pivot(df, index="time", columns="id", values="y"),
            df.pivot(index="time", columns="id", values="y"),
            check_exact=True)

    def test_duplicate_keys_reach_pandas(self):
        df = pd.concat([make_panel(), make_panel().head(1)], ignore_index=True)
        assert DU._fast_pivot(df, "time", "id", "y") is None
        with pytest.raises(ValueError):
            DU._wide_pivot(df, index="time", columns="id", values="y")

    def test_empty_frame_declines(self):
        df = pd.DataFrame({"id": [], "time": [], "y": [], "d": []})
        assert DU._fast_pivot(df, "time", "id", "y") is None

    def test_single_cell_panel(self):
        df = pd.DataFrame({"id": [0], "time": [1], "y": [1.5], "d": [0]})
        pd.testing.assert_frame_equal(
            DU._wide_pivot(df, index="time", columns="id", values="y"),
            df.pivot(index="time", columns="id", values="y"),
            check_exact=True)
