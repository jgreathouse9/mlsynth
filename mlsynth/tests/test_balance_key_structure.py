"""``balance`` answers from the panel's key codes, not from two hash passes.

Every estimator validates its panel with ``datautils.balance`` and then reads it
with ``datautils.dataprep``, and between them the ``(unit, time)`` key structure
is established three times by three different methods. ``balance`` uses the
slowest of the three -- ``df.duplicated([unit, time])`` and
``df.groupby(unit)[time].nunique()``, two full hash passes -- to reach an answer
that ``pandas.factorize`` plus a count gives for a quarter of the cost. It is 35
to 47 percent of ingestion, and ingestion is 78 percent of a small SDID fit.

Two kinds of test live here, and they are red for different reasons.

The differential tests characterise what ``balance`` decides *today*, against an
oracle that is a verbatim copy of the shipped implementation. They pass before
the change and must pass after it: that is the whole acceptance bar, since 35
estimator modules depend on which error fires for which malformed panel.

The work assertion is the one that starts red. It says a validator may not spend
a hash pass on a question the key codes already answer, which is the contract
rung 5 of the analysis found nobody had written down.

The sharp edges, established by probing the shipped code rather than by reading
it: a NaN in either key column raises "not strongly balanced", because
``groupby`` drops NaN groups and the unit carrying it then looks short; and an
empty frame raises "Units have different numbers of observations", because an
empty ``nunique()`` is 0 and not 1. Both are behaviour, so both are pinned.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.datautils import balance


# --------------------------------------------------------------------------- #
# the oracle: the implementation as it stood before this change
# --------------------------------------------------------------------------- #
def _balance_oracle(df, unit_id_column_name, time_period_column_name):
    """Verbatim copy of the shipped ``balance``, kept so the differential test
    survives the change to the real one."""
    if df.duplicated([unit_id_column_name, time_period_column_name]).any():
        raise MlsynthDataError(
            "Duplicate observations found. Ensure each combination of unit and "
            "time is unique."
        )
    total_unique_time_periods = df[time_period_column_name].nunique()
    observations_per_unit = (
        df.groupby(unit_id_column_name)[time_period_column_name].nunique())
    if not (observations_per_unit == total_unique_time_periods).all():
        raise MlsynthDataError(
            "The panel is not strongly balanced. Not all units have observations "
            "for all unique time periods in the dataset."
        )
    if observations_per_unit.nunique() != 1:
        raise MlsynthDataError(
            "The panel is not strongly balanced. Units have different numbers "
            "of observations."
        )


def _has_blank_key(df):
    """Whether the frame carries a row with no unit or no time.

    Such frames leave the differential's scope. The oracle is the implementation
    as it stood before this rewrite, and #453 changed the verdict on them
    deliberately -- from silently admitted to refused -- so asking whether the
    two agree is the wrong question. ``test_missing_key_rejection.py`` asserts
    the refusal directly instead.
    """
    return bool(df["id"].isna().any() or df["time"].isna().any())


def _outcome(fn, df):
    """``None`` if it passes, else ``(type name, message)``."""
    try:
        fn(df, "id", "time")
        return None
    except Exception as exc:                     # noqa: BLE001 - comparing behaviour
        return type(exc).__name__, str(exc)


def _panel(n_units, n_periods, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": np.repeat([f"u{i:04d}" for i in range(n_units)], n_periods),
        "time": np.tile(np.arange(1, n_periods + 1), n_units),
        "y": rng.standard_normal(n_units * n_periods),
    })


# --------------------------------------------------------------------------- #
# the corpus every differential test runs over
# --------------------------------------------------------------------------- #
def _corpus():
    base = pd.DataFrame({"id": ["a", "a", "b", "b"], "time": [1, 2, 1, 2],
                         "y": 1.0})
    return {
        "balanced": base,
        "balanced-large": _panel(9, 7),
        "duplicate-row": pd.concat([base, base.iloc[[0]]], ignore_index=True),
        "missing-cell": base.drop(index=3),
        "ragged-same-count": pd.DataFrame(
            {"id": ["a", "a", "b", "b"], "time": [1, 1, 1, 2], "y": 1.0}),
        "nan-unit-key": pd.DataFrame(
            {"id": ["a", "a", None, "b"], "time": [1, 2, 1, 2], "y": 1.0}),
        "nan-time-key": pd.DataFrame(
            {"id": ["a", "a", "b", "b"], "time": [1, None, 1, 2], "y": 1.0}),
        "single-unit": pd.DataFrame({"id": ["a", "a"], "time": [1, 2], "y": 1.0}),
        "single-period": pd.DataFrame({"id": ["a", "b"], "time": [1, 1], "y": 1.0}),
        "one-cell": pd.DataFrame({"id": ["a"], "time": [1], "y": 1.0}),
        "unsorted": base.iloc[[3, 1, 2, 0]].reset_index(drop=True),
        "categorical-unit": base.assign(id=base["id"].astype("category")),
        "categorical-time": base.assign(time=base["time"].astype("category")),
        "empty": base.iloc[0:0],
        "mixed-type-time": pd.DataFrame(
            {"id": ["a", "a", "b", "b"], "time": [1, "2", 1, "2"], "y": 1.0}),
        "string-time": pd.DataFrame(
            {"id": ["a", "a", "b", "b"], "time": ["t1", "t2", "t1", "t2"],
             "y": 1.0}),
        "float-time": pd.DataFrame(
            {"id": ["a", "a", "b", "b"], "time": [1.0, 2.5, 1.0, 2.5], "y": 1.0}),
        "duplicate-and-unbalanced": pd.concat(
            [base.drop(index=3), base.iloc[[0]]], ignore_index=True),
        "non-unique-index": pd.concat([base.iloc[:2], base.iloc[2:]]),
    }


CORPUS = _corpus()


# --------------------------------------------------------------------------- #
# 1. smoke
# --------------------------------------------------------------------------- #
class TestSmoke:
    def test_a_balanced_panel_passes(self):
        balance(_panel(5, 4), "id", "time")

    def test_returns_none(self):
        assert balance(_panel(3, 3), "id", "time") is None


# --------------------------------------------------------------------------- #
# 2. the acceptance bar -- same decision, same error, same message
# --------------------------------------------------------------------------- #
class TestDecisionsAreIdentical:
    @pytest.mark.parametrize("name", sorted(CORPUS))
    def test_matches_the_oracle_on_the_corpus(self, name):
        df = CORPUS[name]
        if _has_blank_key(df):
            pytest.skip("verdict changed deliberately by #453; asserted in "
                        "test_missing_key_rejection.py")
        assert _outcome(balance, df) == _outcome(_balance_oracle, df), name

    @pytest.mark.parametrize("seed", range(8))
    def test_matches_the_oracle_on_randomly_damaged_panels(self, seed):
        """Fuzz the ways a panel goes wrong: drop rows, duplicate rows, blank a
        key. The two implementations must agree on every one."""
        rng = np.random.default_rng(seed)
        for _ in range(40):
            df = _panel(int(rng.integers(1, 7)), int(rng.integers(1, 7)), seed)
            damage = rng.integers(0, 4)
            if damage == 1 and len(df) > 1:
                df = df.drop(index=int(rng.integers(len(df)))).reset_index(drop=True)
            elif damage == 2 and len(df):
                pick = int(rng.integers(len(df)))
                df = pd.concat([df, df.iloc[[pick]]], ignore_index=True)
            elif damage == 3 and len(df):
                df = df.copy()
                df.loc[int(rng.integers(len(df))), "id"] = None
            if _has_blank_key(df):
                continue
            assert _outcome(balance, df) == _outcome(_balance_oracle, df)

    def test_an_unused_categorical_level_is_not_a_unit(self):
        """The one place this deliberately diverges, and it removes a
        version-dependence rather than adding one.

        ``groupby``'s ``observed`` default changed from ``False`` to ``True`` in
        pandas 3.0. Below that, a unit column typed as a Categorical carrying a
        level no row uses produced a group with zero observations, and the panel
        was rejected as not strongly balanced; from 3.0 the level is dropped and
        the same panel is accepted. The project supports ``pandas>=2.0.0``, so
        the shipped implementation returned different verdicts for the same
        frame depending on the installed pandas.

        ``factorize`` reports only levels that occur, so the answer is now the
        same on every supported version -- and it is the right one: a category
        no row uses is an annotation on the dtype, not a unit of the panel.
        """
        df = pd.DataFrame({"id": pd.Categorical(["a", "a", "b", "b"],
                                                categories=["a", "b", "c"]),
                           "time": [1, 2, 1, 2], "y": 1.0})
        assert balance(df, "id", "time") is None

        counted = df.groupby("id", observed=False)["time"].nunique()
        assert 0 in set(counted.values), (
            "fixture must carry a level with no rows, which is what the old "
            "implementation counted as a unit")

    def test_error_precedence_is_preserved(self):
        """A panel that is both duplicated and unbalanced reports the duplicate,
        because that check runs first."""
        df = CORPUS["duplicate-and-unbalanced"]
        with pytest.raises(MlsynthDataError, match="Duplicate observations found"):
            balance(df, "id", "time")


# --------------------------------------------------------------------------- #
# 3. the three failures, named
# --------------------------------------------------------------------------- #
class TestFailures:
    def test_duplicates(self):
        with pytest.raises(MlsynthDataError, match="Duplicate observations found"):
            balance(CORPUS["duplicate-row"], "id", "time")

    def test_missing_cell(self):
        with pytest.raises(MlsynthDataError,
                           match="Not all units have observations"):
            balance(CORPUS["missing-cell"], "id", "time")

    def test_units_with_different_counts(self):
        df = pd.DataFrame({"id": ["a", "a", "b"], "time": [1, 2, 1], "y": 1.0})
        with pytest.raises(MlsynthDataError,
                           match="The panel is not strongly balanced"):
            balance(df, "id", "time")

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            balance(_panel(3, 3).drop(columns=["time"]), "id", "time")


# --------------------------------------------------------------------------- #
# 4. the work contract -- what starts red
# --------------------------------------------------------------------------- #
class TestWorkContract:
    def test_no_pairwise_hash_pass(self, monkeypatch):
        """``duplicated`` on the key pair and ``groupby`` on the unit column are
        each a full hash of data the key codes already describe."""
        calls = {"duplicated": 0, "groupby": 0}
        real_dup, real_grp = pd.DataFrame.duplicated, pd.DataFrame.groupby

        def spy_dup(self, *a, **k):
            calls["duplicated"] += 1
            return real_dup(self, *a, **k)

        def spy_grp(self, *a, **k):
            calls["groupby"] += 1
            return real_grp(self, *a, **k)

        monkeypatch.setattr(pd.DataFrame, "duplicated", spy_dup)
        monkeypatch.setattr(pd.DataFrame, "groupby", spy_grp)
        balance(_panel(40, 30), "id", "time")
        assert calls == {"duplicated": 0, "groupby": 0}

    def test_each_key_column_is_scanned_once(self, monkeypatch):
        """Two factorizes, one per key column, and nothing else."""
        seen = []
        real = pd.factorize

        def spy(values, *a, **k):
            seen.append(len(values))
            return real(values, *a, **k)

        monkeypatch.setattr(pd, "factorize", spy)
        df = _panel(40, 30)
        balance(df, "id", "time")
        assert len(seen) == 2, f"{len(seen)} factorize passes over the panel"

    def test_a_sparse_panel_does_not_allocate_the_grid(self, monkeypatch):
        """The rejection path must stay bounded. A panel with one row per unit
        and per period has a ``n_units * n_periods`` grid far larger than the
        data -- 16 million cells for 4000 rows -- so a bincount over it would
        allocate 128 MB to report an error. ``_fast_pivot`` guards this with
        ``grid > 2 * n``; the validator has to as well.
        """
        n = 4000
        df = pd.DataFrame({"id": [f"u{i}" for i in range(n)],
                           "time": np.arange(n), "y": 0.0})
        budget = 4 * n                       # generous: still 1000x under the grid
        real = np.bincount

        def guarded(x, weights=None, minlength=0):
            assert minlength <= budget, (
                f"bincount over {minlength} cells for {n} rows -- the grid guard "
                "is missing")
            return real(x, weights=weights, minlength=minlength)

        monkeypatch.setattr(np, "bincount", guarded)
        with pytest.raises(MlsynthDataError):
            balance(df, "id", "time")


# --------------------------------------------------------------------------- #
# 5. scale -- the answer must not depend on panel size or row order
# --------------------------------------------------------------------------- #
class TestScale:
    @pytest.mark.parametrize("n_units,n_periods", [(1, 1), (1, 50), (50, 1),
                                                   (37, 41), (200, 13)])
    def test_balanced_panels_of_many_shapes_pass(self, n_units, n_periods):
        balance(_panel(n_units, n_periods), "id", "time")

    def test_row_order_does_not_matter(self):
        df = _panel(9, 7)
        shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        assert _outcome(balance, df) == _outcome(balance, shuffled)

    def test_large_period_count_does_not_overflow_the_linear_index(self):
        """The key codes are combined into one integer per row; on a wide panel
        that product must not wrap."""
        df = _panel(3, 20000)
        balance(df, "id", "time")
