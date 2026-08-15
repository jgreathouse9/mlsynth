"""A row without a unit or a time is not an observation, and is refused.

``balance`` counted with ``nunique`` and ``groupby``, both of which skip missing
values, so a panel whose blank keys were symmetric across units satisfied every
check it made. ``_wide_pivot`` then pivoted the same frame and pandas kept the
blank as a level. The two disagreed about what a key is: the validator read a
missing key as *not an observation*, the pivot read it as *a level named nan*.

The consequence was silent and it moved the estimand. On a 3-unit, 4-period
panel treated from period 3, adding rows with a blank time key produced a fifth
period sorted to the front and took ``pre_periods`` from 2 to 3 -- the pre/post
split every estimator works from, moved by one, with nothing raised. A blank
unit key produced a fourth donor column named ``nan`` carrying only missing
observations.

Both ends now refuse it, and they have to be both ends: 38 modules reach
``dataprep`` without ``balance`` anywhere in them, because the estimator class
calls the validator and its setup helper calls ingestion. Ingestion cannot
assume its caller validated.

The refusal costs nothing. ``_fast_pivot`` already computed
``(codes < 0).any()`` to decide whether it could take its fast path, and threw
that answer away by deferring to ``df.pivot`` -- the very call that manufactures
the level. This turns a discarded observation into the error it always was.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.datautils import _wide_pivot, balance, dataprep


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _panel(n_units=3, n_periods=4, treated_from=3):
    rows = [{"id": u, "time": t, "y": float(ord(u) + t),
             "d": int(u == "a" and t >= treated_from)}
            for u in [chr(97 + i) for i in range(n_units)]
            for t in range(1, n_periods + 1)]
    return pd.DataFrame(rows)


def _prep(df):
    return dataprep(df, unit_id_column_name="id", time_period_column_name="time",
                    outcome_column_name="y", treatment_indicator_column_name="d")


# --------------------------------------------------------------------------- #
# 1. smoke -- a clean panel is untouched
# --------------------------------------------------------------------------- #
class TestSmoke:
    def test_clean_panel_still_validates(self):
        assert balance(_panel(), "id", "time") is None

    def test_clean_panel_still_ingests(self):
        prep = _prep(_panel())
        assert prep["Ywide"].shape == (4, 3)
        assert prep["pre_periods"] == 2
        assert list(prep["Ywide"].columns) == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# 2. the defect, at the validator
# --------------------------------------------------------------------------- #
class TestBalanceRefuses:
    def test_missing_time_key_symmetric_across_units(self):
        """The frame that used to pass: every unit missing the same period, so
        the counts agreed on both sides of the comparison."""
        df = pd.DataFrame({"id": ["a", "a", "b", "b"],
                           "time": [1, None, 1, None], "y": 1.0})
        with pytest.raises(MlsynthDataError, match="Missing values"):
            balance(df, "id", "time")

    def test_missing_unit_key(self):
        """A row with no unit formed no group, so it was neither counted nor
        reported."""
        df = pd.DataFrame({"id": ["a", None], "time": [1, 1], "y": 1.0})
        with pytest.raises(MlsynthDataError, match="Missing values"):
            balance(df, "id", "time")

    def test_the_message_names_the_columns_and_counts(self):
        df = pd.DataFrame({"id": ["a", None, None], "time": [1, None, 3],
                           "y": 1.0})
        with pytest.raises(MlsynthDataError) as excinfo:
            balance(df, "id", "time")
        message = str(excinfo.value)
        assert "id" in message and "time" in message
        assert "2" in message and "1" in message

    def test_missing_keys_are_reported_before_duplicates(self):
        """A row that cannot be identified is the more fundamental fault, so it
        is named first even when the frame is also duplicated."""
        df = pd.DataFrame({"id": ["a", "a", None], "time": [1, 1, 2], "y": 1.0})
        with pytest.raises(MlsynthDataError, match="Missing values"):
            balance(df, "id", "time")

    @pytest.mark.parametrize("blank", [None, np.nan, pd.NaT])
    def test_every_spelling_of_missing(self, blank):
        df = pd.DataFrame({"id": ["a", "b"], "time": [1, blank], "y": 1.0})
        with pytest.raises(MlsynthDataError, match="Missing values"):
            balance(df, "id", "time")


# --------------------------------------------------------------------------- #
# 3. the defect, at ingestion -- reached by 38 modules that never call balance
# --------------------------------------------------------------------------- #
class TestDataprepRefuses:
    def test_missing_time_key_no_longer_invents_a_period(self):
        df = _panel()
        for unit in ("a", "b", "c"):
            df.loc[len(df)] = {"id": unit, "time": None, "y": 999.0, "d": 0}
        with pytest.raises(MlsynthDataError, match="Missing values"):
            _prep(df)

    def test_missing_unit_key_no_longer_invents_a_donor(self):
        df = _panel()
        df.loc[len(df)] = {"id": None, "time": 2, "y": 999.0, "d": 0}
        with pytest.raises(MlsynthDataError, match="Missing values"):
            _prep(df)

    def test_pivot_helper_refuses_directly(self):
        """The check sits where the codes are computed, so any caller of the
        pivot is covered and not only ``dataprep``."""
        df = pd.DataFrame({"id": ["a", "b"], "time": [1, None], "y": 1.0})
        with pytest.raises(MlsynthDataError, match="Missing values"):
            _wide_pivot(df, index="time", columns="id", values="y")

    def test_missing_value_in_a_non_key_column_is_still_allowed(self):
        """Only the keys identify an observation. A missing outcome is a
        modelling question, not an ingestion one, and is left alone."""
        df = _panel()
        df.loc[2, "y"] = np.nan
        prep = _prep(df)
        assert prep["Ywide"].shape == (4, 3)
        assert np.isnan(prep["Ywide"].to_numpy(dtype=float)).sum() == 1


# --------------------------------------------------------------------------- #
# 4. the estimand this protects
# --------------------------------------------------------------------------- #
class TestTheEstimandIsProtected:
    def test_pre_periods_cannot_be_moved_by_a_blank_key(self):
        """The failure that motivated this: one blank-keyed row per unit took
        ``pre_periods`` from 2 to 3, silently."""
        clean = _prep(_panel())
        assert clean["pre_periods"] == 2

        damaged = _panel()
        for unit in ("a", "b", "c"):
            damaged.loc[len(damaged)] = {"id": unit, "time": None, "y": 0.0,
                                         "d": 0}
        with pytest.raises(MlsynthDataError):
            _prep(damaged)

    def test_no_nan_labels_survive_into_the_wide_frame(self):
        prep = _prep(_panel())
        assert not any(pd.isna(c) for c in prep["Ywide"].columns)
        assert not any(pd.isna(i) for i in prep["Ywide"].index)


# --------------------------------------------------------------------------- #
# 5. edges -- the refusal must not fire on frames that are merely awkward
# --------------------------------------------------------------------------- #
class TestEdges:
    def test_empty_frame_is_unchanged(self):
        """An empty panel has no missing keys, so it reaches the balance checks
        it reached before."""
        df = _panel().iloc[0:0]
        with pytest.raises(MlsynthDataError,
                           match="not strongly balanced|Missing values"):
            balance(df, "id", "time")

    def test_string_keys_that_look_blank_are_values_not_missing(self):
        """An empty string is a label, not an absence."""
        df = pd.DataFrame({"id": ["a", "a", "", ""], "time": [1, 2, 1, 2],
                           "y": 1.0})
        assert balance(df, "id", "time") is None

    def test_zero_and_negative_keys_are_values(self):
        df = pd.DataFrame({"id": ["a", "a", "b", "b"], "time": [0, -1, 0, -1],
                           "y": 1.0})
        assert balance(df, "id", "time") is None

    def test_categorical_keys_with_unused_levels_are_unaffected(self):
        df = pd.DataFrame({"id": pd.Categorical(["a", "a", "b", "b"],
                                                categories=["a", "b", "c"]),
                           "time": [1, 2, 1, 2], "y": 1.0})
        assert balance(df, "id", "time") is None

    def test_still_rejects_the_faults_it_rejected_before(self):
        """The new refusal is additive: duplicated and short panels are still
        reported, and with their own messages."""
        base = _panel(2, 2)
        with pytest.raises(MlsynthDataError, match="Duplicate observations"):
            balance(pd.concat([base, base.iloc[[0]]], ignore_index=True),
                    "id", "time")
        with pytest.raises(MlsynthDataError, match="not strongly balanced"):
            balance(base.drop(index=0), "id", "time")


# --------------------------------------------------------------------------- #
# 6. cost -- the refusal reuses a scan that already ran
# --------------------------------------------------------------------------- #
class TestCost:
    def test_no_extra_pass_over_the_frame(self, monkeypatch):
        """``balance`` already factorizes both key columns; detecting a missing
        key is reading the codes it produced, not another scan."""
        seen = []
        real = pd.factorize

        def spy(values, *a, **k):
            seen.append(len(values))
            return real(values, *a, **k)

        monkeypatch.setattr(pd, "factorize", spy)
        balance(_panel(20, 20), "id", "time")
        assert len(seen) == 2

    def test_ingestion_adds_no_scan_either(self, monkeypatch):
        """The refusal sits inside ``_fast_pivot``, on codes it had already
        computed to choose its path -- so ingestion gains no pass over the
        frame. Checking ``isna`` in ``_wide_pivot`` instead would have cost one
        scan per key column per pivot, and ``dataprep`` pivots twice.
        """
        calls = {"isna": 0}
        real = pd.Series.isna

        def spy(self, *a, **k):
            calls["isna"] += 1
            return real(self, *a, **k)

        monkeypatch.setattr(pd.Series, "isna", spy)
        _prep(_panel(10, 10))
        assert calls["isna"] == 0, (
            f"{calls['isna']} isna passes -- the check should read the codes")


# --------------------------------------------------------------------------- #
# 7. the config layer -- the earliest refusal, with the best message
# --------------------------------------------------------------------------- #
class TestConfigRefuses:
    """``BaseMAREXConfig`` has rejected missing keys all along, but it is
    referenced 14 times; ``BaseEstimatorConfig`` is referenced 150 and checked
    only that the frame was non-empty. Catching it at construction names the
    problem before any pivot exists, which is a better error than one raised
    from inside ingestion. Ingestion still refuses independently -- 38 modules
    reach it without a config in the call path at all.
    """

    @staticmethod
    def _config(df):
        from mlsynth.config_models import BaseEstimatorConfig
        return BaseEstimatorConfig(df=df, outcome="y", treat="d", unitid="id",
                                   time="time")

    def test_clean_panel_still_constructs(self):
        assert self._config(_panel()) is not None

    def test_missing_time_key_is_refused(self):
        df = _panel()
        df.loc[1, "time"] = None
        with pytest.raises(MlsynthDataError, match="Missing values"):
            self._config(df)

    def test_missing_unit_key_is_refused(self):
        df = _panel()
        df.loc[1, "id"] = None
        with pytest.raises(MlsynthDataError, match="Missing values"):
            self._config(df)

    def test_the_message_names_the_column(self):
        df = _panel()
        df.loc[1, "time"] = None
        with pytest.raises(MlsynthDataError) as excinfo:
            self._config(df)
        assert "time" in str(excinfo.value)

    def test_a_missing_outcome_still_constructs(self):
        """Scoped to the keys. A blank outcome is a modelling question and the
        estimators that tolerate it keep doing so -- which is where this differs
        from ``BaseMAREXConfig``, whose stricter contract covers the outcome
        column too.
        """
        df = _panel()
        df.loc[1, "y"] = np.nan
        assert self._config(df) is not None

    def test_a_missing_treatment_indicator_still_constructs(self):
        df = _panel()
        df.loc[1, "d"] = np.nan
        assert self._config(df) is not None
