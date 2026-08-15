"""The index set is the only thing that says which units exist.

``dataprep`` returns the panel's units -- ``unit_names``, ``donor_names`` -- and
that index is the authority. Everything else that looks like a unit list is
derived and has to be projected onto it before it means anything: the key set of
a ``groupby`` result, a dict built from ``.to_dict()``, the column labels of a
pivot, a ``Categorical``'s ``categories``. Any of them can carry a unit the panel
does not contain, and none is entitled to be believed on its own.

Nothing asserted that until now, and one derived list could disagree. pandas
changed ``groupby``'s ``observed`` default from ``False`` to ``True`` in 3.0, so
below that a unit column typed as a ``Categorical`` with a level no row uses
manufactures a group with no observations. ``pyproject.toml`` asks for
``pandas>=2.0.0``, so both behaviours are in range, and the library left the
argument unset at every one of its call sites.

No wrong number came of it -- every consumption site read at the time reindexed
by the observed unit list first. But that was a property each site happened to
have, guaranteed by nothing, and a new call site that reads a reduction wholesale
would break it silently and only on the older pandas.

These tests are written so they can fail on *any* pandas. The runtime ones force
``observed=False``, which is the older default, and assert the answer is
unaffected. The static one asserts the argument is never left to the default at
all, which is what a future call site would trip.

An assertion on the estimate alone would not do. A phantom unit can appear in a
donor-weight dict while leaving the ATT untouched -- that is the shape of the
worked example in ``agents/agents_tests.md``, where the headline number moved 0.3
percent and the weights were the entire result. So these assert the unit-labelled
outputs, not the number.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from mlsynth import FDID, SDID, VanillaSC
from mlsynth.utils.datautils import dataprep

_REAL_GROUPBY = pd.DataFrame.groupby


@pytest.fixture
def older_pandas_grouping(monkeypatch):
    """Force ``groupby(observed=False)``, the default below pandas 3.0.

    A call site that states ``observed=True`` is unaffected; one that leaves the
    argument out inherits the phantom group, which is the whole point.
    """

    def grouping(self, *args, **kwargs):
        kwargs.setdefault("observed", False)
        try:
            return _REAL_GROUPBY(self, *args, **kwargs)
        except TypeError:                      # a grouper that takes no observed
            kwargs.pop("observed", None)
            return _REAL_GROUPBY(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", grouping)


def _panel(categorical, n_units=8, n_periods=12, n_pre=8, seed=4):
    rng = np.random.default_rng(seed)
    factors = np.cumsum(rng.standard_normal((n_periods, 2)) * 0.3, axis=0)
    loadings = rng.uniform(0.3, 1.1, (n_units, 2))
    outcome = (loadings @ factors.T
               + rng.standard_normal((n_units, n_periods)) * 0.3 + 10.0)
    outcome[0, n_pre:] -= 2.0
    names = [f"u{i}" for i in range(n_units)]
    df = pd.DataFrame([{"id": names[i], "time": t + 1, "y": outcome[i, t],
                        "d": int(i == 0 and t >= n_pre)}
                       for i in range(n_units) for t in range(n_periods)])
    if categorical:
        # A spare level: a category the panel never uses. Legitimate input --
        # it is what survives filtering a larger frame down to a subset.
        df["id"] = pd.Categorical(df["id"], categories=names + ["never_used"])
    return df


def _authoritative_units(df):
    """The index set: the columns of the wide frame ``dataprep`` builds.

    On the single-treated path the same set is spelled
    ``{treated_unit_name} | set(donor_names)``; both are asserted to agree
    below, since two spellings of the authority that disagree would be the
    fault this file is about.
    """
    prep = dataprep(df, unit_id_column_name="id", time_period_column_name="time",
                    outcome_column_name="y", treatment_indicator_column_name="d")
    return set(map(str, prep["Ywide"].columns))


def _fit(cls, df, **extra):
    config = {"df": df, "outcome": "y", "treat": "d", "unitid": "id",
              "time": "time", "display_graphs": False, **extra}
    return cls(config).fit()


ESTIMATORS = [
    pytest.param(SDID, {"vce": "noinference"}, id="SDID"),
    pytest.param(VanillaSC, {"inference": False}, id="VanillaSC"),
    pytest.param(FDID, {}, id="FDID"),
]


# --------------------------------------------------------------------------- #
# 1. the fixture is doing what it claims
# --------------------------------------------------------------------------- #
class TestTheFixtureBites:
    def test_forcing_the_older_default_manufactures_a_group(
            self, older_pandas_grouping):
        """Without this, the tests below would pass on pandas 3.0 whatever the
        call sites did, and prove nothing."""
        df = _panel(categorical=True)
        keys = list(df.groupby("id")["y"].mean().index)
        assert "never_used" in keys
        assert pd.isna(df.groupby("id")["y"].mean()["never_used"])

    def test_an_explicit_observed_true_is_not_overridden(
            self, older_pandas_grouping):
        df = _panel(categorical=True)
        keys = list(df.groupby("id", observed=True)["y"].mean().index)
        assert "never_used" not in keys

    def test_the_spare_level_reaches_the_estimator(self):
        """The panel really does carry a category the data never uses, and
        nothing upstream strips it."""
        df = _panel(categorical=True)
        assert "never_used" in list(df["id"].cat.categories)
        assert "never_used" not in set(df["id"].dropna().astype(str))


# --------------------------------------------------------------------------- #
# 2. the authority itself
# --------------------------------------------------------------------------- #
class TestIndexIsAuthoritative:
    def test_dataprep_reports_only_units_the_panel_contains(self):
        assert _authoritative_units(_panel(categorical=True)) == {
            f"u{i}" for i in range(8)}

    def test_the_two_spellings_of_the_index_agree(self):
        """``Ywide.columns`` and ``{treated} | donors`` are the same set. If a
        phantom unit ever entered one and not the other, the authority would be
        ambiguous and this file's premise would be false."""
        df = _panel(categorical=True)
        prep = dataprep(df, unit_id_column_name="id",
                        time_period_column_name="time", outcome_column_name="y",
                        treatment_indicator_column_name="d")
        spelled = {str(prep["treated_unit_name"])} | set(map(str, prep["donor_names"]))
        assert spelled == set(map(str, prep["Ywide"].columns))

    @pytest.mark.parametrize("cls,extra", ESTIMATORS)
    def test_unit_labelled_outputs_match_the_index(
            self, cls, extra, older_pandas_grouping):
        """Every list of unit names a caller can read equals the index set.

        Asserted on the labels and not on the estimate, because a phantom unit
        can ride in a weights dict without moving a number.
        """
        df = _panel(categorical=True)
        units = _authoritative_units(df)
        result = _fit(cls, df, **extra)

        weights = result.weights.donor_weights if result.weights else None
        if weights:
            assert set(map(str, weights)) <= units
            assert not any("never_used" in str(k) for k in weights)

        for name in ("unit_weights", "time_weights"):
            mapping = getattr(result.weights, name, None) if result.weights else None
            if name == "unit_weights" and mapping:
                assert set(map(str, mapping)) <= units

    @pytest.mark.parametrize("cls,extra", ESTIMATORS)
    def test_a_spare_level_changes_no_number(self, cls, extra,
                                             older_pandas_grouping):
        """The weaker claim, kept because it is the one a user would notice."""
        plain = _fit(cls, _panel(categorical=False), **extra)
        typed = _fit(cls, _panel(categorical=True), **extra)
        assert typed.effects.att == pytest.approx(plain.effects.att,
                                                  rel=0, abs=1e-12)

    def test_counterfactual_carries_no_phantom_period_or_unit(
            self, older_pandas_grouping):
        result = _fit(SDID, _panel(categorical=True), vce="noinference")
        series = np.asarray(result.time_series.counterfactual_outcome, dtype=float)
        assert series.shape[0] == 12
        assert np.all(np.isfinite(series))


# --------------------------------------------------------------------------- #
# 3. the contract, checked statically so a new call site trips it
# --------------------------------------------------------------------------- #
class TestNoCallSiteLeavesItToTheDefault:
    """The runtime tests above cover the estimators they run. This one covers
    the ones they do not, and the ones not written yet.

    ``observed`` is only meaningful for a categorical grouper, so passing it
    where the key is never categorical is redundant -- but it is redundant and
    correct, and the alternative is a rule nobody can check. A grep is a rule
    that holds for call sites nobody has reviewed.
    """

    @staticmethod
    def _close_paren(text, start):
        """Index of the paren closing the one at ``start``, or ``None``.

        String-aware, so a bracket inside a column name does not throw the
        count off.
        """
        depth, i, n = 0, start, len(text)
        in_string, quote = False, ""
        while i < n:
            char = text[i]
            if in_string:
                if char == "\\":
                    i += 2
                    continue
                if char == quote:
                    in_string = False
            elif char in "\"'":
                in_string, quote = True, char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return None

    @classmethod
    def _bare_groupby_sites(cls):
        """Call sites whose argument list omits ``observed``.

        Reads the whole call rather than the line it starts on: several of these
        wrap across lines, and a line-wise check reports them as bare when the
        argument is simply further down.
        """
        root = pathlib.Path(__file__).resolve().parents[1]
        pattern = re.compile(r"\.groupby\(")
        found = []
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.parts:
                continue
            source = path.read_text()
            for match in pattern.finditer(source):
                open_i = match.end() - 1
                close_i = cls._close_paren(source, open_i)
                if close_i is None:                       # unbalanced; not ours
                    continue
                if "observed=" in source[open_i + 1:close_i]:
                    continue
                lineno = source.count("\n", 0, match.start()) + 1
                snippet = source[match.start():close_i + 1].split("\n")[0]
                found.append(f"{path.relative_to(root)}:{lineno}: {snippet.strip()}")
        return found

    def test_every_groupby_states_observed(self):
        bare = self._bare_groupby_sites()
        assert not bare, (
            f"{len(bare)} groupby call(s) leave `observed` to the pandas "
            "default, which changed in 3.0:\n  " + "\n  ".join(bare[:15]))
