"""The DTW kernel against R ``dtw``, pattern by pattern.

DTWSC's warp is only as trustworthy as its alignment. mlsynth implements the
step patterns directly rather than taking a DTW dependency, so the only thing
standing between a mistyped recursion coefficient and a silently wrong warp is
a comparison against the reference implementation.

``benchmarks/reference/dtwsc_dtw/gold_steppatterns.csv`` holds 360 alignments
produced by R ``dtw`` over random series (the generator sits beside it). Every
pattern contributes both admissible cases and cases where the reference is too
short for the pattern's slope constraint, so the error contract is pinned by
the same fixture as the arithmetic.

Why these six patterns: the authors' replication archive picks each placebo
run's step pattern from a grid, and the selected optima span
``symmetricP1``, ``symmetricP2``, ``asymmetricP1``, ``asymmetricP2``,
``typeIc`` and ``mori2006``. Supporting fewer means the paper's design cannot
be reproduced -- see ``docs/replications/dtwsc.rst``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.dtwsc_helpers.dtw import dtw

_GOLD = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
         / "dtwsc_dtw" / "gold_steppatterns.csv")

#: R's own ``(pattern, di, dj, weight)`` tables, transcribed from
#: ``dtw:::<name>``. Duplicated here on purpose: the test is the transcription
#: check, so it must not import the values it is meant to be checking.
R_TABLES = {
    "symmetricP1": ([[1, 1, 2, -1], [1, 0, 1, 2], [1, 0, 0, 1],
                     [2, 1, 1, -1], [2, 0, 0, 2],
                     [3, 2, 1, -1], [3, 1, 0, 2], [3, 0, 0, 1]], "N+M"),
    "asymmetricP2": ([[1, 2, 3, -1], [1, 1, 2, 2 / 3], [1, 0, 1, 2 / 3],
                      [1, 0, 0, 2 / 3],
                      [2, 1, 1, -1], [2, 0, 0, 1],
                      [3, 3, 2, -1], [3, 2, 1, 1], [3, 1, 0, 1],
                      [3, 0, 0, 1]], "N"),
    "symmetricP2": ([[1, 2, 3, -1], [1, 1, 2, 2], [1, 0, 1, 2], [1, 0, 0, 1],
                     [2, 1, 1, -1], [2, 0, 0, 2],
                     [3, 3, 2, -1], [3, 2, 1, 2], [3, 1, 0, 2],
                     [3, 0, 0, 1]], "N+M"),
    "asymmetricP1": ([[1, 1, 2, -1], [1, 0, 1, 0.5], [1, 0, 0, 0.5],
                      [2, 1, 1, -1], [2, 0, 0, 1],
                      [3, 2, 1, -1], [3, 1, 0, 1], [3, 0, 0, 1]], "N"),
    "typeIc": ([[1, 2, 1, -1], [1, 1, 0, 1], [1, 0, 0, 1],
                [2, 1, 1, -1], [2, 0, 0, 1],
                [3, 1, 2, -1], [3, 0, 1, 1], [3, 0, 0, 0]], "N"),
    "mori2006": ([[1, 2, 1, -1], [1, 1, 0, 2], [1, 0, 0, 1],
                  [2, 1, 1, -1], [2, 0, 0, 3],
                  [3, 1, 2, -1], [3, 0, 1, 3], [3, 0, 0, 3]], "M"),
}


def _patterns():
    from mlsynth.utils.dtwsc_helpers.dtw import PATTERNS
    return PATTERNS


@pytest.fixture(scope="module")
def gold():
    if not _GOLD.exists():  # pragma: no cover - the fixture is committed
        pytest.skip(f"missing DTW gold at {_GOLD}")
    return pd.read_csv(_GOLD, keep_default_na=False)


def _series(text):
    return np.array([float(v) for v in text.split(",")])


class TestPatternTables:
    """The tables must match R's, coefficient for coefficient."""

    def test_every_pattern_r_selects_is_available(self):
        assert set(_patterns()) >= set(R_TABLES)

    @pytest.mark.parametrize("name", sorted(R_TABLES))
    def test_table_matches_r(self, name):
        table, _ = R_TABLES[name]
        got = np.asarray(_patterns()[name], dtype=float)
        assert got.shape == (len(table), 4)
        np.testing.assert_allclose(got, np.asarray(table, dtype=float),
                                   rtol=0, atol=1e-15)

    @pytest.mark.parametrize("name", sorted(R_TABLES))
    def test_normalization_mode_matches_r(self, name):
        """``mori2006`` normalises by M, which neither existing pattern does.

        Getting this wrong leaves the alignment correct but every normalised
        distance -- and so every open-end endpoint choice -- wrong.
        """
        from mlsynth.utils.dtwsc_helpers.dtw import norm_mode
        assert norm_mode(_patterns()[name]) == R_TABLES[name][1]

    def test_norm_factor_honours_each_mode(self):
        from mlsynth.utils.dtwsc_helpers.dtw import _norm_factor
        P = _patterns()
        assert _norm_factor(P["symmetricP1"], 7, 4) == 11.0   # N+M
        assert _norm_factor(P["asymmetricP1"], 7, 4) == 7.0   # N
        assert _norm_factor(P["mori2006"], 7, 4) == 4.0       # M


class TestAgainstRGold:
    """Distances, normalised distances, and full paths, against R."""

    def test_the_fixture_covers_every_pattern_both_ways(self, gold):
        """A silently-truncated fixture would make the suite look green."""
        assert set(gold["pattern"]) == set(R_TABLES)
        for name, sub in gold.groupby("pattern"):
            assert set(sub["status"]) == {"ok", "error"}, name

    def test_distances_match(self, gold):
        ok = gold[gold.status == "ok"]
        worst = 0.0
        for _, row in ok.iterrows():
            a = dtw(_series(row["query"]), _series(row["reference"]),
                    step_pattern=_patterns()[row["pattern"]],
                    open_end=bool(row["open_end"]))
            worst = max(worst, abs(a.distance - float(row["distance"])))
            assert a.normalized_distance == pytest.approx(
                float(row["normalized"]), rel=1e-9, abs=1e-9), row["pattern"]
        assert worst < 1e-9, worst

    def test_warping_paths_match_exactly(self, gold):
        """Not just the cost -- the alignment itself.

        Two paths can share a cost and warp a donor differently, so an
        equal-distance check would pass implementations that disagree about
        every speed downstream.
        """
        ok = gold[gold.status == "ok"]
        for _, row in ok.iterrows():
            a = dtw(_series(row["query"]), _series(row["reference"]),
                    step_pattern=_patterns()[row["pattern"]],
                    open_end=bool(row["open_end"]))
            exp1 = np.array([int(v) for v in row["index1"].split(",")]) - 1
            exp2 = np.array([int(v) for v in row["index2"].split(",")]) - 1
            np.testing.assert_array_equal(a.index1, exp1, err_msg=row["pattern"])
            np.testing.assert_array_equal(a.index2, exp2, err_msg=row["pattern"])

    def test_inadmissible_cases_raise_where_r_errors(self, gold):
        """Agreeing about failure matters as much as agreeing about cost.

        The caller treats the raise as "this window is unusable"; returning a
        number where R refuses would silently admit a warp R never made.
        """
        bad = gold[gold.status == "error"]
        for _, row in bad.iterrows():
            with pytest.raises(MlsynthEstimationError):
                dtw(_series(row["query"]), _series(row["reference"]),
                    step_pattern=_patterns()[row["pattern"]],
                    open_end=bool(row["open_end"]))


class TestConfigSurface:
    """The new patterns have to be reachable from the public config."""

    @pytest.mark.parametrize("name", sorted(R_TABLES))
    def test_config_accepts_every_pattern(self, name):
        from mlsynth.utils.dtwsc_helpers.config import DTWSCConfig
        import pandas as pd_

        rows = []
        for unit in ("treated", "d1", "d2"):
            for t in range(12):
                rows.append({"unit": unit, "time": t, "y": 1.0 + t,
                             "treated": int(unit == "treated" and t >= 8)})
        cfg = DTWSCConfig(df=pd_.DataFrame(rows), outcome="y", treat="treated",
                          unitid="unit", time="time", display_graphs=False,
                          step_pattern1=name)
        assert cfg.step_pattern1 == name

    def test_an_unknown_pattern_is_refused(self):
        from mlsynth.utils.dtwsc_helpers.config import DTWSCConfig
        from mlsynth.exceptions import MlsynthConfigError
        import pandas as pd_
        from pydantic import ValidationError

        rows = [{"unit": "a", "time": t, "y": 1.0, "treated": 0}
                for t in range(4)]
        with pytest.raises((ValidationError, MlsynthConfigError)):
            DTWSCConfig(df=pd_.DataFrame(rows), outcome="y", treat="treated",
                        unitid="unit", time="time", display_graphs=False,
                        step_pattern1="notAPattern")


class TestWarpAndRefTooShort:
    """The consumers of an alignment, under every pattern.

    ``warp`` averages tied indices and ``ref_too_short`` rounds the maximum of
    that average, so a half-index difference flips a window from usable to
    unusable. Both were only ever exercised on the two original patterns;
    ``mori2006`` normalises by M, which changes which endpoint an open-end
    alignment selects, and that feeds straight into ``ref_too_short``.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def warp_gold():
        path = _GOLD.parent / "gold_warp.csv"
        if not path.exists():  # pragma: no cover - the fixture is committed
            pytest.skip(f"missing warp gold at {path}")
        return pd.read_csv(path, keep_default_na=False)

    def test_the_fixture_covers_every_pattern(self, warp_gold):
        assert set(warp_gold["pattern"]) == set(R_TABLES)

    def test_warp_matches_r_both_directions(self, warp_gold):
        from mlsynth.utils.dtwsc_helpers.dtw import warp

        ok = warp_gold[warp_gold.status == "ok"]
        assert not ok.empty
        for _, row in ok.iterrows():
            a = dtw(_series(row["query"]), _series(row["reference"]),
                    step_pattern=_patterns()[row["pattern"]])
            for column, index_reference in (("warp_query", False),
                                            ("warp_reference", True)):
                # R is 1-based; mlsynth returns 0-based positions.
                expected = _series(row[column]) - 1.0
                got = warp(a, index_reference=index_reference)
                np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9,
                                           err_msg=f"{row['pattern']}/{column}")

    def test_ref_too_short_matches_r(self, warp_gold):
        """A disagreement here silently changes which windows get warped."""
        from mlsynth.utils.dtwsc_helpers.dtw import ref_too_short

        for _, row in warp_gold.iterrows():
            got = ref_too_short(_series(row["query"]),
                                _series(row["reference"]),
                                step_pattern=_patterns()[row["pattern"]])
            expected = str(row["ref_too_short"]).upper() == "TRUE"
            assert got is expected, f"{row['pattern']} trial {row['trial']}"

    def test_an_empty_series_is_refused(self):
        with pytest.raises(MlsynthEstimationError, match="empty"):
            dtw(np.array([]), np.array([1.0, 2.0]))
