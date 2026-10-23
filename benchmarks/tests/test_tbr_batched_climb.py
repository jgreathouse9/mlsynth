"""The batched TBR climb: every candidate of a step scored at once, the same walk as the exact climb.

``benchmarks/studies/tbr_batched_climb`` replaces the two step functions of
``mlsynth.utils.tbr_helpers.design.search`` -- ``_augment`` and ``_match`` -- with
versions that score a step's candidates as columns of one matrix, the way
``forward_did_select`` scores FDID's. The orchestration in ``greedy_search`` is
untouched. The claim is equality, not closeness: the same partition, the same
trace and the same convergence at every treatment size, because any candidate
whose batched key sits within ``BAND`` of a decision is re-scored by
``score_split`` before anything is decided.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from benchmarks.studies.tbr_batched_climb import batched as B
from mlsynth.utils.tbr_helpers.design import objective as O
from mlsynth.utils.tbr_helpers.design import search as S

N_TEST = 7


def _panel(seed, T=40, N=12):
    """Geo series that share a trend and a weekly cycle, at varied scales."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    common = 1 + 0.2 * np.sin(2 * np.pi * t / 7) + np.cumsum(rng.normal(0, 0.02, size=T))
    scale = np.exp(rng.normal(5, 1, size=N))
    return scale * (common[:, None] * rng.normal(1, 0.2, size=N) + 0.05 * rng.normal(size=(T, N)))


def _eligibility(N, rng=None, forced=0):
    """``A_i`` sets: every geo free by default, ``forced`` geos treatment-only."""
    elig = [frozenset({S.TREATMENT, S.CONTROL, S.UNASSIGNED}) for _ in range(N)]
    for j in range(forced):
        elig[j] = frozenset({S.TREATMENT})
    return elig


def _outcome(o):
    return (o.k, list(o.treatment), list(o.control), list(o.unassigned), o.score.key,
            [tuple(k) for k in o.trace], o.converged)


class TestTheBatchedKeys:
    @settings(max_examples=30, deadline=None)
    @given(st.integers(0, 2**31 - 1), st.integers(6, 20), st.integers(24, 60))
    @example(0, 12, 40)
    def test_an_unflagged_key_is_score_splits(self, seed, N, T):
        Y = _panel(seed, T, N)
        rng = np.random.default_rng(seed)
        treat = list(rng.choice(N, int(rng.integers(1, 4)), replace=False))
        rest = [j for j in range(N) if j not in treat]
        control = set(rng.choice(rest, int(rng.integers(2, len(rest))), replace=False).tolist())
        y, x = Y[:, sorted(treat)].sum(1), Y[:, sorted(control)].sum(1)
        cols = np.array(rest)
        b = np.where(np.isin(cols, list(control)), -1.0, 1.0)
        K, flagged = B.batched_keys(Y, y, x, cols, np.zeros(cols.size), b, N_TEST)
        for i, j in enumerate(rest):
            cand = control ^ {j}
            if flagged[i] or not cand:
                continue
            exact = O.score_split(y, Y[:, sorted(cand)].sum(1), objective="reference", n_test=N_TEST).key
            assert tuple(K[i, :5]) == tuple(float(v) for v in exact[:5])
            assert abs(K[i, 5] - exact[5]) <= 1e-12 * abs(exact[5])

    def test_a_constant_control_aggregate_is_flagged(self):
        Y = _panel(1)
        Y[:, 5] = 100.0                                  # a geo that never moves
        y = Y[:, [0]].sum(1)
        K, flagged = B.batched_keys(Y, y, np.zeros(Y.shape[0]), np.array([5, 6]),
                                    np.zeros(2), np.ones(2), N_TEST)
        assert flagged[0] and not flagged[1]

    @pytest.mark.parametrize("value, near", [(0.8, True), (0.8 + 1e-12, True), (0.81, False),
                                             (0.915, True), (0.915 + 1e-12, True), (0.9149, False),
                                             (0.999999, True), (0.9999995, False), (0.99, False)])
    # 0.9999995 sits past the clip: both scorers read 0.999999 there, so nothing is decided.
    def test_a_correlation_at_a_decision_is_flagged(self, value, near):
        assert bool(B.near_correlation_decision(np.array([value]))[0]) is near

    #: A point far from every threshold, then each term moved onto its own.
    CLEAR = dict(corr=0.93, dw=2.0, excess=-1.0, env_max=3.0, loc=1.0, half=4.0, fpp=0.5)

    @pytest.mark.parametrize("term, value", [
        ("dw", 1.5), ("dw", 2.5 * (1 + 1e-12)), ("excess", 0.0), ("excess", 1e-12),
        ("loc", 4.0), ("loc", -4.0 * (1 + 1e-12)),
    ])
    def test_a_term_at_its_threshold_is_flagged(self, term, value):
        assert not B.near_decision(**self.CLEAR)
        assert B.near_decision(**{**self.CLEAR, term: value})

    def test_the_false_positive_cutoff_matters_only_off_zero(self):
        off_zero = {**self.CLEAR, "loc": 9.0, "half": 4.0}
        assert B.near_decision(**{**off_zero, "fpp": 0.2})
        assert not B.near_decision(**{**off_zero, "fpp": 0.3})
        assert not B.near_decision(**{**self.CLEAR, "fpp": 0.2})      # the interval covers zero


class TestTheBatchedClimb:
    @settings(max_examples=25, deadline=None)
    @given(st.integers(0, 2**31 - 1), st.integers(6, 16), st.integers(1, 3),
           st.sampled_from(["carried", "pool", "best"]), st.integers(0, 1))
    @example(0, 12, 3, "carried", 0)
    def test_the_same_designs_as_the_exact_climb(self, seed, N, k, start, forced):
        Y = _panel(seed, 40, N)
        elig = _eligibility(N, forced=forced)
        kw = dict(max_treatment_size=k, n_test=N_TEST, objective="reference", control_start=start)
        exact = S.greedy_search(Y, elig, **kw)
        fast = B.batched_greedy_search(Y, elig, **kw)
        assert [_outcome(o) for o in fast] == [_outcome(o) for o in exact]

    def test_the_reference_designs_on_geolift(self):
        """The CI case's pinned google/matched_markets designs, reached by the batched climb."""
        from benchmarks.cases import tbrmm

        with B.batched():
            got = tbrmm.run()
        for key, want in tbrmm.EXPECTED.items():
            value = want[0] if isinstance(want, tuple) else want
            assert got[key] == pytest.approx(value, rel=1e-12, abs=0.0), key

    def test_ties_break_to_the_first_geo(self):
        Y = _panel(3, 40, 10)
        Y[:, 9] = Y[:, 8]                                # two controls the score cannot tell apart
        elig = _eligibility(10)
        elig[8] = elig[9] = frozenset({S.CONTROL, S.UNASSIGNED})
        kw = dict(max_treatment_size=2, n_test=N_TEST, objective="reference")
        assert ([_outcome(o) for o in B.batched_greedy_search(Y, elig, **kw)]
                == [_outcome(o) for o in S.greedy_search(Y, elig, **kw)])

    def test_a_near_tie_goes_to_the_exact_scorer(self):
        """The pinned witness: geo 9 is geo 8 times (1 + 5 eps), on seed 1.

        The batched keys of the two toggles tie exactly, so a first-index rule on
        them would take geo 8. The exact keys put geo 9 ahead by one unit in the
        last place, and the exact climb takes it.
        """
        Y = _panel(1, 40, 10)
        Y[:, 9] = Y[:, 8] * (1 + 5 * np.finfo(float).eps)
        control = set(range(1, 8))
        y, x = Y[:, [0]].sum(1), Y[:, sorted(control)].sum(1)
        K, flagged = B.batched_keys(Y, y, x, [8, 9], np.zeros(2), np.ones(2), N_TEST)
        exact = [O.score_split(y, Y[:, sorted(control | {g})].sum(1), objective="reference",
                               n_test=N_TEST).key for g in (8, 9)]
        assert not flagged.any() and K[0, 5] == K[1, 5] and exact[1] > exact[0]
        assert B._select(K, flagged, exact.__getitem__) == 1

    def test_the_selection_takes_the_first_of_equal_keys(self):
        K = np.array([[1, 1, 1, 1, 0.9, 2.0], [1, 1, 1, 1, 0.9, 3.0], [1, 1, 1, 1, 0.9, 3.0]], float)
        exact = {i: tuple(K[i]) for i in range(3)}
        assert B._select(K, np.zeros(3, bool), exact.__getitem__) == 1

    def test_a_perfect_holdout_fit_fails_both_climbs_alike(self):
        """Treatment and control geos with identical series: the exact A/A test divides by a zero scale.

        A defect in the shipped objective, not in this study; the batched climb
        reaches it at the same candidate and raises the same error.
        """
        Y = _panel(3, 40, 10)
        Y[:, 7] = Y[:, 6]
        kw = dict(max_treatment_size=2, n_test=N_TEST, objective="reference")
        with pytest.raises(ZeroDivisionError):
            S.greedy_search(Y, _eligibility(10), **kw)
        with pytest.raises(ZeroDivisionError):
            B.batched_greedy_search(Y, _eligibility(10), **kw)

    def test_the_paper_objective_runs_the_exact_steps(self, monkeypatch):
        calls = []
        monkeypatch.setattr(B, "batched_keys", lambda *a, **k: calls.append(1))
        Y = _panel(4, 40, 10)
        elig = _eligibility(10)
        kw = dict(max_treatment_size=2, n_test=N_TEST, objective="paper")
        assert ([_outcome(o) for o in B.batched_greedy_search(Y, elig, **kw)]
                == [_outcome(o) for o in S.greedy_search(Y, elig, **kw)])
        assert not calls

    @pytest.mark.parametrize("from_pool", [False, True])
    def test_an_augment_that_would_empty_the_control_group_uses_the_pool(self, from_pool):
        """Control is the one geo being added: that candidate's control is re-derived from the pool."""
        Y = _panel(5, 40, 8)
        pools = {"forced": [], "treatment": [0, 3, 4, 5], "control": [0, 1, 2]}
        exact = B.EXACT_AUGMENT(S._Scorer(Y, "reference", N_TEST), [3], {0}, pools, from_pool=from_pool)
        fast = B.augment_step(S._Scorer(Y, "reference", N_TEST), [3], {0}, pools, from_pool=from_pool)
        assert fast == exact

    def test_a_match_step_never_offers_the_toggle_that_empties_the_control_group(self, monkeypatch):
        """From a one-geo control group, that geo cannot be toggled out; the step ends where the exact one does."""
        Y = _panel(6, 40, 8)
        offered = []
        real = B.batched_keys

        def spy(Yy, y, x, cols, a, b, n_test):
            offered.append(list(cols))
            return real(Yy, y, x, cols, a, b, n_test)

        monkeypatch.setattr(B, "batched_keys", spy)
        start = dict(treatment=[0], control={1}, toggleable=[1, 2, 3, 4])
        fast = B.match_step(S._Scorer(Y, "reference", N_TEST), **start)
        exact = B.EXACT_MATCH(S._Scorer(Y, "reference", N_TEST), **start)
        assert offered and 1 not in offered[0]
        assert (fast[0], [tuple(k) for k in fast[1]], fast[2]) == (exact[0], [tuple(k) for k in exact[1]], exact[2])

    def test_a_window_too_short_for_the_aa_test_fails_both_climbs_alike(self):
        from mlsynth.exceptions import MlsynthDataError

        Y = _panel(7, 9, 8)                              # 9 periods, 7 held out: two left to fit
        K, flagged = B.batched_keys(Y, Y[:, 0], Y[:, 1], [2, 3], np.zeros(2), np.ones(2), N_TEST)
        assert flagged.all()
        kw = dict(max_treatment_size=1, n_test=N_TEST, objective="reference")
        with pytest.raises(MlsynthDataError, match="A/A"):
            S.greedy_search(Y, _eligibility(8), **kw)
        with pytest.raises(MlsynthDataError, match="A/A"):
            B.batched_greedy_search(Y, _eligibility(8), **kw)

    def test_a_match_step_with_nothing_to_toggle_stops(self):
        Y = _panel(8, 40, 6)
        start = dict(treatment=[0], control={1, 2}, toggleable=[])
        fast = B.match_step(S._Scorer(Y, "reference", N_TEST), **start)
        exact = B.EXACT_MATCH(S._Scorer(Y, "reference", N_TEST), **start)
        assert fast[0] == exact[0] == {1, 2} and fast[2] and len(fast[1]) == 1

    def test_an_augment_skips_a_geo_that_would_leave_no_control_at_all(self):
        """Geo 1 is the whole control pool: adding it leaves nothing, even re-derived."""
        Y = _panel(9, 40, 6)
        pools = {"forced": [], "treatment": [1, 2, 3], "control": [1]}
        for from_pool in (False, True):
            fast = B.augment_step(S._Scorer(Y, "reference", N_TEST), [], {1}, pools, from_pool=from_pool)
            exact = B.EXACT_AUGMENT(S._Scorer(Y, "reference", N_TEST), [], {1}, pools, from_pool=from_pool)
            assert fast == exact and fast[0][0] != 1

    def test_an_augment_with_no_candidate_raises_as_the_exact_step_does(self):
        from mlsynth.exceptions import MlsynthDataError

        Y = _panel(10, 40, 4)
        pools = {"forced": [], "treatment": [1], "control": [1]}
        with pytest.raises(MlsynthDataError, match="no treatment-eligible geo"):
            B.augment_step(S._Scorer(Y, "reference", N_TEST), [], {1}, pools)

    def test_the_patch_is_undone(self):
        with B.batched():
            assert S._match is B.match_step and S._augment is B.augment_step
        assert S._match is B.EXACT_MATCH and S._augment is B.EXACT_AUGMENT


class TestTheRunner:
    def test_a_comparison_row_reports_identity_and_both_times(self):
        from benchmarks.studies.tbr_batched_climb import run as R

        row = R.compare(R.synthetic(16, n_days=40), 2, "carried")
        assert row["identical"] is True and row["geos"] == 16 and row["exact_s"] > 0 < row["batched_s"]
        assert row["exact_scores_batched"] < row["exact_scores"]
        assert "exact_s" not in R.compare(R.synthetic(16, n_days=40), 2, "carried", exact=False)

    def test_the_user_facing_fit_returns_the_same_designs(self):
        from benchmarks.studies.tbr_batched_climb import run as R

        row = R.end_to_end(R.synthetic(16, n_days=40), 2)
        assert row["same_designs"] is True and row["geos"] == 16

    def test_the_geolift_panel_is_the_shipped_one(self):
        from benchmarks.studies.tbr_batched_climb import run as R

        df = R.geolift()
        assert df["geo"].nunique() == 40 and df["date"].nunique() == 105

    def test_main_writes_every_row(self, tmp_path, monkeypatch):
        import json

        from benchmarks.studies.tbr_batched_climb import run as R

        small = R.synthetic(12, n_days=40)
        monkeypatch.setattr(R, "OUT", tmp_path / "out.json")
        monkeypatch.setattr(R, "geolift", lambda: small)
        monkeypatch.setattr(R, "synthetic", lambda n, **k: small)
        R.main()
        res = json.loads((tmp_path / "out.json").read_text())
        assert res["search"] and all(r["identical"] for r in res["search"] if "identical" in r)
        assert all(r["same_designs"] for r in res["end_to_end"])


class TestTheSeattlePanel:
    """The R ``microsynth`` package's Seattle Drug Market Intervention panel, at block-group level."""

    def test_block_groups_aggregate_the_pre_intervention_blocks(self):
        import pandas as pd

        from benchmarks.studies.tbr_batched_climb import run as R

        df = R.seattle()
        assert df["geo"].nunique() == 480 and sorted(df["date"].unique()) == list(range(1, 13))
        assert df.groupby("geo")["Y"].agg(lambda s: s.max() > s.min()).all()     # no constant series
        raw = pd.read_parquet(R.SEATTLE)
        assert df["Y"].sum() == raw.loc[raw["time"] <= 12, "any_crime"].sum()    # aggregation loses no crime

    def test_the_dmi_block_groups_are_the_data_s(self):
        from benchmarks.studies.tbr_batched_climb import run as R

        assert R.dmi_block_groups() == list(R.DMI_BLOCK_GROUPS)
        assert set(R.DMI_BLOCK_GROUPS) <= set(R.seattle()["geo"])

    def test_a_comparison_takes_the_window(self):
        from benchmarks.studies.tbr_batched_climb import run as R

        df = R.seattle()
        df = df[df["geo"].isin(sorted(df["geo"].unique())[:20])]
        row = R.compare(df, 2, "carried", n_test=R.SEATTLE_N_TEST)
        assert row["identical"] is True and row["periods"] == 12

    def test_the_seattle_study_reports_designs_and_the_dmi_split(self, tmp_path, monkeypatch):
        import json

        from benchmarks.studies.tbr_batched_climb import run as R

        full = R.seattle()
        keep = sorted(set(sorted(full["geo"].unique())[:17]) | set(R.DMI_BLOCK_GROUPS))
        monkeypatch.setattr(R, "seattle", lambda: full[full["geo"].isin(keep)])
        monkeypatch.setattr(R, "OUT_SEATTLE", tmp_path / "seattle.json")
        res = R.seattle_study(k=2, starts=("carried",))
        assert json.loads((tmp_path / "seattle.json").read_text()) == res
        assert all(r["identical"] for r in res["search"])
        assert [d["k"] for d in res["designs"]] == [1, 2] and res["fit"]["same_designs"]
        dmi = res["dmi_split"]
        assert dmi["treatment"] == list(R.DMI_BLOCK_GROUPS) and len(dmi["key"]) == 6
