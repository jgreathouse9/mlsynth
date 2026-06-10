"""Coverage/stratification quotas and treated-unit size bands for LEXSCM.

Exercises the ``strata`` module in isolation, the config validators, the Stage-1
admissibility branches (enumerate *and* heuristic), the size-band candidate
filter, every infeasibility error path, and end-to-end fits -- including
composition with the spillover constraint and the budget.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import LEXSCM
from mlsynth.config_models import LEXSCMConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.fast_scm_helpers import strata as st
from mlsynth.utils.fast_scm_helpers.lexsearch import select_treated_designs


# =====================================================================
# strata module -- unit tests
# =====================================================================

class TestStrataModule:

    def _ix(self, labels):
        return IndexSet.from_labels(labels)

    def test_build_strata_none(self):
        assert st.build_strata(self._ix([0, 1, 2]), None) is None

    def test_build_strata_codes_and_missing(self):
        ix = self._ix(["a", "b", "c", "d"])
        codes = st.build_strata(ix, {"a": "X", "b": "Y", "c": "X"})  # d missing
        # first-seen ordering: X->0, Y->1; d (absent) -> -1
        assert codes.tolist() == [0, 1, 0, -1]

    def test_build_strata_nan_is_missing(self):
        ix = self._ix([0, 1])
        codes = st.build_strata(ix, {0: float("nan"), 1: "R"})
        assert codes.tolist() == [-1, 0]

    def test_required_codes(self):
        codes = np.array([0, 1, -1, 0, 2])
        assert st.required_codes(codes, [0, 1, 2]).tolist() == [0, 1]   # cand 2 -> -1
        assert st.required_codes(codes, [0, 3, 4]).tolist() == [0, 2]

    def test_within_max(self):
        codes = np.array([0, 0, 1, -1])
        assert st.within_max(None, [0, 1], 1) is True            # no codes
        assert st.within_max(codes, [0, 1], None) is True        # no quota
        assert st.within_max(codes, [0, 1], 1) is False          # two from stratum 0
        assert st.within_max(codes, [0, 2], 1) is True
        assert st.within_max(codes, [0, 3], 1) is True           # -1 ignored

    def test_satisfies_full(self):
        codes = np.array([0, 0, 1, 2])
        req = np.array([0, 1, 2])
        assert st.satisfies(None, [0, 1], 1, 1, req) is True
        assert st.satisfies(codes, [0, 2, 3], None, 1, req) is True       # min off
        assert st.satisfies(codes, [0, 1, 2], None, 1, req) is False      # max breach
        assert st.satisfies(codes, [0, 2, 3], 1, None, req) is True       # covers 0,1,2
        assert st.satisfies(codes, [0, 1, 2], 1, None, req) is False      # misses 2

    def test_satisfies_many_vectorised(self):
        codes = np.array([0, 0, 1, 2])
        req = np.array([0, 1, 2])
        combs = np.array([[0, 2, 3], [0, 1, 2], [1, 2, 3]])
        # max_per=1: row1 has two from stratum 0 -> fails
        assert st.satisfies_many(codes, combs, None, 1, req).tolist() == [True, False, True]
        # min_per=1 over required {0,1,2}: only rows covering all three
        # ([0,2,3]->{0,1,2} ok; [0,1,2]->{0,1} misses 2; [1,2,3]->{0,1,2} ok)
        assert st.satisfies_many(codes, combs, 1, None, req).tolist() == [True, False, True]
        # no constraints -> all True
        assert st.satisfies_many(codes, combs, None, None, req).all()
        assert st.satisfies_many(None, combs, 1, 1, req).all()

    def test_check_feasible_paths(self):
        codes = np.array([0, 0, 1, 1, 2])
        cand = [0, 1, 2, 3, 4]
        st.check_feasible(None, cand, 2, 1, 1)                      # None -> no-op
        st.check_feasible(codes, cand, 3, 1, None)                  # feasible (m=#strata)
        with pytest.raises(MlsynthConfigError, match="needs at least"):
            st.check_feasible(codes, cand, 2, 1, None)              # 3 strata, m=2
        # min*#strata <= m, but a required stratum has too few candidates
        with pytest.raises(MlsynthConfigError, match="only 1"):
            st.check_feasible(np.array([0, 0, 1]), [0, 1, 2], 4, 2, None)
        with pytest.raises(MlsynthConfigError, match="caps the treatable"):
            st.check_feasible(codes, cand, 4, None, 1)              # cap 3 < m=4


# =====================================================================
# Config validators
# =====================================================================

class TestCoverageSizeConfig:

    def _df(self):
        return pd.DataFrame({"unitid": [0, 1], "time": [0, 0], "y": [1.0, 2.0],
                             "cand": [1, 1]})

    def _cfg(self, **kw):
        return LEXSCMConfig(df=self._df(), outcome="y", unitid="unitid",
                            time="time", candidate_col="cand", m=1, **kw)

    def test_ok_defaults(self):
        c = self._cfg()
        assert c.stratum_col is None and c.size_col is None

    def test_stratum_quota_requires_col(self):
        with pytest.raises(Exception, match="require `stratum_col`"):
            self._cfg(min_per_stratum=1)

    def test_min_exceeds_max_per_stratum(self):
        with pytest.raises(Exception, match="cannot exceed"):
            self._cfg(stratum_col="s", min_per_stratum=3, max_per_stratum=2)

    def test_size_band_requires_col(self):
        with pytest.raises(Exception, match="require `size_col`"):
            self._cfg(min_size=5.0)

    def test_min_size_exceeds_max(self):
        with pytest.raises(Exception, match="cannot exceed max_size"):
            self._cfg(size_col="z", min_size=10.0, max_size=5.0)

    def test_valid_combo(self):
        c = self._cfg(stratum_col="s", min_per_stratum=1, max_per_stratum=2,
                      size_col="z", min_size=1.0, max_size=9.0)
        assert c.min_per_stratum == 1 and c.max_size == 9.0


# =====================================================================
# Stage-1 search with strata (direct, fast) -- enumerate AND heuristic
# =====================================================================

def _gram(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, n))
    X = X - X.mean(1, keepdims=True)
    return X.T @ X


class TestStage1Strata:

    def test_enumerate_max_quota(self):
        G = _gram(6)
        strata = np.array([0, 0, 1, 1, 2, 2])
        out = select_treated_designs(G, list(range(6)), m=3, top_K=10,
                                     strata=strata, max_per_stratum=1, method="enumerate")
        for d in out["top_designs"]:
            codes = strata[d.indices]
            assert len(set(codes.tolist())) == 3            # all distinct strata

    def test_enumerate_min_coverage(self):
        G = _gram(6)
        strata = np.array([0, 0, 1, 1, 2, 2])
        out = select_treated_designs(G, list(range(6)), m=3, top_K=10,
                                     strata=strata, min_per_stratum=1, method="enumerate")
        for d in out["top_designs"]:
            assert set(strata[d.indices].tolist()) == {0, 1, 2}

    def test_heuristic_max_quota(self):
        G = _gram(8)
        strata = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        out = select_treated_designs(G, list(range(8)), m=4, top_K=5,
                                     strata=strata, max_per_stratum=1,
                                     method="heuristic", n_starts=8, random_state=1)
        assert out["top_designs"]
        for d in out["top_designs"]:
            assert len(set(strata[d.indices].tolist())) == 4

    def test_heuristic_min_coverage(self):
        G = _gram(8)
        strata = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        out = select_treated_designs(G, list(range(8)), m=4, top_K=5,
                                     strata=strata, min_per_stratum=1,
                                     method="heuristic", n_starts=12, random_state=2)
        for d in out["top_designs"]:
            assert set(strata[d.indices].tolist()) == {0, 1, 2, 3}

    def test_infeasible_quota_raises(self):
        G = _gram(6)
        strata = np.array([0, 0, 0, 0, 0, 0])      # one stratum, max 1 -> can't reach m=3
        with pytest.raises(MlsynthConfigError):
            select_treated_designs(G, list(range(6)), m=3, top_K=5,
                                   strata=strata, max_per_stratum=1, method="enumerate")

    def test_compose_with_conflict(self):
        G = _gram(6)
        strata = np.array([0, 0, 1, 1, 2, 2])
        conflict = np.zeros((6, 6), bool)
        conflict[0, 1] = conflict[1, 0] = True       # also forbid {0,1} together
        out = select_treated_designs(G, list(range(6)), m=3, top_K=10,
                                     strata=strata, max_per_stratum=2,
                                     conflict=conflict, method="enumerate")
        for d in out["top_designs"]:
            assert not (0 in d.indices and 1 in d.indices)


# =====================================================================
# End-to-end LEXSCM
# =====================================================================

def _panel(n=18, T=40, T_post=10, seed=0, sizes=None, n_cand=12, n_regions=4):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, 2)); nu = rng.normal(size=(T, 2))
    Y = 100 + nu @ g.T + 0.1 * rng.normal(size=(T, n))
    rows = []
    for i in range(n):
        for t in range(T):
            rows.append({"unitid": i, "time": t, "y": Y[t, i],
                         "post": int(t >= T - T_post), "candidate": int(i < n_cand),
                         "region": f"R{i % n_regions}",
                         "size": (sizes[i] if sizes is not None else 100 + i)})
    return pd.DataFrame(rows)


class TestCoverageSizeE2E:
    _BASE = dict(outcome="y", unitid="unitid", time="time", candidate_col="candidate",
                 post_col="post", top_K=5, verbose=False)

    def _regmap(self, df):
        return df.groupby("unitid")["region"].first()

    def test_coverage_every_region(self):
        df = _panel()
        res = LEXSCM({"df": df, **self._BASE, "m": 4, "stratum_col": "region",
                      "min_per_stratum": 1}).fit()
        reg = self._regmap(df)
        assert len({reg.loc[int(u)] for u in res.selected_units}) == 4

    def test_quota_one_per_region(self):
        df = _panel()
        res = LEXSCM({"df": df, **self._BASE, "m": 3, "stratum_col": "region",
                      "max_per_stratum": 1}).fit()
        reg = self._regmap(df)
        regs = [reg.loc[int(u)] for u in res.selected_units]
        assert len(set(regs)) == len(regs)

    def test_size_band_filters_treatment(self):
        sizes = [100 + i for i in range(18)]
        df = _panel(sizes=sizes)
        res = LEXSCM({"df": df, **self._BASE, "m": 3, "size_col": "size",
                      "min_size": 105, "max_size": 113}).fit()
        assert all(105 <= sizes[int(u)] <= 113 for u in res.selected_units)
        # an out-of-band unit may still be a donor
        donors = [int(d) for d in res.design_weights.donor_weights]
        assert any(sizes[d] < 105 or sizes[d] > 113 for d in donors)

    def test_size_band_one_sided(self):
        sizes = [100 + i for i in range(18)]
        df = _panel(sizes=sizes)
        res = LEXSCM({"df": df, **self._BASE, "m": 3, "size_col": "size",
                      "max_size": 108}).fit()
        assert all(sizes[int(u)] <= 108 for u in res.selected_units)

    def test_size_band_nan_excluded(self):
        sizes = [100 + i for i in range(18)]
        sizes[0] = float("nan")          # unit 0 has unknown size -> not treatable
        df = _panel(sizes=sizes)
        res = LEXSCM({"df": df, **self._BASE, "m": 3, "size_col": "size",
                      "min_size": 99}).fit()
        assert 0 not in {int(u) for u in res.selected_units}

    def test_coverage_infeasible_errors(self):
        df = _panel()
        with pytest.raises(MlsynthConfigError):
            LEXSCM({"df": df, **self._BASE, "m": 2, "stratum_col": "region",
                    "min_per_stratum": 1}).fit()

    def test_size_infeasible_errors(self):
        df = _panel(sizes=[100 + i for i in range(18)])
        with pytest.raises(MlsynthConfigError):
            LEXSCM({"df": df, **self._BASE, "m": 3, "size_col": "size",
                    "min_size": 1e9}).fit()

    def test_coverage_plus_size_compose(self):
        # Two treated-side constraints stack: cover every region (min_per=1, m=4)
        # AND every treated unit within the size band.
        sizes = [100 + i for i in range(18)]
        df = _panel(sizes=sizes)
        res = LEXSCM({"df": df, **self._BASE, "m": 4, "stratum_col": "region",
                      "min_per_stratum": 1, "size_col": "size",
                      "min_size": 100, "max_size": 111}).fit()
        reg = self._regmap(df)
        regs = [reg.loc[int(u)] for u in res.selected_units]
        assert len(set(regs)) == 4                        # covers all 4 regions
        assert all(100 <= sizes[int(u)] <= 111 for u in res.selected_units)

    def test_strata_plus_spillover_stage1_feasible(self):
        # coverage (coarse 2-strata) + cluster spillover on a panel where some
        # regions stay treatment-free so donors survive the Stage-2 exclusion.
        df = _panel(n=18, n_cand=8)                       # candidates only i<8 -> R0..R3
        df["half"] = df["region"].map({"R0": "A", "R1": "A", "R2": "B", "R3": "B"})
        res = LEXSCM({"df": df, **self._BASE, "m": 2, "stratum_col": "half",
                      "min_per_stratum": 1, "cluster_col": "region"}).fit()
        half = df.groupby("unitid")["half"].first()
        halves = [half.loc[int(u)] for u in res.selected_units]
        assert set(halves) == {"A", "B"}                  # both halves covered
