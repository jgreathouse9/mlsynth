"""The Stage-1 feasibility audit: one itemised error naming every binding
constraint (candidate pool / budget / coverage / quota / spillover), each with a
``have vs need`` and a minimal fix -- and reporting them together.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.fast_scm_helpers.feasibility import audit_feasibility
from mlsynth.utils.fast_scm_helpers.conflict import greedy_independent_set_size
from mlsynth.utils.fast_scm_helpers import strata as st


# ---------------------------------------------------------------- helpers

def test_greedy_independent_set_size():
    assert greedy_independent_set_size(None, [0, 1, 2]) == 3        # no graph
    clique = np.ones((4, 4), bool); np.fill_diagonal(clique, False)
    assert greedy_independent_set_size(clique, [0, 1, 2, 3]) == 1   # all conflict
    A = np.zeros((4, 4), bool); A[0, 1] = A[1, 0] = True
    assert greedy_independent_set_size(A, [0, 1, 2, 3]) == 3        # one edge


def test_strata_feasibility_problems_empty_when_ok():
    codes = np.array([0, 0, 1, 1, 2])
    assert st.feasibility_problems(codes, [0, 1, 2, 3, 4], 3, 1, None) == []
    assert st.feasibility_problems(None, [0, 1], 2, 1, 1) == []


# ---------------------------------------------------------------- audit

class TestAuditFeasibility:

    def test_feasible_is_noop(self):
        audit_feasibility([0, 1, 2, 3], m=2)                        # nothing raised

    def test_candidate_pool_with_size_band(self):
        with pytest.raises(MlsynthConfigError) as e:
            audit_feasibility([0, 1], m=3, size_band=(50_000, None))
        msg = str(e.value)
        assert "candidate pool" in msg and "only 2" in msg and "m=3" in msg
        assert "size band [50000" in msg

    def test_budget_reports_cheapest_bill(self):
        costs = np.array([1.0, 2.0, 3.0, 10.0])
        with pytest.raises(MlsynthConfigError) as e:
            audit_feasibility([0, 1, 2, 3], m=3, unit_costs=costs, budget=5.0)
        msg = str(e.value)
        # 3 cheapest = 1+2+3 = 6 > 5, over by 1
        assert "budget" in msg and "$6" in msg and "over the $5" in msg and "by $1" in msg

    def test_coverage_needs_more_than_m(self):
        codes = np.array([0, 1, 2, 3])
        with pytest.raises(MlsynthConfigError, match="needs at least 4"):
            audit_feasibility([0, 1, 2, 3], m=2, strata=codes, min_per_stratum=1)

    def test_coverage_stratum_short(self):
        # enough candidates overall (M=4 >= m=4) and min*#strata = 4 <= m, but a
        # required stratum has only 1 unit while min_per_stratum=2.
        codes = np.array([0, 0, 0, 1])   # stratum 1 has a single candidate
        with pytest.raises(MlsynthConfigError, match="only 1"):
            audit_feasibility([0, 1, 2, 3], m=4, strata=codes, min_per_stratum=2)

    def test_quota_caps_capacity(self):
        codes = np.array([0, 0, 1, 1, 2])
        with pytest.raises(MlsynthConfigError, match="caps the treatable"):
            audit_feasibility([0, 1, 2, 3, 4], m=4, strata=codes, max_per_stratum=1)

    def test_budget_does_not_fire_when_affordable(self):
        costs = np.array([1.0, 1.0, 1.0, 1.0])
        audit_feasibility([0, 1, 2, 3], m=3, unit_costs=costs, budget=10.0)

    def test_reports_all_binding_at_once(self):
        # budget AND coverage both fail -> both appear in the one error
        codes = np.array([0, 1, 2, 3])
        costs = np.array([5.0, 5.0, 5.0, 5.0])
        with pytest.raises(MlsynthConfigError) as e:
            audit_feasibility([0, 1, 2, 3], m=2, unit_costs=costs, budget=3.0,
                              strata=codes, min_per_stratum=1)
        msg = str(e.value)
        assert "budget:" in msg and "coverage:" in msg
        assert msg.count("\n  - ") == 2          # exactly two itemised lines
