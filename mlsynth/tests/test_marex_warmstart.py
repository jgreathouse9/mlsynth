"""Tests for MAREX warm-starting (LEXSCM seed) of the exact MIQP.

The contract: a warm start is a *hint*. A solve that runs to completion returns
the identical proven optimum with or without it (so existing behaviour cannot
regress), a wrong/suboptimal seed cannot corrupt the answer, and the labels are
validated like the other geographic-design inputs. ``time_limit`` caps the SCIP
solve and returns the best incumbent.

Reference: Abadie & Zhao (2026); the warm-start mechanism injects ``z`` as a SCIP
partial solution (see ``marex_helpers.warmstart``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX
from mlsynth.config_models import MAREXConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.marex_helpers.warmstart import (
    lexscm_warm_start, solve_warmstarted,
)


def _panel(J=10, T=16, seed=0, candidate=False):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, (T, 2))
    lam = rng.normal(0, 1, (J, 2))
    Y = lam @ F.T + 0.2 * rng.standard_normal((J, T))
    rows = []
    for j in range(J):
        for t in range(T):
            row = {"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t])}
            if candidate:
                row["candidate"] = 1
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    return _panel()


BASE = dict(outcome="y", unitid="unit", time="time", T0=10, m_eq=2)


def _treated(res):
    return sorted(map(str, res.treated_units))


# ----------------------------------------------------------------------
# Result invariance: a completed solve is unchanged by the warm start
# ----------------------------------------------------------------------
class TestInvariance:
    def test_seed_with_optimum_matches_cold(self, panel):
        cold = MAREX({"df": panel, **BASE}).fit()
        warm = MAREX({"df": panel, **BASE, "warm_start": _treated(cold)}).fit()
        assert _treated(warm) == _treated(cold)

    def test_suboptimal_seed_still_reaches_optimum(self, panel):
        # A deliberately bad seed must not change the proven optimum.
        cold = MAREX({"df": panel, **BASE}).fit()
        bad = [u for u in ["u00", "u01", "u02", "u03"] if u not in _treated(cold)][:2]
        warm = MAREX({"df": panel, **BASE, "warm_start": bad}).fit()
        assert _treated(warm) == _treated(cold)

    def test_default_none_is_baseline(self, panel):
        # warm_start defaults to None and reproduces the un-warm-started fit.
        a = MAREX({"df": panel, **BASE}).fit()
        b = MAREX({"df": panel, **BASE, "warm_start": None}).fit()
        assert _treated(a) == _treated(b)


# ----------------------------------------------------------------------
# time_limit
# ----------------------------------------------------------------------
class TestTimeLimit:
    def test_budget_returns_valid_design(self, panel):
        res = MAREX({"df": panel, **BASE, "time_limit": 30.0,
                     "warm_start": ["u00", "u08"]}).fit()
        assert len(res.treated_units) == 2

    def test_no_incumbent_in_tiny_budget_raises(self):
        # A non-trivial instance with an essentially-zero budget yields no
        # feasible design -> a clear, attributable error (defensive path).
        big = _panel(J=40, T=30, seed=3)
        with pytest.raises(MlsynthEstimationError):
            MAREX({"df": big, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 20, "m_eq": 6, "time_limit": 1e-6}).fit()


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
class TestValidation:
    def test_unknown_warm_unit_rejected(self, panel):
        with pytest.raises(MlsynthConfigError):
            MAREXConfig(df=panel, outcome="y", unitid="unit", time="time",
                        warm_start=["not_a_unit"])

    def test_warm_start_incompatible_with_relaxed(self, panel):
        with pytest.raises(MlsynthConfigError):
            MAREXConfig(df=panel, outcome="y", unitid="unit", time="time",
                        relaxed=True, warm_start=["u00"])

    def test_time_limit_incompatible_with_relaxed(self, panel):
        with pytest.raises(MlsynthConfigError):
            MAREXConfig(df=panel, outcome="y", unitid="unit", time="time",
                        relaxed=True, time_limit=10.0)

    def test_time_limit_must_be_positive(self, panel):
        with pytest.raises(Exception):
            MAREXConfig(df=panel, outcome="y", unitid="unit", time="time",
                        time_limit=0.0)


# ----------------------------------------------------------------------
# LEXSCM -> MAREX warm start
# ----------------------------------------------------------------------
class TestLexscmSeed:
    def test_lexscm_warm_start_helper_duck_typed(self):
        class _Cand:
            treated_weight_dict_full = {"u03": 0.5, "u07": 0.5}

        class _Res:
            class search:
                winner = _Cand()

        assert lexscm_warm_start(_Res()) == ["u03", "u07"]

    def test_lexscm_seed_matches_cold_optimum(self):
        from mlsynth import LEXSCM
        df = _panel(J=10, T=16, seed=1, candidate=True)
        lex = LEXSCM({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                      "candidate_col": "candidate", "m": 2,
                      "display_graph": False}).fit()
        seed = lexscm_warm_start(lex)
        assert len(seed) == 2
        cold = MAREX({"df": df, **BASE}).fit()
        warm = MAREX({"df": df, **BASE, "warm_start": seed}).fit()
        # invariance holds whatever LEXSCM picked
        assert _treated(warm) == _treated(cold)


# ----------------------------------------------------------------------
# Low-level helper
# ----------------------------------------------------------------------
class TestSolveWarmstarted:
    def test_shape_mismatch_raises(self, panel):
        import cvxpy as cp
        z = cp.Variable((3, 1), boolean=True)
        prob = cp.Problem(cp.Minimize(cp.sum(z)), [cp.sum(z) >= 1])
        with pytest.raises(ValueError):
            solve_warmstarted(prob, z, np.zeros((2, 1)))
