"""TDD for PANGEO Q-selection rules and the fixed-Q diagnostic sweep.

Two features:

1. A ``q_selection`` rule for automatic Q. The default ``"mde_min"`` keeps the
   historical behaviour (pick the feasible Q of smallest program MDE). The new
   ``"pareto_1se"`` treats Q-choice as a two-objective problem -- minimise the
   MDE, maximise the pair count K (inference credibility) -- keeps only the
   Pareto-efficient Q on ``(MDE down, K up)``, then applies a 1-SE tie-break:
   among the frontier (subject to an optional ``q_min_pairs`` floor) take the
   *largest K* whose MDE is within one standard error of the frontier's best.
   The SE is the deterministic small-sample SD-of-SD approximation
   ``MDE / sqrt(2 (B - 1))`` on the ``B``-week blank window.

2. ``compute_q_sweep`` -- populate ``metadata["q_sweep"]`` even when Q is fixed
   (previously the sweep existed only in auto mode), so the Pareto rule's
   inputs are auditable regardless of how Q was chosen.

The pure selection logic is pinned deterministically on synthetic sweep rows;
the plumbing is pinned end-to-end through ``PANGEO.fit()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PANGEO
from mlsynth.config_models import PANGEOConfig
from mlsynth.utils.pangeo_helpers.pipeline import _select_q


def _panel(n=12, T=24, seed=0):
    rng = np.random.default_rng(seed)
    F = np.cumsum(rng.standard_normal((T, 2)), axis=0)
    rows = []
    for i in range(n):
        b = rng.standard_normal(2)
        y = 10.0 + i + F @ b + rng.standard_normal(T) * 0.1
        for t in range(T):
            rows.append({"unit": f"u{i}", "time": t, "sales": float(y[t]), "arm": "A"})
    return pd.DataFrame(rows)


# The NE+Mid-Atlantic sweep shape from the analysis: MDE is U-shaped in Q, and
# the frontier is {Q2 (most pairs), Q3 (best MDE)}; Q4-6 are dominated.
_SWEEP = [
    {"q": 2, "feasible": True, "n_program_pairs": 6, "mean_program_mde_pct": 3.75, "mde_se": 0.93},
    {"q": 3, "feasible": True, "n_program_pairs": 4, "mean_program_mde_pct": 3.19, "mde_se": 0.67},
    {"q": 4, "feasible": True, "n_program_pairs": 3, "mean_program_mde_pct": 3.70, "mde_se": 1.08},
    {"q": 5, "feasible": True, "n_program_pairs": 3, "mean_program_mde_pct": 3.77, "mde_se": 0.81},
    {"q": 6, "feasible": True, "n_program_pairs": 2, "mean_program_mde_pct": 3.99, "mde_se": 1.88},
    {"q": 1, "feasible": False, "n_program_pairs": 0, "mean_program_mde_pct": None, "mde_se": None},
]


class TestSelectQPure:
    def test_mde_min_picks_global_minimum(self):
        assert _select_q(_SWEEP, method="mde_min", q_min_pairs=1) == 3

    def test_pareto_1se_prefers_more_pairs_within_one_se(self):
        # best MDE is Q3 (3.19), 1-SE band -> <= 3.86; Q2 (3.75, K=6) is inside
        # it and has the most pairs -> Q2 wins.
        assert _select_q(_SWEEP, method="pareto_1se", q_min_pairs=1) == 2

    def test_pareto_1se_respects_pair_floor(self):
        # require >= 6 pairs: only Q2 qualifies on the frontier.
        assert _select_q(_SWEEP, method="pareto_1se", q_min_pairs=6) == 2

    def test_pareto_1se_falls_back_to_best_when_none_clear_floor(self):
        # an impossible floor cannot be met -> fall back to the frontier best.
        assert _select_q(_SWEEP, method="pareto_1se", q_min_pairs=99) == 2

    def test_dominated_q_never_selected(self):
        # Q4/Q5/Q6 are Pareto-dominated and must not be returned by either rule.
        for m in ("mde_min", "pareto_1se"):
            assert _select_q(_SWEEP, method=m, q_min_pairs=1) in {2, 3}

    def test_no_feasible_returns_none(self):
        infeasible = [{"q": 1, "feasible": False, "n_program_pairs": 0,
                       "mean_program_mde_pct": None, "mde_se": None}]
        assert _select_q(infeasible, method="mde_min", q_min_pairs=1) is None


class TestAutoQSelectionPlumbing:
    def test_default_is_mde_min(self):
        df = _panel(12, 28, seed=1)
        res = PANGEO(PANGEOConfig(
            df=df, outcome="sales", arm="arm", unitid="unit", time="time",
            max_supergeo_size=None, fast=True, display_graphs=False)).fit()
        assert res.metadata["q_selection"] == "mde_min"
        # sweep rows now carry the MDE standard error
        feas = [r for r in res.metadata["q_sweep"] if r["feasible"]]
        assert all("mde_se" in r for r in feas)

    def test_pareto_1se_records_method_and_picks_feasible_q(self):
        df = _panel(12, 28, seed=2)
        res = PANGEO(PANGEOConfig(
            df=df, outcome="sales", arm="arm", unitid="unit", time="time",
            max_supergeo_size=None, fast=True, q_selection="pareto_1se",
            display_graphs=False)).fit()
        assert res.metadata["q_selection"] == "pareto_1se"
        feasible_qs = {r["q"] for r in res.metadata["q_sweep"] if r["feasible"]}
        assert res.metadata["q_selected"] in feasible_qs
        assert res.max_supergeo_size == res.metadata["q_selected"]


class TestFixedQSweep:
    def test_fixed_q_no_sweep_by_default(self):
        df = _panel(12, 24, seed=3)
        res = PANGEO(PANGEOConfig(
            df=df, outcome="sales", arm="arm", unitid="unit", time="time",
            max_supergeo_size=2, fast=True, display_graphs=False)).fit()
        assert "q_sweep" not in res.metadata
        assert "q_auto_selected" not in res.metadata

    def test_compute_q_sweep_populates_sweep_but_keeps_fixed_q(self):
        df = _panel(12, 24, seed=4)
        res = PANGEO(PANGEOConfig(
            df=df, outcome="sales", arm="arm", unitid="unit", time="time",
            max_supergeo_size=2, fast=True, compute_q_sweep=True,
            display_graphs=False)).fit()
        assert "q_sweep" in res.metadata
        assert len([r for r in res.metadata["q_sweep"] if r["feasible"]]) >= 2
        # fixed Q is honoured, not auto-selected
        assert res.max_supergeo_size == 2
        assert "q_auto_selected" not in res.metadata
