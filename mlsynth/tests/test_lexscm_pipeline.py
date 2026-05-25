"""End-to-end regression tests for the rebuilt LEXSCM pipeline.

Covers the m>=4 regime, the new search/power/select stages, and the two
failure modes that crashed the old pipeline (now handled gracefully).
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.lexscm import LEXSCM


def _panel(N=40, T_pre=24, T_post=6, r=3, level=100.0, seed=7):
    rng = np.random.default_rng(seed)
    T = T_pre + T_post
    F = rng.standard_normal((T, r)); Lo = rng.standard_normal((N, r))
    Y = level + 8.0 * (F @ Lo.T) + 2.0 * rng.standard_normal((T, N))
    return pd.DataFrame([
        {"unit": f"u{j:02d}", "time": t + 1, "y": float(Y[t, j]),
         "candidate": 1, "post": int(t >= T_pre)}
        for j in range(N) for t in range(T)
    ])


def _cfg(df, **over):
    base = dict(df=df, outcome="y", unitid="unit", time="time",
                candidate_col="candidate", post_col="post",
                m=4, top_K=8, frac_E=0.7, budget=None,
                n_post_grid=[2, 3, 4], n_sims=200, mde_horizon="early_mean",
                display_graph=False, verbose=False, seed=7)
    base.update(over)
    return base


def test_full_pipeline_m4():
    res = LEXSCM(_cfg(_panel())).fit()
    meta = res.bnb_metadata
    assert meta["stats"]["search"]["method"] in ("enumeration", "multistart_local_search")
    assert meta["recommendation"]["status"] == "OK"
    assert meta["recommendation"]["winner"] is not None
    # exactly m treated units chosen
    assert len(res.best_candidate.identification.treated_idx) == 4
    assert len(res.best_candidate.control_weight_dict) >= 1
    # summary table carries the new design metrics
    for col in ("design_id", "imbalance", "mde_sd", "pareto", "winner"):
        assert col in res.summary.columns
    assert res.summary["winner"].sum() == 1


def test_late_horizon_does_not_require_horizon_8():
    # old pipeline hardcoded mde_8w under "late" and crashed when 8 not in grid;
    # new pipeline uses max(n_post_grid).
    res = LEXSCM(_cfg(_panel(), mde_horizon="late", n_post_grid=[2, 3, 4])).fit()
    assert res.bnb_metadata["recommendation"]["status"] in ("OK", "POWER_NOT_ESTABLISHED")


def test_zero_mean_outcome_is_graceful():
    # old pipeline divided by the outcome level (floored 1e-8) and produced
    # all-NaN MDEs -> empty Pareto -> IndexError; new MDE is SD-based.
    res = LEXSCM(_cfg(_panel(level=0.0))).fit()
    assert res.bnb_metadata["recommendation"]["status"] in ("OK", "POWER_NOT_ESTABLISHED")
    assert res.best_candidate is not None


def test_m_six_runs():
    res = LEXSCM(_cfg(_panel(N=40), m=6, top_K=10)).fit()
    assert len(res.best_candidate.identification.treated_idx) == 6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
