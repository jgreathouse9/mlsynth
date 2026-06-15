"""TDD for opt-in parallelism in GEOLIFT market selection.

The candidate search is embarrassingly parallel: each candidate's power
simulation and deployable design fit are independent and deterministic (the
conformal permutation seed is fixed, not order-dependent). Parallelizing across
candidates and collecting results in candidate order must therefore be
bit-identical to the serial run -- only faster. ``n_jobs=1`` (default) keeps the
exact serial path (no joblib, no overhead). These tests pin parallel == serial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import GEOLIFT
from mlsynth.utils.geolift_helpers.marketselect.helpers.batch import run_simulations
from mlsynth.utils.geolift_helpers.marketselect.helpers.candidates import (
    generate_candidate_markets,
)
from mlsynth.utils.geolift_helpers.marketselect.helpers.similarity import (
    rank_markets_by_correlation,
)


def _panel(n_units=14, T=40, seed=11):
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    post = {t: int(t >= T - 6) for t in range(T)}
    rows = []
    for i in range(n_units):
        s = base + rng.normal(scale=0.5, size=T) + i + 2 * np.sin(np.arange(T) / 6 + i)
        for t in range(T):
            rows.append({"unit": f"u{i}", "time": t, "Y": float(s[t]), "post": post[t]})
    return pd.DataFrame(rows)


def _cfg(**over):
    base = dict(df=_panel(), outcome="Y", unitid="unit", time="time",
                treatment_size=3, durations=[4], effect_sizes=[0.0, 0.1],
                lookback_window=2, ns=100, seed=0, display_graphs=False,
                post_col="post")
    base.update(over)
    return base


class TestParallelMatchesSerial:
    def test_run_simulations_bit_identical(self):
        # Build a wide panel + candidates and drive run_simulations directly.
        df = _panel()
        Ywide = df[df["post"] == 0].pivot(index="time", columns="unit", values="Y")
        ranked = rank_markets_by_correlation(Ywide)
        cands = generate_candidate_markets(ranked, treatment_size=3)
        kw = dict(durations=[4], lookback_window=2, effect_sizes=[0.0, 0.1],
                  how="mean", augment="ridge", ns=100, seed=0)
        serial = run_simulations(Ywide, cands, n_jobs=1, **kw)
        parallel = run_simulations(Ywide, cands, n_jobs=2, **kw)
        pd.testing.assert_frame_equal(serial, parallel)

    def test_geolift_fit_shortlist_identical(self):
        serial = GEOLIFT(_cfg(n_jobs=1)).fit().search.shortlist.reset_index(drop=True)
        parallel = GEOLIFT(_cfg(n_jobs=2)).fit().search.shortlist.reset_index(drop=True)
        pd.testing.assert_frame_equal(serial, parallel)

    def test_n_jobs_default_is_all_cores(self):
        # Default uses all cores (-1); users cap it with a positive int.
        from mlsynth.utils.geolift_helpers.config import GeoLiftConfig
        cfg = GeoLiftConfig(**{k: v for k, v in _cfg().items() if k != "n_jobs"})
        assert cfg.n_jobs == -1
