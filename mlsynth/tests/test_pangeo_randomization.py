"""TDD for design-based (randomization) inference on the PANGEO realized ATT.

PANGEO's realized-effect path (``results.effects``) reports an Augmented-DiD
ATT with a model-based prediction-variance SE. That is the *analysis*-model
inference. Because treatment is randomized *within* each supergeo pair, the
design also supports assumption-light **design-based** inference, which this
adds alongside the ADID numbers:

* a two-sided **permutation (randomization) p-value** -- recompute the
  (weighted) mean of the antisymmetric within-pair DiD contrasts under every
  within-pair sign flip (exact for small pair counts, Monte-Carlo with a fixed
  seed otherwise), and compare to the observed statistic; and
* the **matched-pair (pair-clustered) SE** ``sqrt(sum wn_k^2 (d_k - att)^2) *
  K/(K-1)`` with a ``t_{K-1}`` interval -- the analytic companion that respects
  the pairing and reduces to ``sd(d_k)/sqrt(K)`` under equal weights.

These test the pure inference on hand-built pair records (so the permutation
distribution is known exactly) and the end-to-end plumbing through PANGEO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PANGEO
from mlsynth.config_models import PANGEOConfig
from mlsynth.utils.pangeo_helpers.effects import _randomization_inference
from mlsynth.utils.pangeo_helpers.simulation import make_seasonal_sales_panel


def _rec(dt_treat, dt_ctrl, w=1.0):
    """A pair whose DiD contrast is ``dt_treat - dt_ctrl`` by construction:
    pre means are 0 on both sides, post means are the given changes."""
    return {
        "YT_pre": np.zeros(4), "YC_pre": np.zeros(4),
        "YT_post": np.full(3, float(dt_treat)),
        "YC_post": np.full(3, float(dt_ctrl)),
        "weight": float(w),
    }


class TestRandomizationPure:
    def test_uniform_positive_effect_exact_pvalue(self):
        # 4 pairs, each DiD contrast = +10, equal weights. Statistic
        # 2.5*sum(s_k); |stat| >= observed (10) only for all-+ / all-- => 2/16.
        recs = [_rec(10, 0) for _ in range(4)]
        r = _randomization_inference("program", recs, 0.05,
                                     rng=np.random.default_rng(0))
        assert r.att == pytest.approx(10.0)
        assert r.exact is True
        assert r.n_pairs == 4
        assert r.p_permutation == pytest.approx(2.0 / 16.0)

    def test_cancelling_effects_give_pvalue_one(self):
        recs = [_rec(10, 0), _rec(0, 10), _rec(10, 0), _rec(0, 10)]
        r = _randomization_inference("program", recs, 0.05,
                                     rng=np.random.default_rng(0))
        assert r.att == pytest.approx(0.0)
        assert r.p_permutation == pytest.approx(1.0)

    def test_matched_pair_se_matches_sd_over_sqrt_k_equal_weights(self):
        d = [8.0, 10.0, 12.0, 10.0]
        recs = [_rec(x, 0) for x in d]
        r = _randomization_inference("program", recs, 0.05,
                                     rng=np.random.default_rng(0))
        expected = np.std(d, ddof=1) / np.sqrt(len(d))
        assert r.se_pair == pytest.approx(expected)
        assert r.df == len(d) - 1

    def test_large_k_switches_to_monte_carlo_and_is_deterministic(self):
        recs = [_rec(5, 0) for _ in range(16)]  # 2^16 -> Monte-Carlo path
        r1 = _randomization_inference("p", recs, 0.05, rng=np.random.default_rng(0))
        r2 = _randomization_inference("p", recs, 0.05, rng=np.random.default_rng(0))
        assert r1.exact is False
        assert r1.n_perm == r2.n_perm
        assert r1.p_permutation == r2.p_permutation  # fixed seed -> identical


class TestRandomizationPlumbing:
    def _fit(self):
        df = make_seasonal_sales_panel(
            units_per_arm=6, arms=("A",), T=40, n_post=8, seed=0)
        return PANGEO(PANGEOConfig(
            df=df, outcome="sales", arm="arm", unitid="unit", time="time",
            post_col="post_col", max_supergeo_size=2, fast=True,
            compute_power=False, display_graphs=False)).fit()

    def test_effects_carry_randomization(self):
        res = self._fit()
        rand = res.effects.randomization
        assert "program" in rand
        prog = rand["program"]
        assert prog.n_pairs == sum(len(d.pairs)
                                   for d in res.arm_designs.values())
        assert 0.0 <= prog.p_permutation <= 1.0
        assert prog.exact is True  # few pairs -> exact enumeration

    def test_randomization_summary_table(self):
        res = self._fit()
        tbl = res.effects.randomization_summary()
        assert isinstance(tbl, pd.DataFrame)
        assert "p_permutation" in tbl.columns
        assert "program" in tbl.index
