"""TDD for the weakly-targeted treated-tuple selection in LEXSCM.

LEXSCM's first QP (``lexsearch.select_treated_designs``) is the *fully targeted*
design: it picks the treated m-tuple + simplex weights ``w`` that drive the
weighted treated combination onto the population mean,

    min_{w in simplex(S)}  || sum_j w_j Xtilde_j ||^2 = w' G_SS w,

with ``Xtilde`` f-centred on the population (so the origin is the population
mean). This is Abadie & Zhao's "representative experiment" goal (eq. 3) and
Vives-i-Bastida's eq. (1) term 1, solved lexicographically (xi -> 0).

A *weakly targeted* design relaxes that: add a penalty pulling ``w`` toward the
group's own equal-weight aggregate, so the treated group looks more like itself
than like the population, trading population-representativeness (the ATE) for a
deployable equal-weight group (closer to its own ATT):

    min_{w in simplex(S)}  w' G_SS w + gamma || w - (1/m) 1 ||^2 .

On the simplex ``1'w = 1`` so ``||w - 1/m 1||^2 = w'w - 1/m``; the penalty is
therefore a plain diagonal ridge and the program is exactly

    min_{w in simplex(S)}  w' (G_SS + gamma I) w   (up to the constant gamma/m).

So ``targeting_penalty=gamma`` ridges the Gram diagonal for the treated search,
and the reported imbalance is the *true* targeting distance ``sqrt(w' G w)`` at
the penalized ``w`` (the Abadie-Zhao validity quantity), not the penalized
objective. ``gamma=0`` must be bit-identical to today's design.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.lexsearch import (
    _afw_single,
    select_treated_designs,
)


def _gram(n=12, T=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, n))
    X = X - X.mean(axis=1, keepdims=True)          # f-centred (uniform f)
    return X.T @ X


class TestWeakTargeting:
    def test_gamma_zero_bit_identical(self):
        G = _gram(seed=0); cand = list(range(G.shape[0]))
        base = select_treated_designs(G, cand, m=3, top_K=5, method="enumerate")
        pen0 = select_treated_designs(G, cand, m=3, top_K=5, method="enumerate",
                                      targeting_penalty=0.0)
        assert [d.indices for d in base["top_designs"]] == \
               [d.indices for d in pen0["top_designs"]]
        for a, b in zip(base["top_designs"], pen0["top_designs"]):
            np.testing.assert_array_equal(a.weights, b.weights)
            assert a.loss == b.loss
            assert a.imbalance == b.imbalance

    def test_high_gamma_pulls_weights_to_uniform(self):
        G = _gram(seed=1); cand = list(range(G.shape[0]))
        m = 3
        lo = select_treated_designs(G, cand, m=m, top_K=1, method="enumerate",
                                    targeting_penalty=0.0)["top_designs"][0]
        hi = select_treated_designs(G, cand, m=m, top_K=1, method="enumerate",
                                    targeting_penalty=1e4)["top_designs"][0]
        u = np.full(m, 1.0 / m)
        np.testing.assert_allclose(hi.weights, u, atol=1e-2)
        # weakly-targeted weights are at least as uniform as fully-targeted
        assert np.sum((hi.weights - u) ** 2) <= np.sum((lo.weights - u) ** 2) + 1e-9

    def test_reported_imbalance_is_true_not_penalized(self):
        G = _gram(n=10, T=18, seed=2); cand = list(range(10))
        d = select_treated_designs(G, cand, m=3, top_K=1, method="enumerate",
                                   targeting_penalty=5.0)["top_designs"][0]
        S = d.indices; w = np.asarray(d.weights)
        true_imb = float(np.sqrt(max(w @ G[np.ix_(S, S)] @ w, 0.0)))
        np.testing.assert_allclose(d.imbalance, true_imb, rtol=1e-6, atol=1e-9)
        # penalized objective (loss) >= true imbalance^2
        assert d.loss >= true_imb ** 2 - 1e-9

    def test_ridge_equals_uniform_anchor_penalty(self):
        # min w'(G+gamma I)w == min w'Gw + gamma||w - 1/m 1||^2  (differ by gamma/m).
        G = _gram(n=5, T=12, seed=3); m = 5; gamma = 2.0
        loss_ridge, w, _ = _afw_single(G + gamma * np.eye(m), iters=600, tol=1e-14)
        u = np.full(m, 1.0 / m)
        explicit = float(w @ G @ w + gamma * np.sum((w - u) ** 2))
        assert loss_ridge == pytest.approx(explicit + gamma / m, abs=1e-9)

    def test_negative_gamma_rejected(self):
        G = _gram(seed=4); cand = list(range(G.shape[0]))
        with pytest.raises(Exception):
            select_treated_designs(G, cand, m=3, targeting_penalty=-1.0)


# ---------------------------------------------------------------------------
# Estimator wiring: targeting_penalty flows from config through LEXSCM.fit
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402

from mlsynth import LEXSCM  # noqa: E402
from mlsynth.utils.fast_scm_helpers.config import LEXSCMConfig  # noqa: E402


def _panel(n_units=12, T=40, T_post=10, n_candidates=8, L=2, sigma=0.1,
           seed=0, baseline=100.0):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n_units, L)); nu = rng.standard_normal((T, L))
    Y = baseline + nu @ g.T + sigma * rng.standard_normal((T, n_units))
    rows = []
    for i in range(n_units):
        for t in range(T):
            rows.append({"unitid": f"u{i:02d}", "time": t, "y": Y[t, i],
                         "post": int(t >= T - T_post),
                         "candidate": int(i < n_candidates)})
    return pd.DataFrame(rows)


class TestEstimatorWiring:
    def test_config_default_is_zero(self):
        cfg = LEXSCMConfig(df=_panel(), outcome="y", unitid="unitid", time="time",
                           candidate_col="candidate", m=3)
        assert cfg.targeting_penalty == 0.0

    def test_fit_weakly_targeted_weights_near_uniform(self):
        base = dict(df=_panel(), outcome="y", unitid="unitid", time="time",
                    candidate_col="candidate", m=3, post_col="post",
                    display_graph=False, verbose=False)
        rg = LEXSCM({**base, "targeting_penalty": 1e4}).fit()
        wg = np.array(list(rg.search.candidates[0].treated_weight_dict.values()))
        np.testing.assert_allclose(wg, np.full(len(wg), 1.0 / len(wg)), atol=1e-2)
