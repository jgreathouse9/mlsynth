"""Is the reported ``V`` pinned down by the data, or one of many optima?

The predictor weights are generically non-identified: a whole set of ``V`` can
attain the same upper-level loss. ``synth``'s own help warns the nested space
"may contain local minima" and offers restarts as a robustness check, and the
backends bear that out -- on one county of the Wiltshire panel ``malo`` put
weight on 1.7 of 14 predictors and ``mscmt`` on 9.0, both outer-optimal.

``v_agreement`` already reports whether the two MSCMT canonicalisations agree.
What it does not report is how *concentrated* the returned ``V`` is, which is
the difference a user actually sees between backends. Two measures are added:

``n_predictors_weighted``
    Count above a fixed tolerance. Simple, but tolerance-dependent.
``v_effective_count``
    :math:`1 / \\sum_h v_h^2`, the participation ratio. No tolerance: it is
    ``K`` for uniform weights and ``1`` for a corner, and moves continuously
    between. This is the one to read.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _panel(n_units=8, T=12, pre=9, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        base = float(rng.normal()) * 3.0
        for t in range(T):
            rows.append({
                "unit": f"u{u}", "t": t,
                "y": float(base + 0.4 * t + rng.normal() * 0.2),
                "x1": float(rng.normal()),
                "x2": float(rng.normal()) * 100.0,
                "x3": float(rng.normal()) * 0.01,
                "treat": int(u == 0 and t >= pre),
            })
    return pd.DataFrame(rows)


def _fit(backend, **over):
    from mlsynth import VanillaSC

    cfg = {"df": _panel(), "outcome": "y", "treat": "treat", "unitid": "unit",
           "time": "t", "backend": backend, "inference": False,
           "display_graphs": False}
    if backend != "outcome-only":
        cfg["covariates"] = ["x1", "x2", "x3"]
    cfg.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(cfg).fit()


class TestTheDiagnosticIsReported:
    @pytest.mark.parametrize("backend", ["regression", "malo", "mscmt"])
    def test_the_predictor_backends_report_concentration(self, backend):
        res = _fit(backend)
        extra = res.weights.summary_stats or {}
        assert extra.get("predictor_weights") is not None
        assert extra.get("n_predictors_weighted") is not None
        assert extra.get("v_effective_count") is not None

    @pytest.mark.parametrize("backend", ["regression", "malo", "mscmt"])
    def test_it_reaches_the_spec_block_where_users_look(self, backend):
        res = _fit(backend)
        params = res.method_details.parameters_used or {}
        assert "v_effective_count" in params
        assert params["v_effective_count"] is not None

    def test_outcome_only_reports_no_predictor_weights(self):
        res = _fit("outcome-only")
        extra = res.weights.summary_stats or {}
        params = res.method_details.parameters_used or {}
        assert extra.get("predictor_weights") is None
        assert extra.get("v_effective_count") is None
        assert params.get("v_effective_count") is None


class TestItMeasuresWhatItClaims:
    def test_the_effective_count_is_bounded_by_the_predictor_count(self):
        for backend in ("regression", "malo", "mscmt"):
            res = _fit(backend)
            eff = res.weights.summary_stats["v_effective_count"]
            assert 1.0 - 1e-9 <= eff <= 3.0 + 1e-9, backend

    def test_a_corner_scores_one_and_uniform_scores_K(self):
        from mlsynth.utils.vanillasc_helpers.pipeline import _v_concentration

        n, eff = _v_concentration(np.array([1.0, 0.0, 0.0, 0.0]))
        assert n == 1 and eff == pytest.approx(1.0)
        n, eff = _v_concentration(np.full(4, 0.25))
        assert n == 4 and eff == pytest.approx(4.0)

    def test_it_is_continuous_between_those_ends(self):
        from mlsynth.utils.vanillasc_helpers.pipeline import _v_concentration

        # two predictors carrying almost everything, two carrying a sliver
        _, eff = _v_concentration(np.array([0.48, 0.48, 0.02, 0.02]))
        assert 2.0 < eff < 2.3

    def test_none_in_none_out(self):
        from mlsynth.utils.vanillasc_helpers.pipeline import _v_concentration

        assert _v_concentration(None) == (None, None)

    def test_it_separates_a_corner_backend_from_a_dense_one(self):
        """The point of the diagnostic.

        ``malo`` searches predictor corners, so with a lagged-outcome-like
        predictor present it concentrates; the regression rule weights every
        predictor by a squared coefficient and so generally does not. A user
        comparing the two should be able to see that from the result.
        """
        eff = {b: _fit(b).weights.summary_stats["v_effective_count"]
               for b in ("malo", "regression")}
        assert eff["malo"] != pytest.approx(eff["regression"], abs=1e-6)
