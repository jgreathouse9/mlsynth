"""Tests for the LPCA estimator (Feng 2024).

Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): the pseudo-max distance's metric properties, the
  singular-value ratio rank rule, and the Gram-route reconstruction against a
  plain SVD.
* Layer 2 (data utilities): panel ingestion, the block split, and the guards
  that reject panels the method cannot serve.
* Layer 3 (estimator integration): recovery of a planted effect on a nonlinear
  factor panel, the re-centring invariant, and Feng's Kansas application.
* Layer 4 (public API contracts): import, frozen results, config validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import LPCA
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.lpca_helpers import (
    LPCAInputs,
    LPCAResults,
    local_pca_row,
    neighbour_index,
    prepare_lpca_inputs,
    pseudo_max_distance,
    select_rank,
)


def _nonlinear_panel(n_units=60, T=60, T0=50, effect=2.0, noise=0.05, seed=3,
                     scale=0.1):
    """Panel whose untreated outcome is a nonlinear function of latent factors.

    ``Y_it(0) = exp(-|alpha_i - varpi_t| / scale) + noise`` -- Model 2 of the
    paper's Section 5, which is smooth but has no exact low-rank
    representation. Unit 0 is treated from ``T0`` with a constant effect.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0, 1, n_units)
    varpi = rng.uniform(0, 1, T)
    Y = np.exp(-np.abs(alpha[:, None] - varpi[None, :]) / scale)
    Y = Y + rng.normal(0, noise, (n_units, T))
    D = np.zeros((n_units, T))
    D[0, T0:] = 1.0
    Y[0, T0:] += effect
    rows = [{"unit": f"u{i:02d}", "time": t, "y": Y[i, t], "D": int(D[i, t])}
            for i in range(n_units) for t in range(T)]
    return pd.DataFrame(rows), effect


def _kansas_growth():
    """Feng's Kansas panel, first-differenced in percent as the paper uses it."""
    from pathlib import Path
    base = Path(__file__).resolve().parents[2] / "basedata" / "kansas_taxcut.csv"
    df = pd.read_csv(base)
    df = df.sort_values(["state", "year_qtr"])
    df["growth"] = df.groupby("state")["lngdpcapita"].diff() * 100.0
    return df.dropna(subset=["growth"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Layer 1 -- numerical helpers
# ----------------------------------------------------------------------
class TestDistance:
    def test_zero_diagonal_symmetric_nonnegative(self):
        rng = np.random.default_rng(0)
        D = pseudo_max_distance(rng.standard_normal((12, 40)))
        assert np.allclose(np.diag(D), 0.0)
        assert np.allclose(D, D.T, atol=1e-12)
        assert (D >= 0).all()

    def test_each_unit_is_its_own_nearest_neighbour(self):
        rng = np.random.default_rng(1)
        D = pseudo_max_distance(rng.standard_normal((10, 30)))
        assert (np.argmin(D, axis=1) == np.arange(10)).all()

    def test_matches_the_definition(self):
        """rho(x_i, x_j) = max_{l != i,j} |(1/p)(x_i - x_j)' x_l|."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((6, 15))
        D = pseudo_max_distance(X)
        p = X.shape[1]
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                want = max(abs((X[i] - X[j]) @ X[l] / p)
                           for l in range(6) if l not in (i, j))
                assert D[i, j] == pytest.approx(want, abs=1e-12)

    def test_euclidean_and_average_are_available(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((8, 20))
        for metric in ("euclidean", "average"):
            D = pseudo_max_distance(X, metric=metric)
            assert np.allclose(np.diag(D), 0.0)
            assert np.allclose(D, D.T, atol=1e-12)

    def test_unknown_metric_raises(self):
        with pytest.raises(MlsynthConfigError):
            pseudo_max_distance(np.zeros((4, 4)), metric="cosine")


class TestNeighbourIndex:
    def test_returns_requested_size_without_ties(self):
        rng = np.random.default_rng(4)
        keep = neighbour_index(pseudo_max_distance(rng.standard_normal((20, 40))), 5)
        assert keep.sum(axis=0).tolist() == [5] * 20

    def test_ties_widen_the_neighbourhood(self):
        """The reference rule is a threshold, so exact ties are all kept.

        Binary panels tie constantly, which is why the realised neighbourhood
        has to be reported alongside the requested one.
        """
        D = np.zeros((5, 5))
        keep = neighbour_index(D, 2)
        assert keep.sum(axis=0).tolist() == [5] * 5


class TestRankRule:
    def test_floors_at_one(self):
        assert select_rank(np.array([1.0, 1.0, 1.0]), 49) == 1

    def test_keeps_all_but_one_when_every_gap_is_wide(self):
        assert select_rank(np.array([1e6, 1e3, 1.0]), 49) == 2

    def test_stops_at_the_first_indistinguishable_pair(self):
        # s1/s2 wide, s2/s3 narrow -> keep one factor.
        assert select_rank(np.array([1e6, 10.0, 9.9]), 49) == 1

    def test_never_exceeds_the_supplied_spectrum(self):
        for k in range(2, 6):
            r = select_rank(np.geomspace(1e6, 1.0, k), 49)
            assert 1 <= r <= k - 1

    def test_is_inert_below_sixteen_neighbours(self):
        """The rule compares singular-value ratios against ``log log K``.

        Ratios of a descending spectrum are always at least 1, and
        ``log log K < 1`` for ``K <= 15`` (the threshold crosses 1 at
        ``e^e = 15.15``). Below that the comparison can never fire and the rank
        is pinned at ``len(spectrum) - 1`` whatever the data says. Feng's
        Kansas application sets ``K = 14``, so its rank of 2 is mechanical.
        """
        wide = np.array([1e6, 1e3, 1.0])
        flat = np.array([1.0, 1.0, 1.0])
        for K in (5, 10, 14, 15):
            assert select_rank(wide, K) == 2
            assert select_rank(flat, K) == 2
        assert select_rank(flat, 16) == 1


class TestLocalPCA:
    def test_gram_route_matches_a_plain_svd(self):
        rng = np.random.default_rng(5)
        block = rng.standard_normal((12, 40))
        keep = np.zeros(12, dtype=bool)
        keep[[0, 2, 3, 5, 7, 8, 9, 11]] = True
        got, rank, weights = local_pca_row(block, keep, unit=3, n_neighbours=8,
                                           n_components=3)
        rows = np.flatnonzero(keep)
        U, s, Vt = np.linalg.svd(block[rows], full_matrices=False)
        U, s, Vt = U[:, :3], s[:3], Vt[:3]
        pos = int(np.flatnonzero(rows == 3)[0])
        want = (U[pos, :rank] * s[:rank]) @ Vt[:rank]
        assert np.allclose(got, want, atol=1e-10)
        # The reconstruction is exactly this weighted sum of the neighbours.
        assert np.allclose(weights @ block[rows], got, atol=1e-10)
        assert weights.shape == (rows.size,)

    def test_weights_are_a_projection_not_a_simplex(self):
        """LPCA's implied weights come from a projector, so they are unconstrained.

        The unit's own weight is the diagonal of ``U_r U_r'``, which lies in
        [0, 1] and measures how much the reconstruction leans on the unit's own
        row -- the cells the estimator overwrote for a treated unit.
        """
        rng = np.random.default_rng(11)
        block = rng.standard_normal((14, 50))
        keep = np.ones(14, dtype=bool)
        _, rank, weights = local_pca_row(block, keep, unit=6, n_neighbours=14,
                                         n_components=3)
        assert 0.0 <= weights[6] <= 1.0
        # A projector's trace is its rank, so the self-weights average to r/K.
        selves = [local_pca_row(block, keep, unit=i, n_neighbours=14,
                                n_components=3)[2][i] for i in range(14)]
        assert sum(selves) == pytest.approx(rank, abs=1e-8)

    def test_reconstructs_an_exactly_low_rank_block(self):
        """A rank-1 neighbourhood is recovered whatever rank the rule keeps.

        The rule looks for a gap in the spectrum, and past the true rank the
        singular values are roundoff, so which of them it keeps is not
        determinate. What has to hold is the reconstruction.
        """
        rng = np.random.default_rng(6)
        block = rng.standard_normal((10, 1)) @ rng.standard_normal((1, 25))
        keep = np.ones(10, dtype=bool)
        got, rank, _ = local_pca_row(block, keep, unit=4, n_neighbours=10,
                                     n_components=3)
        assert 1 <= rank <= 2
        assert np.allclose(got, block[4], atol=1e-8)

    def test_narrow_neighbourhood_caps_the_component_count(self):
        rng = np.random.default_rng(7)
        block = rng.standard_normal((3, 20))
        keep = np.ones(3, dtype=bool)
        got, rank, _ = local_pca_row(block, keep, unit=1, n_neighbours=3,
                                     n_components=1)
        assert rank == 1
        assert got.shape == (20,)


# ----------------------------------------------------------------------
# Layer 2 -- ingestion and guards
# ----------------------------------------------------------------------
class TestSetup:
    def test_masks_treated_post_cells(self):
        df, _ = _nonlinear_panel()
        inputs = prepare_lpca_inputs(df, "y", "D", "unit", "time",
                                     match_periods=25)
        assert isinstance(inputs, LPCAInputs)
        assert inputs.T0 == 50
        assert inputs.match_periods == 25
        assert np.isnan(inputs.Y_masked[inputs.treated_idx[0], 50:]).all()
        assert not np.isnan(inputs.Y_masked[inputs.treated_idx[0], :50]).any()

    def test_match_block_must_leave_a_pre_period(self):
        """The PCA block needs pre-treatment periods or there is no fit to check."""
        df, _ = _nonlinear_panel(T0=20)
        with pytest.raises(MlsynthDataError):
            prepare_lpca_inputs(df, "y", "D", "unit", "time", match_periods=20)

    def test_match_block_cannot_swallow_the_panel(self):
        df, _ = _nonlinear_panel()
        with pytest.raises(MlsynthDataError):
            prepare_lpca_inputs(df, "y", "D", "unit", "time", match_periods=60)

    def test_too_few_units_rejected(self):
        df, _ = _nonlinear_panel(n_units=2, T=20, T0=15)
        with pytest.raises(MlsynthDataError):
            prepare_lpca_inputs(df, "y", "D", "unit", "time", match_periods=5)

    def test_treatment_in_the_first_period_rejected(self):
        df, _ = _nonlinear_panel(T0=0)
        with pytest.raises(MlsynthDataError):
            prepare_lpca_inputs(df, "y", "D", "unit", "time", match_periods=10)

    def test_unbalanced_panel_rejected(self):
        df, _ = _nonlinear_panel(n_units=10, T=30, T0=20)
        with pytest.raises(MlsynthDataError):
            prepare_lpca_inputs(df.iloc[:-1], "y", "D", "unit", "time",
                                match_periods=10)

    def test_missing_treatment_cell_rejected(self):
        df, _ = _nonlinear_panel(n_units=10, T=30, T0=20)
        df = df.copy()
        df.loc[7, "D"] = np.nan
        with pytest.raises(MlsynthDataError, match="Treatment indicator"):
            prepare_lpca_inputs(df, "y", "D", "unit", "time", match_periods=10)

    def test_defaults_follow_the_papers_rules(self):
        from mlsynth.utils.lpca_helpers import (
            resolve_match_periods, resolve_neighbours,
        )
        assert resolve_neighbours(50, None) == 14        # round(50 ** (2/3))
        assert resolve_neighbours(1000, None) == 100
        assert resolve_neighbours(50, 7) == 7
        assert resolve_match_periods(104, None, 0.5) == 52
        assert resolve_match_periods(104, 40, 0.5) == 40


class TestPipelineGuards:
    def test_degenerate_neighbourhood_rejected(self):
        from mlsynth.utils.lpca_helpers import run_lpca
        df, _ = _nonlinear_panel(n_units=10, T=30, T0=20)
        inputs = prepare_lpca_inputs(df, "y", "D", "unit", "time",
                                     match_periods=10)
        with pytest.raises(MlsynthDataError, match="at least 2"):
            run_lpca(inputs, n_neighbours=1)


# ----------------------------------------------------------------------
# Layer 3 -- estimator integration
# ----------------------------------------------------------------------
class TestEstimator:
    def test_recovers_a_planted_effect(self):
        df, effect = _nonlinear_panel()
        res = LPCA({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                    "time": "time", "display_graphs": False}).fit()
        assert res.att == pytest.approx(effect, abs=0.25)

    def test_counterfactual_is_on_the_outcome_scale(self):
        """The estimator centres each period internally and must undo it.

        A counterfactual left in centred space is offset by the mean of the
        period means -- the defect that produced the published Kansas SC
        number in the paper's first version. Shape assertions cannot see it,
        so the pre-treatment levels are compared directly.
        """
        df, _ = _nonlinear_panel()
        res = LPCA({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                    "time": "time", "display_graphs": False}).fit()
        observed = np.asarray(res.time_series.observed_outcome, float)
        counterfactual = np.asarray(res.counterfactual, float)
        n_pre = int(res.metadata["pre_periods_reported"])
        assert n_pre > 0
        gap = abs(observed[:n_pre].mean() - counterfactual[:n_pre].mean())
        assert gap < 0.1 * max(abs(observed[:n_pre].mean()), 1e-8) + 0.05

    def test_demeaning_does_not_move_the_effect(self):
        """Centring is a preprocessing convenience, not part of the estimand."""
        df, _ = _nonlinear_panel()
        common = {"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                  "time": "time", "display_graphs": False}
        centred = LPCA({**common, "demean": True}).fit().att
        raw = LPCA({**common, "demean": False}).fit().att
        assert centred == pytest.approx(raw, abs=0.35)

    def test_reports_the_realised_neighbourhood_and_rank(self):
        df, _ = _nonlinear_panel()
        res = LPCA({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                    "time": "time", "display_graphs": False}).fit()
        assert res.n_neighbours == round(60 ** (2 / 3))
        assert res.neighbourhood_size >= res.n_neighbours
        assert 1 <= res.local_rank <= 2
        assert len(res.neighbours) == res.neighbourhood_size
        assert set(res.neighbour_weights) == set(res.neighbours)
        assert 0.0 <= res.self_weight <= 1.0
        assert res.donor_weights == res.neighbour_weights

    def test_series_cover_the_pca_block_only(self):
        df, _ = _nonlinear_panel()
        res = LPCA({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                    "time": "time", "match_periods": 25,
                    "display_graphs": False}).fit()
        assert len(res.counterfactual) == 60 - 25
        assert len(res.time_series.time_periods) == 60 - 25

    def test_reproduces_fengs_kansas_application(self):
        """Path A: the counterfactual sits 0.53 points above observed Kansas.

        Feng (2024) Section 6.1, tuning as the paper's script sets it:
        K = round(n^(2/3)) = 14, the first 40 differenced quarters for
        matching, at most three local components. The full replication bundle
        is benchmarks/reference/lpca_kansas/.
        """
        res = LPCA({"df": _kansas_growth(), "outcome": "growth",
                    "treat": "treated", "unitid": "state", "time": "year_qtr",
                    "match_periods": 40, "display_graphs": False}).fit()
        assert res.att == pytest.approx(-0.5306, abs=0.005)
        assert res.n_neighbours == 14
        assert res.local_rank == 2

    def test_single_donor_panel_still_fits(self):
        df, _ = _nonlinear_panel(n_units=3, T=30, T0=22)
        res = LPCA({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                    "time": "time", "match_periods": 10,
                    "display_graphs": False}).fit()
        assert np.isfinite(res.att)

    def test_plot_smoke(self, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        df, _ = _nonlinear_panel()
        res = LPCA({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                    "time": "time", "display_graphs": False}).fit()
        res.plot()
        plt.close("all")

    def test_display_graphs_draws_on_fit(self, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        drawn = []
        df, _ = _nonlinear_panel()
        res = LPCA({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                    "time": "time", "display_graphs": True}).fit()
        drawn.append(res)
        assert drawn and np.isfinite(res.att)
        plt.close("all")

    def test_local_spectrum_plot(self, monkeypatch, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        from mlsynth.utils.lpca_helpers import plot_local_spectrum
        df, _ = _nonlinear_panel()
        res = LPCA({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                    "time": "time", "display_graphs": False}).fit()
        target = tmp_path / "spectrum.png"
        ax = plot_local_spectrum(res, save=str(target))
        assert ax.get_ylabel() == "Singular value"
        assert target.exists()
        # The inert-rule note appears exactly when K <= 15.
        assert ("inert" in ax.get_title()) is (res.n_neighbours <= 15)
        plt.close("all")


# ----------------------------------------------------------------------
# Layer 4 -- public API
# ----------------------------------------------------------------------
class TestPublicAPI:
    def test_top_level_import(self):
        import mlsynth
        assert "LPCA" in mlsynth.__all__

    def test_results_are_an_effect_result(self):
        from mlsynth.config_models import EffectResult
        df, _ = _nonlinear_panel()
        res = LPCA({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                    "time": "time", "display_graphs": False}).fit()
        assert isinstance(res, (EffectResult, LPCAResults))
        with pytest.raises(Exception):
            res.att = 1.0

    @pytest.mark.parametrize("bad", [
        {"max_components": 0},
        {"max_components": -1},
        {"n_neighbors": 1},
        {"match_fraction": 0.0},
        {"match_fraction": 1.0},
        {"match_periods": 0},
        {"distance": "cosine"},
        {"nonexistent_option": 3},
    ])
    def test_invalid_config_rejected(self, bad):
        df, _ = _nonlinear_panel(n_units=10, T=30, T0=20)
        with pytest.raises(MlsynthConfigError):
            LPCA({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                  "time": "time", **bad}).fit()

    def test_neighbourhood_larger_than_the_panel_rejected(self):
        df, _ = _nonlinear_panel(n_units=10, T=30, T0=20)
        with pytest.raises(MlsynthDataError):
            LPCA({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                  "time": "time", "n_neighbors": 50,
                  "display_graphs": False}).fit()

    def test_config_is_reexported_from_config_models(self):
        from mlsynth.config_models import LPCAConfig
        from mlsynth.utils.lpca_helpers.config import LPCAConfig as Colocated
        assert LPCAConfig is Colocated
