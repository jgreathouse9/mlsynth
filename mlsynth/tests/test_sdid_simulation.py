"""The de Brabander et al. (2025) Monte Carlo DGP.

The simulator behind the Path B benchmark. Its job is narrow -- calibrate a
factor model to a real panel, then draw replications from it -- but a Monte
Carlo is only evidence if the DGP is the paper's, so the pieces that could
silently drift are pinned here rather than left to the aggregate table.

Three of those pieces follow the authors' code where their prose says something
else, and each is asserted below so the choice cannot be quietly reverted:

* the common factor is ``gamma * (1 - 1/t)``, not the paper's ``t*gamma/T0``;
* the idiosyncratic errors use ``sd = sigma**2``, so ``sigma=0.25`` is a
  standard deviation of 0.0625 -- the reading the published tables imply;
* the treated unit's factor loading is the second largest of the draws, which
  keeps it inside the donors' convex hull.
"""
from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from pathlib import Path

_PANEL = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
          / "brabander_sdid_match" / "brexit_panel.csv")


@pytest.fixture(scope="module")
def panel():
    if not _PANEL.exists():  # pragma: no cover - the panel is committed
        pytest.skip(f"missing {_PANEL}")
    return pd.read_csv(_PANEL)


@pytest.fixture(scope="module")
def cal(panel):
    from mlsynth.utils.sdid_helpers.simulate import calibrate_brabander_dgp

    return calibrate_brabander_dgp(panel)


class TestCalibration:
    def test_shapes(self, cal):
        assert cal.delta.shape == (25,)
        assert cal.beta.shape == (6, 25)
        assert cal.covariates.shape == (6, 24)
        assert cal.systematic.shape == (25, 24)
        assert np.isfinite(cal.systematic).all()

    def test_the_treated_unit_comes_first(self, cal):
        assert cal.unit_names[0] == "United Kingdom"
        assert len(cal.unit_names) == 24

    def test_it_is_a_per_period_least_squares_fit(self, cal, panel):
        """The authors' stacked interaction design is block diagonal by period.

        Asserted rather than assumed: the residual of each period's fit must be
        orthogonal to that period's regressors, which is what makes the
        collapsed form the same estimator as the stacked one.
        """
        wide = panel.pivot(index="t", columns="country", values="y")
        Y = wide.loc[202:226, list(cal.unit_names)].to_numpy()
        X = np.vstack([np.ones(24), cal.covariates]).T
        resid = Y - cal.systematic
        np.testing.assert_allclose(resid @ X, 0.0, atol=1e-8)

    def test_a_missing_covariate_column_is_reported(self, panel):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.simulate import calibrate_brabander_dgp

        with pytest.raises(MlsynthDataError, match="real_con"):
            calibrate_brabander_dgp(panel.drop(columns=["real_con"]))

    def test_a_short_panel_is_reported(self, panel):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.simulate import calibrate_brabander_dgp

        with pytest.raises(MlsynthDataError, match="periods"):
            calibrate_brabander_dgp(panel[panel.t <= 210])

    def test_an_absent_treated_unit_is_reported(self, panel):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.simulate import calibrate_brabander_dgp

        with pytest.raises(MlsynthDataError, match="Ruritania"):
            calibrate_brabander_dgp(panel, treated="Ruritania")

    def test_a_hole_in_the_outcomes_is_reported(self, panel):
        """A gap has to fail loudly: the per-period fit would otherwise
        propagate NaN into every replication and surface as an unexplained
        Monte Carlo rather than a data problem."""
        import numpy as np

        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.simulate import calibrate_brabander_dgp

        holed = panel.copy()
        holed.loc[(holed.country == "France") & (holed.t == 210), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="missing outcomes"):
            calibrate_brabander_dgp(holed)

    def test_a_hole_in_the_covariates_is_reported(self, panel):
        import numpy as np

        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.simulate import calibrate_brabander_dgp

        holed = panel.copy()
        holed.loc[(holed.country == "Japan") & (holed.t == 210),
                  "real_con"] = np.inf
        with pytest.raises(MlsynthDataError, match="NaN/inf"):
            calibrate_brabander_dgp(holed)


class TestTheDrawnPanel:
    def test_shapes_and_finiteness(self, cal):
        from mlsynth.utils.sdid_helpers.simulate import simulate_brabander

        rep = simulate_brabander(cal, gamma=1.0, sigma=1.0,
                                 rng=np.random.default_rng(0))
        assert rep.outcome.shape == (25, 24)
        assert np.isfinite(rep.outcome).all()

    def test_no_factor_and_no_noise_is_the_calibrated_panel(self, cal):
        """With both stochastic parts switched off the draw collapses to the
        systematic component, which is the sharpest statement of what the DGP
        adds to the calibration."""
        from mlsynth.utils.sdid_helpers.simulate import simulate_brabander

        rep = simulate_brabander(cal, gamma=0.0, sigma=0.0,
                                 rng=np.random.default_rng(0))
        np.testing.assert_allclose(rep.outcome, cal.systematic, atol=1e-12)

    def test_the_factor_path_is_the_authors_code_not_the_prose(self, cal):
        from mlsynth.utils.sdid_helpers.simulate import simulate_brabander

        rep = simulate_brabander(cal, gamma=1.5, sigma=0.0,
                                 rng=np.random.default_rng(0))
        t = np.arange(1, 26)
        np.testing.assert_allclose(rep.factor_path, 1.5 * (1.0 - 1.0 / t))
        # The paper's t*gamma/T0 would start at gamma/24, not at zero.
        assert rep.factor_path[0] == 0.0

    def test_the_treated_loading_is_the_second_largest(self, cal):
        """Not cosmetic: it puts the treated unit inside the donors' convex
        hull, so the simplex weight problem is feasible without extrapolating."""
        from mlsynth.utils.sdid_helpers.simulate import simulate_brabander

        for seed in range(8):
            rep = simulate_brabander(cal, gamma=1.0, sigma=1.0,
                                     rng=np.random.default_rng(seed))
            mu = rep.loadings
            assert mu[0] == np.sort(mu[1:])[-2]
            assert mu[1:].max() > mu[0]

    def test_the_noise_scale_is_sigma_squared(self, cal):
        """``sigma=0.25`` means a standard deviation of 0.0625.

        The reading is not a guess: the paper's own sigma=0.25 RMSEs are 0.0625
        times its sigma=1 ones, which sd=sigma cannot produce.
        """
        from mlsynth.utils.sdid_helpers.simulate import simulate_brabander

        draws = np.concatenate([
            simulate_brabander(cal, gamma=0.0, sigma=0.25,
                               rng=np.random.default_rng(s)).noise.ravel()
            for s in range(60)])
        assert np.std(draws) == pytest.approx(0.0625, rel=0.05)

    def test_it_is_reproducible(self, cal):
        from mlsynth.utils.sdid_helpers.simulate import simulate_brabander

        a = simulate_brabander(cal, gamma=1.0, sigma=1.0,
                               rng=np.random.default_rng(7))
        b = simulate_brabander(cal, gamma=1.0, sigma=1.0,
                               rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a.outcome, b.outcome)

    def test_the_factor_contributes_only_through_gamma(self, cal):
        """Two draws sharing an rng seed differ by exactly the factor term, so
        a comparison across gamma is paired rather than confounded by noise."""
        from mlsynth.utils.sdid_helpers.simulate import simulate_brabander

        a = simulate_brabander(cal, gamma=0.0, sigma=1.0,
                               rng=np.random.default_rng(3))
        b = simulate_brabander(cal, gamma=1.0, sigma=1.0,
                               rng=np.random.default_rng(3))
        np.testing.assert_allclose(
            b.outcome - a.outcome,
            np.outer(b.factor_path, b.loadings), atol=1e-12)


class TestFailuresAreReported:
    @pytest.mark.parametrize("bad", [-1.0, float("nan")])
    def test_a_bad_sigma(self, cal, bad):
        from mlsynth.exceptions import MlsynthConfigError
        from mlsynth.utils.sdid_helpers.simulate import simulate_brabander

        with pytest.raises(MlsynthConfigError):
            simulate_brabander(cal, gamma=1.0, sigma=bad,
                               rng=np.random.default_rng(0))

    @pytest.mark.parametrize("bad", [-1.0, float("inf")])
    def test_a_bad_gamma(self, cal, bad):
        from mlsynth.exceptions import MlsynthConfigError
        from mlsynth.utils.sdid_helpers.simulate import simulate_brabander

        with pytest.raises(MlsynthConfigError):
            simulate_brabander(cal, gamma=bad, sigma=1.0,
                               rng=np.random.default_rng(0))
