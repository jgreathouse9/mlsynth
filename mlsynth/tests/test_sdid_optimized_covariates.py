"""SDID's third covariate method: Stata's default.

``sdid`` in Stata takes ``covariates(varlist, type)`` with ``optimized`` as the
*default* and ``projected`` as the alternative. mlsynth had only ``projected``
(as ``adjust``, following Kranz 2022), so a paper that simply wrote
``covariates(x)`` -- taking the default -- had a specification the library could
not express.

The two are genuinely different estimators, not two solvers for one thing:

* ``projected`` estimates beta once, by OLS on rows with no treatment in force,
  and subtracts ``X beta`` before SDID ever runs. The weight programs never see
  a covariate.
* ``optimized`` estimates beta *jointly* with the unit and time weights,
  minimising a single objective in ``(beta, omega, lambda)``. Changing beta
  changes the weights and vice versa.

These tests pin the second, and pin that adding it leaves the first alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

_QUOTA = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
          / "sdid_quota" / "quota_panel.csv")

# Clarke, Pailanir, Athey & Imbens (2024), Stata Journal, Table 1.
STATA = {"none": 8.034, "optimized": 8.051, "projected": 8.059}
# Columns 1 and 3 already reproduce to this; see docs/sdid.rst.
SOLVER_TOL = 0.02


@pytest.fixture(scope="module")
def quota():
    if not _QUOTA.exists():  # pragma: no cover - the panel is committed
        pytest.skip(f"missing {_QUOTA}")
    d = pd.read_csv(_QUOTA)
    d["year"] = d["year"].astype(int)
    return d


@pytest.fixture(scope="module")
def block(quota):
    """One adoption cohort's blocks, the unit the solver actually works on."""
    from mlsynth.utils.sdid_helpers.setup import prepare_sdid_inputs

    d = quota.dropna(subset=["lngdp"])
    inputs = prepare_sdid_inputs(df=d, outcome="womparl", treat="quota",
                                 unitid="country", time="year")
    key = min(inputs.cohorts_dict)
    p = inputs.cohorts_dict[key]
    wide = d.pivot_table(index="year", columns="country", values="lngdp")
    order = sorted(wide.index)
    return dict(
        donor_outcomes=np.asarray(p["donor_matrix"], dtype=float),
        treated_outcome=np.asarray(p["y"], dtype=float).mean(axis=1),
        donor_covariates=wide.reindex(
            index=order, columns=list(p["donor_names"])).to_numpy(float)[None],
        treated_covariates=wide.reindex(
            index=order, columns=list(p["treated_indices"])
        ).to_numpy(float).mean(axis=1)[None],
        pre_periods=int(p["pre_periods"]),
        n_treated=len(p["treated_indices"]),
    )


def _att(df, **extra):
    import warnings

    from mlsynth import SDID

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(SDID({
            "df": df, "outcome": "womparl", "treat": "quota",
            "unitid": "country", "time": "year", "vce": "noinference",
            "display_graphs": False, **extra}).fit().effects.att)


class TestItReproducesStata:
    """The reason the option exists."""

    def test_optimized_matches_the_published_estimate(self, quota):
        d = quota.dropna(subset=["lngdp"])
        got = _att(d, covariates={"optimized": ["lngdp"]})
        assert got == pytest.approx(STATA["optimized"], abs=SOLVER_TOL)

    def test_the_two_guard_columns_still_reproduce(self, quota):
        """Columns 1 and 3 of the same table, which already worked.

        They are the regression guards: if adding a third method moved either,
        the new code has reached somewhere it should not."""
        assert _att(quota) == pytest.approx(STATA["none"], abs=SOLVER_TOL)
        d = quota.dropna(subset=["lngdp"])
        assert _att(d, covariates={"adjust": ["lngdp"]}) == pytest.approx(
            STATA["projected"], abs=SOLVER_TOL)

    def test_optimized_and_projected_differ(self, quota):
        """Different estimators, so they must not collapse onto each other.

        On this panel Stata separates them by only 0.008, which is inside the
        solver tolerance -- so this asserts they are computed differently, via
        the fitted coefficients, rather than that the ATTs differ. ``projected``
        fits one coefficient on every untreated row; ``optimized`` fits one per
        adoption cohort, against that cohort's own donors and horizon.
        """
        d = quota.dropna(subset=["lngdp"])
        from mlsynth.utils.sdid_helpers.covariates import twfe_covariate_beta
        from mlsynth.utils.sdid_helpers.setup import prepare_sdid_inputs

        b_proj = twfe_covariate_beta(d[d["quota"] == 0], "womparl", ["lngdp"],
                                     "country", "year")
        inputs = prepare_sdid_inputs(
            df=d, outcome="womparl", treat="quota", unitid="country",
            time="year", optimized_covariates=["lngdp"])
        b_opt = {k: p["optimized_beta"]
                 for k, p in inputs.cohorts_dict.items()}

        assert len(b_opt) > 1, "this panel is staggered; expect one beta each"
        assert all(np.isfinite(b).all() for b in b_opt.values())
        assert not all(np.allclose(b, b_proj) for b in b_opt.values())
        # and the cohorts do not all agree with each other either
        assert len({round(float(b[0]), 6) for b in b_opt.values()}) > 1


class TestWhatTheObjectiveActuallyDoes:
    """``optimized`` descends on an objective; it does not reach its minimum.

    This started as a test that the fitted coefficient beats its neighbourhood
    on the objective. It does not, and the reason is the estimator rather than
    a defect: ``synthdid`` steps beta by ``1/t``, so its total step budget grows
    like ``log(max.iter)`` and the iteration stops at its cap while beta is
    still drifting. Because the objective is very nearly flat in beta, that
    early stop is load-bearing -- it shrinks beta toward zero, and running to
    the true minimum gives a coefficient an order of magnitude larger and an
    ATT that misses every published figure.

    So both halves are pinned: the fit makes progress, and it stops short. A
    future change that silently converges would move published numbers, and
    ``test_it_stops_short_of_the_minimum`` is what catches it.
    """

    def test_the_fit_lowers_the_objective(self, block):
        from mlsynth.utils.sdid_helpers.covariates import (
            optimized_covariate_beta, sdid_covariate_objective)

        beta = optimized_covariate_beta(**block)
        assert sdid_covariate_objective(beta=beta, **block) < (
            sdid_covariate_objective(beta=np.zeros_like(beta), **block))

    def test_it_stops_short_of_the_minimum(self, block):
        """Pins the early stop, rather than leaving it as folklore."""
        from mlsynth.utils.sdid_helpers.covariates import (
            optimized_covariate_beta, sdid_covariate_objective)

        short = optimized_covariate_beta(max_iter=100, **block)
        long = optimized_covariate_beta(max_iter=10_000, **block)
        # still moving in the same direction, and still improving
        assert abs(float(long[0])) > abs(float(short[0]))
        assert (sdid_covariate_objective(beta=long, **block)
                < sdid_covariate_objective(beta=short, **block))

    def test_the_step_budget_grows_only_logarithmically(self, block):
        """Why it stops short: ``sum 1/t`` is a log, so beta crawls.

        Two decades of extra iterations move beta by less than the first
        decade did. That is the shape of the schedule, and it is the reason
        the cap binds instead of the convergence test.
        """
        from mlsynth.utils.sdid_helpers.covariates import (
            optimized_covariate_beta)

        b = {n: float(optimized_covariate_beta(max_iter=n, **block)[0])
             for n in (10, 100, 1000)}
        first = abs(b[100] - b[10])
        later = abs(b[1000] - b[100])
        assert 0.0 < later < first

    def test_zero_covariate_effect_recovers_the_no_covariate_fit(self, quota):
        """A covariate with no relation to the outcome should leave the
        estimate essentially where it was."""
        rng = np.random.default_rng(0)
        d = quota.copy()
        d["noise"] = rng.normal(size=len(d))
        base = _att(d)
        with_noise = _att(d, covariates={"optimized": ["noise"]})
        assert with_noise == pytest.approx(base, abs=0.5)


class TestItComposesWithTheExistingInterface:
    """Third key on the same dict; everything else keeps working."""

    def test_adjust_is_unchanged_by_the_addition(self, quota):
        d = quota.dropna(subset=["lngdp"])
        assert np.isfinite(_att(d, covariates={"adjust": ["lngdp"]}))

    def test_optimized_and_adjust_cannot_share_a_column(self, quota):
        from mlsynth.exceptions import MlsynthConfigError

        d = quota.dropna(subset=["lngdp"])
        with pytest.raises(MlsynthConfigError, match="both"):
            _att(d, covariates={"optimized": ["lngdp"], "adjust": ["lngdp"]})

    def test_a_missing_column_is_reported(self, quota):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="not in"):
            _att(quota, covariates={"optimized": ["nope"]})

    def test_a_non_numeric_column_is_reported(self, quota):
        from mlsynth.exceptions import MlsynthConfigError

        d = quota.copy()
        d["label"] = "x"
        with pytest.raises(MlsynthConfigError, match="numeric"):
            _att(d, covariates={"optimized": ["label"]})

    def test_an_empty_list_is_reported(self, quota):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="empty"):
            _att(quota, covariates={"optimized": []})

    def test_the_method_name_records_the_variant(self, quota):
        import warnings

        from mlsynth import SDID

        d = quota.dropna(subset=["lngdp"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SDID({"df": d, "outcome": "womparl", "treat": "quota",
                        "unitid": "country", "time": "year",
                        "vce": "noinference", "display_graphs": False,
                        "covariates": {"optimized": ["lngdp"]}}).fit()
        assert "X" in res.method_details.method_name


class TestDegenerateBlocksAreRejected:
    """The block solver is reached with arrays, so it validates its own shapes.

    Every one of these is a way a caller can hand it something structurally
    wrong; each must say which block is wrong rather than fail later inside a
    weight solve, where the message would name a quantity the caller never set.
    """

    def _mutate(self, block, **over):
        out = dict(block)
        out.update(over)
        return out

    def test_a_single_pre_period_is_reported(self, block):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import (
            optimized_covariate_beta)

        with pytest.raises(MlsynthDataError, match="pre-treatment"):
            optimized_covariate_beta(**self._mutate(block, pre_periods=1))

    def test_no_post_period_is_reported(self, block):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import (
            optimized_covariate_beta)

        n_t = block["donor_outcomes"].shape[0]
        with pytest.raises(MlsynthDataError, match="post-treatment"):
            optimized_covariate_beta(**self._mutate(block, pre_periods=n_t))

    def test_no_donors_is_reported(self, block):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import (
            optimized_covariate_beta)

        bad = self._mutate(
            block,
            donor_outcomes=block["donor_outcomes"][:, :0],
            donor_covariates=block["donor_covariates"][:, :, :0])
        with pytest.raises(MlsynthDataError, match="donor"):
            optimized_covariate_beta(**bad)

    def test_a_ragged_covariate_block_is_reported(self, block):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import (
            optimized_covariate_beta)

        bad = self._mutate(
            block, donor_covariates=block["donor_covariates"][:, :, :-1])
        with pytest.raises(MlsynthDataError, match="line up"):
            optimized_covariate_beta(**bad)

    def test_a_mis_shaped_outcome_block_is_reported(self, block):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import (
            optimized_covariate_beta)

        bad = self._mutate(block,
                           treated_outcome=block["treated_outcome"][:-1])
        with pytest.raises(MlsynthDataError, match=r"\(T, N0\)"):
            optimized_covariate_beta(**bad)

    def test_a_two_dimensional_covariate_cube_is_reported(self, block):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import (
            optimized_covariate_beta)

        bad = self._mutate(block,
                           donor_covariates=block["donor_covariates"][0])
        with pytest.raises(MlsynthDataError, match=r"\(K, T, N0\)"):
            optimized_covariate_beta(**bad)

    def test_a_non_finite_covariate_is_reported(self, block):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import (
            optimized_covariate_beta)

        X = block["donor_covariates"].copy()
        X[0, 0, 0] = np.nan
        with pytest.raises(MlsynthDataError, match="NaN/inf"):
            optimized_covariate_beta(**self._mutate(block,
                                                    donor_covariates=X))

    def test_a_wrong_length_beta_is_reported(self, block):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import (
            sdid_covariate_objective)

        with pytest.raises(MlsynthDataError, match="one"):
            sdid_covariate_objective(beta=[0.0, 1.0], **block)


class TestTheBlockSolverOnASingleCohortPanel:
    """The non-staggered path, which packs cohorts differently in ``setup``."""

    def _panel(self):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(8):
            for t in range(1, 15):
                x = rng.normal(3.0, 1.0)
                y = (10.0 + i + 0.3 * t + 0.8 * x + rng.normal(0, 0.2)
                     - (2.0 if (i == 0 and t >= 10) else 0.0))
                rows.append((f"u{i}", t, y, x, int(i == 0 and t >= 10)))
        return pd.DataFrame(rows, columns=["unit", "t", "y", "x", "treat"])

    def test_a_single_treated_unit_panel_fits(self):
        import warnings

        from mlsynth import SDID

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SDID({"df": self._panel(), "outcome": "y", "treat": "treat",
                        "unitid": "unit", "time": "t", "vce": "noinference",
                        "display_graphs": False,
                        "covariates": {"optimized": ["x"]}}).fit()
        assert np.isfinite(res.effects.att)

    def test_it_recovers_a_planted_covariate_effect(self):
        """The one setting where the truth is known.

        ``x`` enters the outcome with coefficient 0.8 and is independent of
        treatment, so the fitted coefficient should have the right sign and be
        of the right order -- not exact, since the fit stops early by design.
        """
        from mlsynth.utils.sdid_helpers.setup import prepare_sdid_inputs

        inputs = prepare_sdid_inputs(
            df=self._panel(), outcome="y", treat="treat", unitid="unit",
            time="t", optimized_covariates=["x"])
        beta = next(iter(inputs.cohorts_dict.values()))["optimized_beta"]
        assert 0.0 < float(beta[0]) < 2.0


class TestTheStoppingRuleThatAlmostNeverFires:
    """The convergence test exists; on real panels the cap beats it to it.

    ``TestWhatTheObjectiveActuallyDoes`` pins that the iteration runs to
    ``max_iter`` on the quota panel. That leaves the ``min_decrease`` branch
    untaken, so this constructs the case where it does fire -- a covariate that
    carries no signal at all, which zeroes the gradient and leaves the
    objective to converge on the weight steps alone.
    """

    def _block(self, covariate):
        rng = np.random.default_rng(2)
        T, N0, T0 = 12, 5, 8
        donors = rng.normal(size=(T, N0)) + np.arange(T)[:, None] * 0.2
        treated = donors.mean(axis=1) + rng.normal(0, 0.1, size=T)
        return dict(
            donor_outcomes=donors, treated_outcome=treated,
            donor_covariates=covariate(T, N0)[None],
            treated_covariates=np.zeros((1, T)),
            pre_periods=T0, n_treated=1)

    def test_a_signal_free_covariate_stops_before_the_cap(self):
        from mlsynth.utils.sdid_helpers.covariates import (
            optimized_covariate_beta)

        block = self._block(lambda T, N0: np.zeros((T, N0)))
        # If the break never fired this would run 10**6 iterations; that it
        # returns at all is the assertion, and the coefficient stays at zero.
        beta = optimized_covariate_beta(max_iter=1_000_000, **block)
        assert float(beta[0]) == pytest.approx(0.0, abs=1e-12)


class TestTheObjectiveIsTheDocumentedSum:
    """The objective's formula, checked term by term rather than as a black box.

    Comparing objective values across betas -- which the tests above do -- is
    insensitive to the formula's *shape*: dropping the ridge, or swapping which
    block is divided by the pre-period count and which by the donor count,
    changes every value by roughly the same amount and so flips no comparison.
    These pin the decomposition itself, at the seam where it is assembled.
    """

    def test_it_equals_its_three_documented_terms(self, block):
        from mlsynth.utils.sdid_helpers.covariates import (
            _noise_level, sdid_covariate_objective)
        from mlsynth.utils.sdid_helpers.weights import (fit_time_weights,
                                                        unit_weights)

        beta = np.array([0.4])
        t0 = block["pre_periods"]
        n_post = block["donor_outcomes"].shape[0] - t0
        donors = (block["donor_outcomes"]
                  - np.tensordot(beta, block["donor_covariates"], axes=(0, 0)))
        treated = block["treated_outcome"] - beta @ block["treated_covariates"]

        zeta = ((block["n_treated"] * n_post) ** 0.25) * _noise_level(
            block["donor_outcomes"], t0)
        donor_pre, post_mean = donors[:t0], donors[t0:].mean(axis=0)
        a_o, omega = unit_weights(donor_pre, treated[:t0], zeta)
        a_l, lam = fit_time_weights(donor_pre, post_mean)
        err_o = treated[:t0] - a_o - donor_pre @ omega
        err_l = post_mean - a_l - donor_pre.T @ lam

        n0 = donors.shape[1]
        expected = (err_o @ err_o / t0            # omega block: one per period
                    + err_l @ err_l / n0          # lambda block: one per donor
                    + zeta ** 2 * (omega @ omega))
        assert sdid_covariate_objective(beta=beta, **block) == pytest.approx(
            float(expected), rel=1e-12)

    def test_the_ridge_term_is_present(self, block):
        """Separately, since it is the term a value comparison cannot see."""
        from mlsynth.utils.sdid_helpers.covariates import (
            _noise_level, sdid_covariate_objective)
        from mlsynth.utils.sdid_helpers.weights import (fit_time_weights,
                                                        unit_weights)

        beta = np.array([0.4])
        t0 = block["pre_periods"]
        n_post = block["donor_outcomes"].shape[0] - t0
        donors = (block["donor_outcomes"]
                  - np.tensordot(beta, block["donor_covariates"], axes=(0, 0)))
        treated = block["treated_outcome"] - beta @ block["treated_covariates"]
        zeta = ((block["n_treated"] * n_post) ** 0.25) * _noise_level(
            block["donor_outcomes"], t0)
        donor_pre, post_mean = donors[:t0], donors[t0:].mean(axis=0)
        a_o, omega = unit_weights(donor_pre, treated[:t0], zeta)
        a_l, lam = fit_time_weights(donor_pre, post_mean)
        fit_only = (float((treated[:t0] - a_o - donor_pre @ omega) @
                          (treated[:t0] - a_o - donor_pre @ omega)) / t0
                    + float((post_mean - a_l - donor_pre.T @ lam) @
                            (post_mean - a_l - donor_pre.T @ lam))
                    / donors.shape[1])
        got = sdid_covariate_objective(beta=beta, **block)
        assert got > fit_only
        assert got - fit_only == pytest.approx(
            float(zeta ** 2 * (omega @ omega)), rel=1e-9)


class TestTheRidgeScalesWithTheDesign:
    """``n_treated`` enters only through zeta, so it must change the answer.

    Arkhangelsky et al. set the unit-weight penalty to
    ``(N_tr * T_post)^(1/4) * sigma``. If the treated count or the horizon were
    dropped from that expression the estimator would still run and still look
    reasonable -- it would just be penalising by the wrong amount.
    """

    def test_more_treated_units_change_the_objective(self, block):
        from mlsynth.utils.sdid_helpers.covariates import (
            sdid_covariate_objective)

        one = sdid_covariate_objective(beta=[0.3], **{**block, "n_treated": 1})
        many = sdid_covariate_objective(beta=[0.3], **{**block, "n_treated": 9})
        assert one != pytest.approx(many, rel=1e-9)

    def test_more_treated_units_change_the_fit(self, block):
        from mlsynth.utils.sdid_helpers.covariates import (
            optimized_covariate_beta)

        one = optimized_covariate_beta(max_iter=200,
                                       **{**block, "n_treated": 1})
        many = optimized_covariate_beta(max_iter=200,
                                        **{**block, "n_treated": 9})
        assert not np.allclose(one, many)
