"""Overriding SDID's unit-weight ridge.

SDID regularises the unit weights by :math:`T_{pre}\\zeta^2` with
``zeta = (N_tr * T_post)^(1/4) * sigma_hat``, ``sigma_hat`` the standard
deviation of the first-differenced control outcomes (Arkhangelsky et al. 2021;
``synthdid``'s ``zeta.omega``). That is the right default and it stays the
default.

It is not, however, what every published SDID analysis uses. de Brabander,
Juodis & Miyazato Szini (2025) pass ``zeta.omega = 0`` throughout their Brexit
study, and without a way to say the same thing mlsynth cannot express their
specification at all -- their Table 1 SDID column sits 0.26 to 0.32 percentage
points away purely because of a penalty they switched off. Hence ``zeta``.

The option is a number, not a flag, because the quantity is a number: a caller
reproducing a paper needs whatever value that paper used, and zero is only the
most common such value.

What must not change is everything else. ``zeta=None`` computes the parameter
from the data exactly as before, and the tests below assert that against a
frozen reference rather than against the estimator's own output.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

_KRANZ = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
          / "sdid_kranz")
_BREXIT = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
           / "brabander_sdid_match" / "brexit_panel.csv")


@pytest.fixture(scope="module")
def panel():
    p = _KRANZ / "panel.csv"
    if not p.exists():  # pragma: no cover - gold is committed
        pytest.skip(f"missing {p}")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def brexit():
    if not _BREXIT.exists():  # pragma: no cover - gold is committed
        pytest.skip(f"missing {_BREXIT}")
    return pd.read_csv(_BREXIT)


def _att(df, **extra):
    from mlsynth import SDID

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(SDID({
            "df": df, "outcome": "y", "treat": "treat_exp", "unitid": "i",
            "time": "t", "vce": "noinference", "display_graphs": False,
            **extra}).fit().effects.att)


def _gap_at(df, t, **extra):
    """Per-period gap on the Brexit panel, as a percentage loss.

    The paper reports the effect in a named quarter rather than the post-period
    average, so the comparison has to read the gap at that period.

    ``intercept_adjust=True`` is not a detail. The paper's estimator subtracts
    the time-weighted pre-treatment gap,
    ``tau = y_1T - y_0T'w - lambda'(Y_1,pre - Y_0,pre w)``, which is mlsynth's
    intercept-adjusted counterfactual; the default un-adjusted series is a
    different quantity and comparing it to the published number would be
    comparing a different estimator. It is worth about 0.05 here -- small
    enough to read as a minor disagreement rather than the wrong estimand,
    which is exactly why it is pinned rather than assumed.
    """
    from mlsynth import SDID

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SDID({
            "df": df, "outcome": "y", "treat": "treat", "unitid": "country",
            "time": "t", "vce": "noinference", "display_graphs": False,
            "intercept_adjust": True, **extra}).fit()
    ts = res.time_series
    g = dict(zip(np.asarray(ts.time_periods).ravel().tolist(),
                 np.asarray(ts.estimated_gap, dtype=float).ravel().tolist()))
    return -100.0 * g[t]


class TestTheDefaultIsUnchanged:
    """Adding the option must not move any existing estimate."""

    def test_omitting_zeta_is_identical_to_passing_none(self, panel):
        assert _att(panel) == _att(panel, zeta=None)

    def test_the_frozen_gold_still_reproduces(self, panel):
        """Pinned against the #309 reference, not against the estimator itself."""
        txt = (_KRANZ / "gold.txt").read_text().splitlines()
        gold = {k.strip(): v.strip()
                for k, v in (l.split(":", 1) for l in txt if ":" in l)}
        assert _att(panel) == pytest.approx(
            float(gold["tau_unadjusted_converged"]), abs=1e-6)

    def test_passing_the_computed_value_reproduces_the_default(self, panel):
        """The override is the same quantity the estimator computes, not a
        differently-scaled one. Recomputing it by hand and passing it back must
        return the default answer."""
        from mlsynth.utils.sdid_helpers.weights import compute_regularization

        Y = panel.pivot(index="t", columns="i", values="y")
        treated = sorted(panel.loc[panel.treat_exp == 1, "i"].unique())
        donors = [c for c in Y.columns if c not in treated]
        T0 = 15
        z = compute_regularization(Y.loc[Y.index[:T0], donors].to_numpy(),
                                   Y.shape[0] - T0,
                                   num_treated_units=len(treated))
        assert _att(panel, zeta=float(z)) == pytest.approx(_att(panel), abs=1e-9)


class TestTheOverrideTakesEffect:
    """A different penalty gives a different answer, in the expected direction."""

    def test_zero_differs_from_the_default(self, panel):
        assert abs(_att(panel, zeta=0.0) - _att(panel)) > 1e-6

    def test_a_large_penalty_drives_the_weights_toward_uniform(self, panel):
        """The ridge's own definition, asserted rather than assumed.

        With a penalty far larger than the fit, the objective is dominated by
        ``||w||^2``, whose simplex minimiser is the uniform point.
        """
        from mlsynth import SDID

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SDID({
                "df": panel, "outcome": "y", "treat": "treat_exp", "unitid": "i",
                "time": "t", "vce": "noinference", "display_graphs": False,
                "zeta": 1e6}).fit()
        w = np.array(list(res.donor_weights.values()), dtype=float)
        np.testing.assert_allclose(w, np.full_like(w, 1.0 / w.size),
                                   rtol=0, atol=1e-4)

    def test_the_estimate_is_monotone_in_the_penalty(self, panel):
        """Not a coincidence of one value: the estimate moves steadily from the
        unpenalised fit toward the uniform-weight one."""
        vals = [_att(panel, zeta=z) for z in (0.0, 50.0, 200.0, 1e6)]
        diffs = [abs(v - vals[-1]) for v in vals]
        assert diffs == sorted(diffs, reverse=True)

    def test_weights_stay_on_the_simplex(self, panel):
        from mlsynth import SDID

        for z in (0.0, 100.0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SDID({
                    "df": panel, "outcome": "y", "treat": "treat_exp",
                    "unitid": "i", "time": "t", "vce": "noinference",
                    "display_graphs": False, "zeta": z}).fit()
            w = np.array(list(res.donor_weights.values()), dtype=float)
            assert w.min() >= -1e-9
            assert w.sum() == pytest.approx(1.0, abs=1e-6)


class TestItReproducesThePublishedBrexitNumbers:
    """The reason the option exists.

    de Brabander et al. (2025) Table 1, no covariates, treatment 2016Q3, SDID
    case (ii): 2.79 percent at 2018Q4 and 3.92 at 2019Q4. Their case (ii) fits
    on a panel running to the evaluation period, so the panel is truncated to
    match. With mlsynth's own ridge the estimates land at 2.53 and 3.60 --
    0.26 and 0.32 below the published values -- which is what motivated this.

    Not gated to the paper's two printed decimals. ``synthdid_estimate`` stops
    at ``min.decrease`` where mlsynth solves the weight programs exactly, a
    difference already documented in ``sdid_helpers/weights.py``, and worth
    about 0.015 on this panel. The band below is a little wider than that and
    an order of magnitude tighter than what the default ridge produces, so it
    separates the two specifications without claiming solver-level agreement.

    The full table -- all seven estimators, both dates, and the paper's
    in-sample placebo -- is pinned as a durable benchmark case rather than
    here; see ``benchmarks/cases/brabander_brexit_table1.py``. What these tests
    guard is narrower: that ``zeta`` is what makes that specification
    reachable at all.
    """

    @pytest.mark.parametrize("t,published", [(236, 2.79), (240, 3.92)])
    def test_zero_zeta_moves_the_estimate_toward_the_published_value(
            self, brexit, t, published):
        d = brexit[brexit.t <= t]
        default = _gap_at(d, t)
        zeroed = _gap_at(d, t, zeta=0.0)
        assert abs(zeroed - published) < abs(default - published)

    @pytest.mark.parametrize("t,published", [(236, 2.79), (240, 3.92)])
    def test_it_lands_within_a_fiftieth_of_a_point(self, brexit, t, published):
        d = brexit[brexit.t <= t]
        assert _gap_at(d, t, zeta=0.0) == pytest.approx(published, abs=0.02)

    @pytest.mark.parametrize("t,published", [(236, 2.79), (240, 3.92)])
    def test_the_default_is_an_order_of_magnitude_further_away(
            self, brexit, t, published):
        """The contrast that makes the test above meaningful: without the
        override the specification simply is not expressible."""
        d = brexit[brexit.t <= t]
        assert abs(_gap_at(d, t) - published) > 0.2


class TestStaggeredAdoption:
    """Every cohort takes the override, not only the one that sets the pre/post
    counts on the inputs object.

    Worth its own test because the default is deliberately not one number: it
    is recomputed per cohort from that cohort's donors and horizon. Overriding
    it means asking for a single penalty across cohorts, and a loop that stops
    at the first payload would silently give the later cohorts the data-driven
    value instead.
    """

    @staticmethod
    def _staggered():
        from mlsynth.tests.test_sdid import _make_staggered_panel

        return _make_staggered_panel(seed=3)

    def test_the_override_reaches_every_cohort(self):
        from mlsynth.utils.sdid_helpers.setup import prepare_sdid_inputs

        inputs = prepare_sdid_inputs(
            df=self._staggered(), outcome="y", treat="treated",
            unitid="state", time="year", zeta=7.0)
        assert len(inputs.cohorts_dict) > 1
        assert all(p["zeta_override"] == 7.0
                   for p in inputs.cohorts_dict.values())

    def test_no_key_is_written_when_none_is_given(self):
        """The payload stays exactly as it was, so nothing downstream has to
        distinguish 'absent' from 'None'."""
        from mlsynth.utils.sdid_helpers.setup import prepare_sdid_inputs

        inputs = prepare_sdid_inputs(
            df=self._staggered(), outcome="y", treat="treated",
            unitid="state", time="year")
        assert all("zeta_override" not in p
                   for p in inputs.cohorts_dict.values())

    def test_a_staggered_fit_moves_with_the_penalty(self):
        from mlsynth import SDID

        def att(**extra):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(SDID({
                    "df": self._staggered(), "outcome": "y", "treat": "treated",
                    "unitid": "state", "time": "year", "vce": "noinference",
                    "display_graphs": False, **extra}).fit().effects.att)

        assert abs(att(zeta=0.0) - att()) > 1e-9
        assert np.isfinite(att(zeta=0.0))


class TestFailuresAreReported:
    """Invalid values are rejected when the estimator is constructed.

    Construction, specifically, and the tests below never call ``fit``. A
    negative penalty is also caught downstream by ``unit_weights``, so an
    assertion on the fitted call would pass with no config validation at all;
    what is being pinned here is that the caller learns the value is wrong
    before any fitting happens, and while the message can still name the option
    they passed rather than an internal argument.
    """

    def _construct(self, df, **extra):
        from mlsynth import SDID

        return SDID({
            "df": df, "outcome": "y", "treat": "treat_exp", "unitid": "i",
            "time": "t", "vce": "noinference", "display_graphs": False,
            **extra})

    def test_a_valid_zeta_constructs(self, panel):
        """The control for the two below: construction alone is what is being
        tested, and it succeeds for a good value."""
        assert self._construct(panel, zeta=0.0).zeta == 0.0

    def test_a_negative_zeta(self, panel):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="non-negative"):
            self._construct(panel, zeta=-1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_non_finite_zeta(self, panel, bad):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="finite"):
            self._construct(panel, zeta=bad)

    def test_the_message_names_the_option(self, panel):
        """A caller who passed ``zeta=-1`` should not be told about
        ``regularization_parameter_zeta``, which is not something they can set."""
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError) as exc:
            self._construct(panel, zeta=-1.0)
        assert "zeta" in str(exc.value)
        assert "regularization_parameter_zeta" not in str(exc.value)
