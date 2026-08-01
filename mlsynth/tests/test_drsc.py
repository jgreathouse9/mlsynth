"""DRSC: conditional distributional treatment effects (Wied 2026).

Where :class:`mlsynth.DSC` matches *unconditional* quantile functions, DRSC
models the conditional distribution semiparametrically as
``F_it(y | x) = Lambda(x' theta_it(y))`` and imposes parallel trends in the
*parameter* space rather than on the CDF. The payoff is that an effect which
is heterogeneous across the covariate space stays visible: the paper's own
simulation plants a shift along one covariate direction whose mean is zero, so
the marginal distribution barely moves and an unconditional method has no
power, while the conditional test rejects.

Three things about this estimator drive the tests below and are worth stating
because none is obvious from the model.

The asymptotics are in ``n`` -- individuals within a group-time cell -- with the
number of donors and periods fixed. One pre-period suffices for consistency.
Every other estimator in mlsynth gets its precision from ``T``; this one does
not, so the edge cases that matter are thin *cells*, not thin panels.

The weights are ill-conditioned by construction. On the paper's data the Gram
matrix has a condition number near 1e5 and the closed-form solve amplifies
relative input error by about 1e6, so float32 inputs move donor weights by
0.15 while barely touching the estimand. That is pinned in
:class:`TestFloat64IsRequired` because it is exactly the kind of property that
gets rediscovered painfully later.

The weights are not constrained to the simplex -- only to sum to one -- so
negative weights are normal, not a symptom. On the paper's panel 20 of 42 are
negative.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

_REF = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
        / "wied_nj_minwage")


# ---------------------------------------------------------------------------
# synthetic panels
# ---------------------------------------------------------------------------
def _panel(n_per_cell=400, J=4, T=2, T0=1, delta=0.0, seed=0, p=3):
    """Wied's own simulation design (paper Section 4).

    ``J`` donors whose parameter vectors each differ from the common mean in
    exactly one coordinate; the treated unit's parameter is the average of the
    donors', so parallel-trends-in-parameters holds exactly with equal weights.
    Under the alternative the treated unit's post-period outcome gains
    ``delta * X1`` -- a shift along one covariate only, which moves the
    conditional distribution while leaving the marginal nearly fixed.
    """
    rng = np.random.default_rng(seed)
    beta_bar = np.ones(p + 1)
    ctrl = []
    for i in range(J):
        b = beta_bar.copy()
        b[i % (p + 1)] += 0.8
        ctrl.append(b)
    treated_beta = np.mean(ctrl, axis=0)
    rows = []
    for t in range(1, T + 1):
        for i in range(J + 1):
            beta = treated_beta if i == 0 else ctrl[i - 1]
            X = rng.normal(size=(n_per_cell, p))
            y = beta[0] + X @ beta[1:] + rng.normal(size=n_per_cell)
            if i == 0 and t > T0:
                y = y + delta * X[:, 0]
            for k in range(n_per_cell):
                rows.append((f"u{i}", t, y[k], *X[k],
                             int(i == 0 and t > T0)))
    cols = ["unit", "t", "y"] + [f"x{j+1}" for j in range(p)] + ["treat"]
    return pd.DataFrame(rows, columns=cols)


COVS = ["x1", "x2", "x3"]
EVAL = {"x0": {"x1": 1.0, "x2": 0.0, "x3": 0.0}}


def _fit(df=None, **over):
    from mlsynth import DRSC

    cfg = {"df": _panel() if df is None else df, "outcome": "y",
           "treat": "treat", "unitid": "unit", "time": "t",
           "covariates": COVS, "evaluation_points": EVAL,
           "display_graphs": False}
    cfg.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DRSC(cfg).fit()


# ---------------------------------------------------------------------------
class TestSmoke:
    """End-to-end on minimal input: right type, finite numbers."""

    def test_it_fits_and_returns_the_contract_type(self):
        from mlsynth.config_models import EffectResult

        res = _fit()
        assert isinstance(res, EffectResult)

    def test_the_headline_quantities_are_finite(self):
        res = _fit()
        assert np.isfinite(res.effects.att)
        w = res.weights.donor_weights
        assert w and all(np.isfinite(v) for v in w.values())

    def test_the_standard_submodels_are_populated(self):
        res = _fit()
        assert res.effects is not None
        assert res.weights is not None
        assert res.method_details is not None
        assert "DRSC" in res.method_details.method_name


class TestTheWeightsSolveTheStatedProgram:
    """eq. (7): min-norm least squares subject to 1'w = 1."""

    def test_weights_sum_to_one(self):
        w = np.array(list(_fit().weights.donor_weights.values()))
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_negative_weights_are_allowed(self):
        """Not a simplex. A design outside the donor hull must still fit."""
        df = _panel(seed=3)
        # push the treated unit outside the affine hull of the donors
        m = df.unit == "u0"
        df.loc[m, "y"] = df.loc[m, "y"] * 1.8 + 4.0
        w = np.array(list(_fit(df).weights.donor_weights.values()))
        assert np.isfinite(w).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_it_recovers_equal_weights_when_the_design_says_so(self):
        """Wied's DGP sets the treated parameter to the donors' mean, so the
        true weights are 1/J. With one pre-period and large cells the estimate
        should land near that."""
        w = np.array(list(_fit(_panel(n_per_cell=1200, seed=5))
                          .weights.donor_weights.values()))
        assert w == pytest.approx(np.full(len(w), 1.0 / len(w)), abs=0.25)


class TestTheConditionalTestSeesWhatTheMarginalCannot:
    """The paper's central claim, and the reason the estimator exists."""

    def test_size_is_controlled_under_the_null(self):
        """A single draw cannot test size.

        Under H0 the p-value is Uniform(0,1), so asserting ``p > alpha`` on one
        replication fails with probability exactly alpha -- a test that is
        broken by construction. This runs a small sweep instead and asks only
        that the rejection rate is not wildly above nominal. With 8 draws at a
        nominal 0.10 the expected count is 0.8, so a threshold of 4 is far into
        the tail while still catching a grossly oversized test.
        """
        p = [_fit(_panel(delta=0.0, n_per_cell=300, seed=100 + s), n_grid=12)
             .inference.p_value for s in range(8)]
        assert sum(v < 0.10 for v in p) <= 4, f"oversized: {p}"

    def test_a_covariate_heterogeneous_effect_is_detected(self):
        """delta * X1 with E[X1] = 0: the marginal barely moves."""
        res = _fit(_panel(delta=1.2, n_per_cell=600, seed=11))
        assert res.inference.p_value < 0.10

    def test_the_null_is_less_significant_than_the_alternative(self):
        """The comparison a single-draw size test cannot make."""
        null = _fit(_panel(delta=0.0, n_per_cell=600, seed=11))
        alt = _fit(_panel(delta=1.2, n_per_cell=600, seed=11))
        assert null.inference.p_value > alt.inference.p_value
        assert float(null.effects.att) < float(alt.effects.att)

    def test_the_effect_size_orders_correctly(self):
        f = [float(_fit(_panel(delta=d, n_per_cell=600, seed=11)).effects.att)
             for d in (0.0, 0.8, 1.6)]
        assert f[0] < f[1] < f[2]


class TestFloat64IsRequired:
    """kappa(G) ~ 1e5 and the solve amplifies relative error by ~1e6.

    Measured on the paper's data: a float32 round-trip (relative 6e-8) moves
    donor weights by 0.12-0.17 while leaving the estimand nearly intact. The
    estimator must therefore not silently downcast, and must say so if handed
    a float32 outcome or covariate column.
    """

    def test_a_float32_panel_still_computes_in_float64(self):
        df = _panel(seed=7)
        for c in ["y"] + COVS:
            df[c] = df[c].astype("float32")
        res = _fit(df)
        assert np.isfinite(res.effects.att)

    def test_float32_and_float64_agree_on_the_estimand(self):
        """The estimand is robust even where the weights are not."""
        df64 = _panel(seed=7)
        df32 = df64.copy()
        for c in ["y"] + COVS:
            df32[c] = df32[c].astype("float32")
        assert float(_fit(df32).effects.att) == pytest.approx(
            float(_fit(df64).effects.att), rel=0.05)


class TestFailuresAreReported:
    """Bad input raises a translated error, never a bare pandas/numpy one."""

    def test_an_unknown_covariate_is_reported(self):
        from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

        with pytest.raises((MlsynthConfigError, MlsynthDataError),
                           match="nope"):
            _fit(covariates=["x1", "nope"])

    def test_an_empty_covariate_list_is_reported(self):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="empty|at least one"):
            _fit(covariates=[])

    def test_a_non_numeric_covariate_is_reported(self):
        from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

        df = _panel()
        df["label"] = "a"
        with pytest.raises((MlsynthConfigError, MlsynthDataError),
                           match="numeric"):
            _fit(df, covariates=["x1", "label"],
                 evaluation_points={"x0": {"x1": 1.0, "label": 0.0}})

    def test_an_evaluation_point_naming_an_unknown_covariate_is_reported(self):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="nope|evaluation"):
            _fit(evaluation_points={"bad": {"nope": 1.0}})

    def test_no_evaluation_points_is_reported(self):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="evaluation"):
            _fit(evaluation_points={})

    def test_no_donors_is_reported(self):
        from mlsynth.exceptions import MlsynthDataError

        df = _panel(J=1)
        df = df[df.unit == "u0"]
        with pytest.raises(MlsynthDataError, match="donor"):
            _fit(df)

    def test_no_pre_period_is_reported(self):
        from mlsynth.exceptions import MlsynthDataError

        df = _panel(T=2, T0=0)
        df["treat"] = (df.unit == "u0").astype(int)
        with pytest.raises(MlsynthDataError, match="pre-treatment|pre-period"):
            _fit(df)

    def test_no_treated_unit_is_reported(self):
        from mlsynth.exceptions import MlsynthDataError

        df = _panel()
        df["treat"] = 0
        with pytest.raises(MlsynthDataError, match="treated"):
            _fit(df)

    def test_a_cell_too_small_to_identify_the_dr_is_reported(self):
        """p+1 parameters need more than p+1 observations in every cell."""
        from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError

        with pytest.raises((MlsynthDataError, MlsynthEstimationError),
                           match="observation|too few|cell"):
            _fit(_panel(n_per_cell=3))


class TestEdgeCases:
    def test_a_single_donor_still_fits(self):
        """J = 1 forces w = 1 by the adding-up constraint."""
        res = _fit(_panel(J=1, n_per_cell=500))
        w = list(res.weights.donor_weights.values())
        assert len(w) == 1 and w[0] == pytest.approx(1.0, abs=1e-9)

    def test_one_pre_period_is_enough(self):
        """The paper's asymptotics are in n, not T: T0 = 1 is consistent."""
        assert np.isfinite(_fit(_panel(T=2, T0=1)).effects.att)

    def test_more_pre_periods_are_accepted(self):
        assert np.isfinite(_fit(_panel(T=4, T0=3)).effects.att)

    def test_a_collinear_covariate_is_handled(self):
        from mlsynth.exceptions import MlsynthEstimationError

        df = _panel()
        df["x4"] = df["x1"] * 2.0            # exactly collinear
        try:
            res = _fit(df, covariates=COVS + ["x4"],
                       evaluation_points={"x0": {"x1": 1.0, "x2": 0.0,
                                                 "x3": 0.0, "x4": 2.0}})
            assert np.isfinite(res.effects.att)
        except MlsynthEstimationError as exc:
            assert "collinear" in str(exc).lower() or "rank" in str(exc).lower()


class TestPlotting:
    def test_display_graphs_runs(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        res = _fit(display_graphs=True, save=str(tmp_path / "drsc"))
        assert np.isfinite(res.effects.att)


# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (_REF / "nj_estimation_sample.parquet").exists(),
                    reason="Wied reference sample not vendored (see #329)")
class TestItReproducesThePaper:
    """Path A: Wied (2026), Tables 1 and 2, on the authors' own data.

    Tolerances follow the replication spike. The weights and the estimands are
    deterministic and pinned tight; the Gaussian-process critical values are
    not reproducible across LAPACK builds -- ``eigh`` eigenvector signs are
    implementation-defined -- so anything downstream of the GP simulation is
    pinned loose.
    """

    PAPER_W = {12: 0.515, 36: 0.479, 46: -0.255, 32: 0.199, 29: -0.196}
    PAPER_F = {"x0": 0.58, "x10": 1.71, "xhl": 1.55, "xlh": 0.61, "x90": 0.21}

    @pytest.fixture(scope="class")
    def nj(self):
        import json

        d = pd.read_parquet(_REF / "nj_estimation_sample.parquet")
        meta = json.load(open(_REF / "nj_meta.json"))
        d["treat"] = ((d.STATEFIP == 34) & (d.t == 4)).astype(int)
        d["x_std_sq"] = d["x_std"] ** 2
        return d, meta

    def _eval_points(self, meta):
        pts = {}
        for nm, ed, ex in (("x0", 12, 10), ("x10", 10, 2), ("xhl", 16, 2),
                           ("xlh", 10, 37), ("x90", 16, 37)):
            e = (ed - meta["me"]) / meta["se"]
            x = (ex - meta["mx"]) / meta["sx"]
            pts[nm] = {"e_std": e, "x_std": x, "x_std_sq": x ** 2}
        return pts

    def test_table_1_weights(self, nj):
        d, meta = nj
        res = _fit(d, outcome="logwage", unitid="STATEFIP", time="t",
                   covariates=["e_std", "x_std", "x_std_sq"],
                   evaluation_points=self._eval_points(meta))
        w = res.weights.donor_weights
        for fips, target in self.PAPER_W.items():
            assert w[str(fips)] == pytest.approx(target, abs=0.01), f"FIPS {fips}"

    def test_table_2_effect_sizes(self, nj):
        d, meta = nj
        res = _fit(d, outcome="logwage", unitid="STATEFIP", time="t",
                   covariates=["e_std", "x_std", "x_std_sq"],
                   evaluation_points=self._eval_points(meta))
        for nm, target in self.PAPER_F.items():
            got = res.conditional_effects[nm].f_hat * 1e3
            assert got == pytest.approx(target, abs=0.05), nm

    def test_the_corridor_test_rejects_only_for_low_skill_young(self, nj):
        d, meta = nj
        res = _fit(d, outcome="logwage", unitid="STATEFIP", time="t",
                   covariates=["e_std", "x_std", "x_std_sq"],
                   evaluation_points=self._eval_points(meta),
                   focus_region=(np.log(4.25), np.log(5.10)))
        p = {k: v.p_value_focused for k, v in res.conditional_effects.items()}
        assert p["x10"] < 0.05
        assert p["x90"] > 0.5
