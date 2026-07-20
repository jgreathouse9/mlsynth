"""Per-unit prediction intervals for PPSCM (CFPT/SCPI, the engine MSQRT uses).

PPSCM's pooled inference (bootstrap / jackknife over units) cannot give a single
treated unit its own interval. This adds a per-unit band built from each unit's own
post-period effect path and pre-period residuals via mlsynth's out-of-sample CFPT
interval engine -- the same machinery MSQRT uses -- so PPSCM's per-geo bands are
methodologically consistent with MSQRT's and correctly account for the in-sample
fit (a naive permutation over the QP-optimised pre-residuals over-rejects; this
does not).

The gate is finite-sample coverage: under a no-effect null the interval must cover
zero at ~the nominal rate (conservative is acceptable, under-coverage is not). The
rest pins monotonicity in ``alpha``, power against a real effect, the bracketing
invariant, ragged-horizon handling, and the end-to-end wiring onto ``PPSCMUnitFit``.

The same out-of-sample call also yields the per-period (pointwise, TSUS) band for
each treated unit -- the pointwise counterpart of the time-averaged (TAUS) band --
so PPSCM can report a band at every post-treatment horizon, not just the average.
Those bands are wider than the average (the average gets the ``sqrt(L)`` shrink
under ``time_dependence="iid"``; a single period does not) and land on
``PPSCMUnitFit.tau_lower`` / ``tau_upper``, aligned with ``tau``.
"""
import numpy as np
import pytest

from mlsynth.utils.ppscm_helpers.inference import per_unit_intervals


def _resid_cols(rng, n_pre, n_post, effect=0.0):
    """One unit's (M column, tau_rel row): iid pre + post residuals, +effect post."""
    M = rng.normal(0.0, 1.0, size=(n_pre, 1))
    tau = rng.normal(0.0, 1.0, size=(1, n_post)) + effect
    return M, tau


def test_perunit_null_coverage_is_at_least_nominal():
    # THE GATE: under no effect, the 90% per-unit interval covers 0 at >= ~90%
    # (CFPT is a sub-Gaussian prediction band -- conservative is fine, under-
    # coverage is the failure mode we must rule out).
    rng = np.random.default_rng(0)
    alpha, reps, covered = 0.10, 400, 0
    for _ in range(reps):
        M, tau = _resid_cols(rng, n_pre=24, n_post=8, effect=0.0)
        lo, hi, *_ = per_unit_intervals(M, tau, alpha=alpha)
        if lo[0] <= 0.0 <= hi[0]:
            covered += 1
    cov = covered / reps
    assert 0.85 <= cov <= 1.0, cov


def test_perunit_ci_widens_as_alpha_shrinks():
    rng = np.random.default_rng(1)
    M, tau = _resid_cols(rng, n_pre=30, n_post=10, effect=0.0)
    lo80, hi80, *_ = per_unit_intervals(M, tau, alpha=0.20)
    lo95, hi95, *_ = per_unit_intervals(M, tau, alpha=0.05)
    assert lo95[0] <= lo80[0] and hi95[0] >= hi80[0]        # 95% contains 80%


def test_perunit_detects_a_large_effect():
    rng = np.random.default_rng(2)
    M, tau = _resid_cols(rng, n_pre=40, n_post=12, effect=6.0)
    lo, hi, p, *_ = per_unit_intervals(M, tau, alpha=0.10)
    assert lo[0] > 0.0                                       # band clears zero
    assert p[0] < 0.10


def test_perunit_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    M, tau = _resid_cols(rng, n_pre=30, n_post=9, effect=1.5)
    lo, hi, *_ = per_unit_intervals(M, tau, alpha=0.10)
    att = float(np.nanmean(tau))
    assert lo[0] <= att <= hi[0]


def test_perunit_handles_ragged_horizons_and_multiple_units():
    # tau may carry NaN past a unit's horizon; the engine is called per unit, so
    # the two units' bands come back finite regardless of the ragged shape.
    rng = np.random.default_rng(4)
    M = np.column_stack([rng.normal(size=24), rng.normal(size=24)])
    tau = np.vstack([np.r_[rng.normal(size=6), [np.nan, np.nan]],
                     rng.normal(size=8)])
    lo, hi, p, *_ = per_unit_intervals(M, tau, alpha=0.10)
    assert lo.shape == hi.shape == p.shape == (2,)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


# ---- end-to-end: the fields land on PPSCMUnitFit --------------------------

def _block_panel(rng, n_donors=8, n_treated=3, T=30, T0=22, effect=0.0):
    """Long (unit, time, y, treat) panel: a block design, donors + treated."""
    import pandas as pd
    rows = []
    donors = {f"d{i}": rng.normal(50, 5, size=T).cumsum() / 5 + rng.normal(0, 1, T)
              for i in range(n_donors)}
    base = np.mean(list(donors.values()), axis=0)
    for i in range(n_donors):
        for t in range(T):
            rows.append(("d%d" % i, t, donors["d%d" % i][t], 0))
    for j in range(n_treated):
        y = base + rng.normal(0, 1, T)
        y[T0:] += effect                                    # post effect
        for t in range(T):
            rows.append(("t%d" % j, t, y[t], 1 if t >= T0 else 0))  # 0/1 dummy
    return pd.DataFrame(rows, columns=["unit", "time", "y", "treat"])


def test_ppscm_fit_populates_per_unit_interval_fields():
    from mlsynth import PPSCM
    from mlsynth.config_models import PPSCMConfig
    rng = np.random.default_rng(7)
    df = _block_panel(rng, effect=3.0)
    res = PPSCM(PPSCMConfig(df=df, outcome="y", treat="treat", unitid="unit",
                            time="time", run_inference=True, alpha=0.10,
                            display_graphs=False)).fit()
    assert res.per_unit, "expected per-unit fits"
    for key, uf in res.per_unit.items():
        assert uf.ci_lower is not None and uf.ci_upper is not None
        assert uf.p_value is not None and 0.0 <= uf.p_value <= 1.0
        assert uf.ci_lower <= uf.att <= uf.ci_upper           # band brackets point


def test_bootstrap_pooled_and_scpi_perunit_coexist():
    # the GEDI contract: pooled CI from the bootstrap, per-unit CIs from SCPI,
    # in one fit.
    from mlsynth import PPSCM
    from mlsynth.config_models import PPSCMConfig
    rng = np.random.default_rng(11)
    df = _block_panel(rng, effect=3.0)
    res = PPSCM(PPSCMConfig(df=df, outcome="y", treat="treat", unitid="unit",
                            time="time", run_inference=True,
                            inference_method="bootstrap", n_boot=200, seed=0,
                            alpha=0.10, display_graphs=False)).fit()
    # pooled inference is the bootstrap, and its band brackets the pooled ATT
    assert res.inference_detail.method == "bootstrap"
    lo, hi = res.inference_detail.ci
    assert lo <= res.inference_detail.att <= hi
    # per-unit CIs are present (SCPI) and bracket each unit's own ATT
    for uf in res.per_unit.values():
        assert uf.ci_lower is not None and uf.ci_lower <= uf.att <= uf.ci_upper


def test_ppscm_no_inference_leaves_per_unit_bands_none():
    from mlsynth import PPSCM
    from mlsynth.config_models import PPSCMConfig
    rng = np.random.default_rng(8)
    df = _block_panel(rng, effect=2.0)
    res = PPSCM(PPSCMConfig(df=df, outcome="y", treat="treat", unitid="unit",
                            time="time", run_inference=False,
                            display_graphs=False)).fit()
    for uf in res.per_unit.values():
        assert uf.ci_lower is None and uf.ci_upper is None and uf.p_value is None


# ---- per-period (pointwise, TSUS) bands: the same call already computes them ----

def test_perunit_period_bands_shape_and_bracket_each_period():
    # per-period bands come back aligned with tau (J, H) and bracket each period's
    # own effect.
    rng = np.random.default_rng(20)
    M, tau = _resid_cols(rng, n_pre=30, n_post=9, effect=1.5)
    lo, hi, _, tlo, thi = per_unit_intervals(M, tau, alpha=0.10)
    assert tlo.shape == thi.shape == tau.shape
    assert np.all(tlo[0] <= tau[0]) and np.all(tau[0] <= thi[0])


def test_perunit_period_bands_wider_than_time_average():
    # TSUS uses sigma; TAUS (iid) uses sigma / sqrt(L) -> per-period is wider.
    rng = np.random.default_rng(21)
    M, tau = _resid_cols(rng, n_pre=40, n_post=9, effect=0.0)
    lo, hi, _, tlo, thi = per_unit_intervals(M, tau, alpha=0.10)
    avg_hw = 0.5 * (hi[0] - lo[0])
    period_hw = 0.5 * (thi[0] - tlo[0])                 # constant across periods
    assert np.all(period_hw > avg_hw)


def test_perunit_period_bands_widen_as_alpha_shrinks():
    rng = np.random.default_rng(22)
    M, tau = _resid_cols(rng, n_pre=30, n_post=8, effect=0.0)
    *_, tlo80, thi80 = per_unit_intervals(M, tau, alpha=0.20)
    *_, tlo95, thi95 = per_unit_intervals(M, tau, alpha=0.05)
    assert np.all(tlo95[0] <= tlo80[0]) and np.all(thi95[0] >= thi80[0])


def test_perunit_period_null_coverage_is_at_least_nominal():
    # THE PER-PERIOD GATE: under no effect the pointwise band covers 0 at >= nominal
    # (sub-Gaussian, conservative is fine; under-coverage is the failure mode).
    rng = np.random.default_rng(23)
    alpha, reps, covered, total = 0.10, 200, 0, 0
    for _ in range(reps):
        M, tau = _resid_cols(rng, n_pre=24, n_post=6, effect=0.0)
        _, _, _, tlo, thi = per_unit_intervals(M, tau, alpha=alpha)
        covered += int(np.sum((tlo[0] <= 0.0) & (0.0 <= thi[0])))
        total += tau.shape[1]
    assert 0.85 <= covered / total <= 1.0, covered / total


def test_perunit_period_bands_respect_ragged_nan():
    # NaN past a unit's horizon stays NaN in the per-period band; finite elsewhere.
    rng = np.random.default_rng(24)
    M = np.column_stack([rng.normal(size=24), rng.normal(size=24)])
    tau = np.vstack([np.r_[rng.normal(size=6), [np.nan, np.nan]],
                     rng.normal(size=8)])
    _, _, _, tlo, thi = per_unit_intervals(M, tau, alpha=0.10)
    assert tlo.shape == thi.shape == tau.shape
    nanmask = ~np.isfinite(tau)
    assert np.all(np.isnan(tlo[nanmask])) and np.all(np.isnan(thi[nanmask]))
    assert np.all(np.isfinite(tlo[~nanmask])) and np.all(np.isfinite(thi[~nanmask]))


def test_perunit_accepts_1d_single_unit_input():
    # a single unit may be passed as 1-D arrays (M shape (d,), tau shape (H,));
    # they are promoted to one column / one row and every output is single-unit.
    rng = np.random.default_rng(25)
    M = rng.normal(size=20)                             # 1-D pre residuals
    tau = rng.normal(size=7) + 1.0                      # 1-D post effects
    lo, hi, p, tlo, thi = per_unit_intervals(M, tau, alpha=0.10)
    assert lo.shape == hi.shape == p.shape == (1,)
    assert tlo.shape == thi.shape == (1, 7)
    assert np.all(tlo[0] <= tau) and np.all(tau <= thi[0])


def test_ppscm_fit_populates_per_unit_period_bands():
    # end-to-end: tau_lower / tau_upper land on PPSCMUnitFit, aligned with tau,
    # bracketing each period's effect and NaN past the horizon.
    from mlsynth import PPSCM
    from mlsynth.config_models import PPSCMConfig
    rng = np.random.default_rng(30)
    df = _block_panel(rng, effect=3.0)
    res = PPSCM(PPSCMConfig(df=df, outcome="y", treat="treat", unitid="unit",
                            time="time", run_inference=True, alpha=0.10,
                            display_graphs=False)).fit()
    assert res.per_unit
    for uf in res.per_unit.values():
        assert uf.tau_lower is not None and uf.tau_upper is not None
        assert uf.tau_lower.shape == uf.tau_upper.shape == uf.tau.shape
        fin = np.isfinite(uf.tau)
        assert np.all(uf.tau_lower[fin] <= uf.tau[fin])
        assert np.all(uf.tau[fin] <= uf.tau_upper[fin])
        assert np.all(np.isnan(uf.tau_lower[~fin])) and np.all(np.isnan(uf.tau_upper[~fin]))


def test_ppscm_no_inference_leaves_per_unit_period_bands_none():
    from mlsynth import PPSCM
    from mlsynth.config_models import PPSCMConfig
    rng = np.random.default_rng(31)
    df = _block_panel(rng, effect=2.0)
    res = PPSCM(PPSCMConfig(df=df, outcome="y", treat="treat", unitid="unit",
                            time="time", run_inference=False,
                            display_graphs=False)).fit()
    for uf in res.per_unit.values():
        assert uf.tau_lower is None and uf.tau_upper is None
