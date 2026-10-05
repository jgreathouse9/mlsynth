"""The harness that measures mlsynth's TBR against Kerman section 5.2.

The study beside it established the paper's coverage result against a port
written from the paper. This harness runs the same design through the shipped
estimator, which is a different body of code, so the thing under test here is
the harness and not the estimator: the assignment schemes, the injected truth,
and the interval criterion the paper uses to judge coverage.

What the tests hold it to is behaviour. The one number they pin is the identity
the injected truth has to satisfy -- a constant true response and a constant
cost whose ratio is the true iROAS -- because every coverage figure the case
reports is a statement about that number.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmarks.studies.tbr_geo import mlsynth_coverage as mc
from mlsynth import TBR
from mlsynth.config_models import TBRConfig


# --------------------------------------------------------------------------- smoke
def test_one_replication_comes_back_with_the_three_quantities():
    hit90, hit50, iroas = mc.one_replication(np.random.default_rng(0), 20, 0.5, 0.25)
    assert hit90 in (True, False)
    assert hit50 in (True, False)
    assert np.isfinite(iroas)


def test_a_small_grid_runs_and_reports_every_cell():
    rows = mc.run_grid(reps=4, rhos=(0.0, 0.5), cs=(0.25,), pres=(20,))
    assert len(rows) == 2
    for r in rows:
        assert 0.0 <= r["cov90"] <= 1.0 and 0.0 <= r["cov50"] <= 1.0
        assert r["reps"] == 4 and np.isfinite(r["median_iroas"])


# ------------------------------------------------------------------ unit invariants
@pytest.mark.parametrize("scheme", ["permutation", "stratified"])
def test_an_assignment_takes_half_the_geos_exactly_once(scheme):
    rng = np.random.default_rng(3)
    y = mc.dgp.panel(20, 20, 4, 0.5, 0.25, rng)
    treated = mc.assign(y[:20], 20, rng, scheme)
    assert len(treated) == 10
    assert len(set(treated.tolist())) == 10
    assert set(treated.tolist()) <= set(range(20))


def test_stratified_assignment_takes_one_geo_from_each_volume_pair():
    """The reference's GeoStrata default: sort by volume, pair, one per group."""
    rng = np.random.default_rng(5)
    y = mc.dgp.panel(20, 20, 4, 0.5, 0.25, rng)
    order = np.argsort(-y[:20].sum(0))
    pairs = order.reshape(-1, 2)
    treated = set(mc.assign(y[:20], 20, rng, "stratified").tolist())
    for pair in pairs:
        assert len(treated & set(pair.tolist())) == 1


def test_stratified_assignment_balances_volume_better_than_a_free_draw():
    """Which is the whole reason the reference stratifies."""
    rng = np.random.default_rng(7)
    gaps = {"permutation": [], "stratified": []}
    for _ in range(80):
        y = mc.dgp.panel(20, 20, 4, 0.5, 0.25, rng)
        vol = y[:20].sum(0)
        for scheme in gaps:
            t = mc.assign(y[:20], 20, rng, scheme)
            share = vol[t].sum() / vol.sum()
            gaps[scheme].append(abs(share - 0.5))
    assert np.mean(gaps["stratified"]) < np.mean(gaps["permutation"])


def test_the_injected_truth_satisfies_the_iroas_identity():
    """True response over true cost is the true iROAS, by construction."""
    for n_test in (2, 4, 8):
        response, cost = mc.injected_truth(n_test)
        assert response / cost == pytest.approx(mc.TRUE_IROAS, rel=1e-12)
        assert response > 0.0 and cost > 0.0


def test_the_injected_truth_does_not_depend_on_the_realised_panel():
    """A truth derived from realised volume correlates with test-period noise.

    The study beside this one measured that mistake at twice the bias floor, so
    the truth is a constant and takes no panel argument at all.
    """
    import inspect
    params = inspect.signature(mc.injected_truth).parameters
    assert list(params) == ["n_test"]


def test_the_same_seed_gives_the_same_grid():
    a = mc.run_grid(reps=3, rhos=(0.5,), cs=(0.25,), pres=(20,), seed=2)
    b = mc.run_grid(reps=3, rhos=(0.5,), cs=(0.25,), pres=(20,), seed=2)
    assert a[0]["cov90"] == b[0]["cov90"]
    assert a[0]["median_iroas"] == b[0]["median_iroas"]


def test_the_beta_criterion_brackets_the_empirical_rate_and_tightens_with_n():
    """Kerman (2011)'s neutral prior: Beta(1/3 + y, 1/3 + n - y)."""
    widths = []
    for n in (100, 1000, 10000):
        lo, hi = mc.beta_interval(int(0.9 * n), n)
        assert lo < 0.9 < hi
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]


def test_the_beta_criterion_rejects_a_rate_that_is_plainly_off():
    lo, hi = mc.beta_interval(700, 1000)        # 70% where 90% is claimed
    assert not (lo <= 0.90 <= hi)


def test_summarise_counts_the_cells_whose_coverage_misses_nominal():
    def cell(hits90, hits50):
        return {"cov90": hits90 / 1000, "cov50": hits50 / 1000, "reps": 1000,
                "hits90": hits90, "hits50": hits50, "median_iroas": 2.0,
                "mean_iroas": 2.0, "mse_iroas": 0.25}

    rows = [cell(900, 500), cell(700, 500)]
    s = mc.summarise(rows)
    assert s["cov90_cells_off_nominal"] == 1
    assert s["cov50_cells_off_nominal"] == 0
    assert s["n_cells"] == 2


# ------------------------------------------------------------------------ edge cases
def test_the_shortest_admissible_pretest_still_runs():
    hit90, _, iroas = mc.one_replication(np.random.default_rng(1), 10, 0.0, 0.5)
    assert hit90 in (True, False) and np.isfinite(iroas)


def test_a_single_pair_of_geos_is_a_matched_market_test():
    hit90, _, iroas = mc.one_replication(np.random.default_rng(1), 20, 0.5, 0.25,
                                         n_geos=2)
    assert hit90 in (True, False) and np.isfinite(iroas)


def test_zero_correlation_and_the_highest_noise_still_produce_a_fit():
    rows = mc.run_grid(reps=4, rhos=(0.0,), cs=(0.5,), pres=(10,))
    assert np.isfinite(rows[0]["median_iroas"])


# --------------------------------------------------------------------------- failure
def test_an_unknown_assignment_scheme_is_refused():
    rng = np.random.default_rng(0)
    y = mc.dgp.panel(20, 20, 4, 0.5, 0.25, rng)
    with pytest.raises(ValueError, match="scheme"):
        mc.assign(y[:20], 20, rng, "matched-pairs-by-vibes")


def test_an_odd_geo_count_is_refused_for_the_stratified_scheme():
    rng = np.random.default_rng(0)
    y = mc.dgp.panel(21, 20, 4, 0.5, 0.25, rng)
    with pytest.raises(ValueError, match="even"):
        mc.assign(y[:20], 21, rng, "stratified")


@pytest.mark.parametrize("reps", [0, -1])
def test_a_non_positive_replication_count_is_refused(reps):
    with pytest.raises(ValueError, match="replication"):
        mc.run_grid(reps=reps, rhos=(0.5,), cs=(0.25,), pres=(20,))


def test_a_beta_interval_on_more_hits_than_trials_is_refused():
    with pytest.raises(ValueError, match="trials"):
        mc.beta_interval(11, 10)


# --------------------------------------------------------------- the scaling claim
def test_a_fixed_cost_iroas_interval_is_the_cumulative_one_rescaled():
    """Why coverage is measured on the cumulative effect and not through iROAS.

    Section 5.2 takes the incremental cost as a known constant, which is
    section 3.4's fixed-cost branch. There the iROAS posterior is the
    cumulative response posterior divided by that constant, so coverage of one
    at its true value is coverage of the other at the true iROAS. Measuring the
    cumulative effect is therefore the same statement, at one fit per
    replication instead of two.

    This pins the rescaling once. The cross-validation case already checks the
    iROAS numbers themselves against the reference.
    """
    rng = np.random.default_rng(4)
    n_pre, n_test, n_geos = 30, mc.N_TEST, mc.N_GEOS
    y = mc.dgp.panel(n_geos, n_pre, n_test, 0.5, 0.25, rng)
    treated = mc.assign(y[:n_pre], n_geos, rng, "permutation")
    response, cost = mc.injected_truth(n_test)
    y = y.copy()
    y[n_pre:, treated] += response / n_test / len(treated)

    df = mc._frame(y, treated, n_pre, n_geos)
    treated_names = {f"g{j:02d}" for j in treated}
    post = df["post"] == 1
    df["spend"] = 0.0
    df.loc[post & df["geo"].isin(treated_names), "spend"] = (
        cost / n_test / len(treated))

    report = TBR(TBRConfig(
        df=df, outcome="sales", unitid="geo", time="t",
        treatment_col="is_treat", control_col="is_ctrl", post_col="post",
        cost_col="spend", level=0.9)).fit().report

    assert report.iroas is not None
    assert report.iroas.fixed_cost is True
    incr = report.iroas.total_incremental_cost
    assert incr == pytest.approx(cost, rel=1e-9)

    cum = report.cumulative
    assert report.iroas.estimate == pytest.approx(cum.estimate[-1] / incr, rel=1e-9)
    assert report.iroas.lower == pytest.approx(cum.lower[-1] / incr, rel=1e-9)
    assert report.iroas.upper == pytest.approx(cum.upper[-1] / incr, rel=1e-9)


# ------------------------------------------- a cell belongs to itself (RCA, #729)
# A single generator consumed across the grid made a cell's draws depend on which
# cells preceded it. The same (rho, c, n_pre=20) cell returned 192 hits of 200
# when requested alone and 179 when requested inside the full 36-cell grid, and
# that 13-hit swing was large enough to read as a difference between the two
# assignment schemes when the two arms had been run over different cell sets.

def test_a_cell_answers_the_same_alone_and_inside_a_larger_grid():
    alone = mc.run_grid(reps=6, rhos=(0.5,), cs=(0.25,), pres=(20,), seed=29)
    inside = mc.run_grid(reps=6, rhos=(0.0, 0.5), cs=(0.15, 0.25),
                         pres=(10, 20), seed=29)
    match = [r for r in inside if r["rho"] == 0.5 and r["c"] == 0.25
             and r["n_pre"] == 20]
    assert len(match) == 1
    for key in ("hits90", "hits50", "median_iroas", "mean_iroas", "mse_iroas"):
        assert alone[0][key] == match[0][key], key


@settings(max_examples=12, deadline=None)
@given(rho=st.sampled_from(mc.RHOS), c=st.sampled_from(mc.CS),
       n_pre=st.sampled_from(mc.PRES), seed=st.integers(0, 2 ** 16),
       scheme=st.sampled_from(["permutation", "stratified"]))
def test_any_cell_is_independent_of_the_request_it_arrived_in(
        rho, c, n_pre, seed, scheme):
    """Over the whole grid domain, not at one fixture."""
    kw = dict(reps=3, seed=seed, scheme=scheme)
    alone = mc.run_grid(rhos=(rho,), cs=(c,), pres=(n_pre,), **kw)[0]
    padded = mc.run_grid(rhos=(0.0, rho), cs=(0.5, c), pres=(40, n_pre), **kw)
    match = [r for r in padded
             if (r["rho"], r["c"], r["n_pre"]) == (rho, c, n_pre)]
    assert alone["hits90"] == match[0]["hits90"]
    assert alone["hits50"] == match[0]["hits50"]


def test_different_cells_draw_from_different_streams():
    first = mc.cell_rng(11, 0.5, 0.25, 20, "permutation").normal(size=5)
    for other in (mc.cell_rng(11, 0.8, 0.25, 20, "permutation"),
                  mc.cell_rng(11, 0.5, 0.50, 20, "permutation"),
                  mc.cell_rng(11, 0.5, 0.25, 40, "permutation"),
                  mc.cell_rng(11, 0.5, 0.25, 20, "stratified"),
                  mc.cell_rng(12, 0.5, 0.25, 20, "permutation")):
        assert not np.allclose(first, other.normal(size=5))


def test_a_cell_stream_is_the_same_one_every_time_it_is_asked_for():
    a = mc.cell_rng(11, 0.5, 0.25, 20, "permutation").normal(size=5)
    b = mc.cell_rng(11, 0.5, 0.25, 20, "permutation").normal(size=5)
    assert np.allclose(a, b)


def test_every_cell_derives_its_stream_from_its_own_identity(monkeypatch):
    """Not just order-independence: the stream has to come from ``cell_rng``.

    Re-seeding identically inside the loop is also order-independent, so the
    equality above passes while every cell draws the same numbers. That costs
    independence between cells, which no coverage figure shows and which
    understates the standard error of the pooled rate. Both defects bypass
    ``cell_rng``, so asserting on the derivation catches both.
    """
    seen = []
    real = mc.cell_rng

    def spy(seed, rho, c, n_pre, scheme):
        seen.append((seed, rho, c, n_pre, scheme))
        return real(seed, rho, c, n_pre, scheme)

    monkeypatch.setattr(mc, "cell_rng", spy)
    mc.run_grid(reps=2, rhos=(0.0, 0.5), cs=(0.25,), pres=(20, 40), seed=29,
                scheme="stratified")

    assert seen == [(29, 0.0, 0.25, 20, "stratified"),
                    (29, 0.0, 0.25, 40, "stratified"),
                    (29, 0.5, 0.25, 20, "stratified"),
                    (29, 0.5, 0.25, 40, "stratified")]
    assert len(set(seen)) == len(seen)
