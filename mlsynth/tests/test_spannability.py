"""Can the selected donor cluster still span the treated unit?

CLUSTERSC picks donors by trajectory similarity, then fits weights against the
survivors. Where the weight objective is the convex hull (``simplex``), those
two steps can disagree: a cluster that is tight in functional-PC space may have
dropped the donors the treated unit actually needs to be reachable.

West Germany is the case that motivated this module. The FPCA cluster excludes
the USA, Switzerland and Greece, which carry 0.549 of the optimal convex weight
on the full donor pool, so the best achievable pre-period RMSE inside the
cluster is 522.7 against 60.8 on the pool -- a factor of 8.6, present in the
raw data before any denoising. Under the default ``nnls`` objective the same
cluster reports 78.2 by extrapolating to ``sum(w) = 1.148``, so nothing moved
and no pinned value caught it.

The ladder below is the root-cause analysis for that incident, one rung per
level of the dependency chain.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.clustersc_helpers.spannability import (
    SPANNABILITY_WARN_RATIO,
    _best_convex_fit,
    SpannabilityReport,
    assess_spannability,
    warn_if_poorly_spanned,
)


# --------------------------------------------------------------------------
# Fixtures: panels whose spannability we know by construction.
# --------------------------------------------------------------------------

def _panel(T0=24, seed=0):
    """Five donors; only d0 and d1 sit at the treated unit's level.

    The treated series is near 0.5*d0 + 0.5*d1 but not exactly it, so the pool
    optimum keeps a real residual. A fixture whose pool fit is exact has no
    interior: every ratio collapses to solver tolerance and the invariants
    below lose their power to fail.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, T0)
    d0 = 10.0 + 4.0 * t
    d1 = 6.0 + 1.0 * np.sin(6.0 * t)
    d2 = 2.0 + 0.5 * t                       # far below the treated level
    d3 = 1.0 + 0.2 * rng.normal(size=T0)     # noise, also far below
    d4 = 3.0 - 1.0 * t
    pool = np.column_stack([d0, d1, d2, d3, d4])
    treated = 0.5 * d0 + 0.5 * d1 + 0.25 * np.cos(9.0 * t)
    return pool, treated


# ------------------------------------------------------------------- smoke


def test_report_runs_and_is_finite_on_a_minimal_panel():
    pool, treated = _panel()
    rep = assess_spannability(pool, treated, [0, 1, 2])
    assert isinstance(rep, SpannabilityReport)
    for v in (rep.cluster_rmse, rep.pool_rmse, rep.ratio, rep.excluded_mass):
        assert np.isfinite(v)
    assert rep.n_cluster == 3
    assert rep.n_pool == 5


# ----------------------------------------------- rung 1: it separates cases


def test_a_cluster_keeping_the_spanning_donors_scores_about_one():
    pool, treated = _panel()
    rep = assess_spannability(pool, treated, [0, 1, 4])
    assert rep.ratio == pytest.approx(1.0, abs=1e-6)
    assert rep.excluded_mass == pytest.approx(0.0, abs=1e-6)


def test_a_cluster_dropping_a_spanning_donor_scores_far_above_one():
    """The Germany shape: the cluster keeps d1 but loses d0."""
    pool, treated = _panel()
    rep = assess_spannability(pool, treated, [1, 2, 3])
    assert rep.ratio > 10.0
    assert rep.excluded_mass > 0.3


def test_excluded_mass_is_the_pool_optimum_weight_on_the_dropped_donors():
    pool, treated = _panel()
    full = assess_spannability(pool, treated, [0, 1, 2, 3, 4])
    assert full.excluded_mass == pytest.approx(0.0, abs=1e-6)
    scale = max(float(np.abs(treated).max()), float(np.abs(pool).max()))
    pool_w, _ = _best_convex_fit(pool, treated, scale)
    half = assess_spannability(pool, treated, [1, 2, 3, 4])
    assert half.excluded_mass == pytest.approx(float(pool_w[0]), abs=1e-3)


# ------------------------------------------- rung 3: invariants that must hold


def test_ratio_is_never_below_one_because_a_cluster_is_a_subset():
    """A subset cannot fit better than the pool it came from."""
    pool, treated = _panel()
    for idx in ([0], [0, 1], [2, 3], [1, 4], [0, 2, 4], [0, 1, 2, 3, 4]):
        rep = assess_spannability(pool, treated, idx)
        assert rep.ratio >= 1.0 - 1e-8, f"cluster {idx} scored {rep.ratio}"


@pytest.mark.parametrize("scale", [1e-3, 1.0, 37_548.0])
def test_the_diagnostic_is_scale_free(scale):
    """The incident's first hypothesis was solver conditioning, because the
    error tracked the outcome scale. It did not: rescaling the panel leaves
    the ratio identical, so a scale-dependent reading is a wrong reading.
    """
    pool, treated = _panel()
    base = assess_spannability(pool, treated, [1, 2, 3])
    scaled = assess_spannability(pool * scale, treated * scale, [1, 2, 3])
    assert scaled.ratio == pytest.approx(base.ratio, rel=1e-4)
    assert scaled.excluded_mass == pytest.approx(base.excluded_mass, abs=1e-4)


def test_the_diagnostic_does_not_depend_on_donor_column_order():
    pool, treated = _panel()
    rep = assess_spannability(pool, treated, [1, 2, 3])
    perm = np.array([4, 1, 3, 0, 2])
    inv = {old: new for new, old in enumerate(perm)}
    rep_p = assess_spannability(pool[:, perm], treated, [inv[i] for i in (1, 2, 3)])
    assert rep_p.ratio == pytest.approx(rep.ratio, rel=1e-4)
    assert rep_p.excluded_mass == pytest.approx(rep.excluded_mass, abs=1e-4)


def test_duplicating_a_donor_inside_the_cluster_does_not_change_the_ratio():
    pool, treated = _panel()
    rep = assess_spannability(pool, treated, [0, 1])
    dup = np.column_stack([pool, pool[:, 0]])
    rep_d = assess_spannability(dup, treated, [0, 1, 5])
    assert rep_d.ratio == pytest.approx(rep.ratio, rel=1e-4)


# --------------------------------------------------- rung 4: edges & failures


def test_the_whole_pool_as_the_cluster_is_exactly_one():
    pool, treated = _panel()
    rep = assess_spannability(pool, treated, list(range(pool.shape[1])))
    assert rep.ratio == pytest.approx(1.0, abs=1e-9)
    assert rep.excluded_mass == pytest.approx(0.0, abs=1e-9)


def test_a_single_donor_cluster_is_reported_not_refused():
    pool, treated = _panel()
    rep = assess_spannability(pool, treated, [2])
    assert rep.n_cluster == 1
    assert rep.ratio > 1.0


def test_a_treated_unit_equal_to_one_donor_is_perfectly_spanned():
    """Both fits reach the unit, so the ratio is 1.0 and not 0/0.

    Neither RMSE is exactly zero -- CLARABEL leaves its own tolerance behind --
    so the assertion is that both sit under the solver floor and the ratio is
    reported as costing nothing, not that either is literally 0.
    """
    pool, treated = _panel()
    floor = 1e-5 * max(float(np.abs(pool).max()), float(np.abs(pool[:, 3]).max()))
    rep = assess_spannability(pool, pool[:, 3].copy(), [3, 0])
    assert rep.cluster_rmse < floor
    assert rep.pool_rmse < floor
    assert rep.ratio == pytest.approx(1.0, abs=1e-9)


def test_collinear_donors_do_not_break_the_solve():
    pool, treated = _panel()
    collinear = np.column_stack([pool[:, 0], 2.0 * pool[:, 0], 3.0 * pool[:, 0]])
    rep = assess_spannability(collinear, treated, [0, 1])
    assert np.isfinite(rep.ratio)


def test_an_empty_cluster_is_refused():
    pool, treated = _panel()
    with pytest.raises(MlsynthDataError, match="cluster"):
        assess_spannability(pool, treated, [])


def test_a_cluster_index_outside_the_pool_is_refused():
    pool, treated = _panel()
    with pytest.raises(MlsynthDataError, match="index"):
        assess_spannability(pool, treated, [0, 99])


def test_a_length_mismatch_is_refused():
    pool, treated = _panel()
    with pytest.raises(MlsynthDataError, match="length"):
        assess_spannability(pool, treated[:-1], [0, 1])


def test_a_non_2d_pool_is_refused():
    _pool, treated = _panel()
    with pytest.raises(MlsynthDataError, match="2D"):
        assess_spannability(treated, treated, [0])


# ------------------------------------------- the warning is reported, not dropped


def test_a_poorly_spanned_cluster_warns():
    pool, treated = _panel()
    rep = assess_spannability(pool, treated, [1, 2, 3])
    assert rep.ratio > SPANNABILITY_WARN_RATIO
    with pytest.warns(UserWarning, match="span"):
        warn_if_poorly_spanned(rep)


def test_the_warning_names_the_ratio_and_the_excluded_mass():
    pool, treated = _panel()
    rep = assess_spannability(pool, treated, [1, 2, 3])
    with pytest.warns(UserWarning) as rec:
        warn_if_poorly_spanned(rep)
    msg = str(rec[0].message)
    assert f"{rep.ratio:.1f}" in msg
    assert f"{rep.excluded_mass:.2f}" in msg


def test_a_well_spanned_cluster_does_not_warn():
    pool, treated = _panel()
    rep = assess_spannability(pool, treated, [0, 1, 4])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_poorly_spanned(rep)


# ------------------------------------- the incident, on the panel it came from


_BASEDATA = Path(__file__).resolve().parents[2] / "basedata"

#: The 11 donors FPCA put in West Germany's cluster. The five it dropped --
#: Greece, Portugal, Spain, Switzerland, USA -- carry 0.549 of the pool-optimal
#: convex weight, the USA alone 0.343.
_GERMANY_FPCA_CLUSTER = [
    "Australia", "Austria", "Belgium", "Denmark", "France", "Italy", "Japan",
    "Netherlands", "New Zealand", "Norway", "UK",
]


def _germany():
    df = pd.read_csv(_BASEDATA / "german_reunification.csv")
    wide = df.pivot(index="year", columns="country", values="gdp").dropna(axis=1)
    pre = wide.loc[wide.index < 1990]
    donors = [c for c in pre.columns if c != "West Germany"]
    return pre[donors].values.astype(float), pre["West Germany"].values.astype(float), donors


@pytest.mark.skipif(
    not (_BASEDATA / "german_reunification.csv").exists(),
    reason="german_reunification.csv not present",
)
def test_the_west_germany_cluster_is_flagged_as_unable_to_span():
    """The incident: rmse_pre 510.4 under simplex, 78.2 under nnls.

    The diagnostic has to separate this from a cluster that merely fits less
    well, so both numbers are pinned: the ratio, and the optimal weight the
    cluster threw away.
    """
    pool, treated, names = _germany()
    idx = [names.index(n) for n in _GERMANY_FPCA_CLUSTER]
    rep = assess_spannability(pool, treated, idx)
    assert rep.n_cluster == 11
    assert rep.ratio > 8.0
    assert rep.excluded_mass > 0.5
    assert rep.ratio > SPANNABILITY_WARN_RATIO
    with pytest.warns(UserWarning, match="span"):
        warn_if_poorly_spanned(rep)


@pytest.mark.skipif(
    not (_BASEDATA / "german_reunification.csv").exists(),
    reason="german_reunification.csv not present",
)
def test_the_full_german_donor_pool_costs_nothing():
    """The same panel with no selection scores 1.0, so the flag is the cluster."""
    pool, treated, names = _germany()
    rep = assess_spannability(pool, treated, list(range(len(names))))
    assert rep.ratio == pytest.approx(1.0, abs=1e-6)
    assert rep.excluded_mass == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(
    not (_BASEDATA / "basque_data.csv").exists(),
    reason="basque_data.csv not present",
)
def test_the_basque_cluster_is_not_flagged():
    """Basque is the control: a three-donor cluster that gives up nothing.

    Without it the Germany assertion has no power -- a diagnostic that fires
    on every cluster would pass that test and be useless.
    """
    df = pd.read_csv(_BASEDATA / "basque_data.csv")
    wide = df.pivot(index="year", columns="regionname", values="gdpcap").dropna(axis=1)
    pre = wide.loc[wide.index < 1975]
    treated_name = "Basque Country (Pais Vasco)"
    names = [c for c in pre.columns if c != treated_name]
    pool = pre[names].values.astype(float)
    treated = pre[treated_name].values.astype(float)
    idx = [names.index(n) for n in
           ("Baleares (Islas)", "Cataluna", "Madrid (Comunidad De)")]
    rep = assess_spannability(pool, treated, idx)
    assert rep.ratio < SPANNABILITY_WARN_RATIO
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_poorly_spanned(rep)


# --------------------------------------- wired into the estimator, both families


def _germany_long():
    df = pd.read_csv(_BASEDATA / "german_reunification.csv")
    s = df[["country", "year", "gdp"]].copy()
    s["treat"] = ((s.country == "West Germany") & (s.year >= 1990)).astype(int)
    return s


def _basque_long():
    df = pd.read_csv(_BASEDATA / "basque_data.csv")
    t = "Basque Country (Pais Vasco)"
    s = df[["regionname", "year", "gdpcap"]].copy()
    s["treat"] = ((s.regionname == t) & (s.year >= 1975)).astype(int)
    return s


_NEEDS_GERMANY = pytest.mark.skipif(
    not (_BASEDATA / "german_reunification.csv").exists(),
    reason="german_reunification.csv not present",
)


@_NEEDS_GERMANY
def test_clustersc_rpca_warns_on_the_west_germany_cluster():
    from mlsynth import CLUSTERSC
    cfg = dict(df=_germany_long(), outcome="gdp", treat="treat", unitid="country",
               time="year", method="rpca", weight_objective="simplex",
               display_graphs=False)
    with pytest.warns(UserWarning, match="span"):
        CLUSTERSC(cfg).fit()


@_NEEDS_GERMANY
def test_the_report_is_a_field_on_the_result_not_only_a_warning():
    """A caller must be able to read the number, not only see it go by."""
    from mlsynth import CLUSTERSC
    cfg = dict(df=_germany_long(), outcome="gdp", treat="treat", unitid="country",
               time="year", method="rpca", display_graphs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CLUSTERSC(cfg).fit()
    used = res.method_details.parameters_used or {}
    assert "spannability_ratio" in used
    assert used["spannability_ratio"] > 8.0
    assert used["spannability_excluded_mass"] > 0.5


@_NEEDS_GERMANY
def test_the_unconstrained_default_reports_the_same_ratio():
    """nnls hides the damage in the fit; the diagnostic still has to show it."""
    from mlsynth import CLUSTERSC
    base = dict(df=_germany_long(), outcome="gdp", treat="treat", unitid="country",
                time="year", method="rpca", display_graphs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = CLUSTERSC({**base}).fit()
        b = CLUSTERSC({**base, "weight_objective": "simplex"}).fit()
    ra = (a.method_details.parameters_used or {})["spannability_ratio"]
    rb = (b.method_details.parameters_used or {})["spannability_ratio"]
    assert ra == pytest.approx(rb, rel=1e-6)


@pytest.mark.skipif(
    not (_BASEDATA / "basque_data.csv").exists(),
    reason="basque_data.csv not present",
)
def test_clustersc_does_not_warn_on_basque():
    from mlsynth import CLUSTERSC
    cfg = dict(df=_basque_long(), outcome="gdpcap", treat="treat",
               unitid="regionname", time="year", method="rpca",
               weight_objective="simplex", display_graphs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CLUSTERSC(cfg).fit()
    used = res.method_details.parameters_used or {}
    assert used["spannability_ratio"] < SPANNABILITY_WARN_RATIO


@_NEEDS_GERMANY
def test_the_pcr_family_reports_it_too():
    """`cluster_donors` is shared, so the PCR path has the same exposure."""
    from mlsynth import CLUSTERSC
    cfg = dict(df=_germany_long(), outcome="gdp", treat="treat", unitid="country",
               time="year", method="pcr", display_graphs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CLUSTERSC(cfg).fit()
    used = res.method_details.parameters_used or {}
    assert "spannability_ratio" in used
    assert np.isfinite(used["spannability_ratio"])


@_NEEDS_GERMANY
def test_turning_clustering_off_reports_no_restriction():
    """With no selection there is nothing to give up, so no key and no warn."""
    from mlsynth import CLUSTERSC
    cfg = dict(df=_germany_long(), outcome="gdp", treat="treat", unitid="country",
               time="year", method="pcr", clustering=False, display_graphs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        res = CLUSTERSC(cfg).fit()
    assert "spannability_ratio" not in (res.method_details.parameters_used or {})
