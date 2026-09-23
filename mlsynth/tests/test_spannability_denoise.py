"""Whether the denoiser moved the donors' hull away from the treated unit.

``assess_spannability`` compares the selected cluster against the full donor
pool, on raw outcomes, before anything is denoised. That catches a cluster
which dropped donors the treated unit needed -- West Germany, where the FPCA
cluster costs a factor of 8.6 in convex reach.

It is blind to the opposite failure. Basque's three-donor FPCA cluster costs
nothing at all against the full sixteen-donor pool: the best achievable
convex pre-period RMSE is 0.0842 either way, and undenoised that cluster
reproduces Abadie-Gardeazabal's published weights (Cataluna 0.840, Madrid
0.160, nothing on Baleares) to within 0.01. Running the default PCP over it
moves the best achievable fit to 0.2786 and hands Baleares -- the outlier --
a plurality of 0.538. The cluster was right and the denoiser spoiled it, and
the pre-denoising check cannot see that by construction.

This module measures the second step: the best achievable convex fit against
the denoised donors, against the same quantity on the raw ones. It also
reports whether the denoised block still identifies a unique weight vector,
because a low-rank denoiser applied to a handful of donors leaves the fitted
weights one arbitrary point in a continuum.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.clustersc_helpers.spannability import (
    SPANNABILITY_WARN_RATIO,
    assess_denoise_spannability,
    warn_if_denoising_shrank_the_hull,
)

_T0 = 24


@pytest.fixture(scope="module")
def panel():
    """Donors spanning a 4-D subspace, treated unit genuinely off the hull.

    The offset is what gives these tests power: with the treated unit sitting
    exactly inside the hull every ratio below collapses to solver noise and
    the assertions pass for reasons unrelated to denoising.
    """
    rng = np.random.default_rng(4)
    t = np.linspace(0.0, 1.0, _T0)
    basis = np.column_stack([np.ones(_T0), t, np.sin(3.0 * t), t ** 2])
    donors = basis @ rng.normal(size=(4, 6)) * 3.0 + 40.0
    donors += rng.normal(scale=0.4, size=donors.shape)   # full column rank
    w = np.zeros(6); w[[0, 2, 4]] = (0.5, 0.3, 0.2)
    treated = donors @ w + 2.5 + rng.normal(scale=0.05, size=_T0)
    return treated, donors


def _rank_project(M, r):
    """Keep the top ``r`` singular directions -- a stand-in denoiser."""
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    return (u[:, :r] * s[:r]) @ vt[:r]


# --------------------------------------------------------------------------
# Smoke
# --------------------------------------------------------------------------
def test_report_is_finite_and_well_formed(panel):
    treated, donors = panel
    rep = assess_denoise_spannability(donors, _rank_project(donors, 2), treated)
    assert np.isfinite(rep.raw_rmse) and np.isfinite(rep.denoised_rmse)
    assert np.isfinite(rep.ratio) and rep.ratio > 0
    assert rep.n_donors == donors.shape[1]


# --------------------------------------------------------------------------
# Unit: what the ratio means
# --------------------------------------------------------------------------
def test_an_identity_denoiser_costs_exactly_nothing(panel):
    """The control. If this does not read 1.0 the ratio measures noise."""
    treated, donors = panel
    rep = assess_denoise_spannability(donors, donors.copy(), treated)
    assert rep.ratio == pytest.approx(1.0, abs=1e-6)
    assert rep.denoised_rmse == pytest.approx(rep.raw_rmse, rel=1e-6)


def test_collapsing_the_donors_onto_one_direction_is_reported(panel):
    """A rank-1 projection leaves a hull that is nearly a point."""
    treated, donors = panel
    rep = assess_denoise_spannability(donors, _rank_project(donors, 1), treated)
    assert rep.ratio > 2.0


def test_a_denoiser_that_helps_reports_a_ratio_below_one():
    """Stripping noise that pushed the donors away moves the hull closer.

    The check has to be able to report an improvement, or it detects
    denoising and not damage. Built the other way round from the fixture:
    the treated unit sits inside the CLEAN donors' hull, noise pushes that
    hull off it, and recovering the clean rank brings it back.
    """
    rng = np.random.default_rng(9)
    t = np.linspace(0.0, 1.0, _T0)
    basis = np.column_stack([np.ones(_T0), t, np.sin(3.0 * t), t ** 2])
    clean = basis @ rng.normal(size=(4, 6)) * 3.0 + 40.0
    treated = clean @ np.array([0.4, 0.0, 0.25, 0.0, 0.35, 0.0])   # inside the hull
    noisy = clean + rng.normal(scale=1.5, size=clean.shape)
    rep = assess_denoise_spannability(noisy, _rank_project(noisy, 4), treated)
    assert rep.ratio < 1.0


def test_the_ratio_is_invariant_to_a_common_rescaling(panel):
    """Outcome magnitudes span five orders across the panels in basedata."""
    treated, donors = panel
    base = assess_denoise_spannability(donors, _rank_project(donors, 2), treated)
    for factor in (1e-3, 1e4):
        scaled = assess_denoise_spannability(
            donors * factor, _rank_project(donors, 2) * factor, treated * factor)
        assert scaled.ratio == pytest.approx(base.ratio, rel=1e-4)


# --------------------------------------------------------------------------
# Unit: whether the denoised block still identifies the weights
# --------------------------------------------------------------------------
def test_a_rank_deficient_denoised_block_does_not_identify_the_weights(panel):
    """Basque's case: three donors through a low-rank denoiser.

    The reported weights are then one arbitrary point of a continuum, which
    a reader has no way to tell from a determinate answer.
    """
    treated, donors = panel
    three = donors[:, :3]
    rep = assess_denoise_spannability(three, _rank_project(three, 1), treated)
    assert rep.weights_identified is False


def test_a_full_rank_denoised_block_identifies_the_weights(panel):
    treated, donors = panel
    rep = assess_denoise_spannability(donors, donors.copy(), treated)
    assert rep.weights_identified is True


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------
def test_a_single_donor_is_a_point_hull_and_still_reports(panel):
    treated, donors = panel
    one = donors[:, :1]
    rep = assess_denoise_spannability(one, one * 0.5, treated)
    assert np.isfinite(rep.ratio)
    assert rep.n_donors == 1


def test_a_raw_hull_that_already_reaches_the_unit_does_not_divide_by_noise(panel):
    """When the raw cluster reaches the treated unit the ratio is 0/0.

    Report 1.0 when the denoised block also reaches it and infinity when the
    denoiser pushed it away, instead of dividing the solver's residual floor
    by itself.
    """
    _treated, donors = panel
    exact = donors @ np.array([0.4, 0.0, 0.25, 0.0, 0.35, 0.0])
    same = assess_denoise_spannability(donors, donors.copy(), exact)
    assert same.ratio == pytest.approx(1.0, abs=1e-9)
    worse = assess_denoise_spannability(donors, _rank_project(donors, 1), exact)
    assert worse.ratio == float("inf")


def test_a_constant_denoised_block_is_reported_not_crashed(panel):
    treated, donors = panel
    flat = np.full_like(donors, 7.0)
    rep = assess_denoise_spannability(donors, flat, treated)
    assert np.isfinite(rep.ratio) or rep.ratio == float("inf")
    assert rep.weights_identified is False


# --------------------------------------------------------------------------
# The warning
# --------------------------------------------------------------------------
def test_a_harmless_denoiser_is_silent(panel):
    treated, donors = panel
    rep = assess_denoise_spannability(donors, donors.copy(), treated)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_denoising_shrank_the_hull(rep)


def test_a_damaging_denoiser_warns_and_names_the_numbers(panel):
    treated, donors = panel
    rep = assess_denoise_spannability(donors, _rank_project(donors, 1), treated)
    with pytest.warns(UserWarning, match="denois"):
        warn_if_denoising_shrank_the_hull(rep)
    assert rep.ratio > SPANNABILITY_WARN_RATIO


def test_an_unidentified_weight_vector_is_reported_but_does_not_warn(panel):
    """Rank deficiency is reported, not warned about.

    Four donors with one an exact affine combination of the others. The
    hull is untouched -- the denoiser is the identity, so the ratio is
    exactly 1.0 -- and the weights are still a continuum, so the field must
    say so. It must not warn: measured across three panels and six
    denoiser/clustering combinations this is False in eleven of twelve,
    and a warning at that rate buries the ratio warning that does
    discriminate.
    """
    treated, donors = panel
    a, b, c = donors[:, 0], donors[:, 1], donors[:, 2]
    dependent = np.column_stack([a, b, c, a + b - c])
    rep = assess_denoise_spannability(dependent, dependent.copy(), treated)
    assert rep.ratio == pytest.approx(1.0, abs=1e-6)
    assert rep.weights_identified is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_denoising_shrank_the_hull(rep)


# --------------------------------------------------------------------------
# Failure: bad input is reported
# --------------------------------------------------------------------------
def test_non_2d_raw_block_raises(panel):
    treated, donors = panel
    with pytest.raises(MlsynthDataError, match="2D"):
        assess_denoise_spannability(donors[:, 0], donors, treated)


def test_mismatched_shapes_between_raw_and_denoised_raise(panel):
    treated, donors = panel
    with pytest.raises(MlsynthDataError, match="same shape|mismatch"):
        assess_denoise_spannability(donors, donors[:, :3], treated)


def test_pre_period_length_mismatch_raises(panel):
    treated, donors = panel
    with pytest.raises(MlsynthDataError, match="mismatch"):
        assess_denoise_spannability(donors, donors.copy(), treated[:-3])


# --------------------------------------------------------------------------
# Through the pipeline, on the panels that motivated it
# --------------------------------------------------------------------------
def _panel_arrays(file, unit, time, outcome, treated, t0):
    import pandas as pd
    from pathlib import Path
    from mlsynth.utils.datautils import dataprep
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "basedata" / file)
    df["treat"] = ((df[unit] == treated) & (df[time] >= t0)).astype(int)
    prep = dataprep(df, unit, time, outcome, "treat")
    return dict(treated_outcome=np.asarray(prep["y"], float).ravel(),
                donor_outcomes=np.asarray(prep["donor_matrix"], float),
                donor_names=list(prep["donor_names"]),
                T0=int(prep["pre_periods"]))


@pytest.fixture(scope="module")
def basque_arrays():
    return _panel_arrays("basque_data.csv", "regionname", "year", "gdpcap",
                         "Basque Country (Pais Vasco)", 1975)


@pytest.fixture(scope="module")
def germany_arrays():
    import pandas as pd
    from pathlib import Path
    from mlsynth.utils.datautils import dataprep
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "basedata" / "german_reunification.csv")
    df["treat"] = df["Reunification"].astype(int)
    prep = dataprep(df, "country", "year", "gdp", "treat")
    return dict(treated_outcome=np.asarray(prep["y"], float).ravel(),
                donor_outcomes=np.asarray(prep["donor_matrix"], float),
                donor_names=list(prep["donor_names"]),
                T0=int(prep["pre_periods"]))


def test_basque_default_path_reports_the_denoiser_moved_the_hull(basque_arrays):
    """The case this exists for.

    Basque's FPCA cluster costs nothing at selection, so the existing
    pre-denoising check is silent. PCP over that cluster then moves the
    best achievable convex fit from 0.0843 to 0.2786 and hands Baleares --
    the outlier -- a plurality of 0.538, taking the ATT from -0.7015 to
    -0.9204 against Abadie-Gardeazabal's -0.6996.
    """
    from mlsynth.utils.clustersc_helpers.rpca.pipeline import run_rpca
    with pytest.warns(UserWarning, match="Denoising moved"):
        fit = run_rpca(**basque_arrays, rpca_method="PCP")
    assert fit.metadata["spannability_denoise_ratio"] > SPANNABILITY_WARN_RATIO


def test_germany_fgrc_hsvt_does_not_report_a_denoising_problem(germany_arrays):
    """The control. HSVT on the fGRC cluster moves the hull towards the
    treated unit, so a check that fired here would be useless."""
    from mlsynth.utils.clustersc_helpers.rpca.pipeline import run_rpca
    fit = run_rpca(**germany_arrays, cluster_method="fgrc",
                   rpca_method="HSVT", fgrc_k=2)
    assert fit.metadata["spannability_denoise_ratio"] < 1.0


def test_the_denoise_report_reaches_the_result_metadata(basque_arrays):
    from mlsynth.utils.clustersc_helpers.rpca.pipeline import run_rpca
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = run_rpca(**basque_arrays, rpca_method="PCP")
    for key in ("spannability_denoise_ratio", "spannability_denoise_raw_rmse",
                "spannability_denoise_rmse", "spannability_weights_identified"):
        assert key in fit.metadata, key


def test_the_check_runs_even_when_no_donors_were_dropped(basque_arrays):
    """The selection check is skipped when the cluster is the whole pool.
    Denoising still happens, so this one must not be skipped with it."""
    from mlsynth.utils.clustersc_helpers.rpca.pipeline import run_rpca
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = run_rpca(**basque_arrays, rpca_method="PCP", k_clusters=1)
    assert "spannability_ratio" not in fit.metadata      # no donors dropped
    assert "spannability_denoise_ratio" in fit.metadata  # but denoising ran
