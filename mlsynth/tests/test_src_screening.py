"""Tests for SRC's SIRS donor screening (Zhu 2023, Algorithm 2).

Written test-first. Covers the SIRS statistic, the screening count, the
donor-selection mechanism, the config surface, and the end-to-end effect on a
wide (donors >= pre-periods) panel where screening restores a well-posed fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SRC
from mlsynth.config_models import SRCConfig
from mlsynth.exceptions import MlsynthConfigError


# --------------------------------------------------------------------------- #
# unit: the SIRS statistic and the screening count / selection
# --------------------------------------------------------------------------- #
def _import_screening():
    from mlsynth.utils.src_helpers import screening
    return screening


def test_sirs_scores_shape_nonneg_finite_deterministic():
    scr = _import_screening()
    rng = np.random.default_rng(0)
    T0, J = 20, 8
    treated = rng.normal(size=T0)
    donors = rng.normal(size=(T0, J))
    s1 = scr.sirs_scores(donors, treated)
    s2 = scr.sirs_scores(donors, treated)
    assert s1.shape == (J,)
    assert np.all(np.isfinite(s1))
    assert np.all(s1 >= 0.0)
    assert np.array_equal(s1, s2)  # deterministic, no RNG


def test_sirs_prefers_comoving_and_antimoving_over_constant():
    """A donor that (anti-)tracks the treated ranks above a flat donor."""
    scr = _import_screening()
    rng = np.random.default_rng(1)
    T0 = 30
    treated = np.linspace(0, 5, T0) + 0.01 * rng.normal(size=T0)
    comove = treated + 0.01 * rng.normal(size=T0)         # tracks treated
    antimove = -treated + 0.01 * rng.normal(size=T0)      # anti-tracks (still useful: theta<0)
    constant = np.full(T0, 3.0)                           # no variation
    donors = np.column_stack([comove, antimove, constant])
    s = scr.sirs_scores(donors, treated)
    assert s[0] > s[2]          # co-moving beats constant
    assert s[1] > s[2]          # anti-moving also beats constant (SIRS captures dependence, not sign)
    assert s[2] == pytest.approx(0.0, abs=1e-12)  # zero-variance donor screens to 0


def test_screen_count_matches_paper_formula_and_is_well_posed():
    scr = _import_screening()
    # k = min(floor(T0 / log(T0/2)), T0 - 1); always < T0 so the screened fit is well posed
    for T0, expected in [(15, 7), (20, 8), (40, 13), (60, 17)]:
        assert scr.screen_count(T0) == expected
    for T0 in range(3, 80):
        k = scr.screen_count(T0)
        assert 1 <= k <= T0 - 1


def test_screen_count_tiny_T0_is_safe():
    scr = _import_screening()
    assert scr.screen_count(2) >= 1
    assert scr.screen_count(3) >= 1


def test_screen_donors_selects_top_k_indices():
    scr = _import_screening()
    rng = np.random.default_rng(2)
    T0, J = 25, 12
    treated = np.linspace(0, 1, T0)
    donors = rng.normal(size=(T0, J))
    # inject three donors that clearly track the treated -> must be kept
    for j in (3, 7, 11):
        donors[:, j] = treated * (j + 1) + 0.001 * rng.normal(size=T0)
    k = scr.screen_count(T0)
    keep = scr.screen_donors(donors, treated, n_screen=None)
    assert keep.shape == (k,)
    assert len(set(keep.tolist())) == k                    # unique
    assert np.all((keep >= 0) & (keep < J))                # in range
    assert set([3, 7, 11]).issubset(set(keep.tolist()))    # trackers survive
    # exactly the top-k by score
    scores = scr.sirs_scores(donors, treated)
    assert set(keep.tolist()) == set(np.argsort(scores)[::-1][:k].tolist())


def test_screen_donors_noop_when_pool_not_larger_than_k():
    scr = _import_screening()
    rng = np.random.default_rng(3)
    T0, J = 40, 5                    # k(40)=13 > J=5 -> keep all
    treated = rng.normal(size=T0)
    donors = rng.normal(size=(T0, J))
    keep = scr.screen_donors(donors, treated, n_screen=None)
    assert np.array_equal(keep, np.arange(J))


def test_screen_donors_respects_n_screen_override():
    scr = _import_screening()
    rng = np.random.default_rng(4)
    T0, J = 25, 12
    treated = rng.normal(size=T0)
    donors = rng.normal(size=(T0, J))
    keep = scr.screen_donors(donors, treated, n_screen=3)
    assert keep.shape == (3,)


# --------------------------------------------------------------------------- #
# config surface
# --------------------------------------------------------------------------- #
def _df():  # minimal balanced panel for config-construction tests
    rows = []
    for u in range(4):
        for t in range(6):
            rows.append({"unit": f"u{u}", "t": t, "y": float(u + t),
                         "treat": int(u == 0 and t >= 4)})
    return pd.DataFrame(rows)


def test_config_accepts_screen_sirs():
    cfg = SRCConfig(df=_df(), outcome="y", treat="treat", unitid="unit",
                    time="t", screen="sirs")
    assert cfg.screen == "sirs"


def test_config_screen_defaults_to_none():
    cfg = SRCConfig(df=_df(), outcome="y", treat="treat", unitid="unit", time="t")
    assert cfg.screen == "none"


def test_config_rejects_bogus_screen():
    with pytest.raises(Exception):
        SRCConfig(df=_df(), outcome="y", treat="treat", unitid="unit",
                  time="t", screen="bogus")


def test_config_rejects_nonpositive_n_screen():
    with pytest.raises(Exception):
        SRCConfig(df=_df(), outcome="y", treat="treat", unitid="unit",
                  time="t", screen="sirs", n_screen=0)


# --------------------------------------------------------------------------- #
# integration: screening restores a well-posed fit on a wide panel
# --------------------------------------------------------------------------- #
def _wide_panel(seed=0):
    """A panel with J=10 donors > T0=8 pre-periods -> unscreened SRC degenerates."""
    rng = np.random.default_rng(seed)
    T, T0, N, r = 12, 8, 11, 2
    F = rng.normal(size=(T, r))
    L = rng.normal(size=(N, r))
    Y = F @ L.T + 0.1 * rng.normal(size=(T, N))
    Y[T0:, 0] += -3.0            # treatment effect on unit 0 after T0
    rows = []
    for u in range(N):
        for t in range(T):
            rows.append({"unit": f"u{u}", "t": t, "y": float(Y[t, u]),
                         "treat": int(u == 0 and t >= T0)})
    return pd.DataFrame(rows), T0


def test_fit_screening_reduces_pool_and_fixes_degeneracy():
    from mlsynth.utils.src_helpers import screening
    df, T0 = _wide_panel()
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="t",
                display_graphs=False)

    res_no = SRC({**base, "screen": "none"}).fit()
    res_sc = SRC({**base, "screen": "sirs"}).fit()

    J = len(res_no.weights.donor_weights)
    k = screening.screen_count(T0)
    assert J == 10                                   # full pool
    assert len(res_sc.weights.donor_weights) == k    # screened pool
    assert k < J
    # unscreened is degenerate (donors >= pre-periods => sigma^2 ~ 0);
    # screening restores a real noise estimate.
    assert res_no.fit.sigma2 < 1e-8
    assert res_sc.fit.sigma2 > 1e-6


def test_fit_screening_is_deterministic():
    df, _ = _wide_panel()
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="t",
                display_graphs=False, screen="sirs")
    a = SRC(base).fit()
    b = SRC(base).fit()
    assert a.att == pytest.approx(b.att)
    assert a.weights.donor_weights == b.weights.donor_weights


def test_fit_screen_none_matches_default():
    df, _ = _wide_panel()
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="t",
                display_graphs=False)
    default = SRC(base).fit()
    explicit = SRC({**base, "screen": "none"}).fit()
    assert default.att == pytest.approx(explicit.att)
    assert default.weights.donor_weights == explicit.weights.donor_weights


# --------------------------------------------------------------------------- #
# FPCA + clustering donor selection (reuses ClusterSC's helpers)
# --------------------------------------------------------------------------- #
def _fpca_cluster_arrays(seed=0):
    """Two clean shape-clusters: treated + group A ~ sine; group B ~ parabola."""
    rng = np.random.default_rng(seed)
    T0 = 12
    t = np.linspace(0.0, 1.0, T0)
    f1 = np.sin(2 * np.pi * t)          # cols 0..3 shape (with treated)
    f2 = (t - 0.5) ** 2                 # cols 4..8 shape
    treated = f1 + 0.02 * rng.normal(size=T0)
    A = np.column_stack([f1 + 0.02 * rng.normal(size=T0) for _ in range(4)])
    B = np.column_stack([f2 + 0.02 * rng.normal(size=T0) for _ in range(5)])
    donors = np.column_stack([A, B])
    return donors, treated, [0, 1, 2, 3], [4, 5, 6, 7, 8]


def test_fpca_donors_selects_treated_shape_cluster():
    scr = _import_screening()
    donors, treated, groupA, groupB = _fpca_cluster_arrays()
    keep = scr.fpca_donors(donors, treated)
    assert keep.size >= 1
    assert set(keep.tolist()).issubset(set(groupA))      # only the similar-shape donors
    assert set(keep.tolist()).isdisjoint(set(groupB))    # none of the dissimilar ones
    assert keep.size < donors.shape[1]                   # genuinely reduced the pool


def test_fpca_donors_deterministic_and_valid_indices():
    scr = _import_screening()
    donors, treated, _, _ = _fpca_cluster_arrays()
    k1 = scr.fpca_donors(donors, treated)
    k2 = scr.fpca_donors(donors, treated)
    assert np.array_equal(k1, k2)                        # seeded k-means -> deterministic
    assert np.array_equal(k1, np.sort(k1))               # ascending
    assert len(set(k1.tolist())) == k1.size              # unique
    assert np.all((k1 >= 0) & (k1 < donors.shape[1]))


def test_fpca_donors_keeps_all_when_no_shape_structure():
    """With no cross-sectional shape variation (identical paths -> FPC rank 0),
    clustering is degenerate and the whole pool is kept (a no-op)."""
    scr = _import_screening()
    T0, J = 12, 6
    t = np.linspace(0.0, 1.0, T0)
    shape = np.sin(2 * np.pi * t)
    treated = shape.copy()
    donors = np.column_stack([shape.copy() for _ in range(J)])   # identical -> rank 0
    keep = scr.fpca_donors(donors, treated)
    assert np.array_equal(keep, np.arange(J))            # no-op: all donors kept


def test_config_accepts_screen_fpca():
    cfg = SRCConfig(df=_df(), outcome="y", treat="treat", unitid="unit",
                    time="t", screen="fpca")
    assert cfg.screen == "fpca"


def test_fit_screen_fpca_reduces_pool_and_is_deterministic():
    """End-to-end: FPCA selects the treated unit's cluster, deterministically."""
    rng = np.random.default_rng(2)
    T, T0 = 16, 12
    t = np.linspace(0.0, 1.0, T)
    f1 = np.sin(2 * np.pi * t)
    f2 = (t - 0.5) ** 2
    cols = {"u0": f1 + 0.02 * rng.normal(size=T)}        # treated
    for j in range(4):
        cols[f"a{j}"] = f1 + 0.02 * rng.normal(size=T)   # similar
    for j in range(5):
        cols[f"b{j}"] = f2 + 0.02 * rng.normal(size=T)   # dissimilar
    rows = []
    for u, series in cols.items():
        for ti in range(T):
            rows.append({"unit": u, "t": ti, "y": float(series[ti]),
                         "treat": int(u == "u0" and ti >= T0)})
    df = pd.DataFrame(rows)
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="t",
                display_graphs=False)
    full = SRC({**base, "screen": "none"}).fit()
    fpca = SRC({**base, "screen": "fpca"}).fit()
    assert len(full.weights.donor_weights) == 9
    assert len(fpca.weights.donor_weights) < 9           # pool reduced to the cluster
    assert all(lbl.startswith("a") for lbl in fpca.weights.donor_weights)  # only 'a' donors
    again = SRC({**base, "screen": "fpca"}).fit()
    assert fpca.att == pytest.approx(again.att)
    assert fpca.weights.donor_weights == again.weights.donor_weights
