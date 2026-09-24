"""fGRC's own subspace, used as the donor denoiser.

``optim_grc`` fits a cluster subspace ``A1`` and a disturbing subspace ``A2``
together with the partition, and returns all three. ``fgrc_cluster`` keeps the
labels and drops ``A`` on the line that computes it, after which the RPCA
pipeline derives a second, unrelated low-rank structure with PCP. Yamamoto and
Hwang's method is a joint estimate of the subspace and the clustering, so
running a different denoiser afterwards discards half of it.

``fgrc_lowrank`` closes that: the denoised donor matrix is ``G A A^T`` mapped
back through the B-spline basis.

Which half to keep is not a free choice and the tests pin the measured answer.
On the Basque panel, out-of-sample RMSE over eight placebo windows is 0.2419
keeping all of ``A`` and 0.4490 keeping only ``A1``. Projecting out the
disturbing direction is worse, and under a simplex weight objective it is
catastrophic -- the treated unit stops being reachable by any convex
combination and the effect comes back with the wrong sign. A direction can be
disturbing for clustering and still carry the signal a counterfactual needs.
"""

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.clustersc_helpers.rpca.fgrc import (
    FGRCSubspace,
    fgrc_cluster,
    fgrc_lowrank,
    fgrc_subspace,
)

_T, _J = 40, 12


def _panel(seed=0, n_time=_T, n_units=_J):
    """Two groups of trajectories on a shared trend plus group-specific shape."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_time)
    trend = 5.0 + 3.0 * t
    rows = []
    for j in range(n_units):
        shape = np.sin(4.0 * t) if j % 2 == 0 else np.cos(4.0 * t)
        rows.append(trend + 1.5 * shape + 0.05 * rng.normal(size=n_time))
    return np.asarray(rows)


# ------------------------------------------------------------------- smoke


def test_subspace_runs_and_carries_the_pieces_the_clustering_fit():
    X = _panel()
    sub = fgrc_subspace(X, c1=2, c2=1, k=2, n_knots=8, n_random=8, nstart=8)
    assert isinstance(sub, FGRCSubspace)
    assert sub.labels.shape == (_J,)
    assert sub.A.shape[1] == 3            # c1 + c2
    assert sub.G.shape[0] == _J
    assert np.isfinite(sub.loss)


def test_lowrank_returns_the_panel_shape_and_is_finite():
    X = _panel()
    sub = fgrc_subspace(X, c1=2, c2=1, k=2, n_knots=8, n_random=8, nstart=8)
    L = fgrc_lowrank(sub)
    assert L.shape == X.shape
    assert np.all(np.isfinite(L))


# --------------------------------------------- it agrees with the clustering


def test_the_labels_match_what_fgrc_cluster_returns():
    """Same engine, same seed: the accessor must not fit a different thing."""
    X = _panel()
    kw = dict(c1=2, c2=1, k=2, n_knots=8, order=4, seed=0, n_random=8, nstart=8)
    labels_only, loss_only = fgrc_cluster(X, **kw)
    sub = fgrc_subspace(X, **kw)
    assert np.array_equal(sub.labels, labels_only)
    assert sub.loss == pytest.approx(loss_only, rel=1e-12)


# ------------------------------------------------ what the reconstruction is


def test_keeping_the_whole_subspace_is_a_rank_c1_plus_c2_reconstruction():
    X = _panel()
    sub = fgrc_subspace(X, c1=2, c2=1, k=2, n_knots=8, n_random=8, nstart=8)
    L = fgrc_lowrank(sub, keep="all")
    assert np.linalg.matrix_rank(L - L.mean(axis=0), tol=1e-8) <= 3


def test_dropping_the_disturbing_subspace_lowers_the_rank():
    X = _panel()
    sub = fgrc_subspace(X, c1=2, c2=1, k=2, n_knots=8, n_random=8, nstart=8)
    full = fgrc_lowrank(sub, keep="all")
    cluster_only = fgrc_lowrank(sub, keep="cluster")
    r_full = np.linalg.matrix_rank(full - full.mean(axis=0), tol=1e-8)
    r_c1 = np.linalg.matrix_rank(cluster_only - cluster_only.mean(axis=0), tol=1e-8)
    assert r_c1 < r_full


def test_dropping_the_disturbing_subspace_shrinks_between_donor_spread():
    """The measured reason keep='cluster' is the wrong default.

    A2 carries between-donor spread. Removing it flattens the donors toward a
    common profile, and a convex weight step then cannot reach a treated unit
    that sits away from that profile.
    """
    X = _panel()
    sub = fgrc_subspace(X, c1=2, c2=1, k=2, n_knots=8, n_random=8, nstart=8)
    full = fgrc_lowrank(sub, keep="all")
    cluster_only = fgrc_lowrank(sub, keep="cluster")
    assert cluster_only.std(axis=0).mean() < full.std(axis=0).mean()


def test_the_reconstruction_tracks_the_panel_it_was_fit_on():
    X = _panel()
    sub = fgrc_subspace(X, c1=2, c2=1, k=2, n_knots=8, n_random=8, nstart=8)
    L = fgrc_lowrank(sub)
    rel = np.linalg.norm(L - X) / np.linalg.norm(X)
    assert rel < 0.2


# ------------------------------------------------------------- invariants


def test_the_reconstruction_is_equivariant_under_a_common_scale():
    """Equivariant to 3e-4 relative, not to solver precision.

    The ALS stopping rule in ``_als_once`` compares successive loss values
    against an absolute ``eps``, so a panel scaled by 100 carries a loss
    scaled by 10000 and stops a little earlier: the loss ratio comes out
    9994.97 where 10000 is exact. The reconstruction inherits that as a
    3.2e-4 relative difference. Asserting 1e-6 would assert something the
    engine does not do. Making ``eps`` relative would change
    ``fgrc_cluster`` too, which is already pinned by `fgrc_toy_subspace`,
    so it is a separate change.

    Run at the production restart counts on purpose. Cutting them to 8
    widens the gap to 5.4e-3, because the two scaled problems then settle
    in different local optima -- the multi-restart search is scale-
    sensitive for the same reason the stopping rule is.
    """
    X = _panel()
    kw = dict(c1=2, c2=1, k=2, n_knots=8, seed=0)
    a = fgrc_lowrank(fgrc_subspace(X, **kw))
    b = fgrc_lowrank(fgrc_subspace(100.0 * X, **kw))
    assert np.allclose(b, 100.0 * a, rtol=2e-3, atol=1e-6)


def test_repeat_fits_from_the_same_seed_agree():
    X = _panel()
    kw = dict(c1=2, c2=1, k=2, n_knots=8, seed=3, n_random=8, nstart=8)
    assert np.allclose(fgrc_lowrank(fgrc_subspace(X, **kw)),
                       fgrc_lowrank(fgrc_subspace(X, **kw)))


# ------------------------------------------------------ edges and failures


def test_no_disturbing_subspace_is_allowed_and_keep_all_equals_keep_cluster():
    X = _panel()
    sub = fgrc_subspace(X, c1=2, c2=0, k=2, n_knots=8, n_random=8, nstart=8)
    assert np.allclose(fgrc_lowrank(sub, keep="all"),
                       fgrc_lowrank(sub, keep="cluster"))


def test_an_unknown_keep_is_refused():
    X = _panel()
    sub = fgrc_subspace(X, c1=2, c2=1, k=2, n_knots=8, n_random=8, nstart=8)
    with pytest.raises(MlsynthConfigError, match="keep"):
        fgrc_lowrank(sub, keep="A1")


def test_a_non_2d_panel_is_refused():
    with pytest.raises(MlsynthDataError, match="2D"):
        fgrc_subspace(np.arange(10.0), c1=1, c2=0, k=2, n_knots=6)


def test_more_clusters_than_units_is_refused():
    X = _panel(n_units=3)
    with pytest.raises(MlsynthConfigError, match="k"):
        fgrc_subspace(X, c1=1, c2=0, k=5, n_knots=6)


# ------------------------------------------ wired into CLUSTERSC as a method

import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

_BASEDATA = Path(__file__).resolve().parents[2] / "basedata"
_NEEDS_BASQUE = pytest.mark.skipif(
    not (_BASEDATA / "basque_data.csv").exists(), reason="basque_data.csv not present"
)


def _basque_long():
    df = pd.read_csv(_BASEDATA / "basque_data.csv")
    t = "Basque Country (Pais Vasco)"
    s = df[["regionname", "year", "gdpcap"]].copy()
    s["treat"] = ((s.regionname == t) & (s.year >= 1975)).astype(int)
    return s


@_NEEDS_BASQUE
def test_fgrc_is_selectable_as_an_rpca_method():
    from mlsynth import CLUSTERSC
    cfg = dict(df=_basque_long(), outcome="gdpcap", treat="treat",
               unitid="regionname", time="year", method="rpca",
               rpca_method="FGRC", cluster_method="fgrc", display_graphs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CLUSTERSC(cfg).fit()
    assert np.isfinite(res.effects.att)
    assert np.isfinite(res.fit_diagnostics.rmse_pre)
    assert res.method_details.method_name.endswith("(FGRC)")


@_NEEDS_BASQUE
def test_the_method_records_which_half_of_the_subspace_it_kept():
    from mlsynth import CLUSTERSC
    cfg = dict(df=_basque_long(), outcome="gdpcap", treat="treat",
               unitid="regionname", time="year", method="rpca",
               rpca_method="FGRC", cluster_method="fgrc", display_graphs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CLUSTERSC(cfg).fit()
    used = res.method_details.parameters_used or {}
    assert used.get("fgrc_keep") == "all"


@_NEEDS_BASQUE
def test_an_unknown_rpca_method_is_still_refused():
    from mlsynth import CLUSTERSC
    from mlsynth.exceptions import MlsynthConfigError
    cfg = dict(df=_basque_long(), outcome="gdpcap", treat="treat",
               unitid="regionname", time="year", method="rpca",
               rpca_method="NOPE", display_graphs=False)
    with pytest.raises(MlsynthConfigError):
        CLUSTERSC(cfg).fit()


@_NEEDS_BASQUE
def test_dropping_the_disturbing_block_puts_the_treated_unit_outside_the_hull():
    """The mechanism behind the wrong-signed ATT at ``fgrc_keep="cluster"``.

    Diagnosed rather than asserted in prose. On Basque the projection leaves
    the mean level untouched (3.6622 before and after, so the reconstruction's
    mean handling is not at fault) and removes 75% of the between-donor
    spread. The donor envelope shrinks from [1.243, 7.973] to [1.763, 5.740]
    while the treated unit reaches 7.105, so it sits above every donor in
    every pre-period. A convex combination is bounded by the largest donor, so
    no feasible fit exists: the solver pins all the weight on one donor and the
    counterfactual falls short at every date. The ATT that comes back is that
    shortfall, not an effect.
    """
    from mlsynth.utils.clustersc_helpers.rpca.fgrc import fgrc_lowrank, fgrc_subspace

    df = pd.read_csv(_BASEDATA / "basque_data.csv")
    treated = "Basque Country (Pais Vasco)"
    wide = df.pivot(index="year", columns="regionname", values="gdpcap").dropna(axis=1)
    T0 = int((wide.index < 1975).sum())
    y = wide[treated].values[:T0].astype(float)
    Y = wide[[c for c in wide.columns if c != treated]].values.T.astype(float)

    sub = fgrc_subspace(Y, c1=2, c2=1, k=2, n_knots=max(4, T0 // 2 - 2),
                        order=4, seed=0)
    keep_all = fgrc_lowrank(sub, keep="all")[:, :T0]
    keep_c1 = fgrc_lowrank(sub, keep="cluster")[:, :T0]
    raw = Y[:, :T0]

    # the level survives -- this is not a mean-handling bug
    assert keep_c1.mean() == pytest.approx(raw.mean(), rel=1e-9)
    assert keep_all.mean() == pytest.approx(raw.mean(), rel=1e-9)

    # the spread does not
    assert keep_all.std(axis=0).mean() > 0.9 * raw.std(axis=0).mean()
    assert keep_c1.std(axis=0).mean() < 0.4 * raw.std(axis=0).mean()

    # and that is what puts the treated unit out of reach
    assert float((y > raw.max(axis=0)).mean()) == 0.0
    assert float((y > keep_all.max(axis=0)).mean()) == 0.0
    assert float((y > keep_c1.max(axis=0)).mean()) == 1.0


# --------------------------------------- why keep="cluster" collapses the donors


@_NEEDS_BASQUE
def test_the_cluster_subspace_discriminates_groups_and_carries_no_variance():
    """The structural reason ``fgrc_keep="cluster"`` flattens the donors.

    fGRC splits the subspace by what discriminates clusters. ``A1`` holds the
    separating directions, ``A2`` the high-variance ones that do not separate.
    That split is the method working: it stops a variance-dominant direction
    swamping the grouping.

    It also means the two blocks are near-opposites in content. On Basque,
    ``A1``'s columns are about 70% between-cluster while carrying under 4% of
    the variance; ``A2`` carries over 90% of the variance and almost all of it
    is within cluster.
    """
    from mlsynth.utils.clustersc_helpers.rpca.fgrc import fgrc_subspace

    df = pd.read_csv(_BASEDATA / "basque_data.csv")
    treated = "Basque Country (Pais Vasco)"
    wide = df.pivot(index="year", columns="regionname",
                    values="gdpcap").dropna(axis=1)
    T0 = int((wide.index < 1975).sum())
    X = wide[[c for c in wide.columns if c != treated]].values.T.astype(float)
    sub = fgrc_subspace(X, c1=2, c2=1, k=2, n_knots=max(4, T0 // 2 - 2),
                        order=4, seed=0)
    scores = sub.G @ sub.A
    labels = sub.labels

    def between_share(v):
        grand = v.mean()
        between = sum(((v[labels == g].mean() - grand) ** 2) * (labels == g).sum()
                      for g in np.unique(labels))
        within = sum(((v[labels == g] - v[labels == g].mean()) ** 2).sum()
                     for g in np.unique(labels))
        return between / (between + within)

    energy = (scores ** 2).sum(axis=0) / (sub.G ** 2).sum()

    for j in range(sub.c1):                      # A1: discriminates, tiny variance
        assert between_share(scores[:, j]) > 0.5
        assert energy[j] < 0.10
    for j in range(sub.c1, sub.A.shape[1]):      # A2: variance, no discrimination
        assert between_share(scores[:, j]) < 0.2
        assert energy[j] > 0.80


@_NEEDS_BASQUE
def test_the_collapse_is_the_direction_not_the_rank():
    """A rank-2 projection is not the problem; *which* rank-2 subspace is.

    Top-2 PCA of ``G`` keeps essentially all the between-donor spread at the
    same rank fGRC's ``A1`` keeps a fifth of it. Without this the obvious
    reading is that the donors simply need three dimensions, which would make
    the cluster/disturbing split irrelevant to the failure.
    """
    from mlsynth.utils.clustersc_helpers.rpca.fgrc import fgrc_subspace

    df = pd.read_csv(_BASEDATA / "basque_data.csv")
    treated = "Basque Country (Pais Vasco)"
    wide = df.pivot(index="year", columns="regionname",
                    values="gdpcap").dropna(axis=1)
    T0 = int((wide.index < 1975).sum())
    X = wide[[c for c in wide.columns if c != treated]].values.T.astype(float)
    sub = fgrc_subspace(X, c1=2, c2=1, k=2, n_knots=max(4, T0 // 2 - 2),
                        order=4, seed=0)
    G = sub.G
    raw_spread = G.std(axis=0).mean()

    def kept(basis):
        return (G @ basis @ basis.T).std(axis=0).mean() / raw_spread

    pca2 = np.linalg.svd(G, full_matrices=False)[2][:2].T
    assert kept(sub.A[:, :sub.c1]) < 0.35      # fGRC's cluster subspace, rank 2
    assert kept(pca2) > 0.90                   # plain PCA, same rank
    assert kept(sub.A) > 0.90                  # fGRC's full subspace, rank 3


@_NEEDS_BASQUE
def test_the_reconstruction_preserves_the_spread_ratio_of_the_projection():
    """The back-map is not where the spread goes.

    ``fgrc_lowrank`` inverts the Gram root and the B-spline basis, either of
    which could in principle shrink the panel. The ratio it produces in time
    coordinates tracks the ratio the projection produced in ``G``, so the loss
    is the projection's and the reconstruction is exonerated.
    """
    from mlsynth.utils.clustersc_helpers.rpca.fgrc import fgrc_lowrank, fgrc_subspace

    df = pd.read_csv(_BASEDATA / "basque_data.csv")
    treated = "Basque Country (Pais Vasco)"
    wide = df.pivot(index="year", columns="regionname",
                    values="gdpcap").dropna(axis=1)
    T0 = int((wide.index < 1975).sum())
    X = wide[[c for c in wide.columns if c != treated]].values.T.astype(float)
    sub = fgrc_subspace(X, c1=2, c2=1, k=2, n_knots=max(4, T0 // 2 - 2),
                        order=4, seed=0)

    for keep, basis in (("all", sub.A), ("cluster", sub.A[:, :sub.c1])):
        in_g = ((sub.G @ basis @ basis.T).std(axis=0).mean()
                / sub.G.std(axis=0).mean())
        in_time = (fgrc_lowrank(sub, keep=keep)[:, :T0].std(axis=0).mean()
                   / X[:, :T0].std(axis=0).mean())
        assert abs(in_time - in_g) < 0.10, f"{keep}: {in_g:.4f} vs {in_time:.4f}"
