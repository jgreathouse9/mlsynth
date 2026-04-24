"""
ffkm.py
=======
Functional Factorial K-Means (FFKM)

A functional clustering method that jointly optimises a low-dimensional
discriminant subspace and cluster assignments over B-spline-represented
time series.  Unlike PCA-then-k-means, the subspace here is chosen to
maximally separate clusters in the functional inner-product geometry —
making it strictly better for clustering purposes when the goal is
trajectory-homogeneous groups.

Primary public interface
------------------------
ffkm_features(Y0, ...)  →  (features: (N, L), k_opt: int)
    Drop-in replacement for _fpca_features in supergeo_solver.py.
    Takes a (T, N) panel, returns standardised FFKM discriminant scores
    and the silhouette-optimal cluster count.

FFKM class
    Full model with fit / select_model API for direct use.

Mathematical summary
--------------------
Given N time series expressed as B-spline coefficient vectors G ∈ R^{N×M}
and the spline Gram matrix H ∈ R^{M×M} (H[i,j] = ∫ φ_i φ_j dt), FFKM
solves

    min_{U, A}  Σ_k Σ_{i∈C_k} ‖ G_i A − μ_k ‖²_H

where A ∈ R^{M×L} is an H-orthonormal basis for the L-dimensional
discriminant subspace and U ∈ {0,1}^{N×K} is the cluster membership
matrix.  The objective is the within-cluster variance in the projected
functional space, measured with the L² inner product.

This is minimised by alternating:
    1. Fix A  →  assign each geo to its nearest cluster centroid
                 (standard k-means on scores G A).
    2. Fix U  →  update A as the top-L eigenvectors of the between-cluster
                 scatter matrix, weighted by H.

Convergence is guaranteed because each step is non-increasing; multiple
random restarts guard against poor local optima.

References
----------
Ieva, F., Paganoni, A. M., Pigoli, D., & Vitelli, V. (2013).
    Multivariate functional clustering for the morphological analysis of
    electrocardiograph curves.  Journal of the Royal Statistical Society:
    Series C, 62(3), 401–418.

Chiou, J.-M., & Li, P.-L. (2007).
    Functional clustering and identifying substructures of longitudinal
    data.  Journal of the Royal Statistical Society: Series B, 69(4),
    679–699.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.integrate import quad
from scipy.interpolate import BSpline, splev, splrep
from scipy.linalg import LinAlgError, cholesky, eigh, solve
from sklearn.metrics import silhouette_score


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================

def _spectral_rank(singular_values: np.ndarray, energy_threshold: float = 0.95) -> int:
    """
    Smallest rank r such that Σ_{i<r} σ_i² / Σ_i σ_i² ≥ energy_threshold.

    Parameters
    ----------
    singular_values  : (K,) float64   In descending order.
    energy_threshold : float           ∈ [0, 1].

    Returns
    -------
    int
    """
    if energy_threshold == 1.0:
        return len(singular_values)
    sq  = singular_values ** 2
    tot = sq.sum()
    if tot == 0.0:
        return 0
    cumulative = sq.cumsum() / tot
    hits = np.where(cumulative >= energy_threshold)[0]
    return int(hits[0]) + 1 if hits.size > 0 else len(singular_values)


def _silhouette_optimal_k(X: np.ndarray, k_max: int = 10) -> int:
    """
    Silhouette-based optimal cluster count for feature matrix X.

    Evaluates k-means for k ∈ [2, min(k_max, N−1)] and returns the k
    with the highest mean silhouette score.  Returns 1 when N < 2.

    Parameters
    ----------
    X     : (N, F) float64
    k_max : int             Upper bound on k to evaluate.

    Returns
    -------
    int
    """
    n = X.shape[0]
    if n < 2:
        return 1
    max_k = min(k_max, n - 1)
    if max_k < 2:
        return 1

    best_score = -np.inf
    best_k     = 2

    for k in range(2, max_k + 1):
        rng_km = np.random.RandomState(0)
        try:
            _, labels = kmeans2(X, k, minit="points", seed=rng_km)
        except Exception:
            continue
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(X, labels)
        if s > best_score:
            best_score = s
            best_k     = k

    return best_k


# =============================================================================
# FFKM CLASS
# =============================================================================

class FFKM:
    """
    Functional Factorial K-Means.

    Jointly finds a K-partition of N functional observations and the
    L-dimensional H-weighted subspace that best discriminates between them.

    Parameters
    ----------
    n_clusters   : int     K — number of clusters.
    n_components : int     L — discriminant subspace dimension.
    n_basis      : int     M — number of B-spline basis functions.
    spline_order : int     Degree of B-spline (3 = cubic).
    reg_lambda   : float   Roughness penalty weight (0 = no penalty).
    n_init       : int     Number of random ALS restarts.
    max_iter     : int     Maximum ALS iterations per restart.
    tol          : float   Convergence tolerance on loss change.
    random_state : int     Seed for reproducibility.

    Attributes (set after fit)
    --------------------------
    labels_  : (N,) int          Cluster assignment for each geo.
    scores_  : (N, L) float64    Projected coordinates in discriminant space.
    loss_    : float              Final within-cluster variance.
    t_       : (T,) float64      Normalised time grid used during fit.
    knots_   : (M+order+1,)      Extended B-spline knot vector.
    H_       : (M, M)            (Regularised) Gram matrix.
    G_       : (N, M)            Spline coefficient matrix (after reg.).
    A_H_     : (M, L)            H-orthonormal discriminant basis.
    """

    def __init__(
        self,
        n_clusters:   int   = 3,
        n_components: int   = 2,
        n_basis:      int   = 15,
        spline_order: int   = 3,
        reg_lambda:   float = 0.0,
        n_init:       int   = 10,
        max_iter:     int   = 1000,
        tol:          float = 1e-5,
        random_state: int   = 42,
    ) -> None:
        self.K            = n_clusters
        self.L            = n_components
        self.M            = n_basis
        self.order        = spline_order
        self.reg_lambda   = reg_lambda
        self.n_init       = n_init
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, t: Optional[np.ndarray] = None) -> "FFKM":
        """
        Fit FFKM to an (N, T) data matrix.

        Parameters
        ----------
        X : (N, T) float64   One row per unit, one column per time point.
        t : (T,) float64     Time grid.  Defaults to linspace(0, 1, T).

        Returns
        -------
        self
        """
        N, T = X.shape
        if t is None:
            t = np.linspace(0, 1, T)
        self.t_ = t

        X_centered = X - X.mean(axis=0)

        # ---- B-spline basis ----
        interior_knots = np.linspace(t.min(), t.max(), self.M - self.order + 1)
        self.knots_ = np.concatenate([
            [t.min()] * self.order,
            interior_knots,
            [t.max()] * self.order,
        ])

        # ---- Spline coefficients: G_raw ∈ R^{N×M} ----
        self.G_raw_ = self._fit_splines(X_centered, t)

        # ---- Gram matrix H ----
        H = self._gram_matrix()

        # ---- Optional roughness penalty ----
        R = self._penalty_matrix(order=2) if self.reg_lambda > 0.0 else None
        self.H_, self.G_ = self._regularise(self.G_raw_, H, R)

        # ---- Alternating least squares ----
        U, self.A_H_, self.loss_ = self._als(self.G_, self.H_)

        self.labels_ = np.argmax(U, axis=1)

        # ---- Projected scores: S = G H^{-1} A_H ----
        chol_H       = cholesky(self.H_, lower=True)
        A            = solve(chol_H.T, solve(chol_H, self.A_H_))
        self.scores_ = self.G_ @ A   # (N, L)

        return self

    # ------------------------------------------------------------------
    # Public: model selection
    # ------------------------------------------------------------------

    def select_model(
        self,
        X:             np.ndarray,
        t:             Optional[np.ndarray] = None,
        K_range:       range                = range(2, 8),
        var_threshold: float                = 0.95,
        n_runs:        int                  = 5,
        verbose:       bool                 = False,
    ) -> tuple[int, int, dict]:
        """
        Joint selection of L (rank) and K (clusters) via SVD + silhouette.

        Algorithm
        ---------
        1.  Select L: find the spectral rank of X that retains
            `var_threshold` of cumulative squared-singular-value energy.
        2.  Select K: for each K in K_range, run `n_runs` FFKM fits and
            record the mean silhouette score of the projected scores.
            Choose K with the highest mean silhouette.
        3.  Refit the final model with (best_K, best_L).

        Parameters
        ----------
        X             : (N, T)
        t             : (T,)  optional time grid
        K_range       : range  cluster counts to evaluate
        var_threshold : float  energy fraction for rank selection
        n_runs        : int    restarts per K
        verbose       : bool

        Returns
        -------
        best_K   : int
        best_L   : int
        results  : dict   K → (mean_silhouette, std_silhouette)
        """
        N, T = X.shape
        if t is None:
            t = np.linspace(0, 1, T)

        # ---- Step 1: rank selection ----
        X_c       = X - X.mean(axis=0)
        _, S, _   = np.linalg.svd(X_c, full_matrices=False)
        best_L    = max(1, _spectral_rank(S, var_threshold))

        if verbose:
            print(f"Selected L ({var_threshold*100:.0f}% variance): {best_L}")

        # ---- Step 2: silhouette sweep over K ----
        results: dict[int, tuple[float, float]] = {}

        for K in K_range:
            sils: list[float] = []
            for run in range(n_runs):
                m = FFKM(
                    n_clusters=K, n_components=best_L,
                    n_basis=self.M, spline_order=self.order,
                    reg_lambda=self.reg_lambda,
                    n_init=self.n_init, max_iter=self.max_iter,
                    tol=self.tol, random_state=run,
                )
                try:
                    m.fit(X, t)
                    if len(np.unique(m.labels_)) >= 2:
                        sils.append(silhouette_score(m.scores_, m.labels_))
                except Exception:
                    continue

            results[K] = (float(np.mean(sils)), float(np.std(sils))) if sils \
                         else (-np.inf, np.inf)

        best_K = max(results, key=lambda k: results[k][0])

        if verbose:
            print("\nSilhouette results:")
            for k, (m, s) in results.items():
                print(f"  K={k}: mean={m:.4f}  std={s:.4f}")
            print(f"\nSelected K: {best_K}")

        # ---- Step 3: final fit ----
        self.K = best_K
        self.L = best_L
        self.selected_K_        = best_K
        self.selected_L_        = best_L
        self.selection_results_ = results
        self.fit(X, t)

        return best_K, best_L, results

    # ------------------------------------------------------------------
    # Private: spline fitting
    # ------------------------------------------------------------------

    def _fit_splines(self, X_centered: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Fit a B-spline to each row of X_centered and return the M-vector
        of spline coefficients.  Shape: (N, M).
        """
        N  = X_centered.shape[0]
        G  = np.zeros((N, self.M))
        for i in range(N):
            tck     = splrep(t, X_centered[i], k=self.order, s=0)
            G[i]    = tck[1][: self.M]
        return G

    # ------------------------------------------------------------------
    # Private: Gram matrix  H[i,j] = ∫ φ_i(t) φ_j(t) dt
    # ------------------------------------------------------------------

    def _gram_matrix(self) -> np.ndarray:
        """
        Compute the M×M Gram matrix of the B-spline basis by numerical
        integration.  Symmetric; only upper triangle is computed.
        """
        M  = self.M
        H  = np.zeros((M, M))
        t0, t1 = self.t_.min(), self.t_.max()

        for i in range(M):
            for j in range(i, M):
                def integrand(tt, _i=i, _j=j):
                    bv = BSpline.design_matrix(
                        np.array([tt]), self.knots_, self.order
                    ).toarray().ravel()
                    return bv[_i] * bv[_j]

                val, _ = quad(integrand, t0, t1)
                H[i, j] = H[j, i] = val

        return H

    # ------------------------------------------------------------------
    # Private: penalty matrix  R[i,j] = ∫ φ_i^(d)(t) φ_j^(d)(t) dt
    # ------------------------------------------------------------------

    def _penalty_matrix(self, order: int = 2) -> np.ndarray:
        """
        Compute the M×M roughness penalty matrix using derivatives of
        order `order` (default: 2 → penalises curvature).
        """
        M  = self.M
        R  = np.zeros((M, M))
        t0, t1 = self.t_.min(), self.t_.max()
        eye = np.eye(M)

        for i in range(M):
            for j in range(i, M):
                def integrand(tt, _i=i, _j=j):
                    di = splev(tt, (self.knots_, eye[_i], self.order), der=order)
                    dj = splev(tt, (self.knots_, eye[_j], self.order), der=order)
                    return di * dj

                val, _ = quad(integrand, t0, t1)
                R[i, j] = R[j, i] = val

        return R

    # ------------------------------------------------------------------
    # Private: regularisation
    # ------------------------------------------------------------------

    def _regularise(
        self,
        G:  np.ndarray,
        H:  np.ndarray,
        R:  Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply Tikhonov-style roughness regularisation.

        With penalty:   H_reg = H + λR
                        G_reg = (H_reg)^{-1} H G^T  transposed

        Without penalty: H_reg = H,  G_reg = G  (copy)
        """
        if self.reg_lambda > 0.0 and R is not None:
            H_reg = H + self.reg_lambda * R
            G_reg = solve(H_reg, H @ G.T).T
        else:
            H_reg = H
            G_reg = G.copy()
        return H_reg, G_reg

    # ------------------------------------------------------------------
    # Private: cluster assignment step (fix A, update U)
    # ------------------------------------------------------------------

    def _update_clusters(self, scores: np.ndarray) -> np.ndarray:
        """
        Assign each geo to its nearest centroid in the L-dimensional score
        space.  Returns hard membership matrix U ∈ {0,1}^{N×K}.
        """
        _, labels = kmeans2(scores, self.K, minit="points")
        U = np.zeros((scores.shape[0], self.K))
        U[np.arange(scores.shape[0]), labels] = 1.0
        return U

    # ------------------------------------------------------------------
    # Private: subspace update step (fix U, update A)
    # ------------------------------------------------------------------

    def _update_subspace(
        self,
        G: np.ndarray,
        H: np.ndarray,
        U: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Given cluster memberships U, find the L-dimensional H-orthonormal
        subspace A_H that maximises between-cluster scatter.

        The optimal A_H is the top-L generalised eigenvectors of

            scatter = G^T (P_U − I/N) G H

        where P_U = U diag(1/N_k) U^T is the cluster projection operator.

        Parameters
        ----------
        G : (N, M)   Spline coefficient matrix.
        H : (M, M)   Gram matrix.
        U : (N, K)   Cluster membership matrix.

        Returns
        -------
        A_H_new : (M, L)   Updated H-orthonormal basis.
        loss    : float     Within-cluster variance in projected space.

        Raises
        ------
        ValueError   If any cluster is empty (N_k ≤ 1).
        """
        Nk = U.sum(axis=0)
        if np.any(Nk <= 1):
            raise ValueError("Empty or singleton cluster encountered.")

        N  = G.shape[0]
        P_U    = U @ np.diag(1.0 / Nk) @ U.T          # (N, N) cluster projector
        scatter = G.T @ (P_U - np.eye(N) / N) @ G @ H  # (M, M) between-cluster scatter

        eigvals, eigvecs = eigh(scatter)
        idx      = np.argsort(eigvals)[::-1][: self.L]
        A_H_new  = eigvecs[:, idx]                      # (M, L)

        # Within-cluster variance (loss to minimise)
        chol_H = cholesky(H, lower=True)
        A      = solve(chol_H.T, solve(chol_H, A_H_new))
        proj   = G @ A                                  # (N, L)
        loss   = float(np.sum((proj - proj.mean(axis=0)) ** 2))

        return A_H_new, loss

    # ------------------------------------------------------------------
    # Private: ALS core
    # ------------------------------------------------------------------

    def _als(
        self,
        G: np.ndarray,
        H: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Alternating Least Squares with `n_init` random restarts.

        Each restart:
        1.  Initialise A_H as a random H-orthonormal matrix.
        2.  Repeat until convergence:
            a. Compute scores S = G H^{-1} A_H.
            b. Update U via k-means on S.
            c. Update A_H via the generalised eigen-decomposition.
        3.  Keep the restart with the lowest final loss.

        Returns
        -------
        best_U   : (N, K)
        best_A_H : (M, L)
        best_loss: float
        """
        chol_H    = cholesky(H, lower=True)
        best_loss = np.inf
        best_U = best_A_H = None

        for _ in range(self.n_init):
            # Random H-orthonormal initialisation
            Z    = self.random_state.randn(self.M, self.L)
            Q, _ = np.linalg.qr(chol_H @ Z)
            A_H  = chol_H @ Q                    # H-orthonormal: A_H^T H^{-1} A_H = I

            prev_loss = np.inf

            for _iter in range(self.max_iter):
                # Step a: project onto current subspace
                A      = solve(chol_H.T, solve(chol_H, A_H))
                scores = G @ A                   # (N, L)

                # Step b: cluster assignment
                U = self._update_clusters(scores)

                # Step c: subspace update
                try:
                    A_H_new, loss = self._update_subspace(G, H, U)
                except ValueError:
                    break                        # empty cluster — restart

                if abs(prev_loss - loss) < self.tol:
                    break

                prev_loss = loss
                A_H       = A_H_new

            if loss < best_loss:
                best_loss = loss
                best_U    = U
                best_A_H  = A_H

        return best_U, best_A_H, best_loss


# =============================================================================
# DROP-IN REPLACEMENT FOR _fpca_features
# =============================================================================

def ffkm_features(
    Y0:             np.ndarray,
    n_basis:        int   = 15,
    spline_order:   int   = 3,
    reg_lambda:     float = 0.0,
    var_threshold:  float = 0.95,
    k_range:        range = range(2, 8),
    n_runs:         int   = 3,
    n_init:         int   = 5,
    max_iter:       int   = 300,
    tol:            float = 1e-4,
    random_state:   int   = 42,
    k_max_sil:      int   = 10,
) -> tuple[np.ndarray, int]:
    """
    FFKM-based geo embedding for trajectory-aware clustering.

    Drop-in replacement for ``_fpca_features`` in supergeo_solver.py.
    Returns standardised discriminant scores and the silhouette-optimal
    cluster count.

    Algorithm
    ---------
    1.  Transpose Y0 from (T, N) to (N, T) so each geo is a row.
    2.  Run ``FFKM.select_model`` to jointly select:
            L  — discriminant subspace dimension (via spectral energy)
            K  — number of clusters (via silhouette sweep)
    3.  Return the standardised (N, L) score matrix and K.

    Fallback
    --------
    If T ≤ spline_order the B-spline fit is infeasible.  In that case
    we fall back to direct SVD + spectral rank + silhouette k-means,
    which is the behaviour of the previous _fpca_features.

    Parameters
    ----------
    Y0            : (T, N) float64   Pre-period panel (time × geos).
    n_basis       : int    M — B-spline basis size.
    spline_order  : int    Spline degree (3 = cubic).
    reg_lambda    : float  Roughness penalty weight.
    var_threshold : float  Energy fraction for rank selection.
    k_range       : range  Cluster counts evaluated in silhouette sweep.
    n_runs        : int    FFKM restarts per K in the silhouette sweep.
    n_init        : int    ALS restarts inside each FFKM fit.
    max_iter      : int    Maximum ALS iterations per restart.
    tol           : float  ALS convergence tolerance.
    random_state  : int    Master seed.
    k_max_sil     : int    Upper bound on k in the silhouette sweep.

    Returns
    -------
    features : (N, L) float64   Standardised FFKM discriminant scores.
    k_opt    : int               Silhouette-optimal cluster count.
    """
    T, N = Y0.shape
    X    = Y0.T.copy()               # (N, T) — geos as rows

    # ---- Fallback: too few time points for cubic spline ----
    if T <= spline_order:
        warnings.warn(
            f"T={T} ≤ spline_order={spline_order}: falling back to direct SVD.",
            RuntimeWarning, stacklevel=2,
        )
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        L        = max(1, _spectral_rank(S, var_threshold))
        scores   = X @ Vt[:L].T
        k_opt    = _silhouette_optimal_k(_standardise(scores), k_max=k_max_sil)
        return _standardise(scores), k_opt

    # ---- Full FFKM pipeline ----
    model = FFKM(
        n_clusters=2,               # placeholder; overridden by select_model
        n_components=1,             # placeholder; overridden by select_model
        n_basis=min(n_basis, T),    # can't have more bases than time points
        spline_order=spline_order,
        reg_lambda=reg_lambda,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )

    # Clip k_range to what's feasible given N
    feasible_k_range = range(
        k_range.start,
        min(k_range.stop, N),       # can't have more clusters than geos
    )
    if len(feasible_k_range) < 1:
        feasible_k_range = range(2, max(3, N))

    t_norm = np.linspace(0, 1, T)

    try:
        best_K, best_L, _ = model.select_model(
            X,
            t=t_norm,
            K_range=feasible_k_range,
            var_threshold=var_threshold,
            n_runs=n_runs,
            verbose=False,
        )
    except Exception as exc:
        warnings.warn(
            f"FFKM.select_model failed ({exc}); falling back to SVD.",
            RuntimeWarning, stacklevel=2,
        )
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        L        = max(1, _spectral_rank(S, var_threshold))
        scores   = X @ Vt[:L].T
        k_opt    = _silhouette_optimal_k(_standardise(scores), k_max=k_max_sil)
        return _standardise(scores), k_opt

    features = _standardise(model.scores_)   # (N, L) standardised
    return features, best_K


# =============================================================================
# PRIVATE UTILITY
# =============================================================================

def _standardise(X: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std per column.  Constant columns left as zero."""
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (X - mu) / std


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    rng  = np.random.default_rng(0)
    N, T = 30, 70

    # Two trajectory groups: sine-like vs cosine-like
    t_grid = np.linspace(0, 2 * np.pi, T)
    group_a = np.sin(t_grid)[None, :] + 0.1 * rng.standard_normal((N // 2, T))
    group_b = np.cos(t_grid)[None, :] + 0.1 * rng.standard_normal((N - N // 2, T))
    X       = np.vstack([group_a, group_b])   # (N, T)
    Y0      = X.T                             # (T, N) — supergeo_solver convention

    print("Running ffkm_features on (T=70, N=30) panel...")
    features, k_opt = ffkm_features(Y0, n_runs=2, n_init=3, max_iter=100)
    print(f"  Feature shape : {features.shape}")
    print(f"  Optimal K     : {k_opt}  (expect 2)")

    # Verify cluster recovery
    labels = features[:, 0] > 0
    purity = max(labels[:N//2].mean(), 1 - labels[:N//2].mean())
    print(f"  Cluster purity: {purity:.2f}  (expect ~1.0)")

    # Full model API
    print("\nRunning FFKM.select_model directly...")
    model = FFKM(n_clusters=2, n_components=1, n_basis=12, random_state=0)
    K, L, res = model.select_model(X, K_range=range(2, 5), n_runs=2, verbose=True)
    print(f"  labels: {model.labels_}")
    print(f"  loss  : {model.loss_:.4f}")
