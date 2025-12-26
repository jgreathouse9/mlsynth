import numpy as np
from scipy.interpolate import BSpline, splev, splrep
from scipy.integrate import quad
from scipy.linalg import cholesky, solve, eigh
from scipy.cluster.vq import kmeans2
from numpy.random import default_rng
class FFKM:
    """
    Refactored Functional Factorial K-means (FFKM) for clustering functional data.
    Default behavior: two-step PCA + ALS for low-rank subspace estimation and clustering.
    Deterministic via fixed random_state.
    """

    def __init__(self, n_clusters, n_components=None, n_basis=15, spline_order=3,
                 reg_lambda=0.0, n_init=10, max_iter=1000, tol=1e-5, random_state=42,
                 pca_var=0.95):
        self.K = n_clusters
        self.L = n_components  # if None, determined by PCA in two-step
        self.M = n_basis
        self.order = spline_order
        self.reg_lambda = reg_lambda
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = default_rng(random_state)
        self.pca_var = pca_var  # target variance explained for two-step PCA

    def fit(self, X, t=None):
        N, T = X.shape
        if t is None:
            t = np.linspace(0, 1, T)
        self.t_ = t

        # Center data
        X_centered = X - X.mean(axis=0)

        # B-spline basis
        interior_knots = np.linspace(t.min(), t.max(), self.M - self.order + 1)
        self.knots_ = np.concatenate([[t.min()] * self.order,
                                      interior_knots,
                                      [t.max()] * self.order])

        self.G_raw_ = self._fit_splines_to_curves(X_centered, t, interior_knots)

        # Gram matrix
        H = self._compute_gram_matrix()

        # Optional regularization
        R = self._compute_penalty_matrix(order=2) if self.reg_lambda != 0 else None
        self.H_, self.G_ = self._apply_regularization(self.G_raw_, H, X_centered, t, R)

        # ------------------ TWO-STEP PCA + ALS (default) ------------------ #
        # Determine number of components automatically if not specified
        scores_pca, A_H_pca, Q = self._functional_pca(self.G_, self.H_, self.pca_var)
        if self.L is None:
            self.L = Q  # set L to number of components explaining desired variance

        scores_pca, A_H_pca, Q = self._functional_pca(self.G_, self.H_, self.pca_var)

        # Enforce consistency
        self.L = Q

        # ALS step using Q components
        U, A_H_red, loss = self._als_core(self.G_, self.H_, Q)

        # Align subspaces safely
        self.A_H_ = self._procrustes_alignment(A_H_pca[:, :Q], A_H_red[:, :Q])

        self.labels_ = np.argmax(U, axis=1)
        self.loss_ = loss

        chol_H = cholesky(self.H_, lower=True)
        A = solve(chol_H.T, solve(chol_H, self.A_H_))
        self.scores_ = self.G_ @ A
        self.weight_functions_ = self._construct_weight_functions(A)

        return self

    # ---------------- Internal Methods ---------------- #
    def _fit_splines_to_curves(self, X_centered, t, interior_knots):
        N = X_centered.shape[0]
        G = np.zeros((N, self.M))
        for i in range(N):
            tck = splrep(t, X_centered[i], k=self.order, s=0)
            G[i] = tck[1][:self.M]
        return G

    def _compute_gram_matrix(self):
        M = self.M
        H = np.zeros((M, M))
        for i in range(M):
            for j in range(i, M):
                def integrand(tt):
                    basis_vals = BSpline.design_matrix(np.array([tt]), self.knots_, self.order).toarray().flatten()
                    return basis_vals[i] * basis_vals[j]
                val, _ = quad(integrand, self.t_.min(), self.t_.max(), epsabs=1e-8)
                H[i, j] = H[j, i] = val
        return H

    def _compute_penalty_matrix(self, order=2):
        M = self.M
        R = np.zeros((M, M))
        for i in range(M):
            for j in range(i, M):
                def integrand(tt):
                    d_i = splev(tt, (self.knots_, np.eye(M)[i], self.order), der=order)
                    d_j = splev(tt, (self.knots_, np.eye(M)[j], self.order), der=order)
                    return d_i * d_j
                val, _ = quad(integrand, self.t_.min(), self.t_.max(), epsabs=1e-8)
                R[i, j] = R[j, i] = val
        return R

    def _apply_regularization(self, G, H, X_centered, t, R):
        if self.reg_lambda != 0:
            if self.reg_lambda == 'auto':
                self.reg_lambda = self._select_lambda_gcv(X_centered, t, H, R)
            H_reg = H + self.reg_lambda * R
            G_reg = solve(H_reg, H @ G.T).T
        else:
            H_reg = H
            G_reg = G.copy()
        return H_reg, G_reg

    def _functional_pca(self, G, H, var_thresh):
        cov = G.T @ G @ H / G.shape[0]
        eigvals, eigvecs = eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        cumvar = np.cumsum(eigvals[idx]) / eigvals[idx].sum()
        Q = np.searchsorted(cumvar, var_thresh) + 1
        A_H = eigvecs[:, idx[:Q]]
        scores = G @ solve(H, A_H)
        return scores, A_H, Q

    def _procrustes_alignment(self, A_H_orig, A_H_red):
        U, _, Vt = np.linalg.svd(A_H_orig.T @ A_H_red, full_matrices=False)
        return A_H_orig @ U @ Vt

    def _construct_weight_functions(self, A):
        weight_functions = []
        for l in range(self.L):
            a_l = A[:, l]
            def eval_func(tt, a=a_l):
                return splev(tt, (self.knots_, a, self.order))
            weight_functions.append(eval_func)
        return weight_functions

    def _update_clusters(self, scores):
        seed_val = int(self.random_state.integers(0, 2**31))
        _, labels = kmeans2(scores, self.K, minit='points', seed=seed_val)
        U = np.zeros((scores.shape[0], self.K))
        U[np.arange(scores.shape[0]), labels] = 1
        return U, labels

    def _update_subspace(self, G, H, U):
        Nk = U.sum(axis=0)
        if np.any(Nk <= 1):
            raise ValueError("One or more clusters are empty.")
        P_U = U @ np.diag(1.0 / Nk) @ U.T
        scatter = G.T @ (P_U - np.eye(G.shape[0]) / G.shape[0]) @ G @ H
        eigvals, eigvecs = eigh(scatter)
        idx = np.argsort(eigvals)[::-1][:self.L]
        A_H_new = eigvecs[:, idx]
        proj = G @ solve(H, A_H_new)
        loss = np.sum((proj - proj.mean(axis=0))**2)
        return A_H_new, loss

    def _als_core(self, G, H, dim):
        best_loss = np.inf
        best_U = best_A_H = None
        chol_H = cholesky(H, lower=True)

        for init in range(self.n_init):
            # Basis dimension
            M = H.shape[0]

            # Initialize Z in the correct space: (M x L)
            Z = self.random_state.standard_normal((M, self.L))

            # Project into H-orthonormal component space
            Qz, _ = np.linalg.qr(chol_H @ Z)
            A_H = chol_H @ Qz

            prev_loss = np.inf
            loss = np.inf
            for iter_num in range(self.max_iter):
                A = solve(chol_H.T, solve(chol_H, A_H))
                U, labels = self._update_clusters(G @ A)
                try:
                    A_H_new, loss = self._update_subspace(G, H, U)
                except ValueError:
                    break

                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss
                A_H = A_H_new

            if loss < best_loss:
                best_loss = loss
                best_U = U
                best_A_H = A_H

        return best_U, best_A_H, best_loss
