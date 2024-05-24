import numpy as np

def universal_rank(s, ratio):
    omega = 0.56 * ratio ** 3 - 0.95 * ratio ** 2 + 1.43 + 1.82 * ratio
    t = omega * np.median(s)
    rank = max(len(s[s > t]), 1)
    return rank


def spectral_rank(s, t=0.95):
    if t == 1.0:
        rank = len(s)
    else:
        total_energy = (s ** 2).cumsum() / (s ** 2).sum()
        rank = list((total_energy > t)).index(True) + 1
    return rank


def svt(self, X, max_rank=None, t=None):
    (n1, n2) = X.shape
    (u, s, v) = np.linalg.svd(X, full_matrices=False)
    if max_rank is not None:
        rank = max_rank
    elif t is not None:
        rank = self.spectral_rank(s, t=t)
    else:
        ratio = min(n1, n2) / max(n1, n2)
        rank = self.universal_rank(s, ratio=ratio)
    s_rank = s[:rank]
    u_rank = u[:, :rank]
    v_rank = v[:rank, :]
    Y0_rank = np.dot(u_rank, np.dot(np.diag(s_rank), v_rank))
    Hu = u_rank @ u_rank.T
    Hv = v_rank.T @ v_rank
    Hu_perp = np.eye(n1) - Hu
    Hv_perp = np.eye(n2) - Hv

    return (Y0_rank, Hu, Hv, Hu_perp, Hv_perp)


def shrink(X, tau):
    """
    Gets our optimal singular values (shrinks) to optimal number,
    'soft impute'
    """
    # 'Soft' Impute
    Y = np.abs(X) - tau
    return np.sign(X) * np.maximum(Y, np.zeros_like(Y))

def SVT(X, tau):
    """
    Does the Singular Value thresholding (the matrix operation)
    """
    # Does Singular Value Thresholding
    U, S, VT = np.linalg.svd(X, full_matrices=0)
    out = U @ np.diag(shrink(S, tau)) @ VT
    return out

def RPCA(X, vallamb=1):
    """
    Gets our low-rank representation into
    Low rank structure, common patterns and noise
    """

    n1, n2 = X.shape
    mu = n1 * n2 / (4 * np.sum(np.abs(X.reshape(-1))))

    thresh = 10 ** (-9) * np.linalg.norm(X)

    S = np.zeros_like(X)  # Individual effects
    Y = np.zeros_like(X)  # Encourages convergence (updates)
    L = np.zeros_like(X)  # Low Rank structure

    count = 0
    lambd = vallamb / np.sqrt(np.maximum(n1, n2))
    while (np.linalg.norm(X - L - S) > thresh) and (count < 1000):
        L = SVT(X - S + (1 / mu) * Y, 1 / mu)
        S = shrink(X - L + (1 / mu) * Y, lambd / mu)
        Y = Y + mu * (X - L - S)
        count += 1
    return L

