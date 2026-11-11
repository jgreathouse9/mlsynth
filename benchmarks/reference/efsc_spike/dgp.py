"""Data-generating processes for the EFSC spike.

Follows Rodriguez-Deniz & Blei (2026), Appendix B.2 (generative model),
C.3 (one-parameter tilts), C.5 (Gaussian benchmarks) and C.6 (placebo panels).

Where the paper leaves a scale unstated the choice is recorded here:

* C.5 does not give the Gaussian generating scales. The defaults below
  (unit 0.15, time 0.25, factors 0.35, rank 2, shared across the two
  components) reproduce the ranges in the paper's Figure 14:
  ``eta[1]`` within about +-0.6 and ``eta[2]`` within about [-2, -0.55].
* The multinomial panel is not in the paper's synthetic section. It is a
  Medicaid-shaped panel (five categories, the last as reference) built for the
  head-to-head with COMPSC; its scales are this spike's choice.

Every generator returns ``(panel, truth)``. ``panel`` is an ``efsc`` PanelData;
``truth`` carries the untreated and treated natural parameters, the masks, the
cell sizes and a long DataFrame for the mlsynth baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from efsc.panel import prepare_panel_data


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _factors(rng, n, T, r, s_unit, s_time, s_fac):
    alpha = rng.normal(0.0, s_unit, n)
    gamma = rng.normal(0.0, s_time, T)
    theta = rng.normal(0.0, s_fac, (n, r))
    beta = rng.normal(0.0, s_fac, (T, r))
    z = alpha[:, None] + gamma[None, :] + theta @ beta.T
    return z, theta, beta


def _cell_sizes(rng, n, T, m_fixed, lam):
    if m_fixed is not None:
        return np.full((n, T), int(m_fixed))
    return 1 + rng.poisson(lam, (n, T))


def _masks(n, T, treated_idx, post):
    treated_post = np.zeros((n, T), bool)
    treated_post[np.ix_(treated_idx, np.arange(T - post, T))] = True
    return treated_post


def _long_frame(values, cell_index, n, T):
    unit = cell_index // T
    time = cell_index % T
    return pd.DataFrame({"y": values, "unit": unit, "time": time})


def _treated_units(rng, n, k, theta, select):
    if select == "last":
        return np.arange(n - k, n)
    if select == "loading":
        # Confounded assignment: the units with the largest first loading.
        return np.sort(np.argsort(theta[:, 0])[-k:])
    raise ValueError(select)


def gaussian_panel(
    n=32, T=128, r=2, *, m_fixed=None, lam=55, treated=None, post=None,
    intervention="tilt", tau=0.0, kappa=(0.0, 0.0),
    scales=(0.15, 0.25, 0.35), seed=0,
):
    """Two-parameter Gaussian panel (Appendix C.5)."""
    rng = np.random.default_rng(seed)
    treated = treated if treated is not None else int(round(0.2 * n))
    post = post if post is not None else int(round(0.2 * T))
    z, theta, beta = _factors(rng, n, T, r, *scales)
    eta = np.stack([z, -np.exp(z)], axis=-1)  # shared factorisation, eq. 128-130
    treated_idx = np.arange(n - treated, n)
    tp = _masks(n, T, treated_idx, post)

    eta_t = eta.copy()
    if intervention == "tilt":
        eta_t[..., 0] = np.where(tp, eta[..., 0] + tau, eta[..., 0])
    elif intervention == "structured":
        w1 = rng.normal(size=r); w1 /= np.linalg.norm(w1)
        w2 = rng.normal(size=r); w2 /= np.linalg.norm(w2)
        g = _sigmoid(theta @ w1)[:, None] * _sigmoid(beta @ w2)[None, :]
        eta_t[..., 0] = np.where(tp, eta[..., 0] + kappa[0] * g, eta[..., 0])
        eta_t[..., 1] = np.where(tp, eta[..., 1] - kappa[1] * g, eta[..., 1])
    else:
        raise ValueError(intervention)

    sig2 = -1.0 / (2.0 * eta_t[..., 1])
    mu = eta_t[..., 0] * sig2
    m = _cell_sizes(rng, n, T, m_fixed, lam)
    cell = np.repeat(np.arange(n * T), m.ravel())
    y = mu.ravel()[cell] + np.sqrt(sig2.ravel()[cell]) * rng.standard_normal(cell.size)
    frame = _long_frame(y, cell, n, T)
    panel = prepare_panel_data(frame, outcome="y", unit="unit", time="time", family="gaussian")
    truth = dict(eta_ctrl=eta, eta_treat=eta_t, treated_post=tp,
                 counterfactual_mask=~tp, treated_idx=treated_idx, post=post,
                 m=m, frame=frame, theta=theta, beta=beta)
    return panel, truth


def poisson_panel(
    n=32, T=64, r=2, *, m_fixed=None, lam=55, treated=6, post=13, tau=0.0,
    baseline=1.0, scales=(0.05, 0.05, 0.05), select="last", seed=0,
):
    """One-parameter Poisson panel (Appendix C.3 / C.6)."""
    rng = np.random.default_rng(seed)
    z, theta, beta = _factors(rng, n, T, r, *scales)
    eta = (baseline + z)[..., None]
    treated_idx = _treated_units(rng, n, treated, theta, select)
    tp = _masks(n, T, treated_idx, post)
    eta_t = eta.copy()
    eta_t[..., 0] = np.where(tp, eta[..., 0] + tau, eta[..., 0])
    m = _cell_sizes(rng, n, T, m_fixed, lam)
    cell = np.repeat(np.arange(n * T), m.ravel())
    y = rng.poisson(np.exp(eta_t[..., 0]).ravel()[cell]).astype(float)
    frame = _long_frame(y, cell, n, T)
    panel = prepare_panel_data(frame, outcome="y", unit="unit", time="time", family="poisson")
    truth = dict(eta_ctrl=eta, eta_treat=eta_t, treated_post=tp,
                 counterfactual_mask=~tp, treated_idx=treated_idx, post=post,
                 m=m, frame=frame, theta=theta, beta=beta)
    return panel, truth


# Medicaid-shaped composition: uninsured, Medicaid, employer, private, other.
BASE_PROBS = np.array([0.40, 0.30, 0.17, 0.08, 0.05])


def multinomial_panel(
    n=40, T=12, r=1, *, m_fixed=100, treated=16, post=6,
    shift=(-0.4, 0.6, 0.0, 0.0), scales=(0.3, 0.1, 0.3), seed=0,
):
    """Multinomial panel with C-1 reference-category logits (C.7.3 model)."""
    rng = np.random.default_rng(seed)
    C = BASE_PROBS.size
    base = np.log(BASE_PROBS[:-1] / BASE_PROBS[-1])
    logits = np.empty((n, T, C - 1))
    for c in range(C - 1):  # a separate factorisation per logit, as in C.7.3
        z, _, _ = _factors(rng, n, T, r, *scales)
        logits[..., c] = base[c] + z
    treated_idx = np.arange(n - treated, n)
    tp = _masks(n, T, treated_idx, post)
    logits_t = logits + np.where(tp[..., None], np.asarray(shift)[None, None, :], 0.0)

    def probs(lg):
        full = np.concatenate([lg, np.zeros(lg.shape[:-1] + (1,))], axis=-1)
        full = np.exp(full - full.max(axis=-1, keepdims=True))
        return full / full.sum(axis=-1, keepdims=True)

    p_ctrl, p_treat = probs(logits), probs(logits_t)
    m = np.full((n, T), int(m_fixed))
    counts = np.stack([[rng.multinomial(m[i, j], p_treat[i, j]) for j in range(T)] for i in range(n)])
    cats = np.arange(C)
    cell = np.repeat(np.arange(n * T), m.ravel())
    labels = np.concatenate([np.repeat(cats, counts[i, j]) for i in range(n) for j in range(T)])
    frame = _long_frame(labels, cell, n, T)
    panel = prepare_panel_data(frame, outcome="y", unit="unit", time="time",
                               family="multinomial", categories=list(cats))
    truth = dict(eta_ctrl=logits, eta_treat=logits_t, p_ctrl=p_ctrl, p_treat=p_treat,
                 counts=counts, treated_post=tp, counterfactual_mask=~tp,
                 treated_idx=treated_idx, post=post, m=m, frame=frame)
    return panel, truth
