"""On-demand fetch of SucreRouge's ``synth_control`` Bayesian Robust SC reference.

``SucreRouge/synth_control`` (https://github.com/SucreRouge/synth_control) is a
Python implementation of Robust Synthetic Control -- hard singular-value
thresholding (HSVT) of the donor matrix followed, in its ``method="bayesian"``
branch, by the closed-form Gaussian-conjugate posterior over the donor weights
of Amjad, Shah & Shen (*Robust Synthetic Control*, JMLR 19(22):1-51, 2018). It
ships without a licence file, so instead of vendoring it this helper fetches the
tree at a pinned commit into the gitignored ``benchmarks/reference/.cache`` and
imports the upstream ``synth_functions`` from there (mirroring
``clone_tslib`` / ``clone_proximal``). If git or the network is unavailable the
benchmark skips gracefully.

The Bayesian branch lives in ``synth_functions.learn(method="bayesian")``:

* ``threshold(donor_matrix, num_sv=k)`` -- rank-``k`` HSVT of the donor matrix
  (all periods; scaled by ``1/p_hat``, which is ``1`` on a fully observed panel);
* ``sigma^2 = var(y_pre, ddof=1)``, ``inv_var = 1 / sigma^2`` -- observation
  noise from the treated pre-series;
* ``prior_param = forward_chain(A, y, "ridge") * inv_var`` -- the prior precision
  is data-driven (a forward-chaining ridge-CV penalty), not the value the caller
  passes in (the ``bayesian`` branch overwrites it);
* posterior covariance ``Sigma = inv(prior_param I + inv_var A' A)`` and mean
  ``beta = inv_var Sigma A' y``.

:func:`build_bayesian_rsc` reproduces that branch on a supplied panel using the
upstream ``threshold`` and ``forward_chain`` unmodified, and returns the learned
weights together with the two data-driven plug-ins (``sigma^2`` and the prior
precision) the case needs to reconstruct the posterior with mlsynth's kernel.

The pinned commit (``_COMMIT``) freezes the reference; bump it deliberately.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Sequence

import numpy as np

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/SucreRouge/synth_control.git"
_COMMIT = "a55ee1482d5db8cda32e18327de0072e7ba5e0b6"  # master @ 2024
_CACHE = Path(__file__).resolve().parent / ".cache" / "synth_control"


def _ensure_clone() -> Path:
    """Fetch (or reuse) the ``synth_control`` tree pinned at ``_COMMIT``.

    The clone lands at ``.cache/synth_control`` so that ``import synth_functions``
    resolves there; returns that directory (added to ``sys.path``).
    """
    marker = _CACHE / "synth_functions.py"
    if not marker.exists():
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        fetch_pinned_repo(_REPO, _COMMIT, _CACHE)    # git clone, else codeload tarball
        if not marker.exists():  # pragma: no cover - defensive
            raise BenchmarkSkipped("synth_control clone missing synth_functions.py")
    return _CACHE


def import_synth_functions() -> ModuleType:
    """Import the upstream ``synth_functions`` module (threshold, forward_chain)."""
    cache = _ensure_clone()
    if str(cache) not in sys.path:
        sys.path.insert(0, str(cache))
    try:
        return importlib.import_module("synth_functions")
    except ImportError as exc:  # pragma: no cover - missing numpy/pandas/sklearn
        raise BenchmarkSkipped(
            f"reference synth_control import failed ({exc}); "
            f"install its deps (`pip install numpy scikit-learn scipy matplotlib`)"
        ) from exc


def build_bayesian_rsc(
    target_full: np.ndarray,
    donor_full: np.ndarray,
    donor_names: Sequence[str],
    year: int,
    k: int,
) -> Dict[str, object]:
    """Fit SucreRouge's Bayesian Robust SC and return weights + plug-ins.

    Reproduces ``synth_functions.learn(method="bayesian")`` using the upstream
    ``threshold`` and ``forward_chain`` unmodified: rank-``k`` HSVT of the full
    donor matrix, a data-driven prior precision, and the closed-form Gaussian
    posterior over the weights.

    Parameters
    ----------
    target_full : np.ndarray
        Treated unit's outcomes over all periods, shape ``(T,)``.
    donor_full : np.ndarray
        Donor outcomes, columns = donors, shape ``(T, J)``.
    donor_names : sequence of str
        Length-``J`` donor labels (column order of ``donor_full``).
    year : int
        Number of pre-intervention periods ``T0`` (the ``times.dep`` split).
    k : int
        Retained singular-value rank (``num_sv``).

    Returns
    -------
    dict
        ``weights`` (donor -> posterior-mean weight), ``obs_noise_var``
        (``sigma^2``), ``prior_precision`` (the data-driven ``prior_param``),
        the full-period ``counterfactual``, ``pre_rmse``, ``post_cf_mean``,
        ``att``, and ``k``.
    """
    sf = import_synth_functions()

    target_full = np.asarray(target_full, dtype=float).ravel()
    donor_full = np.asarray(donor_full, dtype=float)
    donor_names = [str(n) for n in donor_names]

    # Rank-k HSVT of the full donor matrix (units x periods), the reference's
    # convention: threshold operates on (J, T) and denoises all periods.
    M_hat = sf.threshold(donor_full.T, num_sv=k)          # (J, T)
    y = target_full[:year]
    A = M_hat[:, :year].T                                  # (T0, J)

    var = (1.0 / (len(y) - 1)) * np.sum((y - y.mean()) ** 2)
    inv_var = 1.0 / var
    prior_param = float(sf.forward_chain(A, y, "ridge") * inv_var)

    donor_size = A.shape[1]
    sigma_d = np.linalg.inv(prior_param * np.eye(donor_size) + inv_var * A.T.dot(A))
    beta = inv_var * np.dot(sigma_d, np.dot(A.T, y))

    cf = M_hat.T.dot(beta)                                 # (T,)
    pre_rmse = float(np.sqrt(np.mean((y - A.dot(beta)) ** 2)))
    post_cf_mean = float(np.mean(cf[year:]))
    att = float(np.mean(target_full[year:] - cf[year:]))

    return {
        "weights": dict(zip(donor_names, beta.tolist())),
        "obs_noise_var": float(var),
        "prior_precision": prior_param,
        "counterfactual": cf.tolist(),
        "pre_rmse": pre_rmse,
        "post_cf_mean": post_cf_mean,
        "att": att,
        "k": float(k),
    }
