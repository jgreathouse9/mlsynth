#!/usr/bin/env python3
"""Live reference run: the TASC authors' ``TimeAwareSC`` on California Prop 99.

Cross-validation reference for mlsynth's ``TASC`` estimator. Runs the authors'
own implementation -- ``srho1/tasc`` (Rho, Illick, Narasipura, Abadie, Hsu &
Misra, "Time-Aware Synthetic Control," AISTATS 2026, arXiv:2601.03099) -- on the
Abadie-Diamond-Hainmueller (2010) Proposition 99 panel and emits the values the
mlsynth case pins against.

Driving (matches the authors' own California test,
``test/prop99_aistats_final/prop99_aistats_final_run_tests.py::run_california_test_no_cim``):
``set_seed(1)``, latent dimension ``d = 2``, ``naive`` initialization with
``random_seed=1``, and ``N1 = 1000`` pre-period EM iterations (``em_pre`` only).
The target unit (California, row 0) post-period outcomes are treated as missing;
``make_prediction()`` returns the smoother-based counterfactual whose first
column is the target's path.

Panel: the same matrix the mlsynth case uses -- ``basedata/smoking_data.csv``
(the 39-state ADH Prop 99 pool, 1970-2000; California treated from 1989),
California as row 0 and the 38 donor states as rows 1..38, in the authors'
``Y in R^{N x T}`` orientation. So both implementations see the identical ``Y``,
the identical ``T0`` and the identical latent dimension ``d``.

The reference code ships without a LICENSE, and pulling its full package would
drag in tensorflow / causalimpact, so the four modules this driver needs are
vendored under ``vendor/`` with a ``NOTICE`` recording provenance; nothing here
is redistributed beyond what the cross-validation requires.

Run via ``python benchmarks/reference/generate.py tasc_prop99`` (the manifest's
``command`` invokes this file). It prints a ``== REFERENCE VALUES ==`` block of
tab-separated key/value lines, then ``== SESSION INFO ==``.
"""
from __future__ import annotations

import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE / "vendor"))

import torch  # noqa: E402
from tasc import TimeAwareSC, set_seed  # noqa: E402  (vendored authors' code)

# --- Panel: the exact matrix the mlsynth case uses -------------------------
START_YEAR, INTERVENTION_YEAR = 1970, 1989
D = 2          # latent state dimension (authors' California choice)
N1 = 1000      # pre-period EM iterations (em_pre)
SEED = 1


def prequential_loglik(A, H, Q, R, m0, P0, Y_pre) -> float:
    """Gaussian data log-likelihood of ``Y_pre`` under the LGSSM, via the
    Kalman innovations decomposition, in double precision.

    This is the proper data log-likelihood ``log p(Y_pre | theta)`` -- the same
    quantity for whichever ``theta`` is passed in -- so it is an apples-to-apples
    yardstick for comparing the optimum two EM implementations reach (the
    reference's own ``log_likelihood`` uses a different float32 ``Q``-function
    form, so we report this one for the cross-implementation comparison). Higher
    is a better fit. The state is initialized as the model's first emission step
    (predict ``m0`` one step), matching the authors' ``y_seq`` convention where
    the latent chain starts one step before the first observation.

    Parameters
    ----------
    A, H, Q, R, m0, P0 : array-like
        State-space parameters; shapes ``(d,d)``, ``(N,d)``, ``(d,d)``,
        ``(N,N)``, ``(d,)``, ``(d,d)``.
    Y_pre : np.ndarray
        Pre-period outcome matrix ``(N, T0)``.
    """
    A = np.asarray(A, float); H = np.asarray(H, float); Q = np.asarray(Q, float)
    R = np.asarray(R, float); m0 = np.asarray(m0, float); P0 = np.asarray(P0, float)
    Np, T0p = Y_pre.shape
    m = A @ m0
    P = A @ P0 @ A.T + Q
    ll = 0.0
    for t in range(T0p):
        y = Y_pre[:, t]
        v = y - H @ m
        S = H @ P @ H.T + R
        S = 0.5 * (S + S.T) + 1e-8 * np.eye(Np)
        _, logdet = np.linalg.slogdet(S)
        Sinv = np.linalg.inv(S)
        ll += -0.5 * (Np * np.log(2 * np.pi) + logdet + v @ Sinv @ v)
        K = P @ H.T @ Sinv
        m = m + K @ v
        P = P - K @ H @ P
        m = A @ m
        P = A @ P @ A.T + Q
    return float(ll)


def build_Y() -> tuple[np.ndarray, np.ndarray, int]:
    df = pd.read_csv(_ROOT / "basedata" / "smoking_data.csv")
    wide = df.pivot(index="year", columns="state", values="cigsale")
    years = wide.index.to_numpy()
    T0 = int((years < INTERVENTION_YEAR).sum())
    target = wide["California"].to_numpy(dtype=float)
    donors = wide.drop(columns="California").to_numpy(dtype=float).T  # (n, T)
    Y = np.vstack([target.reshape(1, -1), donors])                   # (N, T)
    return Y, target, T0


def main() -> int:
    Y, target, T0 = build_Y()
    T = Y.shape[1]

    set_seed(SEED)
    model = TimeAwareSC(Y=torch.tensor(Y, dtype=torch.float32), d=D,
                        dtype=torch.float32)
    model.initialize_theta(method="naive", random_seed=SEED)
    model.T0 = T0
    model.em_pre(T0=T0, N1=N1)

    log_like = float(model.log_likelihood(T=T0).item())
    with torch.no_grad():
        target_pred, _, _ = model.make_prediction()
    cf = target_pred.detach().numpy()             # length T, target counterfactual

    cf_pre, cf_post = cf[:T0], cf[T0:]
    obs_post = target[T0:]
    att = float(np.mean(obs_post - cf_post))
    pre_rmse = float(np.sqrt(np.mean((cf_pre - target[:T0]) ** 2)))

    # Apples-to-apples data log-likelihood of the reference fit (double
    # precision innovations form); the mlsynth case computes the same quantity
    # for its own fit and the higher value reached the better optimum.
    preq_ll = prequential_loglik(
        model.A.detach().numpy(), model.H.detach().numpy(),
        model.Q.detach().numpy(), model.R.detach().numpy(),
        model.m0.detach().numpy(), model.P0.detach().numpy(),
        Y[:, :T0],
    )

    print("== REFERENCE VALUES ==")
    post_years = list(range(INTERVENTION_YEAR, START_YEAR + T))
    for yr, c in zip(post_years, cf_post):
        print(f"cf_{yr}\t{float(c):.10f}")
    print(f"tasc_att\t{att:.10f}")
    print(f"tasc_pre_rmse\t{pre_rmse:.10f}")
    print(f"tasc_pre_loglik\t{log_like:.10f}")
    print(f"tasc_pre_loglik_prequential\t{preq_ll:.10f}")

    print("== SESSION INFO ==")
    print(f"python\t{platform.python_version()}")
    print(f"torch\t{torch.__version__}")
    print(f"numpy\t{np.__version__}")
    print(f"pandas\t{pd.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
