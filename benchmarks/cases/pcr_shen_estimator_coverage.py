r"""Coverage of the confidence intervals PCR actually returns.

``rsc_shen_coverage`` validates the Shen et al. (2023) variance algebra: it calls
``_var_homo`` and ``_var_jack`` directly, on coefficients from ``sklearn``, and
compares them cell-for-cell against the authors' ``var.py``. That is a check on
the formulas, and it needs the authors' repository, so it skips offline.

This case asks the other question. It runs the paper's data-generating process
through :func:`~mlsynth.utils.clustersc_helpers.pcr.inference.shen_inference` --
the function a ``CLUSTERSC`` fit reaches -- so the rank truncation, the PCR
weights, the batched variance and the interval assembly are all inside what is
being measured. Nothing is downloaded: the panel ships in ``basedata``.

The data-generating process
---------------------------
Section 5.2.1 of the paper, at the West Germany study. The donor block over the
pre-period is replaced by its rank-``k`` approximation, ``k`` being the smallest
number of singular values holding 99.9 per cent of the spectral energy, and that
low-rank block is held fixed across replications. Each replication draws

.. code-block:: text

    y_n  ~ N(Y0 beta,   sigma_n I)      the treated unit's pre-period
    y_t  ~ N(Y0' alpha, sigma_t I)      the donors, at each post-period

with ``alpha`` and ``beta`` the horizontal and vertical regressions of the
observed data, and the two noise levels taken from the residual off each
projector, as the paper specifies.

The three estimands follow Theorem 3: ``mu_hz = <y_n, Hv alpha>`` is random
through the treated unit's own history, ``mu_vt = <y_t, Hu beta>`` through the
donors, and ``mu_dr = <alpha, Y0' beta>`` is fixed. The treated unit's
post-period outcome is set to the estimand under test, so the reported gap is
``estimand - prediction`` and its interval covers exactly when it contains zero.
That is the paper's coverage criterion, expressed in the gaps this function
returns.

What it finds
-------------
Per period, the shipped path is calibrated: all nine pairings of variance
estimator and estimand land between 0.926 and 0.952 against a nominal 0.95.

The three variance estimators are not separable here. Online-appendix Lemma 8
establishes the jackknife is conservative, and a 300-replication run appeared to
show it covering highest; at 2000 replications that ordering reverses, and the
spread across estimators is inside two Monte Carlo standard errors. So this case
records that they agree at this design and makes no claim about their ordering.
Separating them needs a design where the errors are heteroskedastic, which this
DGP's homoskedastic draw cannot supply.

The ATT interval is a different matter, and it is the reason this case exists.
``shen_inference`` forms it as ``sqrt(mean(v_t) / T1)``, an assumption the code
already flags as the library's own -- the paper derives no multi-period closed
form. That assumption holds under the horizontal model and fails under the
vertical one, because ``w_vt`` is computed once from ``y_n`` and reused at every
post-period, so the vertical gaps are driven by a single estimated weight vector
and do not average. The measured ratio of the true spread of the ATT error to
the reported standard error is about ``sqrt(T1)`` under VT and about one under
HZ, so the vertical ATT interval is too narrow by roughly the whole averaging
factor it claims. Its coverage falls from 0.95 at one post-period to under 0.5
at ten. The doubly robust interval, mixing the two, sits in between.

This case pins that failure rather than hiding it. Correcting the ATT variance
is a separate change; see the issue this case was written for.
"""
from __future__ import annotations

_PANEL = "german_reunification.csv"
_TREATED = "West Germany"
_T_TREAT = 1990
_ENERGY = 0.999          # spectral energy retained, as in Section 5.2.1
_N_ITERS = 2000
_T1_GRID = (1, 5, 10)
_VARIANCES = ("homoskedastic", "jackknife", "hrk")
_SEED = 0


def _spectral_rank(sv, energy: float = _ENERGY) -> int:
    """Smallest rank holding ``energy`` of the spectral mass."""
    import numpy as np

    cum = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    return int(np.searchsorted(cum, energy) + 1)


def _calibrate(n_post: int):
    """The paper's DGP, fitted to the shipped West Germany panel."""
    from pathlib import Path

    import numpy as np
    import pandas as pd

    base = Path(__file__).resolve().parents[2] / "basedata" / _PANEL
    wide = pd.read_csv(base).pivot(index="year", columns="country", values="gdp")
    donors = [c for c in wide.columns if c != _TREATED]

    pre = wide.index < _T_TREAT
    Y0_obs = wide.loc[pre, donors].values                    # (T0, J)
    y_n_obs = wide.loc[pre, _TREATED].values                 # (T0,)
    post = wide.loc[~pre, donors].values[:n_post]            # (T1, J)

    U, sv, Vt = np.linalg.svd(Y0_obs, full_matrices=False)
    k = _spectral_rank(sv)
    U_k, Vt_k = U[:, :k], Vt[:k]
    Y0 = (U_k * sv[:k]) @ Vt_k                               # rank-k, held fixed
    P_time = U_k @ U_k.T                                     # (T0, T0)
    P_unit = Vt_k.T @ Vt_k                                   # (J, J)

    beta = np.linalg.lstsq(Y0_obs, y_n_obs, rcond=None)[0]
    alphas = np.array([np.linalg.lstsq(Y0_obs.T, post[t], rcond=None)[0]
                       for t in range(post.shape[0])])
    T0, J = Y0_obs.shape
    import numpy.linalg as la
    sigma_n = la.norm((np.eye(T0) - P_time) @ y_n_obs) ** 2 / max(T0 - k, 1)
    sigma_t = la.norm((np.eye(J) - P_unit) @ post[0]) ** 2 / max(J - k, 1)
    return dict(Y0=Y0, P_time=P_time, P_unit=P_unit, beta=beta, alphas=alphas,
                k=k, sigma_n=sigma_n, sigma_t=sigma_t, T0=T0, J=J)


def _replicate(cal, rng):
    """One draw: the simulated panel and the three estimands on it."""
    import numpy as np

    Y0, beta, alphas = cal["Y0"], cal["beta"], cal["alphas"]
    T1 = alphas.shape[0]
    y_n = Y0 @ beta + rng.normal(0.0, np.sqrt(cal["sigma_n"]), cal["T0"])
    Y_post = np.array([Y0.T @ alphas[t]
                       + rng.normal(0.0, np.sqrt(cal["sigma_t"]), cal["J"])
                       for t in range(T1)])
    estimands = {
        "hz": np.array([y_n @ (cal["P_time"] @ alphas[t]) for t in range(T1)]),
        "vt": np.array([Y_post[t] @ (cal["P_unit"] @ beta) for t in range(T1)]),
        "dr": np.array([alphas[t] @ (Y0 @ beta) for t in range(T1)]),
    }
    return np.vstack([Y0, Y_post]), y_n, estimands


def _measure(n_post: int, variance: str, n_iters: int = _N_ITERS, seed: int = _SEED):
    """Coverage of each interval for its own estimand, through the shipped path."""
    import numpy as np

    from mlsynth.utils.clustersc_helpers.pcr.inference import shen_inference

    cal = _calibrate(n_post)
    rng = np.random.default_rng(seed)
    per_hit = {s: 0 for s in ("hz", "vt", "dr")}
    att_hit = {s: 0 for s in ("hz", "vt", "dr")}
    att_err = {s: [] for s in ("hz", "vt", "dr")}
    att_rep = {s: [] for s in ("hz", "vt", "dr")}
    n_per = 0
    for _ in range(n_iters):
        donors, y_n, estimands = _replicate(cal, rng)
        for source, mu in estimands.items():
            res = shen_inference(np.concatenate([y_n, mu]), donors, cal["T0"],
                                 cal["k"], variance=variance)
            ci = getattr(res, f"per_period_ci_{source}")
            per_hit[source] += int(np.sum((ci[:, 0] <= 0.0) & (0.0 <= ci[:, 1])))
            lo, hi = getattr(res, f"att_ci_{source}")
            att_hit[source] += int(lo <= 0.0 <= hi)
            att_err[source].append(res.att)          # the truth here is zero
            att_rep[source].append(getattr(res, f"att_se_{source}"))
        n_per += n_post
    out = {}
    for s in ("hz", "vt", "dr"):
        out[f"per_{s}"] = per_hit[s] / n_per
        out[f"att_{s}"] = att_hit[s] / n_iters
        out[f"ratio_{s}"] = float(np.std(att_err[s]) / np.mean(att_rep[s]))
    return out


def run() -> dict:
    import numpy as np

    out = {}
    # Arm 1: per-period coverage, every variance estimator, one post-period.
    for variance in _VARIANCES:
        m = _measure(1, variance)
        tag = variance[:4]
        for source in ("hz", "vt", "dr"):
            out[f"per_{tag}_{source}"] = m[f"per_{source}"]
    # Every pairing lands near nominal, and the estimators are not separable at
    # this design; the band is pinned, their ordering deliberately is not.
    cells = [out[f"per_{v[:4]}_{s}"] for v in _VARIANCES for s in ("hz", "vt", "dr")]
    out["per_min_coverage"] = min(cells)
    out["per_max_coverage"] = max(cells)

    # Arm 2: ATT coverage against the post-period length.
    for n_post in _T1_GRID:
        m = _measure(n_post, "homoskedastic")
        for source in ("hz", "vt", "dr"):
            out[f"att{n_post}_{source}"] = m[f"att_{source}"]
        if n_post > 1:
            out[f"ratio{n_post}_vt_over_sqrt_t1"] = (m["ratio_vt"]
                                                     / float(np.sqrt(n_post)))
            out[f"ratio{n_post}_hz"] = m["ratio_hz"]
    # At one post-period the ATT is the single gap, so the two arms must agree
    # exactly; a drift here means the aggregation changed, not the coverage.
    out["att1_matches_per_period"] = max(
        abs(out[f"att1_{s}"] - out[f"per_homo_{s}"]) for s in ("hz", "vt", "dr"))
    out["att_vt_falls_with_horizon"] = out["att1_vt"] - out["att10_vt"]
    return out


# Filled from a run of this module. The replication count is 2000, so the Monte
# Carlo standard error is about 0.005 near the nominal rate and about 0.011 at
# the low end; tolerances are roughly four of those, wide enough to survive a
# different BLAS and tight enough that a real change in the intervals trips them.
# The ratio metrics are noisier, being a ratio of two estimated spreads.
EXPECTED = {
    # Arm 1 -- per-period, through shen_inference, every variance estimator.
    "per_homo_hz": (0.9295, 0.03),
    "per_homo_vt": (0.9460, 0.03),
    "per_homo_dr": (0.9340, 0.03),
    "per_jack_hz": (0.9460, 0.03),
    "per_jack_vt": (0.9415, 0.03),
    "per_jack_dr": (0.9520, 0.03),
    "per_hrk_hz": (0.9285, 0.03),
    "per_hrk_vt": (0.9260, 0.03),
    "per_hrk_dr": (0.9325, 0.03),
    "per_min_coverage": (0.9260, 0.03),
    "per_max_coverage": (0.9520, 0.03),

    # Arm 2 -- the ATT interval against the post-period length. att*_hz stays at
    # nominal; att*_vt is the failure this case exists to record.
    "att1_hz": (0.9295, 0.03),
    "att1_vt": (0.9460, 0.03),
    "att1_dr": (0.9340, 0.03),
    "att5_hz": (0.9385, 0.03),
    "att5_vt": (0.6080, 0.06),
    "att5_dr": (0.8665, 0.06),
    "att10_hz": (0.9480, 0.03),
    "att10_vt": (0.4580, 0.06),
    "att10_dr": (0.7445, 0.06),

    # The diagnosis: the VT standard error is understated by almost exactly the
    # sqrt(T1) the aggregation assumes, and the HZ one is correctly sized. Both
    # are pinned at 1, so either drifting says the aggregation changed.
    "ratio5_vt_over_sqrt_t1": (1.006, 0.10),
    "ratio10_vt_over_sqrt_t1": (0.9828, 0.10),
    "ratio5_hz": (1.013, 0.10),
    "ratio10_hz": (0.9937, 0.10),

    # Internal consistency, exact: at one post-period the ATT is the single gap.
    "att1_matches_per_period": (0.0, 1e-12),
    # The direction, stated as its own quantity so a regression cannot pass by
    # moving both endpoints together.
    "att_vt_falls_with_horizon": (0.488, 0.08),
}
