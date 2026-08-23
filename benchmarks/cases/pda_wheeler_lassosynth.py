"""PDA cross-validation: the resampled cumulative band against Wheeler's LassoSynth.

``cumulative_method="resample"`` builds the cumulative band by accumulating
resampled conformity scores. That construction is Andrew Wheeler's, from his
LassoSynth post: fit a positive lasso on the donors, take leave-one-out conformity
scores, mirror them into a symmetric pool (``np.concatenate([cs, -cs])``), draw one
pool element per post period, accumulate, and read equal-tailed quantiles off the
accumulated paths.

    https://github.com/apwheele/Blog_Code/tree/master/Python/LassoSynth

mlsynth generalises it in one respect: it draws whole blocks and flips each block's
sign, because a panel's period errors are serially correlated and drawing periods
independently understates how fast a running total's variance grows. The
generalisation has to reduce to the original, and that reduction is what this case
pins.

Three quantities, in decreasing strength of claim.

``parity_max_gap`` is the exact one. Wheeler's construction and mlsynth's, given
the *same* conformity scores and ``block=1``, are the same distribution: the
multiset ``{+e_i, -e_i}`` is the multiset ``{+|e_i|, -|e_i|}``, so mirroring the
pool and sign-flipping a centred score coincide. The case runs Wheeler's code path
and mlsynth's ``block_error_paths`` on one score vector and reports the largest
endpoint disagreement as a fraction of band width -- Monte Carlo noise and nothing
else, at four tenths of a percent.

It is measured from horizon two, because horizon one is not yet a continuous
distribution. With ``m`` scores the one-period band is a quantile of ``2m``
discrete atoms, so both implementations land exactly on an atom and sampling noise
decides which. ``parity_h1_atoms_apart`` records how far apart those atoms are, in
atoms, and pins that the horizon-one difference is discreteness, not
disagreement: adjacent atoms, not a shifted distribution. From horizon two the
convolution smooths the pool and the gap collapses.

``width_ratio`` is the end-to-end one and is weaker on purpose. Run whole, the two
differ in where the scores come from -- Wheeler calibrates on leave-one-out
residuals over the pre-period points, mlsynth on rolling-origin out-of-sample
windows of exactly the reported horizon. Those are different calibration sets, so
the bands are not expected to agree to Monte Carlo error, only to be of a
comparable size.

It comes out above one, and the direction is the informative part. A leave-one-out
residual is scored by a model that saw every pre-period point but one, so it
measures interpolation; a rolling-origin window is scored by a model trained only
on what came before it, from as few as a third of the pre-period. The second is
the harder question and gives the larger errors, so the mlsynth band is the wider
of the two. The two calibration sets are not interchangeable for a cumulative
horizon, and this records by how much.

``block_widening`` is the generalisation working. On a persistent panel the
whole-horizon block must give a wider band than the independent draw, since that
is the correlation the blocks exist to carry.

Cross-validation (scenario: reference implementation available). Wheeler's
construction is transcribed here instead of imported, since the post ships a
script and not a package; the transcription is the four lines quoted above.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_T0, _T1, _N0, _ALPHA, _NSIM = 40, 8, 10, 0.10, 200_000


def _panel(seed=0, rho=0.0, effect=1.0):
    """A factor panel with one treated unit adopting at ``_T0``."""
    rng = np.random.default_rng(seed)
    T = _T0 + _T1
    f = rng.standard_normal((T, 2))
    load = rng.standard_normal((_N0, 2)) * 0.5
    e = rng.standard_normal((_N0 + 1, T)) * 0.3
    if rho:
        for t in range(1, T):
            e[:, t] = rho * e[:, t - 1] + np.sqrt(1 - rho ** 2) * e[:, t]
    y = f @ load.mean(axis=0) + e[0]
    y[_T0:] += effect
    rows = [{"unit": "treated", "t": t, "y": float(y[t]), "d": int(t >= _T0)}
            for t in range(T)]
    for j in range(_N0):
        s = f @ load[j] + e[j + 1]
        rows += [{"unit": f"d{j}", "t": t, "y": float(s[t]), "d": 0} for t in range(T)]
    return pd.DataFrame(rows)


def _wide(df):
    w = df.pivot(index="t", columns="unit", values="y").sort_index()
    donors = [c for c in w.columns if c != "treated"]
    return w["treated"].to_numpy(), w[donors].to_numpy()


def _wheeler_scores(y, X):
    """Leave-one-out residuals, as MAPIE ``cv=-1`` supplies them to Wheeler.

    Signed, not absolute. Wheeler takes ``abs`` before mirroring, which is a no-op:
    mirroring symmetrises either way, and ``{+r, -r}`` is ``{+|r|, -|r|}`` as a
    multiset. Keeping the sign matters for the comparison, because mlsynth centres
    what it is given, and centring absolute values shrinks their spread where
    centring signed residuals does not.
    """
    from sklearn.linear_model import Lasso, LassoCV

    ypre, Xpre = y[:_T0], X[:_T0]
    a = LassoCV(fit_intercept=True, positive=True, max_iter=5000).fit(Xpre, ypre).alpha_
    mk = lambda: Lasso(alpha=a, fit_intercept=True, positive=True, max_iter=5000)
    cs = np.array([
        ypre[i] - mk().fit(np.delete(Xpre, i, 0), np.delete(ypre, i))
        .predict(Xpre[i:i + 1])[0]
        for i in range(_T0)
    ])
    full = mk().fit(Xpre, ypre)
    return cs, y[_T0:] - full.predict(X[_T0:])


def _wheeler_band(scores, tau, seed=0):
    """Wheeler's cumulative band, transcribed: mirror the pool, draw, accumulate."""
    rng = np.random.default_rng(seed)
    pool = np.concatenate([scores, -scores])
    L = tau.size
    sims = np.cumsum(tau.reshape(L, 1) + rng.choice(pool, (L, _NSIM)), axis=0)
    return np.quantile(sims, [_ALPHA / 2, 1 - _ALPHA / 2], axis=1)


def _mlsynth_band_from_scores(scores, tau, block=1, seed=1):
    """The same accumulation, through mlsynth's block draw."""
    from mlsynth.utils.conformal import block_error_paths

    paths = block_error_paths(scores - scores.mean(), horizon=tau.size, block=block,
                              n_sim=_NSIM, rng=np.random.default_rng(seed))
    cum = np.cumsum(tau[None, :] + paths, axis=1)
    return np.quantile(cum, [_ALPHA / 2, 1 - _ALPHA / 2], axis=0)


def _fitted_band(df, **kw):
    from mlsynth import PDA

    cfg = dict(df=df, outcome="y", treat="d", unitid="unit", time="t",
               method="LASSO", display_graphs=False, cumulative_band=True,
               cumulative_method="resample", cumulative_n_sim=20_000)
    cfg.update(kw)
    res = PDA(cfg).fit()
    fit = res.fits["lasso"] if isinstance(res.fits, dict) else res.fits[0]
    return fit.cumulative_band


def run() -> dict:
    warnings.filterwarnings("ignore")
    df = _panel()
    y, X = _wide(df)

    # (1) exact construction parity, on one shared score vector at block = 1
    scores, tau = _wheeler_scores(y, X)
    scores = scores - scores.mean()
    lo_w, hi_w = _wheeler_band(scores, tau)
    lo_m, hi_m = _mlsynth_band_from_scores(scores, tau)
    width = np.abs(hi_w - lo_w)
    parity = float(max(
        max(abs(lo_m[h] - lo_w[h]), abs(hi_m[h] - hi_w[h])) / width[h]
        for h in range(1, tau.size)))                  # horizon 1 is discrete

    # horizon one: both endpoints sit on an atom of the 2m-point pool, so the
    # question is how many atoms apart, not how far apart
    atoms = np.sort(np.unique(np.concatenate([scores, -scores]))) + tau[0]
    h1_apart = float(abs(int(np.argmin(np.abs(atoms - lo_w[0])))
                         - int(np.argmin(np.abs(atoms - lo_m[0])))))

    # (2) end to end: different calibration sets, comparable widths
    fitted = _fitted_band(df, cumulative_block=1)
    mlsynth_half = float(fitted.upper[-1] - fitted.lower[-1]) / 2.0
    wheeler_half = float(hi_w[-1] - lo_w[-1]) / 2.0

    # (3) the generalisation: blocks carry persistence the iid draw discards
    persistent = _panel(seed=3, rho=0.7)
    iid = _fitted_band(persistent, cumulative_block=1)
    blocked = _fitted_band(persistent, cumulative_block=0)

    return {
        "parity_max_gap": parity,
        "parity_h1_atoms_apart": h1_apart,
        "width_ratio": mlsynth_half / wheeler_half,
        "block_widening": float(blocked.se[-1] / iid.se[-1]),
    }


# Deterministic: fixed panel seeds, fixed draw seeds, 200k paths. parity_max_gap is
# Monte Carlo noise on a shared pool, so its tolerance is set by the draw count and
# not by the method. width_ratio and block_widening are properties of the two
# calibration sets and of the DGP's persistence, so theirs absorb solver and RNG
# drift.
EXPECTED = {
    "parity_max_gap": (0.004, 0.012),        # the two constructions coincide at block=1
    "parity_h1_atoms_apart": (1.0, 1.0),     # horizon 1 differs by at most one pool atom
    "width_ratio": (1.43, 0.35),             # rolling-origin scores exceed leave-one-out
    "block_widening": (1.31, 0.20),          # blocks carry AR(0.7) persistence
}
