"""Single treated unit -- Pang, Liu & Xu (2022), Appendix Tables A6 and A7.

Pang, X., Liu, L., & Xu, Y. (2022), *A Bayesian Alternative to Synthetic
Control for Comparative Case Studies*, Political Analysis 30(2):269-288,
`10.1017/pan.2021.22 <https://doi.org/10.1017/pan.2021.22>`_. Replication
package: Harvard Dataverse `10.7910/DVN/B6SWA1
<https://doi.org/10.7910/DVN/B6SWA1>`_.

The paper's appendix runs three single-treated-unit designs and reports bias,
standard deviation, RMSE, coverage and runtime for synth, gsynth and the
Bayesian DM-LFM over six panel sizes each. Two of the three drivers are in
hand -- ``9_sim_single_r8.R`` (Table A6: eight weak factors, no covariates)
and ``10_sim_single_X.R`` (Table A7: three strong factors plus six covariates)
-- and this case covers those two. The third, ``sim_single_r3``, is Table A5;
its driver was not in the supplied material, so its published cells are left
unpinned instead of run against a guessed specification.

The published cells sit in ``benchmarks/reference/dmlfm_germany``, extracted
from the authors' own saved ``tempdata/sim_single_{X,r8}.RData``. The companion
case ``dmlfm_germany`` is the empirical half of the same paper.

The designs generate no treatment effect
----------------------------------------
``simulateCalib.R`` builds ``eff`` as a zero matrix and the block that would
have filled it is commented out, so ``true.eff`` is identically zero in both
drivers. The tables are therefore placebo studies: the bias column is the mean
estimate and the coverage column is coverage of zero. That fixes two targets
without the paper -- bias to zero and coverage to 0.95 -- and it is how the
cells here are read.

The arms
--------
``gsynth`` is ``force = "time"``, no cross-validation, and the rank the driver
hands it: 8 in the ``r8`` design, which is the true rank, and 7 in the ``X``
design, which is not. There the true factor count is 3 and the covariate terms
carry rank of their own -- six time-invariant covariates whose coefficients
move with independent AR(1) series, four of them with non-zero loadings, plus
the time-constant part. The driver's ``r + 4`` is over-specification, not the
true rank, and reading it as "gsynth with r known" misdescribes the cell.

``DM-LFM`` is ``re = "time"``, ``r = 10``, shrinkage on the factor loadings
only, and no covariates -- the driver passes ``Xname = NULL`` in both designs,
so the Bayesian arm never sees the covariates it is being credited with
handling.

What the published cells say about themselves
---------------------------------------------
``rmse^2 = (n-1)/n sd^2 + bias^2`` inverts to the replication count, and on
these cells it returns an integer to 2e-9 in all 54 rows. For ``synth`` and
``gsynth`` that integer is 500 everywhere, the drivers' ``nsims``. For the
Bayesian arm six cells return 498 or 499: those replications failed inside the
driver's ``tryCatch`` and dropped out of an ``na.rm = TRUE`` summary. The
``fail`` column that counted them is in ``storage`` and the table script never
prints it, taking columns 1, 3, 4 and 6 and skipping 5.

The coverage column confirms it independently. Multiplied by the count the
identity implies, it is an integer in all 36 cells to 2e-13; multiplied by a
flat 500 it misses by as much as 0.14. Two columns computed from different
quantities agree on the same per-cell denominator.

What the chain costs, and where it is measured
----------------------------------------------
The sampler runs at ``niter = 1500`` here against the drivers' 10000, and the
reference runs pblasso at both lengths on the same eight panels so the cost of
that is measured and not argued.

It costs nothing at this design. The posterior mean moves by 0.16 on average
and 0.48 at worst, against a sampling standard deviation near 3. The credible
interval does not narrow: the width ratio is 1.013 on average, between 0.965
and 1.042, and the short chain is the wider of the two on six of the eight
panels.

The expectation going in was the opposite, from a single panel run at three
chain lengths, where the width climbed 9.79, 10.19, 10.50 over 1000, 2000 and
5000 draws. Eight panels put that sequence inside one chain's own scatter. Had
the case been built on it, the coverage cells below would carry a downward
correction with nothing behind it.

The generator is transcribed here, not read from R, and pinned against the
authors' own on six moments at four panel sizes. Both sides average over the
time effect when those moments are taken, because the drivers hold a single
draw of it fixed within a case and its deterministic part climbs by four a
period; held fixed it would decide the within-unit moments, and the comparison
would report that draw and not the generator. The treated unit's own demeaned
variance is reported separately from the other five, since one unit's variance
over thirty periods does not average down at any affordable number of draws.

The one substitution is the AR(1) draw: ``arima.sim`` becomes an explicit
stationary AR(1), matching in distribution and not path by path. The panel
moments cannot police that substitution, which is why ``ar1_moments.csv``
exists. Starting the recursion at zero instead of at the stationary value
costs 1.9 per cent of E[x^2] at length 30, and once the loadings, the
covariates and an error standard deviation of 5 have diluted it, it moves the
cross-sectional moment by about 1 per cent -- inside the noise, and measured:
600 draws of the correct generator already scatter by 2 per cent on the same
statistic. On the series alone the separation is fortyfold, 0.04 per cent
against 1.9, so that is where it is checked.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd

from mlsynth import DMLFM, GSYNTH

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REF = _ROOT / "benchmarks" / "reference" / "pang_liu_xu_sims"
_CELLS = _ROOT / "benchmarks" / "reference" / "dmlfm_germany" / "montecarlo_single_treated.csv"

AR_COEF = 0.6
NITER, BURN = 1500, 375
SIMS = 60
#: draws the generator seam takes on the Python side; the reference takes 4000
PY_DRAWS = 600
DMLFM_KW = dict(covariates=None, re="time", r=10, ar1=True, niter=NITER, burn=BURN,
                flasso=1, xlasso=0, zlasso=0, alasso=0)

#: (r, lambda.sd, p, beta, the rank the driver hands gsynth)
DESIGNS = {
    "r8": dict(r=8, lambda_sd=2.0, p=0, beta=None, gsynth_r=8),
    "X": dict(r=3, lambda_sd=4.0, p=6, beta=(4.0, 3.0, 2.0, 1.0, 0.0, 0.0), gsynth_r=7),
}
#: the Monte Carlo runs the drivers' case 1, Nco = 30 and T0 = 20.
MC_NCO, MC_T0, MC_POST = 30, 20, 10

EXPECTED = {
    # ---- the generator, against the authors' simulateCalib ----
    # the AR(1) series on their own, where the start-value substitution shows
    "ar1_second_moment_max_rel_diff": (0.0, 0.005),
    # five panel moments that average well; the treated unit's own is separate
    # below because one unit's variance over thirty periods does not
    "dgp_moment_max_rel_diff": (0.0, 0.035),
    "dgp_treated_moment_max_rel_diff": (0.0, 0.09),
    "dgp_treated_count_always_one": (1.0, 0.0),
    # measured on the authors' code over 2000 panels, not read off the source
    "reference_max_abs_effect": (0.0, 0.0),
    # ---- the estimators, on the panels the R arms saw ----
    "gsynth_att_max_abs_diff": (0.0, 0.05),
    "dmlfm_att_max_abs_diff": (0.0, 1.5),
    "dmlfm_ci_width_ratio": (1.0, 0.25),
    "n_seam_panels": (8.0, 0.0),
    # ---- Table A6, Nco = 30 and T0 = 20: gsynth -0.034/3.448/0.974, DM-LFM 0.068/3.087/0.918
    "r8_gsynth_bias": (-0.034, 1.1),
    "r8_gsynth_rmse": (3.448, 0.85),
    "r8_gsynth_cover": (0.974, 0.10),
    "r8_dmlfm_bias": (0.068, 1.1),
    "r8_dmlfm_rmse": (3.087, 0.85),
    "r8_dmlfm_cover": (0.918, 0.12),
    # ---- Table A7, Nco = 30 and T0 = 20: gsynth 0.229/3.713/0.958, DM-LFM 0.335/3.693/0.906
    "X_gsynth_bias": (0.229, 1.1),
    "X_gsynth_rmse": (3.713, 0.95),
    "X_gsynth_cover": (0.958, 0.10),
    "X_dmlfm_bias": (0.335, 1.1),
    "X_dmlfm_rmse": (3.693, 0.95),
    "X_dmlfm_cover": (0.906, 0.12),
    # ---- what the published cells themselves say ----
    # rmse^2 = (n-1)/n sd^2 + bias^2 inverts to the replication count, and it
    # comes back an integer to 2e-9 in all 54 rows
    "published_replication_count": (500.0, 0.0),
    # the DM-LFM rows come back short: six cells lost one or two replications
    # to pblasso failures the drivers swallow in a tryCatch, and the fail
    # column that recorded them is in storage but never printed
    "published_dmlfm_dropped_max": (2.0, 0.0),
    # the coverage column divides by that same per-cell count, not by 500
    "published_coverage_denominator_agrees": (1.0, 0.0),
    # ---- what the designs are for ----
    # A6 is where the paper claims DM-LFM's advantage: many weak factors. The
    # published pair is 3.087 against 3.448, a 10% edge.
    "r8_dmlfm_rmse_edge": (0.36, 0.75),
    # A7 has three strong factors and the published pair is a tie: 3.693 to 3.713.
    "X_rmse_gap_abs": (0.02, 0.75),
    # the chain length, measured on the reference side at the authors' settings:
    # 1500 draws neither narrows nor widens the interval by 5 per cent
    "pblasso_ci_width_ratio_1500_over_10000": (1.013, 0.05),
    "pblasso_att_max_abs_shift_1500_over_10000": (0.48, 0.20),
    "n_mc_replications": (float(2 * SIMS), 0.0),
}


# --------------------------------------------------------------------------- #
# The generator, transcribed from the authors' code/simulateCalib.R.
# --------------------------------------------------------------------------- #

def _ar1(rng: np.random.Generator, TT: int, phi: float = AR_COEF,
         draws: int | None = None) -> np.ndarray:
    """Stationary AR(1) -- the distribution ``arima.sim(ar = phi, n = TT)`` draws from.

    ``draws`` returns that many independent paths as a ``(TT, draws)`` array.
    The recursion and the starting distribution are the ones the panels use, so
    the seam against ``ar1_moments.csv`` measures this code and not a second
    copy of it.
    """
    shape = () if draws is None else (draws,)
    x = np.empty((TT,) + shape)
    x[0] = rng.normal(0.0, 1.0 / np.sqrt(1.0 - phi ** 2), shape)
    for t in range(1, TT):
        x[t] = phi * x[t - 1] + rng.normal(size=shape)
    return x


def _drift(rng: np.random.Generator, TT: int) -> np.ndarray:
    """``getTS(type = "drift")``: an AR(1) plus a deterministic half-unit climb."""
    return _ar1(rng, TT) + 0.5 * np.arange(TT)


def simulate(rng, N, TT, T0, r, lambda_sd, time_eff, p=0, beta=None,
             error_sd=5.0, tr_noise=1.0, tr_coef=0.1):
    """One panel: force = 2, mu = 0, and a treatment effect of exactly zero."""
    lam = rng.normal(0.0, lambda_sd, size=(N, r))
    F = np.column_stack([_ar1(rng, TT) for _ in range(r)])

    fit = np.zeros((TT, N))
    if p > 0:
        W = rng.normal(size=(N, p))                            # time.invariant = TRUE
        for i in range(p):
            xi = _ar1(rng, TT) * beta[i]                       # the driver's own xi
            fit += np.outer(beta[i] + xi, W[:, i])

    ps = tr_coef * (lam[:, 0] + lam[:, 1]) + rng.normal(0.0, tr_noise, N)
    if p > 0:
        ps = ps + tr_coef * (W[:, 0] + W[:, 1])
    # tr.threshold = (N-1)/N keeps exactly the top-ranked unit, and only that one
    treated = int(np.argmax(ps))

    Y = (F @ lam.T + fit + np.outer(time_eff, np.ones(N))
         + rng.normal(0.0, error_sd, size=(TT, N)))
    D = np.zeros((TT, N))
    D[T0:, treated] = 1.0

    d = pd.DataFrame({"id": np.repeat(np.arange(101, 101 + N), TT),
                      "time": np.tile(np.arange(1, TT + 1), N),
                      "Y": Y.T.ravel(), "D": D.T.ravel()})
    if p > 0:
        for i in range(p):
            d[f"X{i + 1}"] = np.repeat(W[:, i], TT)
    return d


def _design(name: str) -> dict:
    return {k: v for k, v in DESIGNS[name].items() if k != "gsynth_r"}


def _moments(Y: np.ndarray, trcol: int, T0: int) -> list[float]:
    dm = Y - Y.mean(axis=1, keepdims=True)
    return [Y.ravel().var(ddof=1), Y.var(axis=1, ddof=1).mean(),
            Y[:T0, trcol].var(ddof=1), dm.var(axis=0, ddof=1).mean(),
            dm[:, trcol].var(ddof=1), Y[:, trcol].mean()]


# --------------------------------------------------------------------------- #
# The arms.
# --------------------------------------------------------------------------- #

def _fit_gsynth(d: pd.DataFrame, r: int):
    res = GSYNTH({"df": d, "outcome": "Y", "unitid": "id", "time": "time", "treat": "D",
                  "r": r, "r_max": max(r, 10), "force": "time", "inference": True,
                  "seed": 1, "display_graphs": False}).fit()
    return res[0] if isinstance(res, list) else res


def _fit_dmlfm(d: pd.DataFrame, seed: int):
    return DMLFM({"df": d, "outcome": "Y", "unitid": "id", "time": "time", "treat": "D",
                  "seed": seed, "display_graphs": False, **DMLFM_KW}).fit()


def run() -> dict:
    warnings.simplefilter("ignore")
    out: dict = {}

    # ---------------- the generator against the authors' ---------------- #
    ar_ref = pd.read_csv(_REF / "ar1_moments.csv")
    rng = np.random.default_rng(31415)
    out["ar1_second_moment_max_rel_diff"] = float(max(
        abs(float((_ar1(rng, int(row.TT), draws=int(row.draws)) ** 2).mean())
            / float(row.mean_x2) - 1.0)
        for _, row in ar_ref.iterrows()))

    ref_mom = pd.read_csv(_REF / "dgp_moments.csv")
    cols = ["var_y", "var_within_time", "var_treated_pre",
            "var_within_unit_demeaned", "var_treated_demeaned", "mean_treated"]
    treated_col = cols.index("var_treated_demeaned")
    steady = [i for i in range(len(cols)) if i != treated_col]
    rng = np.random.default_rng(20220301)
    rel, rel_tr, one_treated = [], [], []
    for _, row in ref_mom.iterrows():
        N, TT, T0 = int(row.N), int(row.TT), int(row.T0)
        cfg = _design(row.design)
        acc = []
        for _ in range(PY_DRAWS):
            te = _drift(rng, TT) * 8
            d = simulate(rng, N, TT, T0, time_eff=te, **cfg)
            Y = d.Y.to_numpy().reshape(N, TT).T
            flags = d.groupby("id", sort=False).D.max().to_numpy()
            one_treated.append(flags.sum() == 1)
            acc.append(_moments(Y, int(np.flatnonzero(flags == 1)[0]), T0))
        r = np.abs(np.mean(acc, axis=0) / row[cols].to_numpy(float) - 1.0)
        rel.append(r[steady].max())
        rel_tr.append(r[treated_col])
    out["dgp_moment_max_rel_diff"] = float(max(rel))
    out["dgp_treated_moment_max_rel_diff"] = float(max(rel_tr))

    eff = pd.read_csv(_REF / "effect_column.csv")
    out["reference_max_abs_effect"] = float(eff.max_abs_eff.abs().max())
    out["dgp_treated_count_always_one"] = float(
        all(one_treated) and eff.min_treated_units.min() == 1
        and eff.max_treated_units.max() == 1)

    # ---------------- the estimators on the R-drawn panels ---------------- #
    ref = pd.read_csv(_REF / "seam_reference.csv")
    g_gap, d_gap, width, n_panels = [], [], [], 0
    for name in DESIGNS:
        panels = pd.read_csv(_REF / f"seam_panels_{name}.csv")
        n_panels += panels.panel.nunique()
        for tag, block in panels.groupby("panel", sort=True):
            d = block.drop(columns="panel").reset_index(drop=True)
            rg = ref[(ref.panel == tag) & (ref.arm == "gsynth")].iloc[0]
            g_gap.append(abs(float(_fit_gsynth(d, DESIGNS[name]["gsynth_r"]).effects.att)
                             - float(rg.att)))
            rb = ref[(ref.panel == tag) & (ref.arm == "pblasso")
                     & (ref.niter == NITER)].iloc[0]
            res = _fit_dmlfm(d, seed=1)
            d_gap.append(abs(float(res.effects.att) - float(rb.att)))
            width.append((float(res.inference.ci_upper) - float(res.inference.ci_lower))
                         / (float(rb.ci_u) - float(rb.ci_l)))
    out["gsynth_att_max_abs_diff"] = float(max(g_gap))
    out["dmlfm_att_max_abs_diff"] = float(max(d_gap))
    out["dmlfm_ci_width_ratio"] = float(np.mean(width))
    out["n_seam_panels"] = float(n_panels)

    short = ref[(ref.arm == "pblasso") & (ref.niter == NITER)].set_index("panel")
    long_ = ref[(ref.arm == "pblasso") & (ref.niter == 10000)].set_index("panel")
    out["pblasso_ci_width_ratio_1500_over_10000"] = float(
        np.mean((short.ci_u - short.ci_l) / (long_.ci_u - long_.ci_l)))
    out["pblasso_att_max_abs_shift_1500_over_10000"] = float(
        (short.att - long_.att).abs().max())

    # ---------------- the Monte Carlo, against the published cells ---------------- #
    N, TT, T0 = MC_NCO + 1, MC_T0 + MC_POST, MC_T0
    reps = 0
    for name in ("r8", "X"):
        cfg = _design(name)
        rng = np.random.default_rng(90210 + sum(map(ord, name)))
        te = _drift(rng, TT) * 8        # drawn once per case, as the drivers do
        g_est, g_cov, b_est, b_cov = [], [], [], []
        for i in range(SIMS):
            d = simulate(rng, N, TT, T0, time_eff=te, **cfg)
            g = _fit_gsynth(d, DESIGNS[name]["gsynth_r"])
            g_est.append(float(g.effects.att))
            g_cov.append(float(g.inference.ci_lower) <= 0.0 <= float(g.inference.ci_upper))
            b = _fit_dmlfm(d, seed=i + 1)
            b_est.append(float(b.effects.att))
            b_cov.append(float(b.inference.ci_lower) <= 0.0 <= float(b.inference.ci_upper))
            reps += 1
        for arm, est, cov in (("gsynth", g_est, g_cov), ("dmlfm", b_est, b_cov)):
            e = np.asarray(est, float)
            out[f"{name}_{arm}_bias"] = float(e.mean())          # the truth is zero
            out[f"{name}_{arm}_rmse"] = float(np.sqrt((e ** 2).mean()))
            out[f"{name}_{arm}_cover"] = float(np.mean(cov))
    out["n_mc_replications"] = float(reps)

    out["r8_dmlfm_rmse_edge"] = out["r8_gsynth_rmse"] - out["r8_dmlfm_rmse"]
    out["X_rmse_gap_abs"] = abs(out["X_gsynth_rmse"] - out["X_dmlfm_rmse"])

    # ---------------- what the published cells themselves say ---------------- #
    cells = pd.read_csv(_CELLS)
    exact = cells.sd ** 2 / (cells.sd ** 2 + cells.bias ** 2 - cells.rmse ** 2)
    n = exact.round()
    out["published_replication_count"] = float(n.max())
    out["published_dmlfm_dropped_max"] = float(
        n.max() - n[cells.method == "bayes"].min())
    has_cover = cells.cover.notna()
    k = cells.cover[has_cover] * n[has_cover]
    out["published_coverage_denominator_agrees"] = float(
        (exact - n).abs().max() < 1e-6 and (k - k.round()).abs().max() < 1e-6)
    return out
