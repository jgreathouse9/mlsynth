r"""Shi, Sridhar, Misra & Blei (2022): what the fine-grained model says about SC.

Path B (scenario 3 -- full replication archive), plus a property layer. Covers
the simulation study of Shi, C., Sridhar, D., Misra, V. and Blei, D. M., "On the
Assumptions of Synthetic Control Methods", AISTATS 2022 (PMLR 151), whose code
is at github.com/claudiashi57/fine-grained-SC.

The paper re-formulates synthetic control at the level of individuals inside
units instead of the units themselves. Each group is a mixture over ``K = 12``
individual types; at each period 2000 individuals are drawn from a group's
mixture, pushed through one of six type-specific nonlinear and time-varying
functions (``sin``, ``cos``, ``sqrt``, ``log``, ``x**0.8``, ``x**1.2``), and the
group-level series is their average. Linearity at the group level is then a
consequence of aggregation and not an assumption about individuals.

The object that decides whether the fitted weights carry to the post period is
the minimal invariant set :math:`S`: loosely, the directions in which the
target's mixture differs from the donors'. Its cardinality :math:`|S|` is a
knob in the simulation and is not observable to any estimator.

What is reproduced
------------------

Table 2 exactly, all six cells. The paper's own estimator there is unconstrained
least squares with no intercept, fit on the first ``3T/4`` periods and scored on
the rest, and every replication is seeded with ``np.random.seed(i)``, so the
table is deterministic, not matched within Monte Carlo noise.

============================  ================  ================
arm                           observed          counterfactual
============================  ================  ================
outcome only                  0.070 (.07)       0.139 (.14)
suitable covariates           0.059 (.06)       0.127 (.13)
unsuitable covariates         0.060 (.06)       0.241 (.24)
============================  ================  ================

Figure 3's claim is reproduced as a direction, since the figure ships as a
notebook plot. With five donors the counterfactual error is flat through
:math:`|S| \le 5` and then climbs by two orders of magnitude, 0.23 to 36.4 at
:math:`|S| = 11`.

The paper also reports that the counterfactual error grows "at a significantly
faster rate than the observed error". Measured over 40 draws that holds near the
donor count and not across the whole range. The ratio of counterfactual to
observed error runs 2.4 at :math:`|S| = 5`, peaks at 25.2 at :math:`|S| = 8`,
and falls back to 2.3 at :math:`|S| = 11`, because past the boundary the
in-sample fit degrades too. In absolute terms the counterfactual error does gain
more across the range, 36.2 against 15.8. Both forms are pinned here, since the
ratio form is the one that fails if it is stated globally.

Where the paper's evidence stops
--------------------------------

The paper is explicit that its simulations use ordinary least squares as the SC
estimator, and its theory is about identification, for which that is the natural
vehicle. Its Prop 99 figure uses the constrained estimator instead. mlsynth
ships the constrained family: ``VanillaSC`` is the simplex, and ``TSSC``'s
variants keep non-negativity throughout. So this case also measures what the
simulation says about the estimator a practitioner would actually run.

Under the simplex the :math:`|S|` story largely disappears from the median and
is replaced by a different failure. Measured over 40 draws at each
:math:`|S|`, the median counterfactual error moves between 0.2 and 0.7 across
:math:`|S| = 2 \ldots 10` with no trend, while the mean ranges from 34 to 566
and is not monotone in :math:`|S|` at all. The reason is convex-hull
membership: the data-generating process places the target group's aggregate
inside the donors' per-period range in only 30% to 60% of draws, and that share
is itself not monotone in :math:`|S|`. Least squares extrapolates through a hull
failure; the simplex cannot, so a handful of draws carry the mean.

The single sharpest draw at :math:`|S| = 2`: the target lies inside the donor
range in 1 period of 20, the simplex leaves a pre-period MSE of 455 and a
counterfactual MSE of 1227, and least squares on the same panel scores 0.16.

None of that contradicts the paper. It bounds how far the simulation evidence
carries: the :math:`|S|` diagnosis is a statement about least squares on this
design, and for a simplex-constrained estimator hull membership dominates it.
The case pins both, so a change to either estimator moves a recorded number.

Provenance
----------

* Paper: Shi, Sridhar, Misra & Blei (2022), AISTATS, PMLR 151. Table 2 is the
  only numeric target the paper prints; Figures 2 and 3 are notebook plots.
* Archive: ``src/dgp.py`` (the data-generating process and both estimators),
  ``src/auxiliary_covariate.ipynb`` (Table 2), ``src/increase_s.ipynb``
  (Figure 3), ``src/linear_v_nonlinear.ipynb`` (Figure 2).
* The DGP below is a transcription of ``dgp.py``. It was checked against the
  archive module directly: ``gen_probs`` agrees at ``|S|`` of 2, 5 and 11,
  ``make_matrix`` agrees under all four summaries, and ``ols_losses`` agrees
  with the archive's ``fit_regression`` to eight decimals, so the transcription
  reproduces the original's random-number stream and not only its shape. The
  archive is not vendored here, so that check is not a test in the suite.
* The legacy ``numpy.random`` global seeding of the original is kept, since the
  table is reproduced cell for cell and any other stream would change it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

K = 12          # individual types
N_IND = 2000    # individuals drawn per group per period
N_DONORS = 5

# ---------------------------------------------------------------------------
# The data-generating process, transcribed from the archive's src/dgp.py.
# ---------------------------------------------------------------------------

_LINKS = (np.sin, np.cos, np.sqrt, np.log,
          lambda x: x ** 0.8, lambda x: x ** 1.2)


def _outcomes(x, t, num):
    """Individual outcomes for type ``num`` at period ``t``, and the covariate."""
    t = (t + 1) / 10 + t % 10
    if num > 5:
        fn = _LINKS[int(num - 6)]
        res, aux = fn(t * x), t * x
    else:
        fn = _LINKS[num]
        res, aux = fn(t + x), t + x
    return res + np.random.randn(res.shape[0]), aux + np.random.randn(aux.shape[0])


def gen_probs(e, k, s):
    """Mixture over the ``k`` types for the target and each of ``e`` donors.

    ``s`` controls how far the donors' support is from the target's: the target
    is Dirichlet over every type, each donor keeps a type with probability
    ``1 - s/k``. Larger ``s`` means sparser, more different donors.
    """
    e = e + 1
    probs = []
    for i in range(e):
        if i == 0:
            probs.append(np.random.dirichlet(np.ones(k), size=1)[0])
            continue
        p = 1 - (s / k)
        index = np.random.binomial(size=k, n=1, p=p)
        while np.sum(index) == 0:
            index = np.random.binomial(size=k, n=1, p=p)
        alphas = index + 0.001
        q = np.random.dirichlet(alphas, size=1)[0]
        while np.isnan(q)[0]:
            q = np.random.dirichlet(alphas, size=1)[0]
        probs.append(q * index / np.sum(q * index))
    return probs


def make_matrix(probs, k, T, n, summary="aggregate"):
    """Group-level panel ``(1 + e, T)``: the chosen summary of each group's draw."""
    tmp = abs(np.random.randn(T)).reshape(-1, 1)
    matrix = np.zeros((len(probs), T))
    for state in range(len(probs)):
        idx_state = np.random.choice(np.arange(1, k + 1), size=n, p=probs[state])
        for t in range(T):
            t_pop, tmps = [], []
            for i in range(k):
                index = idx_state == i + 1
                if np.sum(index) > 0:
                    res, aux = _outcomes(idx_state[index] * tmp[t], t, i)
                    t_pop.append(res)
                    tmps.append(aux)
            pop = np.concatenate(t_pop)
            if summary == "aggregate":
                matrix[state][t] = np.mean(pop)
            elif summary == "median":
                matrix[state][t] = np.quantile(pop, 0.5)
            elif summary == "aux_good":
                matrix[state][t] = np.mean(np.sin(pop) + np.random.randn(pop.shape[0]))
            elif summary == "aux_bad":
                matrix[state][t] = np.mean(np.concatenate(tmps)) / 10
    return matrix


def ols_losses(matrix, T, train=None):
    """The paper's estimator: least squares, no intercept, split at ``3T/4``."""
    donors, target = np.asarray(matrix[1:]).T, np.asarray(matrix[0])
    train = int(3 * T / 4) if train is None else train
    coef, *_ = np.linalg.lstsq(donors[:train], target[:train], rcond=None)
    pre = float(np.mean((donors[:train] @ coef - target[:train]) ** 2))
    post = float(np.mean((donors[train:] @ coef - target[train:]) ** 2))
    return pre, post


# ---------------------------------------------------------------------------


def _panel(matrix, T0):
    """Group panel -> long frame, group 0 treated after ``T0`` with no effect."""
    n, T = matrix.shape
    unit = np.repeat(np.arange(n), T)
    period = np.tile(np.arange(1, T + 1), n)
    return pd.DataFrame({"unit": unit, "time": period, "y": matrix.reshape(-1),
                         "treat": ((unit == 0) & (period > T0)).astype(int)})


def _simplex_losses(matrix, T0):
    """``VanillaSC`` on the same panel: pre- and post-period MSE."""
    from mlsynth.estimators.vanillasc import VanillaSC

    y = matrix[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC(dict(df=_panel(matrix, T0), outcome="y", treat="treat",
                             unitid="unit", time="time",
                             display_graphs=False)).fit()
    cf = np.asarray(res.time_series.counterfactual_outcome, float)
    return (float(np.mean((y[:T0] - cf[:T0]) ** 2)),
            float(np.mean((y[T0:] - cf[T0:]) ** 2)))


def _in_hull(matrix, T):
    """Is the target inside the donors' per-period range at every period?"""
    return float(all(matrix[1:, t].min() <= matrix[0, t] <= matrix[1:, t].max()
                     for t in range(T)))


def _table2(sim=100):
    """Table 2: outcome-only against suitable and unsuitable auxiliary covariates."""
    s, T, a = 5, 15, 10
    train = int(3 * T / 4)
    tr = [[], [], []]
    te = [[], [], []]
    for i in range(sim):
        np.random.seed(i)
        probs = gen_probs(N_DONORS, K, s)
        base = make_matrix(probs, K, T, N_IND, summary="aggregate")
        good = make_matrix(probs, K, T, N_IND, summary="aux_good")
        bad = make_matrix(probs, K, T, N_IND, summary="aux_bad")
        m1 = np.concatenate([base[:, :train], good[:, :a], base[:, train:]], 1)
        m2 = np.concatenate([base[:, :train], bad[:, :a], base[:, train:]], 1)
        for j, (mat, tn) in enumerate(((base, None), (m1, train + a), (m2, train + a))):
            pre, post = ols_losses(mat, T, train=tn)
            tr[j].append(pre)
            te[j].append(post)
    return ([float(np.mean(x)) for x in tr], [float(np.mean(x)) for x in te])


def _sweep(s_values, sim=40, T=20):
    """OLS and simplex losses, and hull membership, at each ``|S|``."""
    T0 = int(3 * T / 4)
    out = {}
    for s in s_values:
        rows = []
        for h in range(sim):
            np.random.seed(h)
            probs = gen_probs(N_DONORS, K, s)
            matrix = make_matrix(probs, K, T, N_IND)
            o_pre, o_post = ols_losses(matrix, T)
            s_pre, s_post = _simplex_losses(matrix, T0)
            rows.append((o_pre, o_post, s_pre, s_post, _in_hull(matrix, T)))
        d = np.asarray(rows, float)
        out[s] = {"ols_obs": float(d[:, 0].mean()), "ols_cf": float(d[:, 1].mean()),
                  "sc_obs_med": float(np.median(d[:, 2])),
                  "sc_cf_med": float(np.median(d[:, 3])),
                  "sc_cf_mean": float(d[:, 3].mean()),
                  "hull": float(d[:, 4].mean())}
    return out


def run() -> dict:
    obs, cf = _table2(sim=100)
    sweep = _sweep([2, 5, 8, 11], sim=40)

    low, high = sweep[2], sweep[11]
    out = {
        # --- Table 2, the paper's own estimator ------------------------------
        "t2_obs_outcome_only": round(obs[0], 3),
        "t2_obs_suitable": round(obs[1], 3),
        "t2_obs_unsuitable": round(obs[2], 3),
        "t2_cf_outcome_only": round(cf[0], 3),
        "t2_cf_suitable": round(cf[1], 3),
        "t2_cf_unsuitable": round(cf[2], 3),
        # Unsuitable covariates hurt the counterfactual while leaving the
        # observed fit alone, which is the table's point.
        "unsuitable_hurts_counterfactual": float(cf[2] > 1.5 * cf[0]),
        "unsuitable_looks_fine_in_sample": float(obs[2] <= obs[0]),
        # --- Figure 3's direction, under least squares ------------------------
        "ols_cf_s2": round(low["ols_cf"], 3),
        "ols_cf_s11": round(high["ols_cf"], 3),
        "ols_cf_blowup_ratio": round(high["ols_cf"] / max(low["ols_cf"], 1e-9), 1),
        # The counterfactual error gains more in absolute terms than the
        # observed error does across the range.
        "ols_cf_absolute_rise_exceeds_obs": float(
            (high["ols_cf"] - low["ols_cf"]) > (high["ols_obs"] - low["ols_obs"])),
        # The gap between them peaks just past the donor count and then closes,
        # which is where the paper's "faster rate" claim actually lives.
        "cf_over_obs_s5": round(sweep[5]["ols_cf"] / max(sweep[5]["ols_obs"], 1e-9), 1),
        "cf_over_obs_s8": round(sweep[8]["ols_cf"] / max(sweep[8]["ols_obs"], 1e-9), 1),
        "cf_over_obs_s11": round(high["ols_cf"] / max(high["ols_obs"], 1e-9), 1),
        "cf_obs_gap_peaks_past_donors": float(
            (sweep[8]["ols_cf"] / max(sweep[8]["ols_obs"], 1e-9))
            > max(sweep[5]["ols_cf"] / max(sweep[5]["ols_obs"], 1e-9),
                  high["ols_cf"] / max(high["ols_obs"], 1e-9))),
        # --- what the simplex does on the same panels -------------------------
        "sc_cf_med_s2": round(low["sc_cf_med"], 3),
        "sc_cf_med_s11": round(high["sc_cf_med"], 3),
        # the mean is carried by hull failures, so it dwarfs the median
        "sc_mean_over_median_s2": round(
            low["sc_cf_mean"] / max(low["sc_cf_med"], 1e-9), 0),
        "hull_rate_s2": round(low["hull"], 2),
        "hull_rate_s11": round(high["hull"], 2),
        "hull_rate_max": round(max(v["hull"] for v in sweep.values()), 2),
    }
    return out


# Deterministic: every replication is seeded with the archive's own
# ``np.random.seed(i)``, so re-running returns identical numbers on a given
# NumPy. The Table 2 tolerances are the printed precision of the paper's own
# cells; the sweep tolerances absorb the heavy right tail of the MSE
# distribution at 40 draws, which is wide by construction and is itself one of
# the findings.
EXPECTED = {
    # Table 2 of the paper prints .07/.06/.06 observed and .14/.13/.24
    # counterfactual. All six reproduce.
    "t2_obs_outcome_only": (0.070, 0.005),
    "t2_obs_suitable": (0.059, 0.005),
    "t2_obs_unsuitable": (0.060, 0.005),
    "t2_cf_outcome_only": (0.139, 0.010),
    "t2_cf_suitable": (0.127, 0.010),
    "t2_cf_unsuitable": (0.241, 0.020),
    "unsuitable_hurts_counterfactual": (1.0, 0.0),
    "unsuitable_looks_fine_in_sample": (1.0, 0.0),
    # Figure 3: with five donors, |S| = 11 is far past the point where the
    # weights stop generalising.
    "ols_cf_s2": (0.230, 0.15),
    "ols_cf_s11": (36.4, 25.0),
    "ols_cf_blowup_ratio": (158.0, 120.0),
    "ols_cf_absolute_rise_exceeds_obs": (1.0, 0.0),
    "cf_over_obs_s5": (2.4, 2.0),
    "cf_over_obs_s8": (25.2, 18.0),
    "cf_over_obs_s11": (2.3, 2.0),
    "cf_obs_gap_peaks_past_donors": (1.0, 0.0),
    # The simplex does not show the same curve in the median.
    "sc_cf_med_s2": (0.68, 0.50),
    "sc_cf_med_s11": (4.68, 4.00),
    "sc_mean_over_median_s2": (50.0, 45.0),
    # Hull membership is what binds, and it never gets close to certain.
    "hull_rate_s2": (0.30, 0.20),
    "hull_rate_s11": (0.42, 0.25),
    "hull_rate_max": (0.60, 0.25),
}
