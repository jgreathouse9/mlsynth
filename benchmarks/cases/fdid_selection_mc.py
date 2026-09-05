r"""FDID property case: forward-selection consistency (Li 2023).

Path C (property; scenario 1 -- paper only). Measures the two theoretical
results of the Forward DiD Web Appendix that concern the *algorithm*, not
the estimate:

* Propositions 2.2 / D.1 -- the subset the empirical forward selection
  algorithm returns coincides with one the theoretical algorithm would
  return, with probability approaching one as :math:`T_1 \to \infty`:
  :math:`\Pr(\widehat{\mathcal{U}} \subset \mathcal{U}^*) \to 1`.
* Lemma B.1 -- the pre-treatment intercept and error variance converge to
  their population values *uniformly over subsets*, at
  :math:`O_p(\sqrt{\log N / T_1})`.

Neither is a published table, so nothing here is a cell match. What is
pinned is the direction each quantity is claimed to move, measured against
a population benchmark computed in closed form by
:mod:`mlsynth.utils.fdid_helpers.population`, whose own agreement with the
simulator is pinned by ``mlsynth/tests/test_fdid_population.py``.

Provenance
----------

Li, Kathleen T. (2023), *"Frontiers: A Simple Forward
Difference-in-Differences Method"*, Marketing Science 43(2), Online
Appendices B and D (Propositions 2.2, D.1; Lemma B.1) and Web Appendix E
(the DGPs, shared with ``fdid_table5``).

What the two designs measure
----------------------------

Under DGP 2 the treated unit loads at :math:`c_0 = 1` and the donor pool
splits into a matched half (:math:`c_1 = 1`) and a mismatched half
(:math:`c_2 = 2`). The theoretical criterion
:math:`V_U = (c_0 - \bar c)^2 \sigma_S^2 + 1 + 1/|U|` then makes
:math:`\mathcal{U}^*` the matched half exactly -- admitting a mismatched
control buys a :math:`1/|U|` gain that the loading gap more than spends --
and that single subset is the target the empirical algorithm has to find.

Under DGP 1 every control matches, so :math:`V_U = 1 + 1/|U|` and
:math:`\mathcal{U}^*` is the whole pool. That optimum is nearly flat near
the top (:math:`1 + 1/14` against :math:`1 + 1/20` is a 2% difference), so
exact recovery is far out of reach on any grid this case can afford; what
is measured instead is that the selected pool grows toward :math:`N`. The
contrast between the two designs is the point: selection consistency
arrives quickly when the criterion separates the optimum sharply and slowly
when it does not, and the case records both, not only the flattering one.
"""

from __future__ import annotations

import itertools

import numpy as np

from mlsynth import FDID
from mlsynth.utils.fdid_helpers.population import (
    group_counts,
    prediction_variance,
    theoretical_forward_selection,
)
from mlsynth.utils.fdid_helpers.simulation import simulate_fdid_sample

# Draw counts. The paper's own Monte Carlo uses 10,000; these trade precision
# for a benchmark that runs in a couple of minutes. At M = 400 the standard
# error on a selection rate near 0.4 is 0.024, which the tolerances absorb.
M_SELECT = 400
M_LEMMA = 150

N_SELECT = 20        # donor pool for the selection designs
N_LEMMA = 8          # Lemma B.1 maxes over all 2^N - 1 subsets, so keep N small
T2 = 10

T1_GRID = (25, 50, 100, 200, 400, 800, 1600)
T1_GRID_LEMMA = (50, 100, 200, 400, 800, 1600)


def _fit_selected(sample) -> list[int]:
    """Donor indices FDID selects, read through the public result accessor."""
    res = FDID({"df": sample.df, "outcome": "y", "treat": "treat",
                "unitid": "unit", "time": "time",
                "display_graphs": False, "verbose": False}).fit()
    return list(res.fdid.selected_indices)


def _selection_curve(dgp: int, N: int, T1: int, M: int, seed: int = 0) -> dict:
    """Selection accuracy against ``U*`` at one ``T1``."""
    star = theoretical_forward_selection(dgp=dgp, N=N)
    target_n1, _ = next(iter(star.optimal_states))
    exact = matched = selected = 0.0
    for j in range(M):
        rng = np.random.default_rng(seed + j)
        idx = _fit_selected(
            simulate_fdid_sample(dgp=dgp, N=N, T1=T1, T2=T2, rng=rng))
        exact += star.contains(idx)
        n1, _ = group_counts(idx, N)
        matched += n1
        selected += len(idx)
    return {
        "exact": exact / M,
        # Share of the target group recovered, and share of the selection that
        # belongs to it -- the two ways an inexact selection can be wrong.
        "recall": matched / (M * target_n1),
        "precision": matched / max(selected, 1.0),
        "size_ratio": selected / (M * N),
    }


def _lemma_b1(dgp: int, N: int, T1: int, M: int, seed: int = 0) -> dict:
    r"""Uniform deviations over every non-empty subset, normalised by the rate.

    Lemma B.1 quantifies over :math:`\mathcal{U}`, the subsets satisfying
    parallel trends (Web Appendix A, eq. A.4) -- not over the path the
    greedy search happens to walk. Under these DGPs the two coincide with
    the full power set: the factors are stationary and zero-mean, so every
    subset's gap from the treated unit is a weakly dependent mean-zero
    process whatever its loading mismatch, and mismatch inflates
    :math:`\sigma^2_U` without costing parallel trends. Enumerating all
    :math:`2^N - 1` subsets is therefore exactly :math:`\mathcal{U}` here,
    and in any design where it were not, it would be a superset and so a
    stronger check. The subset averages are one matrix product against a
    membership matrix, so the cost is a matmul per draw.
    """
    subsets = [U for k in range(1, N + 1)
               for U in itertools.combinations(range(N), k)]
    membership = np.zeros((len(subsets), N))
    for row, U in enumerate(subsets):
        membership[row, list(U)] = 1.0 / len(U)

    # alpha_U = a_0 - 1 for every subset (the factors are zero-mean), and
    # sigma^2_U is the theoretical error variance.
    alpha_true = {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0}[dgp]
    var_true = np.array([prediction_variance(dgp, *group_counts(U, N))
                         for U in subsets])

    d_alpha, d_var = [], []
    for j in range(M):
        rng = np.random.default_rng(seed + j)
        s = simulate_fdid_sample(dgp=dgp, N=N, T1=T1, T2=5, rng=rng)
        resid = s.Y_treated[:T1][None, :] - membership @ s.Y_controls[:, :T1]
        alpha_hat = resid.mean(axis=1)
        var_hat = ((resid - alpha_hat[:, None]) ** 2).mean(axis=1)
        d_alpha.append(np.abs(alpha_hat - alpha_true).max())
        d_var.append(np.abs(var_hat - var_true).max())

    rate = np.sqrt(np.log(N) / T1)
    return {
        "alpha_dev": float(np.mean(d_alpha)),
        "var_dev": float(np.mean(d_var)),
        "alpha_ratio": float(np.mean(d_alpha)) / rate,
        "var_ratio": float(np.mean(d_var)) / rate,
    }


def _is_increasing(values) -> float:
    """1.0 when the sequence never decreases (Monte-Carlo noise aside)."""
    v = list(values)
    return float(all(b >= a - 0.02 for a, b in zip(v, v[1:])))


def run() -> dict:
    out: dict[str, float] = {}

    # --- Propositions 2.2 / D.1 -------------------------------------------
    dgp2 = {T1: _selection_curve(2, N_SELECT, T1, M_SELECT) for T1 in T1_GRID}
    for T1, m in dgp2.items():
        out[f"dgp2_exact_T{T1}"] = m["exact"]
        out[f"dgp2_recall_T{T1}"] = m["recall"]
    out["dgp2_precision_T1600"] = dgp2[1600]["precision"]
    out["dgp2_exact_increasing"] = _is_increasing(
        m["exact"] for m in dgp2.values())
    out["dgp2_recall_increasing"] = _is_increasing(
        m["recall"] for m in dgp2.values())

    dgp1 = {T1: _selection_curve(1, N_SELECT, T1, M_SELECT) for T1 in T1_GRID}
    for T1 in (25, 1600):
        out[f"dgp1_size_ratio_T{T1}"] = dgp1[T1]["size_ratio"]
    out["dgp1_size_increasing"] = _is_increasing(
        m["size_ratio"] for m in dgp1.values())

    # --- Lemma B.1 --------------------------------------------------------
    lemma = {T1: _lemma_b1(2, N_LEMMA, T1, M_LEMMA) for T1 in T1_GRID_LEMMA}
    for T1 in (50, 1600):
        out[f"lemma_b1_alpha_dev_T{T1}"] = lemma[T1]["alpha_dev"]
        out[f"lemma_b1_var_dev_T{T1}"] = lemma[T1]["var_dev"]
    # The bound is the claim: the deviation divided by sqrt(log N / T1) stays
    # put as T1 grows over a 32x range. Report the span, not a single cell --
    # an O_p rate says nothing about the level.
    a_ratios = [m["alpha_ratio"] for m in lemma.values()]
    v_ratios = [m["var_ratio"] for m in lemma.values()]
    out["lemma_b1_alpha_ratio_min"] = min(a_ratios)
    out["lemma_b1_alpha_ratio_max"] = max(a_ratios)
    out["lemma_b1_var_ratio_min"] = min(v_ratios)
    out["lemma_b1_var_ratio_max"] = max(v_ratios)
    # Growth over the grid: > 1 would say the normalisation is too fast.
    out["lemma_b1_alpha_ratio_growth"] = a_ratios[-1] / a_ratios[0]
    out["lemma_b1_var_ratio_growth"] = v_ratios[-1] / v_ratios[0]
    return out


# Tolerances. The case is seeded end to end, so re-running reproduces these
# exactly; the tolerances instead size "how far may this drift before the claim
# has changed". They are set at roughly three Monte-Carlo standard errors at
# M = 400 -- the spread a different set of draws would produce -- so a
# tolerance failure means the selection behaviour moved, not that the seed did.
EXPECTED = {
    # --- Propositions 2.2 / D.1, DGP 2: Pr(U_hat = U*) climbs to 0.77 -------
    # U* is the matched half exactly (10 of 20 controls), so an exact match
    # needs all ten in and all ten mismatched controls out.
    "dgp2_exact_T25": (0.000, 0.02),
    "dgp2_exact_T50": (0.000, 0.02),
    "dgp2_exact_T100": (0.003, 0.02),
    "dgp2_exact_T200": (0.005, 0.03),
    "dgp2_exact_T400": (0.068, 0.04),
    "dgp2_exact_T800": (0.380, 0.07),
    "dgp2_exact_T1600": (0.768, 0.06),
    # Recall -- the share of U* recovered -- moves earlier and more smoothly
    # than the exact rate, so it shows the convergence before T1 = 400.
    "dgp2_recall_T25": (0.413, 0.03),
    "dgp2_recall_T50": (0.508, 0.03),
    "dgp2_recall_T100": (0.606, 0.03),
    "dgp2_recall_T200": (0.717, 0.03),
    "dgp2_recall_T400": (0.825, 0.03),
    "dgp2_recall_T800": (0.920, 0.03),
    "dgp2_recall_T1600": (0.975, 0.02),
    # By T1 = 1600 the selection admits no mismatched control at all.
    "dgp2_precision_T1600": (1.000, 0.01),
    # The geometry the propositions predict. Zero tolerance: these are booleans.
    "dgp2_exact_increasing": (1.0, 0.0),
    "dgp2_recall_increasing": (1.0, 0.0),

    # --- DGP 1: the same convergence, against a criterion that barely ranks -
    # U* is the whole pool and V = 1 + 1/n is nearly flat near the top, so the
    # selected pool grows steadily without reaching N on this grid.
    "dgp1_size_ratio_T25": (0.250, 0.04),
    "dgp1_size_ratio_T1600": (0.817, 0.05),
    "dgp1_size_increasing": (1.0, 0.0),

    # --- Lemma B.1: the uniform deviations, and the rate that bounds them ---
    # Raw deviations fall by ~5.3x from T1 = 50 to 1600, against a sqrt(log N /
    # T1) that falls by 5.66x.
    "lemma_b1_alpha_dev_T50": (0.794, 0.08),
    "lemma_b1_var_dev_T50": (2.259, 0.25),
    "lemma_b1_alpha_dev_T1600": (0.152, 0.02),
    "lemma_b1_var_dev_T1600": (0.409, 0.05),
    # Normalised, they sit in a band over a 32x range of T1 instead of growing.
    # The band's *level* carries no information (an O_p bound hides its
    # constant); its flatness is the claim, which the growth keys below state
    # directly and these four bracket.
    "lemma_b1_alpha_ratio_min": (3.563, 0.60),
    "lemma_b1_alpha_ratio_max": (4.524, 0.60),
    "lemma_b1_var_ratio_min": (11.076, 1.20),
    "lemma_b1_var_ratio_max": (11.839, 1.20),
    "lemma_b1_alpha_ratio_growth": (1.082, 0.25),
    "lemma_b1_var_ratio_growth": (1.024, 0.25),
}
