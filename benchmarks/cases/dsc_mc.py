"""DSC Path-B: the asymptotics Monte Carlo (Zhang, Zhang & Zhang 2026, Sec. 5.1).

Path B (Monte Carlo; scenario 1 -- paper only, no code, no data, and no table).
Reproduces the model-free simulation of Zhang, L., Zhang, X. & Zhang, X. (2026),
*"Asymptotic Properties of the Distributional Synthetic Controls"*
(arXiv:2405.00953v3), the paper whose Algorithm 1 :class:`mlsynth.DSC` follows.
It complements ``disco_tenure``, which reproduces the ``disco`` Stata Journal
article's published weights bit-for-bit and is what establishes that mlsynth
implements this specification. That case runs on an empirical panel, where there
is no known truth; here the optimum is closed-form at all sixteen points, so the
two quantities the theorems concern can be measured against it directly.

What the design measures
------------------------

Two quantities, as the draw count ``M`` grows, at two donor-pool sizes:

* Theorem 1 -- the risk ratio :math:`\\bar R_{T_1}(\\widehat w) / \\inf_{w}
  \\bar R_{T_1}(w)` falls monotonically toward 1.
* Theorem 2 -- the weight error :math:`\\| \\widehat w - w^{\\mathrm{opt}} \\|`
  falls monotonically, and more slowly at the larger ``J``.

Both are measured against a population optimum this benchmark computes in closed
form (the design is time-invariant, so the averaged post-treatment risk is a
single quadratic; see the simulator module).

What is matched, and what is not
--------------------------------

The geometry reproduces in full: at both pool sizes the risk ratio falls
monotonically toward 1 and the weight error falls monotonically, the ``J = 50``
curve sits above the ``J = 20`` curve at every draw count, and the ratio's
excess over 1 falls by roughly an order of magnitude from ``M = 50`` to
``M = 400``.

The risk ratio is a cell match everywhere except at the smallest draw count.
Every cell with ``M >= 200`` lands within 0.0017 of the published value and five
of the eight within 0.0025, against a published curve spanning 1.003 to 1.027.
The two cells at ``M = 50`` sit above it, by 0.006 at ``J = 20`` and 0.028 at
``J = 50`` -- the corner where the donor matrix is square and the constrained
fit is close to interpolating. The gap grows as the draw count falls toward the
donor count, and the published curves depend on the donor pool much less than
the reconstruction does.

The weight-error half is not a cell match anywhere: the published curve is
steeper, starting above the reconstruction at ``M = 50`` and crossing below it
by ``M = 400``. Those cells are pinned as mlsynth's own deterministic output,
and every distance from the paper is reported as a number
(``max_abs_ratio_gap_vs_paper_Mge200``, ``max_abs_ratio_gap_vs_paper_J20``,
``max_abs_ratio_gap_vs_paper``, ``max_abs_norm_gap_vs_paper``) instead of being
absorbed into a wide tolerance.

Two unstated DGP details are the likely source, and both are documented in
:mod:`mlsynth.utils.dsc_helpers.simulate`: whether every unit carries its own
rank sequence, and whether the donor locations are redrawn per replication.

Where the target numbers come from
----------------------------------

Section 5 has no tables -- its four panels are figures. The arXiv source ships
them as R 4.2.2 vector PDFs, so the plotted values are recoverable from the
polyline coordinates in the content stream; the y axis resolves to about 6e-5
per point, negligible against Monte-Carlo noise. The digitised values are the
``_PAPER_*`` tables below.

The paper's second design (Section 5.2, a quantile factor model) is not
reproduced. Its published ratios at ``M = 100`` (1.101 and 1.155, collapsing to
1.008 and 1.038 at ``M = 200``) and its weight errors falling to 0.002 by
``M = 400`` are not recoverable from the stated design -- a faithful port gives
1.011 and 1.022 at ``M = 100``, and 0.048 and 0.057 at ``M = 400`` -- so it is
left out.

Scope: the paper's Monte Carlo generates the pseudo-samples directly and draws
each unit independently, so it never exercises DSC's quantile-estimation step.
What it validates is the constrained least-squares solver
(:func:`~mlsynth.utils.dsc_helpers.weights.solve_simplex_weights`) and the
:math:`\\lambda_t` aggregation -- the two steps of Algorithm 1 the theorems are
about. Feeding the same draws through :meth:`mlsynth.DSC.fit` would sort each
cell, which is the shared-rank coupling this design does not use, and lands on a
different curve.

Determinism: every replication is seeded ``_SEED0 + 1000 * J + M + r``.
Provenance: arXiv:2405.00953v3, Section 5.1, Figures 1 and 2.
"""
from __future__ import annotations

import numpy as np

from mlsynth.utils.dsc_helpers.simulate import model_free_replication

_T0 = 10                    # pre-treatment periods, as in the paper
_JS = (20, 50)
_MS = (50, 100, 200, 400)
_SEED0 = 20_240_953

# Replications. The paper uses 1000; the whole grid at that count runs to about
# three quarters of an hour, so the case uses a reduced count and the pins carry
# the resulting noise. Deterministic either way -- the seeds are fixed, so the
# reported numbers are exactly reproducible at this _R.
_R = 24

# Digitised from Fig22.pdf (Figure 1) and Fig11.pdf (Figure 2) of the arXiv
# source: J -> value at M = 50, 100, 200, 400. Regenerate with
# benchmarks/reference/dsc_mc/digitize_figures.py, which reads the plotted
# polylines out of the figures' content streams.
_PAPER_RATIO = {20: (1.0238, 1.0145, 1.0075, 1.0030),
                50: (1.0274, 1.0180, 1.0099, 1.0043)}
_PAPER_NORM = {20: (0.1856, 0.1221, 0.0668, 0.0276),
               50: (0.2789, 0.2123, 0.1301, 0.0593)}


def _cell(J: int, M: int) -> tuple[float, float]:
    """Mean risk ratio and mean weight error over ``_R`` replications."""
    draws = [model_free_replication(J=J, M=M, T0=_T0, seed=_SEED0 + 1000 * J + M + r)
             for r in range(_R)]
    ratios, norms = zip(*draws)
    return float(np.mean(ratios)), float(np.mean(norms))


def _strictly_decreasing(values) -> float:
    return float(all(b < a for a, b in zip(values, values[1:])))


def run() -> dict:
    out: dict[str, float] = {}
    ratio: dict[int, list[float]] = {}
    norm: dict[int, list[float]] = {}
    for J in _JS:
        ratio[J], norm[J] = [], []
        for M in _MS:
            r, n = _cell(J, M)
            ratio[J].append(r)
            norm[J].append(n)
            out[f"ratio_J{J}_M{M}"] = r
            out[f"norm_J{J}_M{M}"] = n

    # Theorem 1 and 2: both curves fall as the draw count grows, at both pools.
    for J in _JS:
        out[f"ratio_monotone_J{J}"] = _strictly_decreasing(ratio[J])
        out[f"norm_monotone_J{J}"] = _strictly_decreasing(norm[J])

    # Theorem 2: convergence slows as the donor pool grows -- the paper's
    # "w-hat converges faster when J = 20 than J = 50", at every draw count.
    out["J_ordering_holds"] = float(
        all(r50 > r20 for r20, r50 in zip(ratio[20], ratio[50]))
        and all(n50 > n20 for n20, n50 in zip(norm[20], norm[50]))
    )
    # Theorem 1: the optimality gap at M = 400 is an order of magnitude below
    # the gap at M = 50 (the published curves fall by 7.9x and 6.4x).
    for J in _JS:
        out[f"ratio_excess_decay_J{J}"] = (ratio[J][0] - 1.0) / (ratio[J][-1] - 1.0)
    out["excess_decay_exceeds_5x"] = float(
        all(out[f"ratio_excess_decay_J{J}"] > 5.0 for J in _JS))

    # Distance from the published figures, reported, not absorbed into bands.
    # Three nested windows: the cells that match tightly, the J = 20 column,
    # and the whole grid including the corner where the design degenerates.
    out["max_abs_ratio_gap_vs_paper_Mge200"] = max(
        abs(g - p) for J in _JS
        for g, p in zip(ratio[J][2:], _PAPER_RATIO[J][2:]))
    out["max_abs_ratio_gap_vs_paper_J20"] = max(
        abs(g - p) for g, p in zip(ratio[20][1:], _PAPER_RATIO[20][1:]))
    out["max_abs_ratio_gap_vs_paper"] = max(
        abs(g - p) for J in _JS for g, p in zip(ratio[J], _PAPER_RATIO[J]))
    out["max_abs_norm_gap_vs_paper"] = max(
        abs(g - p) for J in _JS for g, p in zip(norm[J], _PAPER_NORM[J]))
    return out


# Tolerances.
#
# Risk-ratio cells are anchored to the paper's digitised figure, with a band set
# by column and not by cell. At M >= 100 the reconstruction is within 0.0070 of
# the published value at all six cells and within 0.0025 at five of them, so
# those carry 0.010 -- roughly the measured gap plus three Monte-Carlo standard
# errors at _R = 24 (per-cell SE ~0.002 against the paper's 1000 replications).
# The M = 50 column is where the two curves part: the draw count there equals or
# barely exceeds the donor count, and the reconstruction sits 0.006 above the
# published value at J = 20 and 0.028 above it at J = 50, the corner where the
# donor matrix is square. A single band covering both is 0.040, and the docs
# page says so instead of presenting the column as a match.
#
# Weight-error cells are mlsynth's own deterministic output at these seeds --
# regression pins, not claims about the paper. Their 0.02 band covers platform
# arithmetic drift in the QP, not sampling noise, since the seeds are fixed. The
# published curve is steeper than the reconstruction: it starts higher at M = 50
# and ends lower at M = 400, and max_abs_norm_gap_vs_paper records how far.
#
# The three nested max-gap metrics carry the precision claim, each read as "no
# cell in this window sits further than tol from the published figure". The
# monotonicity, ordering and decay flags are exact 0/1 quantities with zero
# tolerance -- they are the paper's qualitative claims, and the strongest
# assertions here.
EXPECTED = {
    # Theorem 1, against Figure 1: the six cells that match.
    "ratio_J20_M100": (1.0145, 0.010),
    "ratio_J20_M200": (1.0075, 0.010),
    "ratio_J20_M400": (1.0030, 0.010),
    "ratio_J50_M100": (1.0180, 0.010),
    "ratio_J50_M200": (1.0099, 0.010),
    "ratio_J50_M400": (1.0043, 0.010),
    # ... and the M = 50 column, which does not.
    "ratio_J20_M50": (1.0238, 0.040),
    "ratio_J50_M50": (1.0274, 0.040),
    # How close, in three nested windows.
    "max_abs_ratio_gap_vs_paper_Mge200": (0.0, 0.005),
    "max_abs_ratio_gap_vs_paper_J20": (0.0, 0.006),
    "max_abs_ratio_gap_vs_paper": (0.0, 0.040),
    "max_abs_norm_gap_vs_paper": (0.0, 0.16),
    # The geometry both theorems predict.
    "ratio_monotone_J20": (1.0, 0.0),
    "ratio_monotone_J50": (1.0, 0.0),
    "norm_monotone_J20": (1.0, 0.0),
    "norm_monotone_J50": (1.0, 0.0),
    "J_ordering_holds": (1.0, 0.0),
    "excess_decay_exceeds_5x": (1.0, 0.0),
    # The decay itself is a ratio of two small numbers, so a wide band. The
    # published curves fall by 7.9x (J = 20) and 6.4x (J = 50).
    "ratio_excess_decay_J20": (12.97, 4.0),
    "ratio_excess_decay_J50": (11.12, 4.0),
    # Theorem 2, pinned as mlsynth's own output.
    "norm_J20_M50": (0.1323, 0.02),
    "norm_J20_M100": (0.0798, 0.02),
    "norm_J20_M200": (0.0660, 0.02),
    "norm_J20_M400": (0.0448, 0.02),
    "norm_J50_M50": (0.1509, 0.02),
    "norm_J50_M100": (0.1167, 0.02),
    "norm_J50_M200": (0.0828, 0.02),
    "norm_J50_M400": (0.0552, 0.02),
}
