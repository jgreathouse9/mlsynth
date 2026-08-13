"""Forward-selected PDA estimation (Shi & Huang 2023).

Greedy forward selection of control units: at each step add the donor whose
inclusion maximizes the pre-treatment OLS ``R^2`` (equivalently minimizes the
residual variance ``sigma^2 = mean(e^2)``). Selection **stops** as soon as the
modified information criterion

    IC(r) = log( sigma^2(U_r) ) + B * r,   B = log(log N) * log(T1) / T1

stops decreasing (the stopping rule of Wang, Li & Tsai used in the authors'
``fsPDA`` R package), starting from the intercept-only ``IC = log(var(y1))``.
The counterfactual is the OLS extrapolation on the selected set. This is a
direct port of ``est.fsPDA.R``.

Scoring the step without refitting
----------------------------------
Written as defined, each step refits OLS once per remaining donor and keeps the
smallest ``sigma^2``: ``N`` least-squares solves per step, and the run costs
``O(N T1 r^2)`` for ``r`` selected donors. The step's answer is available in one
matrix-vector product. Let ``r_vec = y1 - P y1`` be the pre-period residual on
the donors already selected (``P`` the projection onto their span, including the
constant when one is fitted) and ``z_j = x_j - P x_j`` donor ``j`` orthogonalised
against that span. Adding ``j`` drops the residual sum of squares by

    gain_j = (z_j' r_vec)^2 / ||z_j||^2 = (x_j' r_vec)^2 / ||z_j||^2,

the second equality because ``r_vec`` is already orthogonal to the selected
span. Minimising ``sigma^2`` over the pool is maximising ``gain_j`` over it, so
``Z' r_vec`` ranks every candidate at once and a rank-one downdate of ``Z``
carries the orthogonalisation into the next step. The cost per step falls to
``O(T1 N)``, and the donor pool no longer multiplies the number of solves.

The ranking is exact in exact arithmetic, so it decides the step; the step's
``sigma^2`` still comes from :func:`_ols_sigma2` on the winning design, which is
what the IC reads and what stops the search. Candidates whose scores sit within
rounding of the leader -- duplicated donors, a donor already inside the selected
span -- are all refit and compared the way the scan compares them, so the search
selects what the definition selects and stops where the definition stops.

The two part company only where the step has no answer to give. A donor pool can
contain donors that fit the pre-period identically -- exact duplicates, or a
pool of rank one, where every donor is a multiple of the same series -- and a
fit that has interpolated (``y1`` exactly in the donor span) puts every
remaining donor in that position, since the residual has fallen to ~1e-16 and
``log(sigma^2)`` keeps the search going. Inside such a set the scan's winner is
whichever candidate its least-squares solve rounded smallest, so reproducing it
means running the scan. What holds is that the two agree on the donors that
explain the outcome and on the counterfactual to machine precision: the members
of an equivalent set are interchangeable, and the extras an interpolating fit
collects carry coefficients at the 1e-16 level. How many extras it collects is
not a property of either rule -- once ``sigma^2`` is at 1e-31 the accept test
compares a ratio of rounding artefacts against ``exp(-B)``, so the step count
there follows the BLAS, for the scan as much as for this search.
Below ``_SHORTLIST_MAX`` equivalent donors the exact comparison settles the tie
the same way the scan settles it.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Width of the band below the leading score whose candidates are refit and
# compared exactly, as a fraction of the residual sum of squares the search
# started from. Scores this close are equal up to rounding, which is the regime
# where an argmax and the definitional scan can part company: exactly duplicated
# donors, or a donor lying in the span of those already selected.
_SHORTLIST_REL = 1e-9

# Ceiling on that shortlist. A pool of mutually collinear donors can put every
# remaining column within rounding of the leader; they fit the pre-period
# identically, so settling the first few by exact comparison is enough and the
# step keeps a bounded number of solves.
_SHORTLIST_MAX = 32

# A candidate whose orthogonalised norm has fallen this far below its starting
# norm lies in the span of the selected donors. It cannot lower the residual, and
# its score would be 0/0.
_RANK_TOL = 1e-12


def _ols_sigma2(y: np.ndarray, Z: np.ndarray) -> float:
    """OLS (pinv) residual variance ``mean(e^2)`` for design ``Z`` (incl. intercept)."""
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ coef
    return float(np.mean(resid ** 2))


def _shortlist(gain: np.ndarray, alive: np.ndarray, scale: float) -> List[int]:
    """Ascending indices of the leading candidates, ties included.

    Returns the donor maximising ``gain`` together with any live donor scoring
    within :data:`_SHORTLIST_REL` of it, capped at :data:`_SHORTLIST_MAX`. The
    band is measured against ``scale``, the residual sum of squares the search
    started from, so it stays an absolute width as the residual shrinks: once a
    fit approaches interpolation every remaining candidate falls inside it and
    the step is settled by exact comparison, which is where a ranking on
    rounding noise would otherwise decide it.
    """
    masked = np.where(alive, gain, -np.inf)
    order = np.argsort(-masked, kind="stable")          # ties keep index order
    best = masked[order[0]]
    cut = best - _SHORTLIST_REL * max(abs(best), scale)
    n_tied = int(np.count_nonzero(masked[order] >= cut))
    n_keep = max(1, min(n_tied, _SHORTLIST_MAX))
    return sorted(int(j) for j in order[:n_keep])


def forward_select(
    y: np.ndarray, X: np.ndarray, T0: int, intercept: bool = False,
) -> Tuple[List[int], np.ndarray, float, np.ndarray]:
    """Forward-select donors with the stop-on-increase IC rule, then refit OLS.

    Returns ``(selected_indices, beta_full, intercept, counterfactual)`` where
    ``beta_full`` is an ``N``-vector with zeros off the selected support.

    Selection is capped at ``T0 - 1 - intercept`` donors so the pre-period fit
    keeps a residual degree of freedom; see the comment on ``max_selected``.

    Parameters
    ----------
    intercept : bool, default False
        When ``False`` the donor regression is fit *without* a constant and the
        IC baseline is ``log(mean(y_pre**2))`` -- the paper's simulation
        convention (Shi & Huang 2023, ``simulation/FS.R``). This is the
        correctly-specified form for the factor DGP (mean-zero factors, no unit
        level shift) and yields valid test size. When ``True`` a constant is
        included and the baseline is ``log(var(y_pre))`` -- the released
        ``fsPDA`` R package convention (``est.fsPDA.R``), appropriate for panels
        with genuine level differences but size-distorting on centered data.
    """
    y_pre, X_pre = y[:T0], X[:T0]
    N = X.shape[1]
    B = np.log(np.log(max(N, 3))) * np.log(T0) / T0
    IC = float(np.log(np.var(y_pre, ddof=1))) if intercept else float(np.log(np.mean(y_pre ** 2)))

    def _design(cols):
        return np.column_stack([np.ones(T0), X_pre[:, cols]]) if intercept else X_pre[:, cols]

    # Candidate donors orthogonalised against the selected span, and the current
    # residual. Fitting a constant means projecting out the vector of ones, which
    # is centring; the span grows by one direction per accepted donor.
    Z = np.array(X_pre, dtype=float, copy=True)
    resid = np.array(y_pre, dtype=float, copy=True)
    if intercept:
        Z -= Z.mean(axis=0)
        resid -= resid.mean()
    norm0 = np.einsum("ij,ij->j", Z, Z)
    norm = norm0.copy()
    rss0 = float(resid @ resid)

    selected: List[int] = []
    alive = np.ones(N, dtype=bool)
    # Leave at least one residual degree of freedom. The IC cannot enforce this
    # itself: it carries log(s2), which diverges to -inf as the fit approaches an
    # exact one, so every further donor keeps "improving" it and selection runs
    # to T0 -- T0 + 1 parameters once the intercept is counted. The resulting
    # residual variance is ~1e-31, which is the scale the Jiang et al. prediction
    # intervals studentize against.
    max_selected = max(0, T0 - 1 - int(intercept))
    for _ in range(max_selected):
        if not alive.any():
            break
        # Score every candidate at once. A donor inside the selected span keeps
        # its score at zero: it cannot lower the residual, and dividing by its
        # vanished norm would produce a NaN that could win the ranking. A donor
        # that never varied has ``norm0 == 0`` and fails the same test.
        usable = alive & (norm > _RANK_TOL * norm0)
        proj = Z.T @ resid
        gain = np.zeros(N)
        gain[usable] = proj[usable] ** 2 / norm[usable]

        best_j, best_s2 = None, np.inf
        for j in _shortlist(gain, alive, rss0):
            s2 = _ols_sigma2(y_pre, _design(selected + [j]))
            if s2 < best_s2:
                best_s2, best_j = s2, j
        IC_new = np.log(best_s2) + B * (len(selected) + 1)
        if not IC_new < IC:               # stop at first non-improvement
            break

        IC = IC_new                       # accept and continue
        selected.append(best_j)
        alive[best_j] = False
        # Carry the exact residual forward, so the next step's scores start from
        # the fit the IC was computed on and no error accumulates along the path.
        design = _design(selected)
        coef, *_ = np.linalg.lstsq(design, y_pre, rcond=None)
        resid = y_pre - design @ coef
        if norm[best_j] > 0.0:            # extend the span by the accepted donor
            q = Z[:, best_j] / np.sqrt(norm[best_j])
            Z -= np.outer(q, q @ Z)
            norm = np.einsum("ij,ij->j", Z, Z)

    if not selected:                          # degenerate: intercept-only / zero
        intercept_val = float(np.mean(y_pre)) if intercept else 0.0
        beta_full = np.zeros(N)
        return [], beta_full, intercept_val, np.full(X.shape[0], intercept_val)

    coef, *_ = np.linalg.lstsq(_design(selected), y_pre, rcond=None)
    beta_full = np.zeros(N)
    if intercept:
        intercept_val = float(coef[0])
        beta_full[selected] = coef[1:]
    else:
        intercept_val = 0.0
        beta_full[selected] = coef
    counterfactual = X @ beta_full + intercept_val
    return selected, beta_full, intercept_val, counterfactual
