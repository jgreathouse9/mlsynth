"""Faithful NumPy port of Kyo and Kawata's ``code_demo_F.txt`` (SDM).

The port reproduces the authors' R script step for step, including the parts
that depart from the published algorithm. Those departures are the finding, so
they are preserved here and reported as separate outputs instead of corrected:

* the working matrix ``X`` is initialised with ``bet / sum(bet)`` while the
  weight bookkeeping vector ``Sbet`` is initialised with ``bet / M``, so the two
  differ by the constant ``M / sum(bet)`` for the whole run;
* the final block renormalises the weights to sum to one before building the
  counterfactual, which the article's Equation (14) does not do;
* the convergence index is computed from the scalar ``a`` left over from the
  inner loop, not from the substantial set ``G`` of Equation (13);
* ``Reg1`` leaves ``V`` unassigned when a donor column has been zeroed, and R's
  lexical scoping supplies the enclosing environment's ``V`` instead of raising.
  ``RScope`` emulates that fall-through and counts how often it fires.

``sdm_fit`` returns the three weight vectors the script implicitly carries so a
caller can compare what the iteration optimises against what it ships.

Source: replication package supplied by the authors (see ``provenance.json``);
the article is Kyo and Kawata (2026), J. Appl. Stat., doi:10/pmrs.
"""
from __future__ import annotations

import numpy as np

__all__ = ["RScope", "reg1", "reg2", "sdm_fit", "eq14"]


class RScope:
    """Stands in for R's enclosing environment for ``Reg1``'s undefined ``V``.

    ``Reg1`` assigns ``V`` only inside ``if (length(NO) > 0)``. When a donor
    column has been driven to zero that branch is skipped and R resolves ``V``
    lexically, finding the ``V <- min(V1, V2)`` left in the global environment
    by the previous inner-loop step. The read is silent, so a run can consume a
    stale objective value with no diagnostic.
    """

    def __init__(self) -> None:
        self.V: float | None = None
        self.fallthroughs = 0


def reg1(y, X, J, T0, scope):
    """One-by-one update; mirrors ``Reg1()``.

    Returns ``(a, e, V)``. ``V`` is the enclosing scope's value when the column
    is dead, matching the R behaviour described in :class:`RScope`.
    """
    x1 = X[:, J]
    x2 = np.delete(X, J, axis=1).sum(axis=1)

    a = 0.0
    e = np.zeros(T0)
    V = None
    if np.abs(x1).max() > 0:
        y1 = y - x2
        a = float(np.sum(x1 * y1) / np.sum(x1**2))
        if a < 0:
            a = 0.0
        e = y1 - a * x1
        V = float(np.mean(e**2))
    if V is None:
        scope.fallthroughs += 1
        V = scope.V
    return a, e, V


def reg2(y, X, J, T0):
    """Pairwise update; mirrors ``Reg2()``.

    Unlike ``Reg1`` this one initialises ``V`` to ``1e10``, so a dead column is
    handled without touching the enclosing scope.
    """
    x1 = X[:, J]
    x2 = np.delete(X, J, axis=1).sum(axis=1)

    a, b, V = 0.0, 1.0, 1e10
    e = np.zeros(T0)
    if np.abs(x1).max() > 0:
        Z = np.column_stack([x1, x2])
        try:
            coef = np.linalg.solve(Z.T @ Z, Z.T @ y)
        except np.linalg.LinAlgError:
            return 0.0, 1.0, e, 1e10
        a, b = float(coef[0]), float(coef[1])
        if a < 0 or b < 0:
            a, b = 0.0, 1.0
        e = y - a * x1 - b * x2
        V = float(np.mean(e**2))
    return a, b, e, V


def eq14(yy, XX, w, T0):
    """The article's Equation (14): intercept, counterfactual, fit, ATT.

    ``alpha = mean(y_1) - sum_j w_j mean(y_j)`` over the pre-period, which is
    the concentrated-out intercept of ``min_{alpha, w} ||y_1 - alpha - Y_0 w||``.
    """
    alpha = float(np.mean(yy[:T0] - XX[:T0] @ w))
    counterfactual = alpha + XX @ w
    pre_mspe = float(np.mean((yy[:T0] - counterfactual[:T0]) ** 2))
    att = float(np.mean(yy[T0:] - counterfactual[T0:]))
    return alpha, counterfactual, pre_mspe, att


def sdm_fit(yy, XX, T0, max_iter=500, rho_tol=1e-4):
    """Run the authors' algorithm on a treated series and donor matrix.

    Parameters
    ----------
    yy : ndarray, shape (T,)
        Treated outcome over the full sample.
    XX : ndarray, shape (T, M)
        Donor outcomes, no missing values.
    T0 : int
        Number of pre-intervention periods.
    max_iter, rho_tol : int, float
        The script's own ``500`` sweeps and ``1e-4`` threshold on ``rho``.

    Returns
    -------
    dict
        ``w_objective`` are the weights encoded in the working matrix, i.e. the
        ones the iteration actually minimises over. ``w_bookkeeping`` is the
        script's ``Sbet``. ``w_shipped`` is ``Sbet / sum(Sbet)``, which is what
        the counterfactual, the plot and the printed weight table are built
        from. ``rho_coded`` and ``rho_paper`` are the stopping statistic as
        coded and as Equation (13) defines it.
    """
    scope = RScope()
    M = XX.shape[1]
    y = yy[:T0] - yy[:T0].mean()
    Xc = XX[:T0] - XX[:T0].mean(axis=0)

    # D-2: univariate slopes against the centred treated series, clamped at zero
    bet = np.zeros(M)
    for j in range(M):
        bet[j] = max(float(np.mean(y * Xc[:, j]) / np.mean(Xc[:, j] ** 2)), 0.0)
    if bet.sum() <= 0:
        raise ValueError("no donor has a positive univariate slope")

    # D-3: X carries bet/sum(bet); Sbet carries bet/M. The two differ by a
    # constant that survives every update and cancels only at the final
    # renormalisation.
    w0 = bet / bet.sum()
    X = Xc * w0
    Sbet = bet / M

    MV = float(np.mean((y - X.sum(axis=1)) ** 2))
    mspe_trace = [MV]
    rho_coded_trace: list[float] = []
    rho_paper_trace: list[float] = []
    a_last = None
    MI = max_iter

    for I in range(2, max_iter + 1):
        accepted: dict[int, float] = {}
        for J in range(M):
            a2, b2, _, V2 = reg2(y, X, J, T0)
            if not a2 > 0:
                V2, b2 = 1e10, 1.0

            a1, _, V1 = reg1(y, X, J, T0, scope)
            b1 = 1.0
            if not a1 > 0:
                V1 = 1e10

            V = min(V1, V2)
            scope.V = V

            # The script applies an update only when it improves the running
            # minimum, so the monotone decline of the objective is enforced by
            # this guard and is not a property of the update itself.
            if V < MV:
                MV = V
                a, b = (a2, b2) if V2 < V1 else (a1, b1)
                other = [k for k in range(M) if k != J]
                X[:, J] *= a
                X[:, other] *= b
                Sbet[J] *= a
                Sbet[other] *= b
                a_last = a
                accepted[J] = a

        # As coded: `NO <- which(a > 0)` reads the scalar `a` surviving the
        # inner loop, so rho is (a_last - 1)^2 for one donor. Equation (13)
        # averages over the whole substantial set.
        rho_coded = (a_last - 1.0) ** 2 if a_last and a_last > 0 else 0.0
        live = [v for v in accepted.values() if v > 0]
        rho_paper = float(np.mean([(v - 1.0) ** 2 for v in live])) if live else 0.0
        rho_coded_trace.append(rho_coded)
        rho_paper_trace.append(rho_paper)
        mspe_trace.append(MV)

        if rho_coded < rho_tol:
            MI = I - 1
            break

    # Recover the weights the working matrix encodes, column by column.
    nz = np.abs(Xc).max(axis=0) > 0
    w_objective = np.zeros(M)
    w_objective[nz] = X[0, nz] / Xc[0, nz]

    return {
        "w_objective": w_objective,
        "w_bookkeeping": Sbet.copy(),
        "w_shipped": Sbet / Sbet.sum(),
        "mspe_objective": MV,
        "iterations": MI,
        "rho_coded": rho_coded_trace,
        "rho_paper": rho_paper_trace,
        "mspe_trace": mspe_trace,
        "scope_fallthroughs": scope.fallthroughs,
        "bet": bet,
        "scale_offset": float(M / bet.sum()),
    }
