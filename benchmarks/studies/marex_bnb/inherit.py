"""A child node's bound from its parent's multipliers, at the cost of one eigendecomposition.

The certificate of ``certified_dnn`` holds for any multipliers with the right
signs. A child's designs are a subset of its parent's, and a child's lift has
a subset of its parent's rows. A dropped row is not always zero on the child's
designs: including market ``j`` makes ``z_j = 1`` (``u_j = v_j = 0``), and
excluding it makes ``u_j = 1`` (``z_j = w_j = 0``). So the parent's vector is
``y_p = E y_c`` for every child design, ``E`` keeping the child's rows,
sending a row whose value is one to the constant row and dropping a row whose
value is zero. Then

    y_p' M_p y_p = y_c' (E' M_p E) y_c,   M_p = C_p - y0 E_00 - H_p,

and ``E' M_p E = C_c - y0 E_00 - E' H_p E`` (the objective lives on the weight
rows, which never fold). The child's multipliers are therefore
``y0 + (E' H_p E)_00`` on ``Y_00 = 1`` and ``E' H_p E`` elsewhere, which keeps
its signs: a folded entry adds a parent multiplier on a product with a row of
value one, and no entry fixed at zero in the parent survives the fold. The
child's certificate is then evaluated exactly as a fresh one would be: its own
integer map, objective and trace bound, the best shift of ``y0``, and an exact
lower bound on the smallest eigenvalue. No solver runs.

``inherit_probe`` measures how tight this is against a fresh lift.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from benchmarks.studies.marex_bnb.certified_dnn import best_shift, certify, objective
from benchmarks.studies.marex_bnb.certify import down


def _unfold(parent_lift, child_lift, I, O):
    """``E``: the parent's vector as a map of the child's, on the child's designs."""
    pos_c = {key: i for i, key in enumerate(child_lift.rows)}
    E = np.zeros((len(parent_lift.rows), len(child_lift.rows)))
    for a, key in enumerate(parent_lift.rows):
        if key in pos_c:
            E[a, pos_c[key]] = 1.0
        elif (key[0] == "z" and key[1] in I) or (key[0] == "u" and key[1] in O):
            E[a, pos_c[("1", None)]] = 1.0              # a row equal to one
    return E


def inherited_bound(panel, parent_lift, mult, child_lift, I, O):
    """The certified bound of ``child_lift`` at the parent's multipliers ``mult``.

    ``I`` and ``O`` are the child's markets fixed in and out.
    """
    E = _unfold(parent_lift, child_lift, set(I), set(O))
    Hf = E.T @ np.asarray(mult["H"], dtype=float) @ E
    y0 = float(mult["y0"]) + Hf[0, 0]
    Hf[0, 0] = 0.0
    shift = mult["shift"]
    C = objective(panel.G, child_lift, shift)
    Cf = C.float()
    Hc = np.where(child_lift.zero, Hf, np.clip(Hf, 0.0, None))
    P = child_lift.P.astype(float)
    S = P.T @ (Cf - Hc) @ P
    S[0, 0] -= y0
    _, d = best_shift((S + S.T) / 2, y0, child_lift.tau)
    lb = certify(C, child_lift, y0 - d, Hc, None)
    return down(max(lb, Fraction(0)) * (1 << shift))
