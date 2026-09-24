"""Path C: is mlsynth's Augmented SCM an instance of the paper's equation (7),
and does the paper's degenerate-hyperparameter trap have a panel analogue?

The paper's alarming empirical finding is that practical tuning selects the
balancing hyperparameter ``delta = 0`` on 56 percent of its draws (Table 1),
where the augmented estimator is numerically identical to OLS -- and where
``delta = 0`` is never optimal across its 36 DGPs. This asks the corresponding
question of the estimator mlsynth ships.

The mapping, derived and then checked numerically below. Writing ``B`` for the
centered donor pre-matrix (T0 x J), ``A`` for the centered treated pre-vector
and ``W`` for base simplex weights, mlsynth's correction is

    W_ridge = M @ (B B^T + lam I)^{-1} @ B,     M = A - B W

so the augmented prediction is ``W . Y_post + M . beta`` with

    beta = (B B^T + lam I)^{-1} B Y_post .

Put ``Xp = B^T`` (donors as rows, pre-periods as features), ``phi_q = A`` and
``Sigma = Xp^T Xp / J = B B^T / J``. Then ``beta`` is the paper's generalized
ridge coefficient at ``Lambda = (lam / J) I``, and ``M`` is the paper's
residual feature shift at weights ``w / J = W``. So mlsynth's ASCM is exactly
equation (7) with a ridge outcome model and simplex balancing weights.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from prop43 import generalized_ridge, l2_balancing_weights, ols_plugin  # noqa: E402

from mlsynth.utils.bilevel.ridge_augment import (  # noqa: E402
    build_matching, ridge_augment_weights, simplex_qp,
)
from check_equivalence import factor_panel  # noqa: E402


def part1_mapping():
    """mlsynth's ASCM equals equation (7) at Lambda = lam / J."""
    print("Part 1 -- is mlsynth's ASCM the paper's equation (7)?")
    print(f"{'donors':>7} {'T0':>5} {'lam':>9} {'mlsynth ASCM':>15}"
          f" {'paper eq (7)':>15} {'abs diff':>10}")
    worst = 0.0
    for n_donors, T0 in ((40, 12), (20, 20), (12, 30)):
        y_pre, Y0_pre, Y0_post = factor_panel(n_donors, T0, seed=7)
        for lam in (0.001, 0.1, 10.0, 1000.0):
            res = ridge_augment_weights(y_pre, Y0_pre, lambda_=lam)
            mlsynth_est = float(res.W @ Y0_post)

            # the paper's side, on the same centered matrices mlsynth uses
            B, A = build_matching(y_pre, Y0_pre, None, None)
            Xp, phi_q = B.T, A
            J = Xp.shape[0]
            W_base = simplex_qp(B, A)
            beta = generalized_ridge(Xp, Y0_post, np.full(T0, lam / J))
            paper_est = float(W_base @ Y0_post + (phi_q - B @ W_base) @ beta)

            worst = max(worst, abs(mlsynth_est - paper_est))
            if lam in (0.1, 1000.0):
                print(f"{n_donors:>7} {T0:>5} {lam:9.3f} {mlsynth_est:15.8f}"
                      f" {paper_est:15.8f} {abs(mlsynth_est - paper_est):10.2e}")
    print(f"  worst over 12 cells: {worst:.3e}\n")
    return worst


def part2_lambda_to_zero():
    """Does lam -> 0 collapse the shipped ASCM onto an unpenalized fit?"""
    print("Part 2 -- the panel analogue of the delta = 0 trap")
    print("  As lam -> 0 the ridge correction stops being regularized. Where")
    print("  B B^T has full rank the augmented fit interpolates the treated")
    print("  pre-window exactly, and the estimate is the unpenalized plug-in.\n")
    print(f"{'donors':>7} {'T0':>5} {'lam':>10} {'ASCM est':>13}"
          f" {'pre-fit RMSE':>13} {'|W|_1':>9}")
    for n_donors, T0 in ((40, 12), (12, 30)):
        y_pre, Y0_pre, Y0_post = factor_panel(n_donors, T0, seed=3)
        B, A = build_matching(y_pre, Y0_pre, None, None)
        rank_note = "full row rank" if n_donors >= T0 else "rank deficient"
        print(f"  --- J={n_donors}, T0={T0}  (B B^T is {rank_note})")
        for lam in (1e-10, 1e-6, 1e-2, 1.0, 100.0):
            res = ridge_augment_weights(y_pre, Y0_pre, lambda_=lam)
            est = float(res.W @ Y0_post)
            rmse = float(np.sqrt(np.mean((A - B @ res.W) ** 2)))
            print(f"{n_donors:>7} {T0:>5} {lam:10.0e} {est:13.6f}"
                  f" {rmse:13.2e} {np.abs(res.W).sum():9.3f}")
        unpen = ols_plugin(B.T, Y0_post, A)
        print(f"    unpenalized plug-in: {unpen:.6f}\n")


def part3_what_cv_picks():
    """Does mlsynth's own CV ever land on the degenerate end of the grid?"""
    print("Part 3 -- what mlsynth's CV actually selects")
    print(f"{'donors':>7} {'T0':>5} {'seeds':>6} {'lam=grid floor':>15}"
          f" {'median lam':>12} {'min lam':>11}")
    for n_donors, T0 in ((40, 12), (20, 20), (12, 30), (8, 40)):
        chosen = []
        floors = 0
        for seed in range(30):
            y_pre, Y0_pre, _ = factor_panel(n_donors, T0, seed=100 + seed)
            res = ridge_augment_weights(y_pre, Y0_pre)      # lambda_=None -> CV
            chosen.append(res.lambda_)
            if res.cv is not None:
                grid = np.asarray(res.cv["lambdas"], dtype=float)
                if grid.size and np.isclose(res.lambda_, grid.min()):
                    floors += 1
        chosen = np.asarray(chosen)
        print(f"{n_donors:>7} {T0:>5} {len(chosen):>6} {floors:>10}/{len(chosen)}"
              f" {np.median(chosen):12.4g} {chosen.min():11.4g}")


if __name__ == "__main__":
    part1_mapping()
    part2_lambda_to_zero()
    part3_what_cv_picks()


def part4_why_no_ols_collapse():
    """Why the paper's OLS collapse does not transfer to simplex-weight ASCM.

    At ``lam -> 0`` with ``B B^T`` invertible the correction is
    ``W_ridge = M (B B^T)^{-1} B``, so

        W_aug . Y = W . Y + (A - B W) (B B^T)^{-1} B Y
                  = A (B B^T)^{-1} B Y  +  W . (I - P) Y,   P = B^T (B B^T)^{-1} B

    The first term is exactly the OLS plug-in. The second is the base weights
    applied to the part of the donors' post-treatment outcome that lies outside
    the row space of the pre-window. Unconstrained l2 balancing weights are of
    the form ``theta B``, which lies in that row space, so the residual term
    vanishes and the paper's collapse follows. Simplex weights are not of that
    form, so the term survives: it is the gap below.
    """
    print("\nPart 4 -- why simplex ASCM does not collapse to OLS")
    print(f"{'donors':>7} {'T0':>5} {'ASCM(lam->0)':>14} {'OLS plug-in':>13}"
          f" {'W.(I-P)Y':>11} {'residual':>10}")
    for n_donors, T0 in ((40, 12), (30, 20), (25, 25)):
        y_pre, Y0_pre, Y0_post = factor_panel(n_donors, T0, seed=3)
        B, A = build_matching(y_pre, Y0_pre, None, None)
        if np.linalg.matrix_rank(B @ B.T) < T0:
            continue
        res = ridge_augment_weights(y_pre, Y0_pre, lambda_=1e-10)
        est = float(res.W @ Y0_post)
        ols = ols_plugin(B.T, Y0_post, A)
        P = B.T @ np.linalg.solve(B @ B.T, B)
        extra = float(res.W_base @ ((np.eye(n_donors) - P) @ Y0_post))
        print(f"{n_donors:>7} {T0:>5} {est:14.6f} {ols:13.6f}"
              f" {extra:11.6f} {abs(est - ols - extra):10.2e}")

    print("\n  The same limit with l2 balancing weights in place of the simplex,")
    print("  through the identical mlsynth code path (base_weights_fn is a hook):")
    print(f"{'donors':>7} {'T0':>5} {'ASCM(l2, lam->0)':>18} {'OLS plug-in':>13}"
          f" {'abs diff':>10}")
    for n_donors, T0 in ((40, 12), (30, 20)):
        y_pre, Y0_pre, Y0_post = factor_panel(n_donors, T0, seed=3)

        def l2_base(Bm, Am, warm_start=None, _d=1e-6):
            return l2_balancing_weights(Bm.T, Am, _d) / Bm.shape[1]

        res = ridge_augment_weights(y_pre, Y0_pre, lambda_=1e-10,
                                    base_weights_fn=l2_base)
        est = float(res.W @ Y0_post)
        B, A = build_matching(y_pre, Y0_pre, None, None)
        ols = ols_plugin(B.T, Y0_post, A)
        print(f"{n_donors:>7} {T0:>5} {est:18.6f} {ols:13.6f} {abs(est - ols):10.2e}")
