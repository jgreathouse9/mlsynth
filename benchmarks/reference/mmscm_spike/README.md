# MMSCM spike — Kato & Ohda, moment-matching synthetic control

Assessment material for a candidate estimator that was not built. The paper
review recommended one decisive prototype; this is that prototype and its
result.

> Kato, M., & Ohda, A. (2025). *Asymptotically Unbiased Synthetic Control
> Methods by Moment Matching*. arXiv:2307.11127v5 (econ.EM).
> Reference implementation: `github.com/MasaKat0/mmscm` (MIT).

## The method

Weights are chosen on the simplex to match `G` raw moments of the treated
unit's pre-period outcome to the weighted sum of the donors' (their eq. 4), on
outcomes rescaled into [0, 1]. Moments are taken over the pre-period time
dimension, so this consumes an ordinary aggregate panel — unlike `DSC`
(Gunsilius), which matches quantile functions and needs micro-level cells.
The claimed gain is asymptotic unbiasedness under a mixture model, addressing
the implicit endogeneity of Ferman & Pinto (2021).

## What was tested

`G` moment equations for `J` weights leaves the objective nearly flat, and the
reference's SLSQP stops at different points from different starts. The
prototype asked whether a stated tie-break rule fixes that, and what the
resulting estimator is worth against baselines the paper does not run.

`mmscm_oracle.py` solves the objective properly (CLARABEL) and then selects one
point of the near-optimal set by a stated rule: `minnorm` takes the point
closest to uniform, `pathfit` takes the one that best fits the pre-period path.
`dgp_verbatim.py` transcribes the authors' own section 7.1 design from their
`Figure2_Figure3_Simulation_Treatment_Effect.ipynb`, since their notebook
differs from the paper's text in three ways that change what is measured (see
that file's docstring). `run_panels.py` runs the three empirical panels;
`run_mc_verbatim.py` runs the simulation. Set `MMSCM_REPO` to a clone of the
authors' repository to include their implementation in the comparison;
without it the runners skip that row.

## Results

Tie-breaking fixes the determinism, completely. Over 8 random starts on Prop 99
the published solve moves by L1 1.0059 in the weights and gives ATTs from
−33.77 to −31.19; with the `pathfit` tie-break the spread is 0.0000 and the ATT
is identical to every digit.

The selector matters more than the tie-break. `minnorm` was the rule the review
proposed, and it is the wrong one: minimising the norm on a simplex pulls
towards uniform, which is where the loose solve already sat.

| Prop 99 | ATT | pre-RMSE | active donors |
|---|---|---|---|
| reference `mmscm.py` (SLSQP) | −32.25 | 8.91 | 34 |
| MMSCM, `minnorm` tie-break | −33.03 | 9.27 | 28 |
| MMSCM, `pathfit` tie-break | −22.79 | 3.66 | 5 |
| Abadie SC | −19.51 | 1.66 | 6 |
| demeaned SC (Ferman & Pinto) | −11.11 | 0.96 | 9 |

Abadie, Diamond & Hainmueller (2010) report about −19. The same ordering holds
on the Basque and German panels, where the moment constraint costs a pre-period
RMSE 3.4 to 4.6 times Abadie SC's.

The reference implementation does not solve its own objective. On Prop 99 its
SLSQP reaches a loss of 1.11e-05 against a true optimum of 3.80e-10, a factor
of about 29,000. On the German panel it returns ATT +837 with a pre-period RMSE
of 711 — the wrong sign, against −1297 for Abadie SC and −533 to −798 for every
properly solved MMSCM variant.

## Why the paper's simulation favours it

On the authors' own DGP at their own 100 replications, an equally weighted
donor average — no fitting, no moments, no optimisation — beats MMSCM in
**9 of 9 cells**, by factors of 1.13 to 1.91. `mc_100reps.out` is that run.

MMSCM's weights sit 0.25 to 0.64 from uniform in L1 (the simplex diameter is
2.0), and the closer they are the smaller its gap to the uniform average. The
DGP generates independent random walks, so averaging `J` of them cuts variance
by about `1/J` while any method that fits the pre-period path — Abadie SC, and
`pathfit` MMSCM alike — chases one realisation and extrapolates it. That is the
whole of the reported advantage over Abadie SC: a diversification effect from
weights that happen to be near-uniform, not a gain from matching moments.

Two further checks fail. The paper reports the error declining as `G` grows;
across `J` in {5, 15, 30} and `G` in {2, 5, 10} the pattern is non-monotone and
at `J = 30` increasing. And the DGP random-walks the outcome's mean with a
per-period step of `10 * sqrt(J + 1)`, so the simulation validating the theory
violates the theory's own Assumption 5.7, which requires a stationary, strongly
mixing error process.

## Conclusion

Pass. Recorded in `agents/future_integrations.md` section 23.
