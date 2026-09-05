# TWSF Path-B gate — Shen, Two-Way Synthetic Forecasting

The verification run that cleared `agents/future_integrations.md` section 17 for
a build. Section 17 parked TWSF because neither replication path could be
completed from v1 as published; v2 answers both blocking questions, and this
directory is the run that checked the harder one.

> Shen, D. Causal Forecasting in Panel Data: A Two-Way Synthetic Forecasting
> Approach. arXiv:2606.18512v2. Single author, no code release.

## What was blocked and what v2 changed

Section 17 recorded two unreported specifications. v2 resolves the case-study
one outright: `case_study.tex:77` states that TWSF is estimated on daily counts
while the figures display cumulative trajectories, and it adds the distinction
the earlier spike got wrong — validation targets a realised trajectory and so
takes a prediction interval `sqrt(V^2 + m sigma^2)`, while the counterfactual
targets a conditional mean and keeps the confidence interval.

For the simulation, v2 now gives the harmonic frequencies
(`2pi/(8 T*)`, `2pi/12`, `2pi/37`, `2pi/91`), the full eight-component basis,
the scaling rule (`max |<u_i, v_t(d)>| <= 0.8`), `sigma = 0.10`, the donor
factor construction, and the population ranks. It still does not give the
numeric entries of `A_0` and `A_1`, only that both are 4 x 8, fixed across
replications, and that the lowest-frequency harmonic is absent under control
and present under treatment.

`dgp_v2.py` therefore zeroes `A_0`'s first two columns and draws the rest once
from a fixed seed. The gate's question is whether that one remaining choice
still matters: with the scaling and noise level specified, does any
non-degenerate pair respecting the stated structure give nominal coverage?

## Result — it does

`gate_1000reps.out` is 100 latent replications x 10 noise replications, the
paper's own budget, with Monte Carlo standard errors clustered over the latent
draws as `simulations.tex` specifies.

| n | h=1 | h=5 | h=10 |
|---|---|---|---|
| 25 | 0.857 | 0.871 | 0.838 |
| 50 | 0.936 | 0.914 | 0.921 |
| 75 | 0.893 | 0.876 | 0.854 |
| 100 | 0.895 | 0.904 | 0.890 |
| 125 | 0.909 | 0.883 | 0.895 |
| 150 | 0.908 | 0.885 | 0.892 |

Nominal is 0.90 and the Monte Carlo standard errors are 0.008 to 0.020. The v1
reconstruction gave 0.39 to 0.87.

The bias now shrinks in `n`, which is what failed before. At `n = 150` it is
0.0005 to 0.0010 against a standard error near 0.011, a ratio of about 0.06;
the v1 reconstruction had bias 0.195 against a standard error of 0.084 at the
same dimension, and it did not shrink.

## The three diagnostics

Section 17's lesson was to separate "algebra wrong" from "variance wrong" from
"design wrong" before concluding anything. All three were re-run.

Algebra. With `sigma = 0` the forecast error is 0 to machine precision at every
`n` and `h` — 1e-13 to 1e-16 — confirming the Page-block layout, the
identification and every step of the recursion.

Variance. Empirical SD over mean plug-in SE runs 0.894 to 1.165 across the
grid, so the paper's plug-in variance formula is sound. Section 17 measured
0.92 to 1.08 on v1 and reached the same conclusion; the variance was never the
problem.

Spectrum. This is what changed. At `n = 150` the Page matrix's retained signal
spans 47x with its smallest direction at 0.586 against a noise floor of 0.10.
The v1 reconstruction spanned 500,000x with its smallest signal direction at
about 1e-4 against a noise floor of 3.7, so rank-8 PCR was inverting noise.

## Where the small-n shortfall comes from

Coverage sits below nominal at `n = 25` (0.838 to 0.871) and dips again at
`n = 75` for the longer horizons (0.854 at `h = 10`). The spectrum explains
both. Lag length equals `n` in this design, so a 25-period lag window cannot
resolve the `2pi/91` harmonic or separate the two `omega_0` directions, and the
Page spectrum is near-degenerate at the oracle rank — its eighth singular value
is 0.017 at `n = 25` against 0.586 at `n = 150`. The theory is asymptotic in
`n`, so a shortfall at the smallest panel with a spectral cause is consistent
with it, not evidence against it.

## One implementation note for the port

`HSVT(A, k)` has rank exactly `k`, so its pseudo-inverse must be built by
inverting those `k` singular values. Calling `np.linalg.pinv` on the truncated
matrix keeps numerically-zero directions under its default tolerance and
inflates the result without bound — at `n = 150` that turned a handful of
plug-in standard errors into ~1e8. `twsf.py` builds both pseudo-inverses from
the truncated factors; a port should do the same.

## Reproducing

    python run_gate.py 100 10        # the full grid, about 4.5 minutes
    python run_gate.py 5 3           # a quick smoke run

Needs only NumPy.
