# Path C spike: augmented balancing weights, applied to mlsynth's ASCM

> Bruns-Smith, D., Dukes, O., Feller, A., & Ogburn, E. L. (2026). "Augmented
> balancing weights as linear regression." *JRSS-B* 88(3), 699-723.
> doi:10.1093/jrsssb/qkaf019. Replication package:
> `github.com/bruns-smith/balance-equiv-jrssb` (Python, MIT).

The paper is cross-sectional and out of scope as an estimator: `X, Y, Z` i.i.d.
with a binary treatment, target `E[Y(1) - Y(0)]`, no time index and no donor
pool. `dataprep` cannot read it.

Its theorems are still about an estimator mlsynth ships. Ridge-augmented
synthetic control (Ben-Michael, Feller & Rothstein 2021) is available as
`VanillaSC(augment="ridge")` and implemented in
`mlsynth/utils/bilevel/ridge_augment.py`, and Feller is an author of both
papers. Section 2.3 names the Synthetic Control Method and "their augmented
analogues" as the nonnegativity-constrained case of its own framework.

This spike asks three questions and answers all three. Nothing here is
registered in `benchmarks/registry.py`: it is a recommendation, not a case.

## Running it

```
python benchmarks/reference/balance_equiv_spike/check_equivalence.py
python benchmarks/reference/balance_equiv_spike/check_ascm.py
python benchmarks/reference/balance_equiv_spike/check_real_panels.py
```

`check_equivalence.py` part 1 needs the authors' `data/nsw_psid_*.csv`; clone
their repo and point `LALONDE` at it. The other two run against `basedata/`
and simulated panels, so they need nothing external. All three are seconds.

## 1. Does Proposition 4.3 hold?

Yes, to machine precision, in both geometries. Ridge outcome model at `Lambda`
augmented with `l2` balancing weights at `delta` is a *single* generalized
ridge at

    gamma_j = delta * lambda_j / (sigma_j + lambda_j + delta)  <=  lambda_j

so augmentation is exactly undersmoothing, with a closed form for how much.

| geometry | cells | worst abs difference |
| --- | --- | --- |
| the authors' LaLonde data, 11 and 171 features | 32 | 1.6e-09 |
| simulated panels, donors as units and pre-periods as features | 27 | 1.3e-12 |

Panels invert the paper's aspect ratio as often as not, so the panel sweep
covers `J = 40, T0 = 12`, `J = 12, T0 = 40` and `J = T0 = 30`. The equivalence
does not care.

The `delta -> 0` collapse to the OLS plug-in also reproduces: 2.98e-08 on the
11-feature LaLonde design, 6.58e-12 on the `J = 40, T0 = 12` panel. Where the
design is rank deficient the two sides are pseudo-inverse solutions and agree
only to 1.6e+00 on 5.2e+03 (171 features) and 7.2e-05 (`J = 12, T0 = 40`).

## 2. Is mlsynth's ASCM an instance of the paper's equation (7)?

Yes, exactly, and the correspondence is explicit. With `B` the centered donor
pre-matrix, `A` the centered treated pre-vector and `W` the base simplex
weights, `ridge_augment_weights` computes `W_ridge = M (B B^T + lam I)^{-1} B`
with `M = A - B W`, so the prediction is `W . Y_post + M . beta` for
`beta = (B B^T + lam I)^{-1} B Y_post`. Put `Xp = B^T` and `Sigma = B B^T / J`:
that `beta` is the paper's generalized ridge coefficient at `Lambda = lam / J`,
and `M` is its residual feature shift. Checked over 12 cells, worst difference
2.3e-12.

So mlsynth's ASCM is equation (7) with a ridge outcome model and **simplex**
balancing weights.

## 3. Does the `delta = 0` trap have a panel analogue here?

This is the finding that matters. The paper's alarming result is that its
tuning picks the degenerate hyperparameter on 56 percent of draws (Table 1),
landing on OLS, which is never optimal across its 36 DGPs. Two things stop
that from transferring.

**The collapse target is not OLS.** At `lam -> 0` with `B B^T` invertible,

    W_aug . Y = A (B B^T)^{-1} B Y  +  W . (I - P) Y,    P = B^T (B B^T)^{-1} B

The first term is the OLS plug-in; the second applies the base weights to the
part of the donors' post-treatment outcome outside the row space of the
pre-window. `l2` balancing weights have the form `theta B`, which lies in that
row space, so the term vanishes and the paper's collapse follows. Simplex
weights do not, so it survives. Measured, the decomposition is exact:

| J | T0 | ASCM at lam -> 0 | OLS plug-in | `W . (I - P) Y` | residual |
| --- | --- | --- | --- | --- | --- |
| 40 | 12 | 4.328869 | 3.923511 | 0.405357 | 1.6e-12 |
| 30 | 20 | -0.502091 | -0.417980 | -0.084111 | 5.4e-10 |

Swapping the simplex for `l2` weights through the *same* mlsynth code path
(`base_weights_fn` is an injectable hook) makes the collapse appear exactly:
3.923511 against 3.923511, difference 8.9e-16. So the nonnegativity constraint
is what breaks the equivalence, which is the case the paper sends to
Supplementary Appendix D.2 and treats as sample trimming.

**mlsynth's CV does not go there anyway.** Over 120 simulated panels the
1-SE rule lands on the grid floor once, and on the three real panels never:

| panel | J | T0 | CV lambda | at grid floor | ATT (CV) | ATT (lam -> 0) | ATT (OLS) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kansas tax cut | 49 | 89 | 0.0787 | no | -0.04006 | -0.06699 | 10.77700 |
| Prop 99 (California) | 38 | 19 | 429.8 | no | -15.95257 | -12.37436 | 86.57067 |
| German reunification | 16 | 30 | 1.49e+05 | no | -1333.11 | -5251.24 | 21220.16 |

Kansas under CV reproduces the `augsynth` R reference of -0.0401 that
`benchmarks/cases/ascm_kansas.py` pins, so the path being probed is the
validated one.

The last column is the paper's warning priced on panels, and it is worse here
than in its own application: the unpenalized plug-in is not merely suboptimal
but nonsensical, +10.8 log points on Kansas and +21220 on Germany against
CV-selected estimates of -0.04 and -1333. The degenerate end costs a factor of
1.7 on Kansas and 3.9 on Germany even without reaching OLS.

## What this settles

- The theorem reproduces, and holds in panel geometry.
- mlsynth's ASCM is an instance of the framework, at `Lambda = lam / J`.
- The failure mode the paper documents does not fire in mlsynth, for two
  independent reasons: the simplex constraint changes the collapse target, and
  the 1-SE CV does not select the degenerate end.

The third point is the one to re-check if the ridge penalty grid, the 1-SE
rule, or the base solver ever changes. `check_ascm.py` part 3 and
`check_real_panels.py` are that check, and take seconds.

There is no estimator to add.
