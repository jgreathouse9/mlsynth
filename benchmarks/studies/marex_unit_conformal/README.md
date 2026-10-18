# Conformal inference for each treated unit of a MAREX design

MAREX tests and bands the aggregate contrast, the synthetic treated minus the
synthetic control. The question here is whether the same inference can be had
for each treated unit separately, beside the aggregate.

Abadie and Zhao's Unit-level design (their formulation 10) gives each treated
unit `j` its own synthetic control `v_.j`, fitted on the fitting periods from the
untreated units, and the aggregate is the `w`-weighted average of the per-unit
contrasts (their eq. 11):

    u_jt = Y_jt - sum_i v_ij Y_it,        tau_hat_t = sum_j w_j u_jt.

Each `u_j` has a blank stretch and a post stretch, so every piece of MAREX's
inference applies to it unit by unit. The argument for Theorem 2 needs only
weights chosen on the fitting periods and exchangeable factors over the blank
and post periods; `v_.j` is such a weight vector, and Assumption 3(i) is read as
an exact fit of unit `j`'s own predictors (or the paper's relaxation,
exchangeable `(theta_t, lambda_t)`).

MAREX today has no per-unit controls. `design="unit_penalized"` fits each unit
against every other unit once, keeps only the residual of that fit as the `xi`
penalty, and reports the effect through the shared control the `standard`
objective fits to the population mean, not eq. 11. The per-unit controls here
are refitted after the design, when the assignment is known, so eq. 10's
constraint `v_ij = 0` for treated `i` can be imposed.

## Design

`units.py`, 300 draws a cell. The panel is `marex_cumulative_conformal`'s:
Abadie and Zhao's Section 5 factor model, `J = 15` units, `N(0, 1)` noise, under
four regimes for the factors: `iid` (Theorem 2's assumption), AR(1) at 0.5 and
0.8, and `sorted` (trending). Each draw solves a `standard` design and a
`unit_penalized` design (`xi = 1`), `m = 3` treated, on the same panel, with 40
fitting and 13 post periods, and 26 blank periods (`mid`) or 12 (`short`, below
the 19-period floor at 5%).

Per treated unit, two contrasts:

- `unit`: `v_.j` refitted on the fitting periods from the untreated units.
- `naive`: the group's shared control, `Y_j - Y v`, what reading one unit off
  today's result gives.

Tests and bands on each, all under no effect except the totals:

- Theorem 2's permutation p-value, 999 drawn arrangements, `(1 + k)/(1 + B)`.
- Theorem 3's pointwise band with the split-conformal quantile (#744).
- The band for the unit's total by inversion, evaluated at the true total under
  a constant effect and a ramp-shaped one: an SC refit on every period, the
  signed post sum, cyclic shifts over the blank and post periods.
- Across the three units: any raw `p_j <= alpha`, Bonferroni, and a joint test,
  `max_j` of `mean_post |u_j| / mean_window |u_j|` with one draw of periods
  applied to every unit. Under no effect at all Holm rejects exactly when
  Bonferroni does.

And the aggregate, the shared contrast against eq. 11's `sum_j w_j u_j`.

Logs: `results/units_{iid,ar5,ar,sorted}.log`.

## Results

### Per-unit inference holds where the aggregate's does

Under `iid` factors, `mid`, rejection under no effect and coverage at
5% / 20%:

| design | per-unit size | pointwise coverage | total, constant | total, ramp |
| --- | --- | --- | --- | --- |
| standard | 0.060 / 0.189 | 0.962 / 0.818 | 0.972 / 0.818 | 0.992 / 0.888 |
| unit_penalized | 0.047 / 0.189 | 0.968 / 0.817 | 0.974 / 0.811 | 0.988 / 0.902 |

Each per-unit size pools 900 unit-draws, a standard error of 0.007 at 5%. With
12 blank periods the 95% pointwise band is infinite, as the floor says it must
be, and the 80% band (floor 4) covers 0.85; the inversion's reference is the 25
blank and post periods, so its band is unaffected (0.966 / 0.80).

### Three units are three tests

Rejecting when any unit's `p_j <= alpha` falsely rejects at
`1 - (1 - alpha)^3`: 0.11-0.17 at 5% and 0.45-0.49 at 20% under `iid`.
Bonferroni and the joint test both hold the family at its level (0.03-0.05 at
5%, 0.16-0.22 at 20%). At three units the joint test gains nothing over
Bonferroni; the units' contrasts share the factors, but not enough to tighten
the maximum.

### A unit is noisier than the average

The per-unit contrast does not average the noise across the treated units:
its blank-period RMS is 3.1-3.6 against 1.5-1.8 for the aggregate, and the 95%
half-width 6.7-7.7 against 3.4-4.0. A per-unit band is about twice as wide as
the aggregate's on the same design.

### The refit matters

The shared control fits the population mean, not unit `j`. Read off it, a unit's
blank RMS is 4.8-6.6 and its half-width 9.5-11.7, against 3.1-3.6 and 6.7-7.7
refitted. Under trending factors the shared control also fails the test: per-unit
size 0.42-0.48 at 5%, against 0.30-0.39 refitted.

### The unit-level penalty buys per-unit precision

`unit_penalized` chooses treated units the others reproduce well, and its
per-unit contrasts are tighter: blank RMS 3.1 against 3.6 under `iid`, 2.2
against 3.0 trending; half-width 6.7 against 7.7.

### Eq. 11 against the shared control

Post-period RMS of the aggregate effect under no effect, `mid`:

| factors | standard: shared | standard: eq. 11 | unit_penalized: shared | unit_penalized: eq. 11 |
| --- | --- | --- | --- | --- |
| iid | 1.42 | 1.59 | 1.70 | 1.65 |
| AR 0.5 | 1.50 | 1.66 | 1.66 | 1.65 |
| AR 0.8 | 1.51 | 1.78 | 1.74 | 1.77 |
| trending | 1.12 | 1.36 | 1.37 | 1.25 |

On a `standard` design the shared control, which the design itself optimised,
is the better aggregate by 11-21%. On a `unit_penalized` design the two are
within 3% except trending, where eq. 11 is 9% better. Size and coverage do not
separate them.

### Serial dependence

Per-unit inference inherits the aggregate's limits. With factors at 0.8 the
per-unit size is 0.22 at 5%, the aggregate's 0.20-0.22; trending, 0.30-0.39
against 0.25-0.34. The per-unit band for the total by inversion holds best, as
it did for the aggregate: 0.94-0.95 at 95% at 0.8, 0.92-0.94 trending, and the
ramp 0.96-0.99.

## Recommendation

Build it, on its own branch after #744's fixes, which it inherits (the
conformal quantile, the floors, the `(1 + k)/(1 + B)` p-values):

1. After the design, refit `v_.j` for each treated unit on the fitting periods
   from the untreated units of its cluster.
2. Run MAREX's inference on each `u_j`: the Theorem 2 p-value, per-period
   p-values, the pointwise band, and the band for the total by inversion.
3. Report the family: raw per-unit p-values, Bonferroni-adjusted ones, and the
   joint test.
4. Report eq. 11's aggregate beside the shared one, with its own inference.

## References

Abadie, A. and J. Zhao (2026). Synthetic controls for experimental design.

Chernozhukov, V., K. Wuthrich and Y. Zhu (2021). An exact and robust conformal
inference method for counterfactual and synthetic controls. Journal of the
American Statistical Association 116(536), 1849-1864.

Lei, J., M. G'Sell, A. Rinaldo, R. J. Tibshirani and L. Wasserman (2018).
Distribution-free predictive inference for regression. Journal of the American
Statistical Association 113(523), 1094-1111.

Phipson, B. and G. K. Smyth (2010). Permutation p-values should never be zero.
Statistical Applications in Genetics and Molecular Biology 9(1), Article 39.
