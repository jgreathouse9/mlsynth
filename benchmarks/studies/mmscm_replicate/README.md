# Moment-matching SCM, demonstrated before it is built

Reference: Masahiro Kato and Akari Ohda (2026), "Asymptotically unbiased
synthetic control methods by moment matching", *Journal of Causal Inference*
14(1):20250008, DOI
[10.1515/jci-2025-0008](https://doi.org/10.1515/jci-2025-0008). Section 5.3 is
the estimator, Assumption 5.6 its identification condition, Theorem 5.10 the
convergence result, Section 7.2 the three empirical applications. Reference
implementation: [MasaKat0/mmscm](https://github.com/MasaKat0/mmscm), cloned at a
pinned commit by `benchmarks/reference/clone_mmscm.py` and never vendored, since
it is GPL-3.0 and mlsynth is MIT.

This is a replication spike. It produces a recommendation, not estimator code.
The recommendation is: do not build it.

```
python -m benchmarks.studies.mmscm_replicate.run
python -m benchmarks.studies.mmscm_replicate.run --covariates
```

## The method

MMSCM chooses simplex weights so that, for each moment `k`, the time average of
the residual is zero:

    loss(w) = sum_k v_k * ( mean_t [ y_k,t - sum_j w_j y_k,j,t ] )^2

Abadie's objective is the same expression with the square inside the average,
`mean_t[ (.)^2 ]`. That transposition is the whole method: Abadie matches the
pre-period path, MMSCM matches `K` scalar averages of it. The motivation is
Ferman and Pinto (2021): the least-squares SC weights are biased because the
donor outcomes correlate with the SC error, and matching moments of a mixture
model instead removes that bias asymptotically.

The moments are the raw features plus powers of the first one. With one feature
and `poly = p` there are `p - 1` moments; with covariates the raw covariate
series are moments too. `poly = 2` appends no powers at all.

## What was asked, and what came back

Three questions, in the order it was cheapest to answer them.

### 1. Does a port reproduce the authors' arithmetic? Yes.

Ingesting through `dataprep` and solving the same program agrees with the
reference to 1e-10 on weights, in both outcome-only and covariate modes:

| panel | max abs weight diff | with covariates |
| --- | --- | --- |
| Prop 99 | 9.1e-11 | 1.7e-10 |
| Basque | 2.2e-09 | 4.7e-08 |
| Germany | 3.2e-10 | 1.5e-08 |

`port.py` is the readable oracle. It is deliberately not an estimator: no
config, no result object, no inference.

### 2. Is the Abadie baseline solved? No.

The paper's comparison runs through `MMSCM(method="Abadie")` in the same file.
That solve stops early on every panel:

| panel | authors' iterations | converged | ATT theirs | ATT converged | error |
| --- | --- | --- | --- | --- | --- |
| Prop 99 | 6 | 115 | −30.66 | −19.41 | −11.25 (58%) |
| Basque | 11 | 79 | −0.763 | −0.896 | +0.134 (15%) |
| Germany | 5 | 129 | −480.5 | −1302.1 | +821.6 (63%) |

The mechanism is a stopping rule that is not scale-invariant. SLSQP's `ftol` is
an absolute tolerance on the objective; the features are max-scaled to about 1
and then averaged, so the objective sits at 1e-4 to 1e-5, where the default
`ftol=1e-6` fires almost immediately. Both runs report `success=True`.

Two things push the objective into that range. The max-feature scaling in
`_data_setup`, which the port shares; and, in the Abadie branch only, a weight
vector built with the wrong length — `init_beta2 = np.ones(len(treated_final.T))`
gives one entry per *period* where one per *moment* is required, so with a
single feature it silently becomes `1/T` and scales the objective down by a
further factor of `T`. On Prop 99 the two contribute 1.080 and 10.358 packs,
summing to the observed 11.438.

The invariant this violates: minimising `f` and `c·f` for `c > 0` must give the
same argmin. Rescaling the Prop 99 objective across six positive constants moves
the ATT by 8.8 packs, and below `c = 1e-2` the solver stops after one iteration.

Converged, the baseline lands on the published answers where the published fit
is outcome-driven:

| panel | converged weights | published |
| --- | --- | --- |
| Prop 99 | Utah .394, Montana .232, Nevada .205, Connecticut .109 | Utah .334, Nevada .234, Montana .199, Colorado .164, Connecticut .069 |
| Germany, covariates | Austria .585, Japan .172, USA .156, Switzerland .088 | Austria .42, USA .22, Japan .16, Switzerland .11, Netherlands .09 |
| Basque, covariates | Cantabria .467, Madrid .317, Cataluna .162 | Catalonia .851, Madrid .149 |

The published weights come from covariate-matched fits with a `V` matrix on
pre-period averages, while this code matches covariate *series* period by
period, so exact agreement is not expected. Prop 99 recovers four of five
donors; Germany four of five with Austria correctly dominant; Basque finds both
published donors but ranks a third above them.

### 3. Are the moment-matched weights identified? Only sometimes.

The system is linear in `w`, so the solution set is an affine subspace of
dimension `J - rank(M)` intersected with the simplex, where `M` stacks the
donor time-averages. Below a null dimension of roughly ten the non-negativity
bounds collapse that intersection to a point; above about twenty-five they
cannot. Spread is the range of the ATT over 25 random simplex starts at a
tightened tolerance, so premature stopping is ruled out:

| panel | J | poly | moments | null dim | cond(M) | ATT spread |
| --- | --- | --- | --- | --- | --- | --- |
| Prop 99 | 38 | 11 | 10 | 27 | 3.4e+09 | **4.95 packs** |
| Prop 99 + retprice | 38 | 11 | 11 | 26 | 4.5e+09 | **4.95 packs** |
| Basque | 16 | 2 | 1 | 14 | 1.0e+00 | 0.33 k$ |
| Basque | 16 | 6 | 5 | 10 | 9.5e+03 | 3.6e-06 |
| Basque + covariates | 16 | 2 | 7 | 9 | 1.0e+02 | 3.4e-06 |
| Germany | 16 | 2 | 1 | 14 | 1.0e+00 | **2237 $** (sign flips) |
| Germany + covariates | 16 | 11 | 13 | 2 | 4.3e+08 | 4.0e-03 |

Germany with one moment is the extreme: the ATT ranges −1514 to +724 depending
on where the solver starts.

The condition numbers say what the rank hides. Powers of a smooth series are
nearly collinear, so `K` counts moments but not information: at cond 1e9 the
later polynomial moments are numerical noise, which is why ten of them buy
Prop 99 nothing while six distinct covariates pin Basque immediately.

## The recommendation

Pass. Two reasons, and only the first is fatal.

The comparison the paper rests on is against a baseline its own code does not
solve, by roughly 60% of the effect on two of three panels. Repairing it moves
classical SC onto the published answers and leaves MMSCM 20–65% away from them,
with 3–6× worse pre-period fit on Prop 99 and Germany.

And identification needs the null dimension small, which means a donor pool
small relative to the moment count. That is the regime where plain SC was never
in difficulty; the large-donor-pool application that motivates synthetic control
is exactly where MMSCM is not identified, and polynomial moments cannot rescue
it because they are collinear.

`DSC` (Gunsilius 2023) stays mlsynth's distributional answer. The gap this paper
aims at — a distributional synthetic control that runs on an aggregate panel
instead of micro data — is real and remains open.

## If anyone ports this anyway

Three fragilities in the reference, beyond the stopping rule:

- `train_param` raises `treated_final[0]` to powers, so the polynomial moments
  come from whichever numeric column is first. Put a covariate before the
  outcome and the moments are of the covariate.
- `_data_setup` treats *every* numeric column as a matched feature, so an
  identifier column (Basque's `regionno`) would silently become a moment.
- `abadie_obj_func2` normalises its argument in place (`beta2 /= np.sum(beta2)`)
  while scipy is iterating on it.

A port would need a `hypothesis` property pinning the invariant the stopping
rule breaks: fitted weights are invariant to a positive rescaling of the
objective.

## Files

```
run.py       the three questions, as one CLI; writes results/*.json
port.py      the readable oracle -- the method in about eighty lines
results/     mmscm_replicate.json, mmscm_replicate_covariates.json
```
