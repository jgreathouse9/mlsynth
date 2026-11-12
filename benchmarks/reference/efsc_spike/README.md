# EFSC spike -- Rodriguez-Deniz & Blei, "Exponential Family Synthetic Controls"

Demonstrate-first spike. No estimator is added. This runs the authors' package
against the paper's own simulations, adds the baselines mlsynth already ships,
and measures the placebo test's size, which the paper does not report.

Sources:

* Paper: Rodriguez-Deniz, H. & Blei, D. M. (2026), arXiv:2609.23970v1.
* Code: <https://github.com/blei-lab/efsc> @ `adab68e` (MIT, PyTorch, one
  commit). Its 264 tests pass here. The manuscript-reproduction repository it
  points to (`blei-lab/efsc-paper-reproduction`) asks for credentials, so the
  paper's exact priors and initialisations come from the appendix text.

## Verdict

Park. The package reproduces the shape of the paper's Tables 1 and 2, and the
advantage it shows is real, but it does not come from the synthetic-control
step. On the counterfactual -- the missing untreated block, which is what a
synthetic control estimates -- EFSC ties `MCNNM` run on cell moments, a
baseline mlsynth already has. EFSC's margin comes from its second fit, a
low-rank smoother over the treated post-treatment cells alone. Pairing the
`MCNNM` counterfactual with that treated-side fit matches EFSC to the third
decimal. The placebo test is conservative under an exchangeable null, has good
power at the paper's effect sizes, and rejects at twice its nominal rate when
treated units are selected on a latent loading.

A build would add a torch-free port of a factorisation whose counterfactual
mlsynth can already produce. The part that is new, smoothing the treated block,
assumes the treatment effect itself is low rank across the treated cells; in
both of the paper's synthetic designs it is so by construction (a constant tilt;
a rank-one sigmoid product), which is why it helps there.

## What was run

`dgp.py` implements the generative model of Appendix B.2 with the settings of
C.3 (one-parameter tilts), C.5 (Gaussian benchmarks) and C.6 (placebo panels).
C.5 does not state its Gaussian generating scales; the ones used reproduce the
ranges in the paper's Figure 14. The multinomial panel is Medicaid-shaped (five
categories, the last as reference, rank 1 per logit) and is this spike's
construction.

`methods.py` calls `efsc.fitting.fit_factorization` twice per panel, with the
masks `EFSC.fit` builds in causal mode (`panel.py:515-516`): a counterfactual
fit without the treated post block, and a treated fit on that block alone.
Paper settings follow C.5 (N(0,1) priors, lr 0.1, 3000 epochs); `pmf_default`
uses the package's defaults (N(0,2^2), 2000 epochs). The two agree to within
0.005 everywhere in Table 1.

Baselines, all on the treated post block:

| name | what it is |
|---|---|
| `sc_means` | simplex SC on cell means (paper's row 1) |
| `mle_sc` | simplex SC on cell-MLE natural parameters (the paper's "EFSC-MLE") |
| `mle_mcnnm` | `mlsynth.MCNNM` on cell-MLE natural parameters |
| `mom_mcnnm` | `mlsynth.MCNNM` on cell means and log unbiased variances |
| `dsc` | `mlsynth.DSC` on the microdata, one treated unit at a time |
| `logit_mcnnm` | Haldane-corrected cell logits, `mlsynth.MCNNM` per logit |
| `compsc`, `compsc_halfm` | `mlsynth.COMPSC` per treated unit; default zero replacement 1e-6, or 0.5/m |
| `pmfcf_mletr`, `momcf_pmftr`, `pmfcf_momtr` | one method's counterfactual paired with another's treated side |

Tables use 3 replications per cell, the placebo study 100 null panels and 60
for each other scenario, each with 19 placebo assignments.

### The package's ELBO

`vi.py` evaluates log q through
`MultivariateNormal(mean, torch.diag(var))`, which forms and factorises a
dense D x D matrix every Adam step (D is about 960 at the paper's Gaussian
size). A fit took 97 s on four threads. `fastvi.py` computes the same diagonal
density as a sum; `check_fastvi.py` shows the fitted natural parameters agree to
5e-7, and the same fit takes 12 s. Every number below uses `fastvi`.

## Q1: does the package reproduce Tables 1 and 2?

Yes, in shape and roughly in level.

Table 1 (exponential tilt on the first natural parameter, N=32, T=128,
m ~ 1+Poisson(55)). Error in recovering the tilt on the natural-parameter
scale, MAE:

| method | tau=0.1 | tau=0.5 | tau=2.0 | paper (PMF / MLE) |
|---|---|---|---|---|
| pmf | 0.126 | 0.131 | 0.202 | 0.092 / 0.101 / 0.194 |
| pmf_default | 0.127 | 0.136 | 0.207 | |
| mle_sc | 0.204 | 0.228 | 0.410 | 0.180 / 0.195 / 0.367 |
| mle_mcnnm | 0.187 | 0.211 | 0.393 | |
| mom_mcnnm | 0.190 | 0.211 | 0.386 | |
| pmfcf_mletr | 0.187 | 0.208 | 0.390 | |

On the outcome scale every method lands between 0.17 and 0.19 at every tau, as
in the paper (0.161-0.169): with 56 draws per cell the treated cell mean is
noisy enough to swamp the differences.

Table 2 (structured latent intervention, kappa = (0.6, 0.5), fixed m), KL MAE:

| method | m=5 | m=25 | m=100 | paper m=5 (PMF / MLE) |
|---|---|---|---|---|
| pmf | 0.078 | 0.026 | 0.010 | 0.056 / 0.648 |
| mle_sc | 0.565 | 0.055 | 0.016 | |
| mle_mcnnm | 0.427 | 0.050 | 0.013 | |
| mom_mcnnm | 0.287 | 0.049 | 0.014 | |
| dsc | 0.541 | 0.067 | 0.021 | |
| pmfcf_mletr | 0.278 | 0.053 | 0.014 | |

The true KL averages 0.009 over the block, so at m=5 every method's error is
several times the quantity being estimated.

## Q2: is the advantage over a competent baseline?

Not on the counterfactual.

Counterfactual error alone, first natural parameter (Table 1 / Table 2):

| method | T1, any tau | T2 m=5 | T2 m=25 | T2 m=100 |
|---|---|---|---|---|
| pmf | 0.069 | 0.166 | 0.089 | 0.052 |
| mle_mcnnm | 0.071 | 0.648 | 0.108 | 0.052 |
| mom_mcnnm | 0.084 | 0.203 | 0.100 | 0.059 |

and the counterfactual mean in Table 2: pmf 0.091 / 0.046 / 0.027,
mom_mcnnm 0.079 / 0.051 / 0.030. EFSC is ahead on one scale and behind on the
other, by margins smaller than the gap between the two baselines.

Swapping sides settles where the KL margin comes from (Table 2, KL MAE):

| counterfactual | treated side | m=5 | m=25 | m=100 |
|---|---|---|---|---|
| EFSC | EFSC | 0.078 | 0.026 | 0.010 |
| MCNNM on moments | EFSC | 0.079 | 0.023 | 0.009 |
| EFSC | moments | 0.258 | 0.050 | 0.014 |
| MCNNM on moments | moments | 0.287 | 0.049 | 0.014 |

The treated-side fit is what moves the error, and its error on the treated
natural parameter shows why: 0.297 against 1.392 for the raw cell moments at
m=5. Table 1 decomposes the same way: `pmfcf_mletr` lands on `mle_mcnnm`.

The smoother is fit to the treated post block with the full unit and time
parameterisation, so it pools across treated cells. That pooling is exact when
the effect is a constant tilt or a smooth low-rank surface, which is what both
designs generate. A heterogeneous effect would be smoothed toward its low-rank
part; the paper has no design that tests this.

The multinomial panel tells the same story at the level users report:

| metric (m=20 / 100 / 1000) | efsc | logit_mcnnm | compsc_halfm | compsc |
|---|---|---|---|---|
| counterfactual prob MAE | 0.036 / 0.017 / 0.008 | 0.031 / 0.017 / 0.010 | 0.045 / 0.030 / 0.017 | 0.078 / 0.032 / 0.017 |
| block-average effect error | 0.013 / 0.004 / 0.003 | 0.015 / 0.005 / 0.002 | 0.017 / 0.010 / 0.002 | 0.030 / 0.010 / 0.002 |
| Medicaid effect (truth 0.184) | 0.202 / 0.181 / 0.186 | 0.193 / 0.177 / 0.185 | 0.181 / 0.164 / 0.181 | 0.149 / 0.162 / 0.181 |

At m=20 EFSC overstates the Medicaid shift by 0.018 (10 percent), the logit
completion by 0.009. Per-cell effect error favours EFSC (0.049 against 0.071 at
m=20), again through the treated-side fit.

`DSC` is behind everywhere in Table 2. It targets quantile functions with
simplex weights and is not built for five draws per cell.

### COMPSC zero replacement

At m=20, 56 percent of cells have a zero category. `COMPSC`'s default zero
replacement of 1e-6 puts log(1e-6) into the log-ratios and pulls the Medicaid
effect to 0.149 against a truth of 0.184; 0.5/m gives 0.181. This is a finding
about `COMPSC`'s default, separate from EFSC, and belongs on its own branch.

## Q3: the placebo test

Algorithm 2 with the C.6 panel (Poisson, N=16, T=32, rank 2, m=100, four
treated units, six post periods, 500 epochs), p = (1 + #{placebo >= observed}) /
20.

| scenario | panels | reject at 0.05 | reject at 0.10 | mean p |
|---|---|---|---|---|
| null, treated = last four units | 100 | 0.01 | 0.05 | 0.57 |
| tilt tau = 0.1 | 60 | 0.90 | 0.95 | 0.06 |
| null, treated = top-4 first loading (factor sd 0.3) | 60 | 0.10 | 0.18 | 0.48 |

Under the exchangeable null the test is conservative. The no-leakage rule drops
the actually treated post block from every placebo fit, so placebo
counterfactuals are trained on fewer cells than the observed one, and placebo
statistics run high (mean 0.0041 against 0.0036 observed). With treated units
chosen on a loading the rejection rate doubles to 0.10 (6 of 60; standard error
about 0.04 at the nominal rate). The paper reports neither number.

In C.7.5 the pre-target fit trains on
the target units' own pretreatment cells, so ECD_pre is an in-sample smoothing
discrepancy while ECD_post is out of sample. Delta is positive under the null
for that reason, which is why the paper calibrates against placebos and not
against zero.

## Departures and gaps in the package

1. Algorithm 2 and the Medicaid posterior-predictive placebo (Algorithm 3) are
   not in the package; `run_placebo.py` implements Algorithm 2 from C.6/C.7.5.
2. The default prior is N(0, 2^2); C.5 uses N(0, 1). It makes no measurable
   difference in Table 1.
3. Optimisation runs a fixed epoch count with no convergence check.
4. The paper's EFSC-MLE and SC baselines are not shipped.

## Files

| file | contents |
|---|---|
| `dgp.py`, `methods.py` | panels and estimators |
| `fastvi.py`, `check_fastvi.py`, `fastvi_check.json` | the diagonal ELBO and its equivalence check |
| `run_gaussian.py` | Table 1, Table 2, the swap |
| `run_multinomial.py` | the multinomial head-to-head |
| `run_placebo.py` | Algorithm 2 size and power |
| `analyse.py`, `results.json` | summaries |
| `*.csv`, `*.log` | raw results |
| `manifest.json`, `provenance.json` | commands, versions, upstream SHA |
