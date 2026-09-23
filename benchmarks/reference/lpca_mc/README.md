# `lpca_mc` — Feng's Section 5 designs, run against his own R

A captured cross-validation bundle for
[`benchmarks/cases/lpca_mc.py`](../../cases/lpca_mc.py). The reference is the
author's simulation harness,
[`Simulation/Feng-2024_LPCA_simul_suppfuns.R`](https://github.com/yingjieum/Replication_NonlinearFactorModel_2023/blob/main/Simulation/Feng-2024_LPCA_simul_suppfuns.R),
pinned at `ca34fba` and run live.

## Why the panels are fixed

Feng's Table 1 draws a fresh panel for each of 2000 replications. That is the
right design for a Monte Carlo and the wrong one for comparing two
implementations: R and NumPy have no shared random stream, so the two sides
would be estimating on different data and any agreement could only be claimed up
to Monte Carlo error.

`make_panels.py` writes the panels once and both implementations read the same
bytes. The sampling channel is gone, so every cell in `reference.json` is an
exact claim about the two estimators and not about their sampling variability.

The statistical reproduction of Table 1 at the paper's own scale is a separate
exercise, recorded in `../lpca_kansas/` (`run_simulation.py`, and the Path B
section of that README): 500 replications at `n = p = 1000`, median
disagreement 0.83 Monte Carlo standard errors across the 48 cells.

## What is here

```
make_panels.py       writes the fixed panels (regenerate: python make_panels.py)
panel_model{1,2,3}.csv.gz    the observed matrices, n = p = 150
latent_model{1,2,3}.csv      alpha and eta; the noise-free surface is a closed
                             form of these, so no second matrix is stored
evaluated_model{1,2,3}.csv   the three units whose last-period cell is zeroed
manifest.json        case metadata + data checksummed by generate.py
reference.R          the reference run
reference.out        its verbatim stdout          (generated)
reference.json       the parsed values the case pins against  (generated)
provenance.json      versions, OS, git SHA, data checksums     (generated)
```

```bash
bash benchmarks/R/install_feng_lpca.sh
python benchmarks/reference/lpca_mc/make_panels.py   # only to redraw the inputs
python benchmarks/reference/generate.py lpca_mc
python benchmarks/run_benchmarks.py --case lpca_mc
```

## Panel size

`n = p = 150` instead of the paper's 1000, so the inputs stay small enough for
version control. The neighbourhood grid that size implies is 14, 28, 42, and
that is a useful accident: the singular-value ratio rule cannot fire at 14 and
does fire at 28 and 42, so a single run covers both branches. The full-scale
grid (49/99/149) exercises only the live one.

Model 3 is binary, so its panel compresses to a few kilobytes while the two
continuous panels take about 150 KB each.

## Statistics

Per model and per `K`, the two Table 1 quantities: the maximum absolute error
over every cell of the held-out block, and the prediction error at the three
units whose latent variable sits at the 0.1, 0.5 and 0.9 sample quantiles. Those
three have their last-period entry zeroed, which is the simulation's stand-in
for a treated cell.

The bundle also records a global-PCA baseline (`HDRFA::PCA_FN` for the factor
count, as Feng's script uses). The case reports it for context, since Table 1's
comparison is against it, but does not pin mlsynth to it — global PCA is not
mlsynth's code.

## Result

Both estimators agree cell for cell. 35 of the 36 pinned cells land within
5e-11, the print precision of `reference.out`; `model2_K14_mae` lands at 1.2e-07,
which is `irlba`'s iterative tolerance surfacing in the statistic most exposed
to it -- a maximum over 22 500 cells reports whichever one the two disagree on
most. mlsynth takes an exact eigendecomposition of the neighbourhood's Gram
matrix where the reference calls `irlba`, so that gap is the reference's
approximation, not the port's.

Two behaviours the paper describes show up on these fixed panels. Model 3 is
binary, so the pseudo-max distance ties and the threshold selection rule widens
a nominal 14-unit neighbourhood to 36 -- which is why the estimator reports the
realised neighbourhood next to the requested one. And model 1, the most
nonlinear design, degrades as `K` grows (1.08 at 14, 1.69 at 28, 3.98 at 42),
the same failure Table 1 records at `K` = 149.
