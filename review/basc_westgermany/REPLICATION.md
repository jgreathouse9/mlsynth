# Replication package for the BASC West Germany referee report

Every numerical claim the report makes is recomputed here from the panel and
checked against the value the report states. Nothing is transcribed from a run
that is not in this package, and nothing is hardcoded.

## One command

```
./run_all.sh                # short chains, about 20 minutes
./run_all.sh --full         # adds the 25000/25000 chains, about 2 hours
./run_all.sh --verify-only  # re-check against the CSVs already here
```

It clones the authors' sampler, installs `mlsynth` from `main`, derives the
patched sampler files from the authors' own code, runs the MCMC diagnostics, and
then recomputes every claim and compares it against the report. It exits
non-zero if any claim fails, and writes `verification.json`. Completed steps are
skipped on a re-run, so an interrupted run resumes.

Rendering the report is not part of this. `verify.py` needs no Quarto and no
LaTeX; if Quarto happens to be installed the report is rendered afterwards as a
convenience, and a failure there does not affect the verification.

The check itself:

```
python verify.py                # the deterministic claims, about a minute
python verify.py --with-mcmc    # also the claims read from the R outputs
```

Each line prints the claim, the value the report states, and the value computed
here. Deterministic claims are held to a tight tolerance; MCMC claims are given
a band, since a chain reproduces to its seed and not to a decimal.

PyPI carries `mlsynth` 1.0.0 and the report needs 2.x, so the install comes from
GitHub. A short run merges into the shipped CSVs on `config`, `N` and `seed`, so
it updates what it recomputed and leaves the long chains in place.

## What is not included

The authors' Gibbs sampler is not redistributed. `scripts/prepare_sampler.py`
derives what the diagnostics need from their own file, so every modification is
visible as code:

```
git clone https://github.com/sll-lee/paper-BASC
python scripts/prepare_sampler.py paper-BASC/BASC_realdata.R
```

That writes three files. `basc_funcs.R` is their function definitions extracted
verbatim. `basc_funcs_g1.R` adds four arguments, one of which skips the loop
resampling the donor-inclusion indicators. `basc_funcs_q.R` additionally
replaces `Dt * alpha.v` with `Dt %*% alpha.v` at the five sites where it
appears, which generalises the post-treatment term from a single constant to a
basis; the published code supports only the constant.

Run with the switches off, the derived code reproduces the original sampler
exactly. That is the control, and it is checked below.

## Requirements

Python 3.11 with `pip install -e ".[bayes]"` for `mlsynth`, plus `jupyter`,
`matplotlib` and `tabulate` (see `requirements.txt`). Quarto 1.7.8 and a LaTeX
engine for the PDF. R 4.3 with `MASS`; the R scripts supply `rinvgamma`,
`rtmvnorm` and `rdist` themselves so no other R package is needed.

Run everything from the package root; the report locates the panel by
walking up for a `basedata` directory.

## Order

```
git clone --depth 1 https://github.com/sll-lee/paper-BASC
pip install "mlsynth[bayes] @ git+https://github.com/jgreathouse9/mlsynth@main"
pip install jupyter matplotlib tabulate cvxpy

python scripts/export_inputs.py basedata/repgermany.dta   # y.csv, x.csv, donors.csv, w_opt.csv
python scripts/prepare_sampler.py paper-BASC/BASC_realdata.R

Rscript scripts/basc_run.R                          # BASC posterior paths and weights
Rscript scripts/run_gamma1.R  2000 2000 200         # the gamma = 1 diagnostic
Rscript scripts/run_q.R       2000 2000 200         # q = 1 against q = 2
Rscript scripts/run_decomp.R  2000 2000 200         # selection and alpha_u at q = 2
Rscript scripts/run_init.R    2000 2000 200         # chain started at the best simplex fit

python scripts/collect_results.py                   # per-seed outputs into data/
python verify.py --with-mcmc                        # check every claim
```

`run_gamma1.R`, `run_q.R` and `run_decomp.R` take burn-in, sampling and seed.
The report uses seeds 100, 200 and 300 at 2000/2000, and seed 200 at
25000/25000. The long chains take about 25 minutes per configuration.

## Controls

Three checks establish that the patched sampler is the authors':

| check | expected |
|---|---|
| `run_gamma1.R` control at 2000/2000, seed 200 | RMSE 169.45, ATT -582.50 |
| `run_gamma1.R` control at 25000/25000, seed 200 | RMSE 195.516, ATT -389.799 |
| `run_q.R` at q = 1, three seeds | RMSE 166.15, 169.45, 222.10 |

The second pair are the two numbers the report used to carry as literals, from
an offline chain run before this package existed. They now come from the script.

## Files

```
basc_westgermany_review.qmd   the report
basedata/
  repgermany.dta              Abadie, Diamond and Hainmueller (2015) panel
requirements.txt
data/
  counterfactuals.csv         every counterfactual in the report, by year
  basc_counterfactual.csv     BASC posterior path + 95% CI (2000/2000)
  basc_counterfactual_500.csv BASC posterior path + 95% CI (500/500)
  bmv_counterfactual.csv      B-MV (bsynth) posterior path + 95% CI
  basc_weights.csv            BASC posterior donor weights
  basc_insample_rmse.txt      BASC 2000/2000 in-sample RMSE
  gamma1_diagnostic.csv       gamma = 1 runs, three configurations
  effect_basis_diagnostic.csv q = 1 against q = 2
  selection_decomposition.csv selection and alpha_u crossed, at q = 2
run_all.sh                    the whole pipeline
verify.py                     recomputes and checks every claim
scripts/
  export_inputs.py            y.csv, x.csv, donors.csv, w_opt.csv from the .dta
  collect_results.py          merges the per-seed outputs into data/
  prepare_sampler.py          derives the sampler files from the authors' code
  basc_run.R                  BASC posterior paths and weights
  run_gamma1.R                gamma forced to 1
  run_q.R                     the post-treatment effect basis
  run_decomp.R                what is left of the gap once q = 2
  run_init.R                  chain started at the best simplex fit
```
