# mlsynth benchmarks

Durable, re-runnable validation of mlsynth estimators against the source
paper's own numbers and against authoritative **reference implementations**
(usually R). This institutionalizes the cross-checks that otherwise get done
once by hand and thrown away.

Each estimator's replication follows one of three "paths" (same vocabulary as
`docs/replications.rst`):

* **Path A** — reproduce the paper's empirical result on the authors' data.
* **Path B** — reproduce the paper's Monte Carlo / simulation table.
* **Cross-validation** — match an authoritative reference implementation
  (e.g. R `Synth`, `synthdid`, `did`) cell by cell.

The **definitions of done** for each path — broken out by what the source gives
you (paper only / code excerpt / full repo) — live in
[`agents/agents_benchmarking.md`](../agents/agents_benchmarking.md). This README
is the mechanics ("what to run"); that doc is the process ("when is it done").

## Layout

    benchmarks/
      run_benchmarks.py     # driver: runs registered cases, prints pass/fail vs tolerance
      registry.py           # the list of benchmark cases (name -> callable -> expected)
      compare.py            # tolerance-based comparison + reporting
      cases/                # one module per benchmark (pure-Python where possible)
        fdid_table5.py      # Path B: Li (2024) Table 5 PMSE grid (no R needed)
        fdid_hongkong.py    # Path A: Li (2024) Hong Kong GDP empirical (no R needed)
        sdid_prop99.py      # cross-val: SDID vs the authors' synthdid R (pinned)
        mcnnm_prop99.py     # cross-val: MC-NNM vs the authors' MCPanel R (pinned)
        spsydid_state_mc.py # cross-val: SpSyDiD vs the authors' repo (per-rep)
        clustersc_subgroups.py      # Path B: ClusterSC vs whole-pool RSC (subgroup regime)
        clustersc_subgroups_ref.py  # cross-val: authors' ClusterSC code vs its own paper
        clustersc_rpca_germany.py   # Path A: RPCA-SC West German reunification (Bayani 2021)
        tssc_brooklyn.py            # Path A: TSSC Brooklyn showroom (Li & Shankar 2024)
        tssc_figure2.py             # Path B: TSSC Figure-2 MSE-ratio grid
        sbc_germany.py              # Path A: SBC German reunification (Shi-Xi-Xie 2025)
        sbc_mc.py                   # Path B: SBC vs SC, Shi-Xi-Xie MSE ratios
        hsc_hongkong.py             # Path A: HSC 1997 Hong Kong handover (Liu & Xu)
        hsc_mc.py                   # Path B: HSC regime adaptation (Liu & Xu)
        rsc_synth_error.py          # Path B: RSC train-error approximates gen-error (ASS 2018)
        rsc_shen_coverage.py        # cross-val: Shen et al. PCR CIs + coverage validity
        lexscm_walmart.py           # Path A: LEXSCM Walmart placebo design (Abadie-Zhao Sec 4)
        lexscm_design_mc.py         # Path B: LEXSCM recovers planted effect (Abadie-Zhao Sec 5)
        scmo_germany.py             # Path A: SCMO West Germany balance (Tian et al. Table 2)
        scmo_concatenated_mc.py     # Path B: concatenated multi-outcome bias (Tian Table 1)
        scmo_averaged_mc.py         # Path B: averaged regime contrast (Sun et al. App. D)
        rescm_brexit.py             # Path A: SCM-relaxation Brexit/UK GDP (Liao-Shi-Zheng)
        rescm_relax_ref.py          # cross-val: mlsynth L2 relaxation vs scmrelax
        rescm_relax_mc.py           # Path B: latent-group MC, L2 relaxation beats SCM (Liao-Shi-Zheng S5)
        linf_crossval_ref.py        # cross-val: mlsynth LINF/L1LINF vs LinfinitySC (Wang-Xing-Ye)
        linf_prop99.py              # Path A: dense L-inf weighting vs sparse SC (Prop 99)
        linf_sim.py                 # Path B: L-inf beats SC in dense DGPs (Wang-Xing-Ye Table 4)
        ascm_kansas.py              # cross-val: ridge-ASCM ladder vs augsynth (Kansas)
      reference/            # Python reference implementations (cloned on demand)
        clone_spsydid.py    # pin + clone serenini/spatial_SDID (no licence -> not vendored)
        clone_linfinitysc.py # pin + clone BioAlgs/LinfinitySC (no packaging -> not vendored)
        spsydid_ref.py      # authors' SDID weights + the notebook's spatial WLS
        clone_clustersc.py  # pin + clone srho1/ClusterSC (MIT; imported not vendored)
        clone_panel_regressions.py  # pin + clone deshen24/panel-data-regressions (Shen CIs)
      R/                    # reference-implementation cross-checks
        requirements.R      # install the reference R packages
        synth_crosscheck.R  # run R's Synth on a dumped panel for cell-by-cell comparison

## Quick start

```bash
# All benchmarks (cases skip themselves if an optional dependency is missing)
python benchmarks/run_benchmarks.py --all

# A single case
python benchmarks/run_benchmarks.py --case fdid_table5

# Reference cross-checks behind a flag (need R + packages)
Rscript benchmarks/R/requirements.R          # one-time
python benchmarks/run_benchmarks.py --with-reference
```

A case **passes** when every reported number is within its declared tolerance
of the expected (paper / reference) value, **fails** when a number is out of
tolerance, and **skips** (`[SKIP]`, never a failure) when an optional
dependency or reference is unavailable. Tolerances are intentionally loose
enough to absorb Monte-Carlo noise (smaller M than the paper) but tight enough
to catch real regressions.

### Optional dependencies

Some cross-validation cases run against a Python reference implementation. They
are part of the default `--all` set but **skip gracefully** when their optional
dependency is absent. Install them **one at a time** (a batched `pip install`
aborts wholesale on the first failure):

```bash
pip install cvxopt             # linf_crossval_ref
pip install cvxpy              # rescm_relax_ref
pip install kneed scikit-learn # clustersc_subgroups_ref (the authors' syclib deps)
pip install toolz scikit-learn # rsc_shen_coverage (the authors' var.py deps)
pip install --ignore-installed beautifulsoup4 && pip install libpysal  # spsydid_state_mc; bs4 pre-empt avoids the apt-managed-RECORD clash
pip install git+https://github.com/PanJi-0/scmrelax.git                # rescm_relax_ref (not on PyPI)
```

The R cross-checks (`masc_basque`, `microsynth_seattle`, `nsc_prop99`,
`pensynth_prop99`, `pda_luxurywatch`, `scmo_concatenated_mc`, `siv_syria_mc`,
and the `cwz_*` cases) need a full R toolchain. In an environment
where CRAN is blocked, install it with apt + the pinned GitHub-mirror scripts:
`R/install_augsynth.sh` and `R/install_pensynth.sh`. The end-to-end
provisioning recipe — network probe, PPA fix, install order, and the gotchas —
lives in [`agents/agents_benchmarking.md`](../agents/agents_benchmarking.md)
under "Provisioning and running the full reference stack".

`spsydid_state_mc`, `clustersc_subgroups_ref`, and `rsc_shen_coverage`
additionally **clone** the authors' reference repos (`serenini/spatial_SDID`,
`srho1/ClusterSC`, `deshen24/panel-data-regressions`) at pinned commits into
`benchmarks/reference/.cache/` (git-ignored) — imported, never vendored. If git
or the network is unavailable the case skips. (The mlsynth-only Path-B cases
`clustersc_subgroups` and `rsc_synth_error` need no clone and always run.)

## Adding a benchmark

1. Add `benchmarks/cases/<name>.py` exposing `run() -> dict[str, float]` and
   `EXPECTED: dict[str, tuple[float, float]]` (value, abs-tolerance).
2. Register it in `registry.py`.
3. If it needs a reference run, either drop an R script under `R/` (add the case
   to `NEEDS_REFERENCE` and have it read the dumped output), or — for a
   pip-installable / clonable Python reference — import it inside `run()` and
   `raise BenchmarkSkipped(...)` when it is missing, so `--all` stays green
   without the extra dependency.
