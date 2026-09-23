DMLFM: German reunification
===========================

:doc:`../dmlfm` is validated by cross-validation against ``pblasso`` 1.0.8 on
the Abadie, Diamond and Hainmueller (2015) panel, reproducing Section 5.1 of
Pang, Liu and Xu (2022).

Why cross-validation and not Path A
-----------------------------------

The paper reports its German result as Figures 5 and 6 and prints no point
estimate anywhere in the text, so there is no published number to match. What
there is instead is the code that drew those figures, shipped in the authors'
Dataverse package.

The estimator in that package is ``pblasso`` 1.0.8. It is not the public
``liulch/bpCausal`` repository, which is a later rename: searching every commit
reachable from all refs of that repository finds none of the arguments the
paper's own scripts pass to it. The tarball inside the replication package is
the only published copy of the code behind the figures, and its sha256 is
recorded in ``benchmarks/reference/dmlfm_germany/provenance.json``.

The specification
-----------------

Six time-invariant covariates, each the unit's mean over all forty-four years
including the post-treatment ones; every covariate entering with both a
constant coefficient and a time-varying one; no unit-varying coefficients; ten
candidate factors; AR(1) dynamics; 25,000 iterations after 5,000 burn-in; flat
priors on the coefficients with shrinkage only on the loading scales.

Two details decide the answer and neither is in the paper. The covariate means
run over the whole sample, so the treated unit's covariates use post-treatment
outcomes. And every covariate is divided by its pooled standard deviation
before fitting (``blasso_default.R:88-91``) -- without that step a covariate
large in level swamps the rest, and on this panel one of them is a mean GDP
running to five figures.

Results
-------

Ten reference seeds against eight from mlsynth:

.. list-table::
   :header-rows: 1

   * - quantity
     - pblasso 1.0.8
     - mlsynth DMLFM
   * - ATT 1990-2003
     - -1597.9 (sd 38.7, n=10)
     - -1565.1 (sd 51.4, n=8)
   * - pre-treatment gap, max abs
     - 117.0
     - 116 to 128
   * - gap 1990
     - +457
     - +428 to +452
   * - gap 2003
     - -4117
     - -3772 to -4087
   * - leading loading scale
     - 3247
     - 2789 to 3775

The means agree at a Welch t of 1.50 (p = 0.16) and the variances at an F-test
p of 0.42.

The path reproduces the shape of the paper's Figure 6(b): flat at zero through
the pre-period with a band too narrow to see, a small positive excursion
peaking just under +1000 around 1991, a zero crossing in 1993, and a decline to
roughly -4000 by 2003.

What is checked exactly
-----------------------

The sampler is stochastic, so the quantities above are compared as means. Two
things are compared exactly, because they are deterministic.

Every object the sampler consumes on this panel -- the outcome vector, the
constant and time-varying design blocks, the treated unit's blocks, the unit
and period codes, both group-break vectors and the time-sort permutation --
matches the reference's own setup code element-wise to machine precision.

And each conditional draw factors into a deterministic mean and covariance plus
a normal draw. Those deterministic halves agree at 1e-10 across nine steps:
group moments, the coefficient posterior including its flat prior on the first
entry, both design expansions, the scale design, both partial fits, the AR(1)
coefficient moments, the per-unit loading posterior, and the per-period state
posterior with its first-period special case.

Reading the tolerances
----------------------

The sampler's seed-to-seed spread is wide relative to the effect it is
measuring. The first five reference seeds gave a standard deviation of 18.0 and
the next five gave 54.7; the pooled figure over ten is 38.7. An early version
of this page fixed a tolerance from the first five, which would have rejected a
correct port. Compare on means across several seeds.

What is not validated
---------------------

The cross-validation covers one configuration: time-varying coefficients only,
no unit-varying block, a single treated unit, a balanced panel. The
unit-varying block, staggered adoption with several treated units, and the
branch of the loading draw that handles unit-level random effects alongside
loadings are exercised by the unit tests but have no reference comparison
behind them.

Reproducing
-----------

.. code-block:: bash

   # the reference, needing pblasso 1.0.8 from the replication package
   Rscript benchmarks/reference/dmlfm_germany/reference.R

   # the comparison
   python benchmarks/run_benchmarks.py --case dmlfm_germany
