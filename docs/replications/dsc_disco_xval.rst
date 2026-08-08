.. _replication-dsc-disco-xval:

DSC against the DiSCos R package (Dube minimum wage)
=====================================================

:Estimator: :doc:`../dsc` -- :class:`mlsynth.DSC`
:Source: Gunsilius, F. (2023), *"Distributional Synthetic Controls,"*
   Econometrica 91(3):1105-1117, as implemented by the authors' R package
   `Davidvandijcke/DiSCos <https://github.com/Davidvandijcke/DiSCos>`_ 0.1.4.
:Replication type: cross-validation against the reference implementation.
:Status: verified -- mlsynth sits closer to the reference's centre than the
   reference sits to itself, in both feasible sets. Maximum donor-weight gap
   0.0079 under the simplex and 0.0103 without it, against the reference's own
   across-seed standard deviations of 0.0160 and 0.0300.
:Benchmark: ``benchmarks/cases/dsc_disco_xval.py``
   (`source <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/dsc_disco_xval.py>`__).

Why this case took so long to exist
-----------------------------------

DSC is mlsynth's port of an estimator with a published reference
implementation, so a cross-validation against that implementation should have
been the first check written. It was not, and the reason is in the reference:
``DiSCo_weights_reg`` draws its quadrature points with ``runif``, so the fitted
weights are a Monte Carlo estimate. On this panel at the package's default
settings they move by more than the quantity under test, and
:doc:`disco_tenure` put the problem plainly -- the package "disagrees with
itself across seeds by up to 0.119, where mlsynth sits 0.044 from any one of
them", so no tolerance against a single R run means anything.

That argument rules out one comparison, not all of them. What is unstable is an
individual run; the mean over runs is not. This case fixes ``M = 10,000``, takes
40 seeded reference runs, and scores mlsynth against their mean, reading the
across-seed standard deviation as the yardstick. The question it asks is not
whether the two agree exactly, which they cannot, but whether mlsynth sits
nearer the reference's centre than a single reference run does.

Result
------

===========  ==================  ===============  ===================  ==========
Mode         max abs weight gap  mean abs gap     reference seed sd    gap / sd
===========  ==================  ===============  ===================  ==========
simplex      0.0079              0.0022           0.0160               0.50
sum-to-one   0.0103              0.0042           0.0300               0.34
===========  ==================  ===============  ===================  ==========

Correlation across the 33 donor weights is 0.9958 and 0.9983. Both ratios sit
below one, which is the claim: the distance between the two implementations is
smaller than the distance between the reference and itself.

Both feasible sets
------------------

The two implementations do not default to the same one, which is why the table
has two rows.

mlsynth constrains the weights to the simplex, the set :math:`\mathcal H` of
Zhang, Zhang & Zhang (2026): non-negative and summing to one. ``DiSCos`` passes
``lb = NULL`` unless ``simplex = TRUE``, so its default is sum-to-one with an
upper bound of one and negatives allowed. mlsynth reaches that second set
through ``weight_constraint="sum_to_one"``, and the agreement there is the
closer of the two once the reference's larger seed noise is accounted for.

The relaxed fit is genuinely a different answer, not a re-parameterisation of
the same one: it puts weight down to :math:`-0.176` on one donor, extrapolating
outside the donors' convex hull, and the reference does the same to
:math:`-0.171`.

What this settles
-----------------

Issue #304 recorded mlsynth and ``DiSCos`` disagreeing on donor weights by up to
0.074 on this same panel, with the two reaching pre-period objective values 4
percent apart. ``dsc_dube`` treats that as the reason its rows cannot be
cross-validated: "pinning agreement before the disagreement is understood would
pin the wrong thing."

The disagreement was a measurement artifact -- one seed, at the package's
default draw count. Raise :math:`M` and average the seed noise away, and it is
0.0079.

Two other results reach the same conclusion from different directions, and
together they close the question:

* :doc:`disco_tenure` reproduces the deterministic Stata implementation's
  published weights bit-for-bit, so mlsynth's fixed grid is not the outlier
  among the three implementations.
* Handed bit-identical design matrices, mlsynth's solver and the reference's
  ``pracma::lsqlincon`` return weights agreeing to 3e-9
  (``benchmarks/reference/dsc_mc/``), so the weight solve was never a candidate
  explanation.

What remained was the quadrature rule, and that is what this case measures.

A caution on how to read a stochastic reference
-----------------------------------------------

The natural way to state agreement with a noisy reference is in standard-error
units, and here it does not work. The simplex solution is sparse: most donors
carry exactly zero weight in all 40 seeds, so their across-seed standard
deviation is zero and any floating-point difference divides by nothing. A
per-donor standard-error statistic reports a worst case of 1,586 on a
comparison that is, in weight units, agreement to 0.008. The case therefore
states the gap in absolute weight units against the reference's largest seed
standard deviation, which is well defined whether or not a given donor is in
the support.

Reproducing it
--------------

The committed dump means the case itself needs no R:

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case dsc_disco_xval

Regenerating the reference does need it (``bash benchmarks/R/install_discos.sh``,
15-25 minutes cold), and takes about 130 seconds for the two modes:

.. code-block:: bash

   python benchmarks/reference/dsc_disco_xval/export_panel.py
   Rscript benchmarks/reference/dsc_disco_xval/reference.R
   python benchmarks/reference/export_comparison.py dsc_disco_xval
   python benchmarks/reference/build_validation.py

The panel is exported to CSV first because the reference cannot read parquet;
that CSV is gitignored, being a 19 MB re-encoding of a 2 MB file the repository
already carries.
