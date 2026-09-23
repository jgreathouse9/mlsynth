TWSF: forecast accuracy, variance calibration and coverage
==========================================================

.. currentmodule:: mlsynth

What is validated
-----------------

``TWSF`` is validated against the simulation design of Shen [TWSF]_, section
7.1 -- Path B in the :doc:`../replications` scheme. The paper ships no code, so
there is no reference implementation to compare against cell by cell, and the
empirical application's data is not redistributable with the library.

The design draws donor factors, forms treated and control time factors from a
shared harmonic basis, scales the signal so that
:math:`\max_{i,t,d} |\langle \mathbf{u}_i, \mathbf{v}_t(d) \rangle| \le 0.8`,
and adds Gaussian noise at :math:`\sigma = 0.10`. Everything in that description
is reported except the numeric entries of the two :math:`4 \times 8` loading
matrices, which are given only structurally: fixed across replications, with the
lowest-frequency harmonic absent under control and present under treatment. The
benchmark draws a non-degenerate pair respecting that structure.

That is a real limitation and it bounds the claim. What is established is that
*a* loading pair respecting the stated structure yields nominal coverage, not
that the author's exact numbers reproduce.

Three properties, in the order that isolates cause
--------------------------------------------------

The sequence matters, because a coverage number alone cannot tell a broken port
from a broken design. Each check rules out one explanation before the next is
read.

Algebra first. With :math:`\sigma = 0` the forecast must be exact, since the
treated time factor is a sum of harmonics and so satisfies a linear recursion of
order at most the lag length. The benchmark measures a maximum absolute error of
about :math:`6 \times 10^{-15}`. An error anywhere in the Page-block layout, the
companion recursion or the bilinear combination surfaces here at machine
precision, not later as a few points of coverage.

Variance second. The empirical standard deviation of the forecast error over
repeated panels, divided by the mean plug-in standard error, is 0.89 at horizon
1 and 0.78 at horizon 5 on the benchmark's budget; the full-budget run gives
0.894 to 1.165 across the grid. This is the diagnostic that separates a wrong
variance formula from a wrong design, and it is the one that matters most here:
during the earlier assessment of this paper the plug-in variance was exact while
coverage was badly broken, so reading coverage without it would have blamed the
wrong component.

Coverage last. Against a nominal 90%, the benchmark measures 0.93 and 0.95 at
horizons 1 and 5, and the full-budget run gives 0.908, 0.885 and 0.892 at
horizons 1, 5 and 10 on the largest panel.

Where coverage falls short, and why
-----------------------------------

Coverage is below nominal on the smallest panels -- 0.838 to 0.871 at 25 units
in the full-budget run -- and the reason is spectral, not statistical.
The design sets the lag length equal to the panel dimension, so a short window
cannot resolve the longest harmonic or separate the two lowest-frequency
directions, and the Page matrix is near-degenerate at the oracle rank: its
eighth singular value is 0.017 at 25 units against 0.586 at 150. The benchmark
pins the comparison directly, asserting that the larger panel's retained signal
is better conditioned.

The theory is asymptotic, so a shortfall at the smallest panels with a spectral
cause is consistent with it, not evidence against it. The benchmark
asserts the large-panel value for that reason, and not a uniform one.

History
-------

This estimator was assessed twice. The first version of the paper left the
simulation's harmonics, scaling and loadings unreported, and a reconstruction
then produced a Page spectrum spanning five orders of magnitude with the
smallest signal direction below the noise floor -- so the spectral truncation
was inverting noise, and coverage came out between 0.39 and 0.87. The estimator
was parked, not built.

The second version reports the harmonics, the scaling rule and the noise level.
Those pin the signal-to-noise ratio that the earlier reconstruction had to
guess, and coverage reaches nominal. ``agents/future_integrations.md`` section
17 records the full history and the gate that cleared it.

Reproducing
-----------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case twsf_coverage_mc
