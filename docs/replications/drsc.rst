.. _replication-drsc:

DRSC — New Jersey's 1992 minimum wage (Wied 2026)
==================================================

:Estimator: :doc:`../drsc` — :class:`mlsynth.DRSC`
:Source: Wied, Dominik (2026), *"A Synthetic Control Approach to Conditional
   Distributional Treatment Effects,"* arXiv:2606.09625.
:Replication type: Path A — the author's empirical result on the author's data.
:Status: Fully verified — Tables 1 and 2 reproduced cell for cell.
:Data: `benchmarks/reference/wied_nj_minwage/
   <https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/reference/wied_nj_minwage>`_

The question
------------

In April 1992 New Jersey raised its minimum wage from $4.25 to $5.05 while the
federal floor stayed at $4.25. Card and Krueger studied the employment effects
of the same reform; this application asks a different question, about the wage
distribution rather than about jobs. Specifically: for which kinds of worker did
the increase bite, and where in their wage distribution?

That is a question an average cannot answer, and it is also one an
*unconditional* distributional method answers only weakly. Low-education young
workers are a small share of the workforce, so an effect concentrated on them
barely registers in the pooled wage distribution. Conditioning on education and
experience is what makes the pattern visible.

Data
----

CPS Basic Monthly, 1989–1993, organised into April–March policy years aligned to
the reform: :math:`t = 1` is Apr 1989–Mar 1990 through :math:`t = 4` is Apr
1992–Mar 1993, the post-treatment year. That periodisation matters — it puts the
April 1992 increase exactly at a period boundary, so no partially treated year
has to be discarded.

The estimation sample is 381,953 workers across 43 states (New Jersey plus 42
donors) and four policy years. Eight states with their own minimum-wage changes
during the window are excluded. New Jersey's cell sizes are 3,852 / 3,998 /
4,031 in the pre-periods and 3,905 in the post period.

The raw CPS extract is not vendored — it is roughly 145 MB compressed and 8.8
million person-records. What is stored is the estimation sample the estimator
actually consumes, produced by the author's own preparation code. New Jersey's
four cell sizes matching the paper to the individual observation is what
establishes the extract is the right one.

Path A — the numbers
--------------------

Five largest synthetic control weights (Table 1):

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - donor
     - paper
     - mlsynth
     - difference
   * - Florida
     - 0.515
     - 0.515
     - 0.000
   * - New York
     - 0.479
     - 0.479
     - 0.000
   * - South Dakota
     - -0.255
     - -0.255
     - 0.000
   * - Nevada
     - 0.199
     - 0.199
     - 0.000
   * - Missouri
     - -0.196
     - -0.196
     - 0.000

Treatment effects at five covariate values (Table 2). :math:`T_n` is the
full-support supremum statistic, :math:`p` its p-value, and
:math:`p_{\mathcal{Y}_0}` the p-value of the focused test restricted to the
minimum-wage corridor :math:`[\log 4.25, \log 5.10]`:

.. list-table::
   :header-rows: 1
   :widths: 30 18 18 18 16

   * - evaluation point
     - :math:`\hat f \times 10^{3}`
     - :math:`T_n`
     - :math:`p`
     - :math:`p_{\mathcal{Y}_0}`
   * - median (ed 12, exp 10)
     - 0.58
     - 26.7
     - 0.294
     - 0.064
   * - low-skill young (ed 10, exp 2)
     - 1.71
     - 73.0
     - 0.054
     - 0.012
   * - high-ed young (ed 16, exp 2)
     - 1.55
     - 37.1
     - 0.686
     - 0.150
   * - low-ed senior (ed 10, exp 37)
     - 0.61
     - 28.6
     - 0.750
     - 0.060
   * - high-skill (ed 16, exp 37)
     - 0.21
     - 27.6
     - 0.824
     - 0.846

Every cell matches the paper. Also reproduced: the active grid size
:math:`m = 32`, the Gram condition number :math:`9.64 \times 10^{4}` against the
paper's :math:`9.6 \times 10^{4}`, and 20 of 42 donors receiving negative
weight.

The substantive finding survives intact. The effect is sharply concentrated
among low-education, low-experience workers and inside the wage corridor the
policy actually operates on; for high-education, high-experience workers — whose
wages sit far above the new floor — there is nothing, which is the falsification
check the paper intends.

What does not reproduce exactly, and why
----------------------------------------

The Gaussian-process critical values. This implementation obtains 245.5 and
240.5 for the two pre-trend transitions where the paper reports 245.3 and 241.7,
despite the seed being fixed in both.

The cause is not the seed. Simulating the limiting process requires a matrix
square root of the estimated covariance kernel, taken here from an
eigendecomposition. Eigenvector signs are implementation-defined: a different
LAPACK build returns a different set of signs for the same matrix, hence a
different factor, hence a different realisation from identical Gaussian draws.
The simulated distribution is unchanged; individual quantiles of it move. The
measured effect is 0.2–0.5 percent, which covers both discrepancies.

Test statistics involve no eigendecomposition and are exact — including the
x-simultaneous pre-trend statistics, 144.9 and 209.6, which match the paper
exactly.

The rule this implies for anything built on this replication: pin
:math:`T_n`, :math:`\hat f`, the confidence bounds and the weights tightly; pin
critical values and p-values loosely.

Precision is not optional here
------------------------------

The replication also measured what the ill-conditioned Gram matrix costs. A
float32 round-trip of the estimation sample — relative error about
:math:`6 \times 10^{-8}` — moves the largest donor weight by about 0.04, taking
Florida from 0.515 to 0.505; random noise of the same magnitude, which does not
partially cancel the way rounding does, moves them by 0.12 to 0.17. Either way
:math:`\hat f` stays within a few percent and the corridor p-value at
:math:`x_{10}` is unchanged at 0.012.

That asymmetry is the signature of near-collinearity: the weight vector is
poorly determined while the fitted counterfactual is well determined. It is why
the estimator requires float64 inputs, and why a benchmark built on this should
lean on the estimands rather than on individual weights if it ever has to choose.

Reproducing it
--------------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case wied_nj_minwage

The author's own script output is stored alongside the data as
``results_empirics.csv``, and an independent port written from the paper's
equations is kept as ``independent_port.py``. Both reproduce the tables, which
is what makes the target implementation-independent rather than a re-run of one
author's code.
