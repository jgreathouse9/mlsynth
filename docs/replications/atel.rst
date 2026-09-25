.. _replication-atel:

ATEL — Average Treatment Effect Localization (Lee 2026)
=======================================================

.. currentmodule:: mlsynth

Validation strategy
-------------------

Lee ([ATEL]_) ships a MATLAB toolbox
(`rueichilee/ATEL <https://github.com/rueichilee/ATEL>`_) and its Section 5
application runs on public data: the Donohue and Ayres right-to-carry panel,
also the source of ``castle.csv`` in ``basedata/``. Both paths are therefore
open, and the replication takes both.

Cross-validation against the toolbox is the primary check, since it pins every
intermediate array and not only the headline. Path A then confirms the published
Table 4 to four decimals. The paper's Monte Carlo (Path B) is not reproduced,
and the reason is given under `Not replicated here`_.

Running the reference
---------------------

The toolbox is MATLAB and this project has no MATLAB, so it was run under
GNU Octave 8.4. Octave lacks two of the toolboxes it calls, and each gap was
filled and then checked against an independent oracle before any of its output
was trusted:

* ``norminv``, ``normcdf`` and ``tcdf`` come from the Statistics package. They
  were reimplemented from ``erf``, ``erfinv`` and ``betainc``, and agree with
  ``scipy.stats`` to ten decimals.
* ``augknt`` and ``spcol`` come from the Curve Fitting Toolbox. They were
  reimplemented by the Cox-de Boor recursion and agree with
  ``scipy.interpolate.BSpline.design_matrix`` to 2.2e-16 at widths 2, 3, 4 and
  5, the last of which exercises an interior knot.

One change to the toolbox source was needed. ``make_bspline_W.m`` calls
``unique(x, 'stable')`` for its third output, which Octave 8.4 does not
implement. The input there is already sorted, so the inverse index is the
cumulative count of first occurrences; that substitution is exact, and was
confirmed on 200 duplicate-heavy random vectors.

Path A — Arizona right-to-carry
-------------------------------

Arizona adopted a right-to-carry law in 1994. The panel is violent crime per
100,000 residents over 1977-2006 against 14 donor states, with poverty rate and
police rate as covariates, and :math:`T_0 = 17`.

One data detail decides whether a port matches. The ``policerate`` the toolbox
uses is not the raw column of that name in its own CSV: it is
``lnlpolicerate``, the log of the one-year-lagged police rate per 100,000. The
two agree nowhere -- the raw column differs by up to 924 and has a 2006 gap
that the lagged one does not -- and matching the lagged column reproduces
``data.mat`` at 0.0e+00.

With that, ``ATEL`` reproduces every row of the paper's Table 4:

.. list-table::
   :header-rows: 1
   :widths: 10 18 18 14 16

   * - :math:`J`
     - ATEL
     - Standard error
     - p-value
     - Published
   * - 2
     - 49.7665
     - 15.0055
     - 0.0069
     - matches
   * - 3
     - 71.9248
     - 12.3949
     - --
     - matches
   * - 4
     - 57.8420
     - 13.3495
     - --
     - matches
   * - 5
     - 99.0401
     - 13.7880
     - --
     - matches

Cell by cell against the toolbox, on the intermediate arrays and not only
the scalars:

.. list-table::
   :header-rows: 1
   :widths: 46 26

   * - Array
     - Max relative difference
   * - Diversified weights ``W`` (15x120)
     - 4.5e-09
   * - Factors ``F`` (30x2)
     - 2.2e-09
   * - Loadings ``beta`` (2x13)
     - 4.7e-09
   * - Counterfactual, pre and post
     - 1.0e-09
   * - Pre-period residuals
     - 2.5e-09

The scalars agree more tightly, from 3.4e-12 at :math:`J = 2` to 5.9e-10 at
:math:`J = 5`.

Two of the four rows are not well-defined
-----------------------------------------

This is the replication's substantive finding, and it is not in the paper.

``DP.m`` slices the weight matrix by columns :math:`1 + (j-1)T : jT` for
:math:`j = 1, \ldots, J`, while ``construct_weights.m`` lays the blocks out as
(basis j, covariate p) with p inner. The projection therefore consumes the first
J of the :math:`J \times P` blocks, and that set is a complete set of (basis,
covariate) pairs only when J is a multiple of P. Otherwise it takes some basis
for the first covariate and not the second, and the estimate depends on the
order the covariates are passed:

.. list-table::
   :header-rows: 1
   :widths: 10 22 22 16

   * - :math:`J`
     - (poverty, police)
     - (police, poverty)
     - Difference
   * - 2
     - 49.7665
     - 49.7665
     - 0.0000
   * - 3
     - 71.9248
     - 49.3082
     - 22.6166
   * - 4
     - 57.8420
     - 57.8420
     - 0.0000
   * - 5
     - 99.0401
     - 27.6103
     - 71.4298

The same table comes out of the toolbox itself, so this is a property of the
method as implemented and not of the port.

Two readings follow. Table 4's :math:`J = 3` and :math:`J = 5` rows report a
quantity that moves by 22.6 and 71.4 under a relabelling, against standard
errors near 13, so they do not support a claim about sensitivity to J.
Restricted to the rows that are invariant, :math:`J = 2` and :math:`J = 4` give
49.77 and 57.84 -- a spread of 8.07, inside one standard error -- so the
estimate is more stable in J than the published table suggests.

``ATELConfig`` refuses a factor count that is not a multiple of the covariate
count, which rules out exactly the ill-defined configurations. The benchmark
case pins the two invariant rows; pinning the other two would pin an argument
order.

The rank criterion does not select
----------------------------------

``select_rank.m`` implements a BIC-type criterion for J, attributed in its
header to [SuWang]_ and in its docstring to Ahn and Horenstein (2013). It is not
called for the published numbers -- the demo hard-codes :math:`J = 2`, and its
selection branch would fail on two undefined variables before printing -- and it
does not work when called.

On the Arizona panel the criterion is monotone over its whole candidate range
:math:`J = 2, \ldots, 8`, so it returns an endpoint under either extremum: the
minimum the source specifies gives 8, and the maximum the code takes gives 2.
Decomposing it explains why. The residual it penalises is the error of
projecting each period's donor cross-section onto :math:`2J` locally estimated
directions, and the design has :math:`2J` columns against :math:`N = 14`
donors, so the projection saturates:

.. list-table::
   :header-rows: 1
   :widths: 8 10 18 16 14

   * - :math:`J`
     - :math:`2J`
     - :math:`V(J)`
     - :math:`\log V`
     - rank
   * - 2
     - 4
     - 9.33e-02
     - -2.372
     - 4
   * - 5
     - 10
     - 1.40e-02
     - -4.270
     - 10
   * - 6
     - 12
     - 4.38e-03
     - -5.430
     - 12
   * - 7
     - 14
     - 4.88e-06
     - -12.230
     - 14
   * - 8
     - 16
     - 1.05e-18
     - -41.394
     - 14

The rank is :math:`\min(2J, N)` exactly, and :math:`\log V` falls by 39.02 over
the grid while the penalty rises by 2.01, a factor of 19 short. The grid's own
upper bound, :math:`\min(8, \min(N,T) - 1)`, counts J columns and not
:math:`2J`, so it permits the saturated range.

Two further measurements bound what a repair would need. The residual is
in-sample: the held-out period carries the maximum kernel weight when the
loading is fit and is then projected onto it. Holding it out cuts
:math:`\log V`'s range from 0.91 to 0.27 on an unsaturated synthetic panel, so
the in-sample fit accounts for about two thirds of the mechanical decline -- but
it overcorrects, and the penalty then dominates, returning the grid's low end
for every true J. Under [SuWang]_'s own :math:`\rho_{NT} = \ln(NT)/(NT)` the
penalty's range is 0.0195 against :math:`\log V`'s 0.91, two orders of magnitude
short.

Whether the criterion identifies J once all of this is repaired is untested
here. Three data-generating processes were tried and none had the power to
answer it: the toolbox's own simulation has a constant mean loading plus white
noise, so its common component is rank 1 whatever ``J_true`` says, and its
covariates are drawn independently of the loadings so the rank condition fails
by construction; two hand-built alternatives had no spectral gap at the true J.
``ATEL`` therefore requires ``n_factors`` and does not port ``select_rank``.

The bandwidth is a corner solution
----------------------------------

The cross-validated bandwidth is 0.95 -- the top of the search grid -- at every
factor count on the Arizona panel, with the criterion still falling where the
grid stops (3340, 3171, 3054 across its top three points at :math:`J = 2`). At
:math:`h = 0.95` with :math:`T_0 = 17` the smoothing window spans 16 of the 17
pre-periods, so the local linear fit is close to a global one and the
time-variation the method exists to capture is barely operative at the selected
bandwidth. ``ATEL`` warns when the search lands on an endpoint.

A defect that changes nothing
-----------------------------

``cal_var.m`` and ``select_rank.m`` both use [SuWang]_'s boundary kernel, which
divides the Epanechnikov weight by its integral over the truncated support. The
left branch divides by :math:`0.75(a + a^3/3 + 2/3)` where that integral is
:math:`0.75(a - a^3/3 + 2/3)`; the right branch has the integral. The error
reaches 50% at :math:`a = 1` and applies to 15 of Arizona's 17 pre-periods, and
``cal_var`` is on the path that produces the published standard error.

It changes nothing. The normalizer is one scalar multiplying the whole kernel
vector for a given period, and it enters only through
:math:`(X^\top K X)^{+} X^\top K y`, which is invariant to :math:`K \to cK`.
Measured both ways the standard error is 15.0054936497 identically, with
residuals differing by 1.5e-10. The port keeps the toolbox's form, and
:mod:`mlsynth.utils.atel_helpers.inference` records why.

Defects in the reference test harness
-------------------------------------

Found while looking for a fixture, and reported here so the next reader does not
spend the time again:

* ``test_simulation.m`` cannot run as shipped. Its DGP defines ``Y`` and it
  calls ``atel(y, ...)``.
* With that fixed it reports the wrong target. ``alpha = 5.0``, the effect size,
  is overwritten by ``alpha = 0.5``, the significance level, before
  ``true_atel = sum(alpha .* kT1) / T1h``. The kernel mass is 0.95983, so the
  script prints 0.4799 where its README documents 4.7992 -- which is
  :math:`5.0 \times 0.95983`, so the README predates the overwrite. The same
  ``alpha`` makes the printed "95% CI" a 50% interval.
* ``select_rank``'s default-``kmax`` guard tests ``nargin < 2`` where it needs
  ``nargin < 4``, so a three-argument call raises on an undefined variable
  instead of applying the documented default.

Fixtures pinned in the suite
----------------------------

``mlsynth/tests/test_atel.py`` pins two panels against the toolbox at
:math:`J = 2` and :math:`J = 4`, and once at a fixed bandwidth so one case
avoids the cross-validation path entirely:

* the bundled ``basedata/fdi_oecd_brexit.csv`` panel, UK treated, 29 donors,
  :math:`T_0 = 22`;
* a deterministic synthetic panel built without a random number generator, so
  the arrays handed to Octave and to Python are bit-identical. Its
  cross-validated bandwidth is 0.90, an interior point, so the grid-edge case is
  not the only one covered.

Agreement is to nine significant figures on the estimate, the standard error and
the p-value.

Not replicated here
-------------------

* The paper's Section 4 Monte Carlo. Its reported coverage (0.9323-0.9407) and
  MSE are a property of the estimator under a data-generating process the paper
  describes but does not ship, and the toolbox's own simulation script is not
  that process -- see the defects above. Reconstructing it is a separate piece
  of work from validating the estimator, and the cross-validation already pins
  every intermediate the Monte Carlo would exercise.
* The paper's interactive-fixed-effects comparison (153.5546, standard error
  40.1636). ``estimate_IFE`` is a vendored third-party module inside the
  toolbox, and mlsynth has :doc:`../gsynth` and :doc:`../fma` for that
  comparison.

References
----------

Donohue, J. J., & Ayres, I. Right-to-carry replication data.
`works.bepress.com/john_donohue/89 <https://works.bepress.com/john_donohue/89/>`_.

Lee, R.-C. (2026). "Average Treatment Effect Localization: Projection
Methods in Synthetic Control." *Econometric Theory*.

Su, L., & Wang, X. (2017). "On Time-Varying Factor Models: Estimation
and Testing." *Journal of Econometrics* 198(1):84-101.
