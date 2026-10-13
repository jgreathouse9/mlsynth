.. _replication-kp-scm-forecast:

Klössner & Pfeifer (2018) — synthetic control as a forecasting method
=====================================================================

:Estimator: :doc:`../vanillasc` — :class:`mlsynth.VanillaSC`
:Source: Klößner, Stefan & Pfeifer, Gregor (2018), *"Outside the box: using
   synthetic control methods as a forecasting technique,"* Applied Economics
   Letters 25(9), 615–618, https://doi.org/10.1080/13504851.2017.1352071.
:Replication type: Path A — the paper's empirical table, on the public FRED
   series it uses.
:Benchmark case: `benchmarks/cases/kp_scm_forecast.py
   <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/kp_scm_forecast.py>`_
:Status: Verified for the ``All`` specification; the ``A,L`` rows are out of the
   suite's runtime budget and the ``A`` row is not identified.

What the paper does
-------------------

Every other replication in this catalogue estimates a treatment effect. This one
does not, and that is why it is here: it uses the synthetic-control machinery to
forecast, with no treatment anywhere in the problem.

The construction removes the panel. In an ordinary synthetic control the donors
are other units — other states, other countries — and the weights are chosen so
their weighted average tracks the treated unit before the intervention. Here
there is only one series, US real GDP growth, and the donors are time-shifted
copies of it: donor :math:`\ell` at date :math:`t` is :math:`Y_{t-\ell}`. The
weights are chosen so the weighted average of the lagged copies tracks the
series itself over a window of :math:`T` recent quarters, and the same weights
then project one quarter past the end of that window.

Since donor :math:`\ell` takes the value :math:`Y_{t+1-\ell}` at the forecast
date, the forecast is

.. math::

   \widehat{Y}_{t+1} \;=\; \sum_{\ell=1}^{H} w_\ell \, Y_{t+1-\ell},
   \qquad w_\ell \ge 0, \quad \sum_\ell w_\ell = 1 .

This is an autoregression of order :math:`H` with no intercept whose
coefficients are confined to the simplex. That constraint is the whole
difference from a textbook AR, and it does two things. It produces sparsity for
free — the optimum of a linear objective over a simplex sits at a face, so most
coefficients are exactly zero — which is the "built-in shrinkage behaviour" the
paper advertises. And it bounds the forecast inside the range of the last
:math:`H` observations, so the method cannot extrapolate past a recent extreme.

Specifications
--------------

The donors are fixed by :math:`H`; what varies is the set of predictors the
donors are matched on. Each predictor is a linear functional of the fitting
window, computed for the series and for every lagged copy:

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Name
     - Predictors
   * - ``A``
     - the window average
   * - ``L``
     - the last value in the window
   * - ``A,L``
     - both of the above
   * - ``F,M,L``
     - the first, middle and last values
   * - ``All``
     - every value in the window

A discount factor :math:`\beta \le 1` weights the outer objective by
:math:`\beta^{T-t}`, putting more weight on recent quarters.

Two identities, verified in the benchmark
-----------------------------------------

The port is cheap because two ingredients turn out to need no solver support.

For ``All``, the predictor-weight search is redundant. Its predictors are the
fitting window itself, so the inner objective coincides with the outer one and
:math:`V = I` already attains the outer optimum. Checked against the full
differential-evolution search on three configurations, the two agree to
:math:`2 \times 10^{-11}`. So :math:`\mathrm{SCM}^\beta_{\mathrm{All}}(H)` is a
discounted simplex least squares, which is what :class:`mlsynth.VanillaSC`
computes with no covariates — and the benchmark confirms that the public
estimator and the engine path return the same forecast to
:math:`5 \times 10^{-12}`.

The discount factor needs no support either. Weighting the outer MSPE by
:math:`\beta^{T-t}` is a rescaling of row :math:`t` by :math:`\beta^{(T-t)/2}`,
because the residual is linear in the row.

Data and its reconstruction
---------------------------

``basedata/fred_gdpc1.csv`` is FRED series ``GDPC1`` — US real GDP, quarterly,
seasonally adjusted — in levels. The paper works with the growth rate, rebuilt
as :math:`((L_t / L_{t-1})^4 - 1) \times 100` and cut to 1947Q2–2015Q1, giving
:math:`n = 272`.

Three independent checks confirm the reconstruction is the paper's series. The
sample bounds are the ones Section 3 states. The range, :math:`(-9.99, 16.68)`,
matches the span of Figure 1. And the forecast windows come out at 1960Q2–2015Q1
for :math:`T = 12` and 1967Q2–2015Q1 for :math:`T = 40`, which are the dates the
paper prints — an agreement that also settles a detail the paper leaves
implicit, namely that the forecast origins are set by the largest lag count in
the grid (40) instead of by each configuration's own :math:`H`, so every
configuration is scored on one common set of target quarters.

Results
-------

Table 2's ``All`` rows, each averaged over the nine window lengths of Table 1:

.. list-table::
   :header-rows: 1
   :widths: 30 18 18 18 18

   * - Specification
     - MAPE (paper)
     - MAPE (mlsynth)
     - RMSPE (paper)
     - RMSPE (mlsynth)
   * - :math:`\mathrm{SCM}^{1}_{\mathrm{All}}(36)`
     - 2.746
     - 2.727
     - 3.646
     - 3.612
   * - :math:`\mathrm{SCM}^{1}_{\mathrm{All}}(40)`
     - 2.758
     - —
     - 3.656
     - 3.627
   * - :math:`\mathrm{SCM}^{1}_{\mathrm{All}}(8)`
     - 2.873
     - —
     - 3.812
     - 3.792
   * - :math:`\mathrm{SCM}^{0.95}_{\mathrm{All}}(28)`
     - 2.823
     - —
     - 3.754
     - 3.681

The match is to about 0.02–0.07, not to display precision, and the gap runs one
way: every reconstructed figure sits slightly below the published one. The cause
is the data vintage. The authors pulled FRED in August 2016; the vendored file
carries a decade of subsequent NIPA revisions to the same series, so the growth
rates differ in the third significant figure and the forecast errors inherit
that. The declared tolerance of 0.12 brackets the largest observed gap while
staying inside the 0.17 spread separating Table 2's best and worst ``All`` rows,
so a configuration mix-up or a solver regression still fails the case.

What is excluded, and why
-------------------------

The ``A,L`` rows are excluded for runtime. They need a genuine two-predictor
:math:`V` search, which costs 82–153 s over the nine window lengths even on the
batched active set, against a suite budget of under a minute per case. An
exhaustive grid over the scale-free :math:`V` was measured against the shipped
differential-evolution search and the two agree to
:math:`1.4 \times 10^{-3}`, so the exclusion is a cost decision and not an
accuracy one.

The ``A(28)`` row is excluded for a different reason. With one predictor and 28
donors, matching a single scalar on the simplex admits a continuum of solutions,
so the answer is fixed by the solver's tie-break instead of by the data.
mlsynth's minimum-norm rule and the reference's disagree there, and that row
misses the paper by 0.92 in RMSPE against 0.02–0.07 for the identified rows.
Pinning it would pin an arbitrary choice.

Reading the result
------------------

The replication holds, and the method's standing is a separate question. On the
paper's own series the ``All`` and ``A,L`` configurations do beat the random
walk, Holt–Winters and ARMA–AIC, and edge ARMA–BIC. But the winning
configuration is chosen from a grid of roughly 1,620 against baselines that get
no comparable search, and the spread among the top six SCM rows is 0.03 in
RMSPE. There is a rolling train/test split per forecast and no
train/validation/test split for choosing the specification, so the reported
margin is an upper bound on what a practitioner would obtain.
