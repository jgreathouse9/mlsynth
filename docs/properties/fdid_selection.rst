.. _property-fdid-selection:

FDID — forward-selection consistency (Li 2023)
==============================================

:Estimator: :doc:`../fdid` — :class:`mlsynth.FDID`
:Source: Li, Kathleen T. (2023), *"Frontiers: A Simple Forward
   Difference-in-Differences Method,"* Marketing Science 43(2) [Li2024]_ —
   Online Appendices A, B and D.
:Results checked: Propositions 2.2 and D.1 (selection consistency);
   Lemma B.1 (uniform convergence of the subset intercept and error
   variance).
:Benchmark case: `benchmarks/cases/fdid_selection_mc.py
   <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/fdid_selection_mc.py>`_
:Status: Reproduced. :math:`\Pr(\widehat{\mathcal{U}} = \mathcal{U}^*)`
   climbs from 0.00 to 0.77 as :math:`T_1` goes from 25 to 1600, and
   Lemma B.1's normalised deviations stay in a band across a 32-fold range
   of :math:`T_1`.

What this page checks, and why it is not a replication
------------------------------------------------------

The :doc:`replication pages <../replications>` ask whether mlsynth
reproduces a number some author printed. This page asks something the
printed numbers cannot answer: whether the estimator's *algorithm* behaves
the way its proofs say it does.

Forward DiD works in two stages. It first searches the donor pool for a
subset of control units whose average tracks the treated unit well before
treatment, adding one control at a time; it then runs an ordinary
difference-in-differences against that subset's average. The paper's
Propositions 2.2 and D.1 concern the first stage alone: they say the subset
the search lands on converges, as the pre-treatment window grows, on the
subset the same search would have chosen if it knew the population
variances instead of estimating them.

That claim is invisible to a replication. mlsynth already reproduces Li's
Hong Kong application cell by cell and the paper's Monte Carlo table (see
:doc:`../replications/fdid`), but both score the *estimate*. A selection
bug that swapped one control for a near-identical one would move the ATT by
almost nothing and pass both — while breaking the property the proofs
establish. Measuring the selection directly is what closes that gap.

The setting
-----------

There are :math:`N` control units and one treated unit, observed over
:math:`T_1` pre-treatment and :math:`T_2` post-treatment periods. For a
subset of controls :math:`U \subseteq \{1, \dots, N\}`, write
:math:`\bar y_{Ut} = |U|^{-1} \sum_{i \in U} y_{it}` for its average
outcome. Forward DiD's fit for that subset is the treated series minus the
subset average, up to a level shift :math:`\alpha_U`; the residual is
:math:`v_{Ut} = y_{tr,t} - \bar y_{Ut} - \alpha_U`.

Two quantities drive everything. The subset's level shift and its error
variance,

.. math::

   \alpha_U = \mathbb{E}\bigl[y_{tr,t} - \bar y_{Ut}\bigr],
   \qquad
   \sigma^2_U = \mathbb{E}\bigl[v_{Ut}^2\bigr],

have sample counterparts :math:`\widehat\alpha_U` (the pre-treatment mean
gap) and :math:`\widehat\sigma^2_U = T_1^{-1}\sum_{t \le T_1}\hat v_{Ut}^2`.
The algorithm never sees :math:`\sigma^2_U`; it minimises
:math:`\widehat\sigma^2_U`, which is what maximising the pre-treatment
:math:`R^2` amounts to when the treated series is held fixed.

Two algorithms, and the gap between them
----------------------------------------

Web Appendix D defines the *theoretical* forward selection algorithm: the
same greedy search, run on :math:`\sigma^2_U` in place of
:math:`\widehat\sigma^2_U`. It is infeasible — nobody knows the population
variances — and exists purely as the benchmark the empirical algorithm is
measured against. Write :math:`\widehat{\mathcal{U}}` for the subset the
empirical search returns and :math:`\mathcal{U}^*` for the collection of
subsets the theoretical search could return. The collection can hold more
than one subset because the theoretical criterion can tie.

Both algorithms run the same three steps (Web Appendix D):

Step 1
   Compute the criterion for each of the :math:`N` single-control models
   and keep the best.

Step 2
   Add each remaining control to that model in turn, giving :math:`N-1`
   two-control models, and keep the best.

Step 3
   Continue to the model holding all :math:`N` controls. This leaves
   :math:`N` nested models, one per size; select the one whose criterion is
   smallest.

The assumptions
---------------

Assumption 2.1 (Forward DiD parallel trends). The subset of control units
selected by the Forward DiD method, :math:`\widehat{\mathcal{N}}_{co}`,
satisfies :math:`y^0_{tr,t} = \alpha + \bar y_{\widehat{\mathcal{N}}_{co},t}
+ v_t` for :math:`t = 1, \dots, T`, where :math:`\alpha` is a constant and
:math:`v_t` is a weakly dependent process with zero mean and finite
variance.

*Remark.* This asks the gap between the treated unit and the selected
comparison group to be stable across the pre- and post-periods up to a
mean-zero shock. Both series may trend arbitrarily, so long as their
difference does not. Web Appendix A collects the subsets meeting it into
:math:`\mathcal{U}`, the set the uniform bounds below quantify over.

Assumption 2. (i) Outcomes follow the appendix's factor model, with
:math:`|b_i| \le c` and :math:`|\lambda_i| \le c` for all
:math:`i \in \mathcal{N}_0 := \{0, 1, \dots, N\}` and :math:`c` a fixed
positive constant. (ii) :math:`\epsilon_{it}` is iid with zero mean, finite
fourth moment, and
:math:`\max_{i \in \mathcal{N}_0} |T_1^{-1}\sum_{t \le T_1} \epsilon^j_{it}
- \mathbb{E}(\epsilon^j_{it})| = O_p(\sqrt{\log N / T_1})` for
:math:`j = 1, 2, 3`. (iii)
:math:`\max_{i \in \mathcal{N}_0}\mathbb{E}[|\epsilon_{it}|^j] \le C` for
:math:`j = 2, 3`. (iv)
:math:`\max_{i,j \in \mathcal{N}_0}|T_1^{-1}\sum_{t \le T_1}
\epsilon_{it}\epsilon_{jt} - \mathbb{E}[\epsilon_{it}\epsilon_{jt}]|
= O_p(\sqrt{\log N / T_1})`. (v)
:math:`0 < c_1 \le \inf_{U \in \mathcal{U}} \sigma^2_U \le
\sup_{U \in \mathcal{U}} \sigma^2_U \le c_2 < \infty`.

Assumption 3. (i) When :math:`f_t` is stationary, it is iid with zero mean
and finite fourth moment, and
:math:`\max_{i \in \mathcal{N}_0}|T_1^{-1}\sum_{t \le T_1}\epsilon_{it}f_t
- \mathbb{E}[\epsilon_{it}f_t]| = O_p(\sqrt{\log N / T_1})`. (ii) If
:math:`f_t` is non-stationary, then :math:`\bar\lambda_U = 0`, where
:math:`\bar\lambda_U = |U|^{-1}\sum_{i \in U}\lambda_i`.

Assumption 4. As :math:`T_1, T_2 \to \infty`, (i)
:math:`\log N / T_2 \to 0` and (ii) :math:`T_2 \log N / T_1 \to 0`.

*Remark.* Assumption 4 lets :math:`N` be large or small but requires
:math:`T_1` to grow faster than :math:`T_2` — the pre-period has to
out-grow the post-period, since the selection is estimated on the former
and evaluated on the latter.

The results
-----------

Lemma B.1. If Assumptions 1 to 4 hold, then

.. math::

   \text{(a)} \quad \max_{U \in \mathcal{U}}
     \bigl|\widehat\alpha_U - \alpha_U\bigr|
     = O_p\!\left(\sqrt{\log N / T_1}\right),
   \qquad
   \text{(b)} \quad \max_{U \in \mathcal{U}}
     \bigl|\widehat\sigma^2_U - \sigma^2_U\bigr|
     = O_p\!\left(\sqrt{\log N / T_1}\right).

*Remark.* The maximum is what makes this a statement about the algorithm
and not about any one fit. The search compares subsets, so a bound holding
for each subset separately would not stop the comparison going wrong at
whichever subset happens to be worst; a bound holding uniformly does. The
:math:`\log N` is the price of taking a maximum over a pool that may grow.

Proposition D.1. Let :math:`\mathcal{U}^*` denote the collection of subsets
that can be selected by the theoretical forward selection algorithm. Under
Assumptions 1 to 4 and with :math:`N` a fixed positive integer,

.. math::

   \Pr\bigl(\widehat{\mathcal{U}} \subset \mathcal{U}^*\bigr) \to 1
   \qquad \text{as } T_1 \to \infty.

Proposition 2.2. Assume the theoretical forward selection algorithm selects
a single subset :math:`\mathcal{N}^*_{co}`, and :math:`N` is a fixed
positive integer. Then under Assumption 2.1 and Assumptions 2 to 4,

.. math::

   \Pr\bigl(\widehat{\mathcal{N}}_{co} = \mathcal{N}^*_{co}\bigr) \to 1
   \qquad \text{as } T_1 \to \infty.

*Remark.* Proposition 2.2 is the special case of D.1 in which nothing ties,
so the containment sharpens to an equality. Which of the two applies is a
property of the design, not of the estimator, and the designs used below
are checked for it, not assumed to have it.

The computable benchmark
------------------------

Measuring any of this needs :math:`\mathcal{U}^*`, which needs the
population variances. Those are available in closed form on the paper's own
Monte Carlo designs (Web Appendix E), which mlsynth already implements in
:mod:`mlsynth.utils.fdid_helpers.simulation` for the
:doc:`Path B replication <../replications/fdid>`.

In those designs the treated unit is
:math:`y_{tr,t} = a_0 + c_0\mathbf{1}'f_t + \varepsilon_{tr,t}` and control
:math:`i` is :math:`y_{it} = 1 + c_{g(i)}\mathbf{1}'f_t + \varepsilon_{it}`,
where :math:`g(i)` puts the first half of the pool in loading group 1 and
the rest in group 2, and the shocks are independent and unit-variance. For
a subset holding :math:`n_1` controls from group 1 and :math:`n_2` from
group 2, with :math:`n = n_1 + n_2` and mean loading
:math:`\bar c = (n_1c_1 + n_2c_2)/n`, the gap is

.. math::

   y_{tr,t} - \bar y_{Ut}
     = (a_0 - 1) + (c_0 - \bar c)\,\mathbf{1}'f_t
       + \varepsilon_{tr,t} - \bar\varepsilon_{Ut}.

The factors are zero-mean, so :math:`\alpha_U = a_0 - 1` for every subset,
and

.. math::

   \sigma^2_U = (c_0 - \bar c)^2\,\sigma_S^2 + 1 + \frac{1}{n},
   \qquad
   \sigma_S^2 = \operatorname{Var}(\mathbf{1}'f_t) = 5.8103,

with :math:`\sigma_S^2` the sum of the three factors' stationary variances
(AR(1) at :math:`\phi = 0.8`; ARMA(1,1) at :math:`\phi = -0.6`,
:math:`\theta = 0.8`; MA(2) at :math:`\theta = (0.9, 0.4)`).

Two things follow. First, :math:`\sigma^2_U` depends on a subset only
through its group counts, so members of a loading group are
interchangeable and :math:`\mathcal{U}^*` can be described by a count pair
instead of an enumeration — which is what makes the theoretical algorithm
runnable in :math:`O(N)` steps where a subset search would take
:math:`O(2^N)`. Second, the criterion trades a loading gap against an
averaging gain, and which one wins is what separates the two designs used
here:

.. list-table::
   :header-rows: 1
   :widths: 12 26 30 32

   * - DGP
     - Loadings
     - :math:`\mathcal{U}^*`
     - How sharply it is separated
   * - 1
     - :math:`c_0 = c_1 = c_2 = 1`
     - the whole pool
     - Barely. :math:`\sigma^2_U = 1 + 1/n` puts :math:`n = 14` within 2%
       of :math:`n = 20`.
   * - 2
     - :math:`c_0 = c_1 = 1`, :math:`c_2 = 2`
     - group 1 exactly
     - Sharply. One mismatched control costs more than the
       :math:`1/n` gain repays.

Both are singletons, so Proposition 2.2's hypothesis holds and the target
is an equality in each case. The population module derives them instead of
taking them on faith, and
``mlsynth/tests/test_fdid_population.py`` checks the derivation against
brute-force enumeration over all :math:`2^N - 1` subsets and against Monte
Carlo draws from the simulator.

What the case measures
----------------------

Propositions 2.2 / D.1, DGP 2 (:math:`N = 20`, so :math:`\mathcal{U}^*` is
a specific 10-unit subset), 400 draws per cell. An exact match needs all
ten matched controls in and all ten mismatched controls out.

.. list-table::
   :header-rows: 1
   :widths: 14 22 22 22

   * - :math:`T_1`
     - :math:`\Pr(\widehat{\mathcal{U}} = \mathcal{U}^*)`
     - Share of :math:`\mathcal{U}^*` recovered
     - Mean :math:`|\widehat{\mathcal{U}}|`
   * - 25
     - 0.000
     - 0.413
     - 4.6
   * - 50
     - 0.000
     - 0.508
     - 5.4
   * - 100
     - 0.003
     - 0.606
     - 6.3
   * - 200
     - 0.005
     - 0.717
     - 7.4
   * - 400
     - 0.068
     - 0.825
     - 8.3
   * - 800
     - 0.380
     - 0.920
     - 9.2
   * - 1600
     - 0.768
     - 0.975
     - 9.7

By :math:`T_1 = 1600` the selection admits no mismatched control at all
(precision 1.000): what remains between 0.768 and 1 is the last matched
control or two, dropped because the criterion's gain from adding them is
small next to the sampling noise at that window length.

Lemma B.1, DGP 2 at :math:`N = 8` so that the maximum can run over all
:math:`2^N - 1 = 255` subsets, 150 draws per cell. Under these designs the
factors are stationary and zero-mean, so every subset's gap is a weakly
dependent mean-zero process whatever its loading mismatch: mismatch
inflates :math:`\sigma^2_U` without costing parallel trends, and
:math:`\mathcal{U}` is the whole power set. Enumerating all subsets is
therefore exactly the lemma's index set here.

.. list-table::
   :header-rows: 1
   :widths: 12 22 22 22 22

   * - :math:`T_1`
     - :math:`\max_U|\widehat\alpha_U - \alpha_U|`
     - :math:`\div\sqrt{\log N/T_1}`
     - :math:`\max_U|\widehat\sigma^2_U - \sigma^2_U|`
     - :math:`\div\sqrt{\log N/T_1}`
   * - 50
     - 0.794
     - 3.89
     - 2.259
     - 11.08
   * - 100
     - 0.514
     - 3.56
     - 1.686
     - 11.70
   * - 200
     - 0.413
     - 4.05
     - 1.162
     - 11.40
   * - 400
     - 0.295
     - 4.09
     - 0.836
     - 11.60
   * - 800
     - 0.231
     - 4.52
     - 0.604
     - 11.84
   * - 1600
     - 0.152
     - 4.21
     - 0.409
     - 11.34

The raw deviations fall by a factor of 5.2 and 5.5 while
:math:`\sqrt{\log N / T_1}` falls by 5.66. Normalised, both sit in a band
across a 32-fold range of :math:`T_1` — the intercept ratio between 3.56
and 4.52, the variance ratio between 11.08 and 11.84 — instead of growing.

What is established, and what is not
------------------------------------

Both results are one-sided: an :math:`O_p` bound can be contradicted by
measurement but never confirmed by it. What the case pins is that the
bound held at every point on its grid, and that the quantity each
proposition says should move did move, in the claimed direction, at a rate
consistent with the claimed one. It does not establish the rate, and the
level of the normalised band carries no information at all — an
:math:`O_p` statement hides its constant, so only the band's flatness is
the claim.

The two designs disagree about how fast the convergence arrives, and the
disagreement is the useful part. Under DGP 1 the exact-match rate is 0.000
across the whole grid; what moves is the size of the selected pool, from
25% of the donors at :math:`T_1 = 25` to 82% at :math:`T_1 = 1600`. Nothing
is wrong: :math:`\mathcal{U}^*` is the whole pool and the criterion ranks a
14-control model within 2% of the 20-control one, so distinguishing them
takes a pre-period far longer than any grid here. Selection consistency is
asymptotic in :math:`T_1`, and how much :math:`T_1` it needs depends on how
sharply the population criterion separates the optimum from its neighbours
— which is a property of the panel a practitioner has, not of the method.
Read the other way: on a real panel where several donor subsets fit almost
equally well, the specific subset Forward DiD reports should be treated as
one of many near-ties, even though the ATT it implies is stable across
them.

Reproducing it
--------------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case fdid_selection_mc

The case is seeded end to end, so it returns the same numbers on every run;
its tolerances are set at roughly three Monte-Carlo standard errors, the
spread a different set of draws would produce. Runtime is about two and a
half minutes.

.. code-block:: python

   from mlsynth.utils.fdid_helpers.population import theoretical_forward_selection

   star = theoretical_forward_selection(dgp=2, N=20)
   star.counts        # (10, 0) -- ten matched controls, no mismatched ones
   star.unique        # True -- U* is a singleton, so Proposition 2.2 applies
   star.variance      # 1.1 = 1 + 1/10
   star.contains([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])   # True
