Contamination triage for a locked design
========================================

.. currentmodule:: mlsynth

When to use this diagnostic
---------------------------

An experimental design of the MAREX kind (:doc:`marex`) commits to its control
markets before the experiment runs. It chooses treated weights
:math:`\mathbf{w}` and control weights :math:`\mathbf{v}` from pre-period data
alone, and the effect it later reports is a comparison against the weighted
average those control weights define. The commitment is the point: weights
chosen after seeing the outcome are weights chosen to produce an answer.

It also means the design cannot react. If an outside event hits one of the
control markets while the experiment is running -- a competitor launches there,
a large store closes, pricing changes and nobody tells the analyst -- the
weighted average moves, and the reported effect moves with it. For an additive
shock :math:`\pi` to control market :math:`k`, which the design gave weight
:math:`v_k`,

.. math::

   \hat{\tau}_{\text{observed}} - \hat{\tau}_{\text{clean}} = -v_k \pi .

The identity is exact, not a first-order approximation, which is what makes it
useful when the call comes in. Three of the questions an analyst asks have
answers available immediately, before anything is re-estimated:

- how much weight does the market carry (``exposure``, :math:`v_k`);
- what does an event of a given size cost (``bias``, :math:`-v_k \pi`);
- how large would the event have to have been to consume the whole measured
  effect (``breakdown_shock``, :math:`|\hat{\tau}| / v_k`).

The third usually settles the matter. A market at :math:`v_k = 0.04` needs an
event twenty-five times the reported effect to overturn it, and most incidents
are not that. A market at :math:`v_k = 0.6` needs an event under twice the
effect, and many are.

Repairing a contaminated estimate is a different and much harder problem. It
needs the size of the event, which is the one thing nobody has, and the
correction arms for it are studied under
`benchmarks/studies/contaminated_control
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/contaminated_control>`_.
This diagnostic answers the prior question of whether a repair is needed.

How it works
------------

:func:`mlsynth.contamination_report` is the primitive: it takes a control-weight
vector and the index of a market, and returns a frozen
:class:`mlsynth.ContaminationReport`. :func:`mlsynth.control_exposure` reads the
weights off a fitted design instead, so it works on any
:class:`mlsynth.config_models.DesignResult` that populates
``design_weights.summary_stats["control_weights_agg"]``, takes the market by
name, and uses the design's own reported effect when none is passed. A market
absent from the control group -- it was treated, or the design did not use it --
reports zero exposure, which is the answer to the question asked.

Alongside those three numbers the report carries four concentration summaries:
``max_weight``, ``effective_sample_size`` (:math:`1/\sum_j v_j^2`),
``herfindahl`` and ``n_carrying_weight``. These describe how badly the design
could ever be hurt this way, and they are knowable at :math:`T_0`. A design
spread over an effective twelve markets has no single market that can move the
answer much. One that put half its control weight in a single market has staked
the experiment on nothing going wrong there.

That second case is a design decision, and MAREX takes it as one:
``max_control_weight`` caps every control weight at a chosen value, so the
exposure is bounded before the experiment starts. See
:ref:`marex-control-weight-cap`.

Example
-------

.. code-block:: python

   from mlsynth import MAREX, control_exposure
   from mlsynth.config_models import MAREXConfig

   res = MAREX(MAREXConfig(df=df, outcome="y", unitid="market", time="week",
                           T0=18, m_eq=3)).fit()

   # A competitor launched in market 3 during the campaign, and the launch is
   # judged to have pushed that market's sales down by 0.9 units a week.
   rep = control_exposure(res, market="3", shock=-0.9)
   rep.exposure                 # 0.4916 -- market 3 is half the control group
   rep.bias                     # 0.4424 -- the effect is overstated by this much
   rep.effective_sample_size    # 3.34 effective control markets out of 6
   rep.n_carrying_weight        # 6

   # Running it backwards, against a reported effect of 2.0:
   control_exposure(res, market="3", att=2.0).breakdown_shock    # 4.068

So a shock of 4.07 would consume the whole effect, and the one estimated at 0.9
accounts for about a fifth of it. Had the same design been fitted with
``max_control_weight=0.2``, market 3 would carry 0.2 instead of 0.4916, the
effective sample size would be 6.85 across 8 markets, the same shock would cost
0.18 instead of 0.4424, and the breakdown shock would be 10.0.

Verification
------------

The identity is measured, not assumed. ``mlsynth/tests/test_contamination.py``
fits a MAREX design, shifts one control market's post-period level by a known
amount, re-estimates, and checks the move against the ``bias`` the report gives
before the re-estimation. The study arm under
`benchmarks/studies/contaminated_control
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/contaminated_control>`_
measures the same identity across six of the library's data-generating
processes, where it holds to :math:`1.6 \times 10^{-14}`.

The invariants the report promises -- the exposure lies in the unit interval,
the bias is linear in the shock, the breakdown shock times the exposure returns
the effect, the Herfindahl index is the reciprocal of the effective sample size,
and padding the weight vector with markets the design did not use changes
nothing -- are asserted generatively in
``mlsynth/tests/test_contamination_properties.py``.

Core API
--------

.. autofunction:: contamination_report

.. autofunction:: control_exposure

.. autoclass:: ContaminationReport
   :members:
