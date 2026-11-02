Specification sensitivity
=========================

When to use
-----------

``SpecificationStudy`` refits a declared list of research designs for one
treated unit. It exposes variation across donor pools, covariates, predictor
aggregation windows and fitting windows. Each run calls the existing
``VanillaSC.fit()`` independently. This initial interface supports classical,
single-treated-unit VanillaSC; augmentation and other estimators require
separate result and diagnostic checks.

The ATT is the mean observed-minus-counterfactual outcome over the fixed
``att_periods``. The outcome, unit and time columns, treatment assignment,
backend and seed cannot vary across designs. ``backend="auto"`` is resolved
once from the baseline, preventing predictor changes from switching solvers.
To compare predictor sets, choose a backend that supports all intended sets.
The fixed ``outcome-only`` backend rejects designs that add covariates.
Per-fit display and saving are disabled, including when enabled in the supplied
baseline. The original baseline is preserved; the registry records the effective
settings. Display or save the figure returned by ``result.plot()`` explicitly.
Other settings, including inference, retain their configured values. Set
``inference=False`` when only the descriptive sensitivity results are needed.
``fit_window`` accepts ``None`` for either endpoint, using the first or last
pre-treatment period respectively. When removing covariates, also clear any
inherited ``covariate_windows`` entries for those covariates.

Assumptions and safeguards
--------------------------

1. Declare the registry before examining post-treatment estimates. Designs
   need substantive justification; they are not automatically equally credible.
2. All fitting and predictor windows must use strictly pre-treatment periods.
   Precomputed covariate columns must themselves exclude future information.
   The runner checks calendar windows, not the provenance of those columns.
3. Mark exactly one unit as treated with a permanent binary treatment indicator.
   Declare other known contaminated donors in ``excluded_donors``. The runner
   fixes treatment identity and timing before donor subsetting and rejects
   any requested donor pool that violates these restrictions.
   Exclusions must be a collection of IDs present in the original panel;
   scalar strings and unknown IDs raise before fitting.
4. No anticipation, no spillovers and stability of the untreated relationship
   remain substantive assumptions. Good pre-treatment fit does not establish them.

Outputs and interpretation
--------------------------

``summary`` has one row per attempted design, with ATT, common-period
``pre_rmse``, design-window ``fit_rmse``, donor count and status. ``paths`` and
``weights`` contain unrounded trajectories and donor weights. The original
native results in ``fits`` preserve predictor weights and solver diagnostics.
``registry`` contains every resolved configuration; ``baseline`` retains the
original configuration and a copy of the panel. ``metadata`` records the fixed
periods, exclusions, backend, estimator and package version. Archive these with
the analysis and software environment.
The package version is ``None`` if distribution metadata is unavailable in a
source checkout. The registry excludes the panel and includes the runner's
``donors`` field: to replay a native fit, remove that field and supply the
original panel restricted to the declared donors, treated unit and common times.

Exceptions appear in ``failures``. Captured warnings, including primary-solver
messages preserved in native ``additional_outputs['solver_warnings']`` and nonconvergence,
appear in ``warnings`` and mark successful finite fits as ``warning``. Their
estimates remain visible. All selected units must have finite outcomes on the
common full trajectory; a missing post-treatment observation fails a design
instead of shortening its ATT period. There is no fit-quality filter.
The common trajectory consists of the treated unit's observed periods; extra
periods belonging only to donors or unrelated units do not expand it. Warnings
include native library diagnostics and need not indicate final nonconvergence.

``dispersion()`` reports counts, the range, median, sample standard deviation
and sign shares among finite estimates, including fits with warnings. The
shares describe the registered designs; they are not causal probabilities.
The spread of estimates is not a confidence interval. ``by_dimension()``
groups estimates and failure counts by one registered choice. These are
descriptive comparisons: other choices can vary within groups, so this table
does not attribute a causal contribution to any dimension.

Proposition 99 example
----------------------

Run from a repository checkout using its bundled panel. The restricted pool
is illustrative; the example does not certify its substantive eligibility.

.. code-block:: python

    import pandas as pd
    from mlsynth import VanillaSC
    from mlsynth.sensitivity import SpecificationStudy

    panel = pd.read_csv("basedata/augmented_cali_long.csv")
    panel["treated"] = (
        (panel.state == "California") & (panel.year >= 1989)
    ).astype(int)
    study = SpecificationStudy(
        estimator=VanillaSC,
        base_config=dict(
            df=panel, outcome="cigsale", treat="treated",
            unitid="state", time="year", backend="outcome-only",
            inference=False, display_graphs=False,
        ),
        specifications=[
            {"name": "baseline"},
            {"name": "late_fit", "fit_window": (1980, 1988)},
            {"name": "restricted", "donors": ["Utah", "Nevada", "Colorado"]},
        ],
        att_periods=range(1989, 2001),
    )
    result = study.fit()
    result.summary
    result.dispersion()
    result.by_dimension("fit_window")
    figure = result.plot()

The curve displays every attempted design. Failed designs are labelled with
no assigned effect value; crosses identify fits with warnings. An unchanged
baseline is added if omitted. The caller explicitly displays or saves the
returned Figure. With ``inference=False`` the study is descriptive; the next
section adds inference across the designs.

Inference across designs
------------------------

A placebo test asks whether the treated unit's effect is unusual among the
effects the same design produces when each untreated unit is pretended to be
the treated one. Ferman, Pinto and Possebom (2020) show that running that test
on several designs and reporting the most favourable one rejects too often:
with seven predictor sets and twelve pre-treatment periods, a test at a
nominal 5 percent finds at least one rejection about 14 percent of the time
when there is no effect. Their remedy keeps the designs and combines them.

With ``inference="placebo"`` in the base configuration, every design's own
VanillaSC fit runs the in-space placebo: each unit of the design's pool is
refit as the treated one, from the same date, on the design's fit periods and
predictors, and the fit records every unit's gaps and donor weights.
``placebo_includes_treated=True`` keeps the actually treated unit in the
placebo units' pools as an ordinary donor, as Abadie, Diamond and Hainmueller
(2010) and Ferman, Pinto and Possebom do; without it the treated unit is left
out of them. The cost is the number of designs times the size of the pool.

Let :math:`j = 1` denote the treated unit, :math:`\mathcal{T}_1` the
pre-treatment periods (:math:`T_0` of them) and :math:`\mathcal{T}_2` the ATT
periods (:math:`T_2` of them). For design :math:`s`, unit :math:`j` has gaps
:math:`\tau_{jt}(s)` when it is the one treated, and the post/pre statistic

.. math::

   r_j(s) \coloneqq
   \frac{T_2^{-1}\sum_{t\in\mathcal{T}_2}\tau_{jt}(s)^2}
        {T_0^{-1}\sum_{t\in\mathcal{T}_1}\tau_{jt}(s)^2}.

``placebo_test()`` reports, per design, :math:`r_1(s)` and its permutation
p-value, the share of units with :math:`r_j(s) \ge r_1(s)`. ``joint_test()``
replaces each unit's statistic by its mean across the chosen designs,
:math:`\bar r_j \coloneqq |\mathcal{S}|^{-1}\sum_{s\in\mathcal{S}} r_j(s)`, and
ranks :math:`\bar r_1` the same way. ``confidence_band()`` inverts the joint
test (Firpo and Possebom 2018): it collects the effect paths
:math:`\bar\tau_t + c`, the average effect across designs shifted by a
constant, that the joint test does not reject. Under each such null the
treated unit's outcome is netted of the hypothesised effect wherever it
appears, including in the placebo units' synthetic controls; the weights stay
fixed because they depend on pre-treatment data only. ``choose()`` ranks
designs by an objective rule instead: ``"dube-zipperer"`` by the placebo
units' mean squared post-treatment gap (Dube and Zipperer 2015),
``"donohue"`` by the treated unit's mean squared gap over a validation window
of pre-treatment periods held out of the fit (Donohue, Aneja and Weber 2019).

5. The p-value treats the treated unit as one draw from the units in the
   pool: it has its nominal level when treatment is as good as randomly
   assigned among them.

   Remark. With :math:`N` units the attainable levels are multiples of
   :math:`1/N`, and a unit whose fit fails is left out of the count. The band
   rejects a null when its p-value is at most ``significance``; Ferman,
   Pinto and Possebom use :math:`2/17` with 17 units.
6. Designs combined in one test share one placebo distribution, so they must
   share one unit set; a design that restricts the donor pool is tested on
   its own.

   Remark. ``joint_test()`` and ``confidence_band()`` raise when the chosen
   designs' unit sets differ.

``identification`` reports, for every fitted design, the range of the ATT
over all donor weights that fit the design's matched quantities exactly as
well as the solved weights :math:`\mathbf{w}^\ast`. A design's loss depends
on the weights only through those fitted values (the V-weighted predictors,
or the outcomes of the fit periods without covariates), so every
:math:`\mathbf{w} \in \Delta^{N_0}` with the same fitted values is an equally
good solution. ``identified`` is false when the range has positive width: the
design does not pin the weights, and the effect, down, and its placebo
p-value depends on which solution the solver returns. Designs with few
predictors or few fit periods are the usual cases. In Ferman, Pinto and
Possebom's Table 3, solving the inner problem exactly moves the two largest
p-values from 0.588 and 0.353 to 0.059.

.. code-block:: python

    study = SpecificationStudy(
        estimator=VanillaSC,
        base_config=dict(
            df=panel, outcome="cigsale", treat="treated",
            unitid="state", time="year", backend="outcome-only",
            inference="placebo", placebo_includes_treated=True,
            display_graphs=False,
        ),
        specifications=[
            {"name": "baseline"},
            {"name": "late_fit", "fit_window": (1980, 1988)},
        ],
        att_periods=range(1989, 2001),
    )
    result = study.fit()
    result.placebo_test()              # one row per design
    result.joint_test()                # p-value of the mean statistic
    band = result.confidence_band(significance=4 / 39)
    result.choose("dube-zipperer")     # first entry is the design chosen
    result.identification              # ATT range per design

``confidence_band`` searches outward from zero in steps of ``initial_step``
(default 1), halving the step ``precision`` times; set ``initial_step`` to the
order of the outcome. A side on which no null is rejected within
``max_steps`` steps is reported as infinite, with a warning.

Verification
------------

``mlsynth/tests/test_sensitivity.py`` checks parity with standalone VanillaSC,
counterfactual reconstruction from donor weights, a constructed panel with a
known ATT, fixed evaluation periods, donor-order invariance, pre-treatment
information boundaries and complete failure retention. The estimator's
empirical verification is documented in :doc:`replications/vanillasc`.

The inference across designs is checked against the authors' own R run of
Ferman, Pinto and Possebom (2020). ``mlsynth/tests/test_sensitivity_inference.py``
rebuilds their West Germany results from the donor weights their Synth 1.1-5
run produced: all fourteen p-values of Table 3 to 1e-12, the joint test
(1/17), both choice rules, and the Figure 2 band. The replication, including
where Synth's optimizer moves the published cells, is written up in
`benchmarks/studies/fpp_cherry_picking
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/fpp_cherry_picking>`_.
``mlsynth/tests/test_sensitivity_placebo.py`` checks that the tests rank the
placebo fits each design's VanillaSC fit records, and that each of those is a
standalone VanillaSC fit with that unit treated and the treated unit in the
pool.

API
---

.. automodule:: mlsynth.sensitivity
   :members: SpecificationStudy

.. autoclass:: mlsynth.sensitivity.SensitivityResult
   :members:

.. automodule:: mlsynth.utils.sensitivity_helpers.inference
   :members:
