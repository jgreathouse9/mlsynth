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

Exceptions appear in ``failures``. Captured warnings, including primary-solver
messages preserved in native ``additional_outputs['solver_warnings']`` and nonconvergence,
appear in ``warnings`` and mark successful finite fits as ``warning``. Their
estimates remain visible. All selected units must have finite outcomes on the
common full trajectory; a missing post-treatment observation fails a design
instead of shortening its ATT period. There is no fit-quality filter.

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
returned Figure. This study is separate from placebo inference and from
pre-treatment holdout validation, which is not implemented here.

Verification
------------

``mlsynth/tests/test_sensitivity.py`` checks parity with standalone VanillaSC,
counterfactual reconstruction from donor weights, a constructed panel with a
known ATT, fixed evaluation periods, donor-order invariance, pre-treatment
information boundaries and complete failure retention. The estimator's
empirical verification is documented in :doc:`replications/vanillasc`.

API
---

.. automodule:: mlsynth.sensitivity
   :members:
