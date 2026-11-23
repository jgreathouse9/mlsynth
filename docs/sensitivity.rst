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
returned Figure. This study is separate from placebo inference.

Check pre-treatment prediction
------------------------------

Add validation windows to the same study to compare prediction accuracy with
the final treatment effect:

.. code-block:: python

    study = SpecificationStudy(
        estimator=VanillaSC,
        base_config=dict(
            df=panel, outcome="cigsale", treat="treated", unitid="state",
            time="year", backend="outcome-only", inference=False,
        ),
        specifications=[
            {"name": "baseline"},
            {"name": "late_fit", "fit_window": (1980, 1988)},
        ],
        att_periods=range(1989, 2001),
        validation_windows=[range(1984, 1989)],
        validation_max_rmse=20,
    )
    result = study.fit()
    result.summary
    result.validation
    figure = result.plot_validation()
    result.dispersion(validation_only=True)

The cutoff of 20 cigarette packs per capita illustrates a declaration; it is
not a recommended standard. Specify any cutoff before viewing estimates.
``validation_pass`` flags that predictive criterion, and the original unfiltered
results remain available. Passing does not certify convergence or identification.

Each fold trains strictly before its first validation period. Fitting windows,
exclusions, covariate windows and default covariate averages are restricted to
that earlier history. Empty matching windows fail without a fallback. Native
scaling and predictor-weight optimization use only training observations.
Precomputed covariates must themselves exclude future information.

The runner copies the fold panel, marks the validation boundary in its treatment
indicator, and disables fold inference and plotting. The native optimizer uses
only the fold's pre-boundary observations. Later outcomes enter prediction and
scoring. The original treatment assignment and actual policy date are preserved;
the final ATT fit uses the unchanged full pre-treatment configuration and inference
settings. Fold native ``effects`` are prediction gaps, not policy effects.

One window gives a holdout. Several disjoint windows give rolling-origin checks,
for example ``[range(1980, 1984), range(1984, 1989)]``. Later folds may train on
earlier held-out observations. All windows must contain observed pre-treatment
periods with earlier training data. Every design is scored on identical periods.
Pooled RMSE gives each held-out period equal weight; any failed fold makes that
design's aggregate unavailable. Successful folds are never selected for aggregation.

``validation`` records RMSE, sample counts, status and exceptions for every
design/fold. ``n_train`` counts available earlier periods; each design's actual
matching windows remain in ``validation_registry``.
Predictions, weights and warnings remain in ``validation_paths``,
``validation_weights`` and ``validation_warnings``. ``validation_registry``
records effective configurations without repeated panel data, including empty
windows that fail. ``validation_fits`` retains native results. Both are keyed by
specification and fold; fold IDs follow the chronological windows in ``metadata``.
The original baseline, registry, backend and seed remain available for reproduction.

``plot_validation()`` labels ATT versus pooled validation RMSE and names designs
missing either quantity. A declared threshold is drawn without filtering the
figure. This reports predictive sensitivity; good pre-treatment forecasts alone
do not establish no spillovers, no anticipation or post-treatment stability.

Verification
------------

``mlsynth/tests/test_sensitivity.py`` checks parity with standalone VanillaSC,
counterfactual reconstruction from donor weights, a constructed panel with a
known ATT, fixed evaluation periods, donor-order invariance, pre-treatment
information boundaries and complete failure retention. The estimator's
empirical verification is documented in :doc:`replications/vanillasc`.
``mlsynth/tests/test_sensitivity_validation.py`` checks native fold parity,
held-out data perturbations, rolling folds, failed-fold retention, pooled scores
and unchanged final policy fits, including generative tests.

API
---

.. automodule:: mlsynth.sensitivity
   :members: SpecificationStudy

.. autoclass:: mlsynth.sensitivity.SensitivityResult
   :members:
