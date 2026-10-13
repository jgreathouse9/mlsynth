DMLFM
=====

Dynamic multilevel latent factor model, after Pang, Liu and Xu (2022).

When to use it
--------------

You have a panel, one treated unit or a few, and enough pre-treatment periods
to learn a factor structure. You want a counterfactual for the treated unit
together with an uncertainty statement you can read as a probability, and you
suspect the untreated outcomes are driven by common shocks that hit units
differently, so a parallel-trends assumption is not credible.

Two situations point here specifically. The first is having many candidate
covariates and no strong view about which matter or whether their influence is
stable: DMLFM lets a covariate's coefficient vary by unit, by time, or neither,
and shrinks the ones that do not earn their place. The second is not knowing
how many latent factors to use. Where :doc:`gsynth` picks a factor count by
cross-validation, DMLFM puts a shrinkage prior on the scale of each loading and
lets the ones it does not need collapse toward zero.

It is a poor choice when the pre-period is short. The paper's own simulations
find the frequentist properties unsatisfactory below about twenty
pre-treatment periods, and the estimator reports a ``short_pre_period`` flag
when you are under that. It is also slow: a fit takes tens of seconds where
:doc:`gsynth` takes under one.

Notation
--------

Units are indexed :math:`i = 1, \ldots, N` and periods :math:`t = 1, \ldots, T`.
Unit :math:`i` adopts treatment at :math:`a_i`, and :math:`y_{it}(0)` is the
outcome it would have had under control. Write :math:`\mathbf{x}_{it}` for the
covariate vector, :math:`\boldsymbol{\gamma}_i` for unit :math:`i`'s loadings on
:math:`r` latent factors and :math:`\mathbf{f}_t` for the factors at :math:`t`.

The untreated outcome is

.. math::

   y_{it}(0) = \mathbf{x}_{it}' \boldsymbol{\beta}_{it}
             + \boldsymbol{\gamma}_i' \mathbf{f}_t + \varepsilon_{it},
   \qquad
   \boldsymbol{\beta}_{it} = \boldsymbol{\beta}
             + \boldsymbol{\alpha}_i + \boldsymbol{\xi}_t,

so each covariate carries a coefficient with a common part, a unit-specific
part and a time-specific part. The time-varying pieces and the factors follow
independent AR(1) processes,
:math:`\boldsymbol{\xi}_t = \Phi_\xi \boldsymbol{\xi}_{t-1} + e_t` and
:math:`\mathbf{f}_t = \Phi_f \mathbf{f}_{t-1} + \nu_t`.

Each varying block is written as a scale times a standardised term --
:math:`\boldsymbol{\gamma}_i = \omega_\gamma \cdot \tilde{\boldsymbol{\gamma}}_i`
with :math:`\tilde{\boldsymbol{\gamma}}_i \sim N(0, I_r)`. The scale
:math:`\omega_\gamma` is what the shrinkage prior acts on: when its
:math:`k`-th entry goes to zero the :math:`k`-th factor leaves the model.

Which prior does that shrinking is set by ``prior``. The default ``lasso`` is
the Bayesian lasso of Pang, Liu and Xu (2022): a single penalty shared by every
coefficient in a block, drawn from a Gamma prior whose shape and rate are the
``a1``--``p2`` settings. The alternative ``horseshoe`` is the global--local
prior of Ma, Gao, Wang, Wang and Zhu (2026), which gives each coefficient its
own local scale on top of a block-wide one,

.. math::

   \beta_j \mid \lambda_j, \tau \sim N(0, \lambda_j^2 \tau^2), \qquad
   \lambda_j \sim \mathrm{C}^+(0, 1), \qquad \tau \sim \mathrm{C}^+(0, 1),

where :math:`\mathrm{C}^+` is the half-Cauchy distribution on the positive
line.

The difference between them is what happens to a coefficient that is genuinely
large. One penalty shared across a block has to be small enough to leave that
coefficient alone and large enough to flatten everything else, so it settles
somewhere between the two and does neither well. A per-coefficient scale is not
under that constraint: :math:`\tau` pulls the block toward zero while a large
:math:`\lambda_j` lets one coefficient escape. That is the regime the 2026
paper's simulations target, and it is the case for the horseshoe when the panel
is sparse -- many candidate covariates or factors, few that matter.

The horseshoe fixes both half-Cauchy scales at one, so it has no
hyperparameters to set and ``a1``--``p2`` are unused under it. The four
``xlasso``/``zlasso``/``alasso``/``flasso`` flags keep their meaning under
either prior: they choose which blocks are shrunk, and ``prior`` chooses how.
Their names predate the second prior.

The treatment effect is
:math:`\delta_{it} = y_{it}(a_i) - y_{it}(0)` for :math:`t \geq a_i`, and the
reported ATT averages it over the treated observations.

Assumptions
-----------

1. No anticipation. Outcomes before adoption do not depend on adopting later,
   so :math:`y_{it}(a_i) = y_{it}(c)` for :math:`t < a_i`.

   Remark. If West Germany's economy adjusted in 1989 to an expected
   reunification, the 1989 outcome is already treated and the pre-period is
   contaminated. This is why the paper backdates its placebo to 1987.

2. Latent ignorability. Conditional on the covariates and a latent vector
   :math:`\mathbf{U}_i`, adoption timing is independent of the untreated
   outcome path.

   Remark. This is weaker than parallel trends, which is the special case where
   :math:`\mathbf{U}_i` is a unit constant. It permits a unit's exposure to a
   common trend to predict when it gets treated, so long as that exposure is
   captured by the latent term.

3. Feasible factor extraction. The latent term admits a low-rank approximation
   :math:`\mathbf{U} = \Gamma' \mathbf{F}` with :math:`r` small relative to
   :math:`\min(N, T)`.

   Remark. This fails when unit-specific trends are idiosyncratic -- when every
   unit moves to its own drummer, there is no common structure to extract and
   no borrowing of strength is possible.

4. Balanced panel. Every unit is observed in every period.

   Remark. The model does not require this and the reference implementation
   tolerates gaps, but mlsynth ingestion carries no observation mask, so
   ``DMLFM`` raises ``MlsynthDataError`` on a ragged panel instead of fitting
   something the result contract cannot describe.

5. Absorbing treatment. A unit that adopts at :math:`a_i` stays treated for
   every :math:`t \ge a_i`.

   Remark. The counterfactual is imputed from adoption onward, so a unit that
   left treatment would have its post-exit outcomes reported as treated.
   Ingestion raises on an indicator that switches back off.

6. Untreated coverage in every period. Each period contributes at least one
   observation with the indicator at zero.

   Remark. The time-varying coefficient :math:`\boldsymbol{\xi}_t` and the
   factor :math:`\mathbf{f}_t` are identified only by untreated observations
   in period :math:`t`. A period in which every unit has already adopted leaves
   both drawn from their priors, and the counterfactual there carries no
   information from the panel. ``DMLFM`` raises ``MlsynthDataError`` naming the
   periods that fail, unless ``r = 0`` and ``re = "none"``, where nothing is
   indexed by period.

Staggered adoption
------------------

Several treated units, adopting at different dates, need nothing of the
sampler. Estimation runs on the control observations -- every :math:`(i, t)`
with the indicator at zero, which is Eq. (A.5) of Pang, Liu and Xu (2022) and
already includes each treated unit's own pre-adoption rows. The estimation set
is cell-level, not unit-level, so no unit has to be reserved as a donor:
a panel in which every unit eventually adopts is estimable up to the period
where assumption 6 fails.

The counterfactual block grows to ``n_treated * n_periods`` rows, and the
result reports per-cohort aggregations beside the pooled ATT:

``effects.additional_effects["cohort_att"]``
   ``{adoption label: mean ATT for the units adopting then}``, over that
   cohort's post-adoption cells.

``effects.additional_effects["event_study"]``
   ``{relative time: mean gap}`` where relative time is :math:`t - a_i`.
   Negative keys are pre-adoption, so they read as a placebo on assumption 1;
   non-negative keys are the dynamic effects.

``method_details.parameters``
   ``treated_units``, ``adoption_periods``, ``staggered``, and
   ``treated_cells``, the number of cells the pooled ATT averages over.

``effects.att`` averages the gap over treated cells, so a unit treated for one
period contributes one cell and a unit treated for ten contributes ten. The
time-series arrays carry one column per treated unit, and the plotter draws one
panel per unit with its own adoption line.

On the election-day-registration panel of Xu (2017) -- nine states adopting in
1976, 1996, 2008 and 2012 -- the pooled ATT is 5.7 percentage points of turnout
with a 95 percent credible interval of [3.1, 8.2], inside the [2.9695, 7.4810]
that Ma, Gao, Wang, Wang and Zhu (2026) report. Event times :math:`-3`,
:math:`-2` and :math:`-1` come in at 0.02, 0.24 and 0.49, against 4.0 and above
from adoption onward.

Inference and diagnostics
-------------------------

The counterfactual is a posterior predictive draw: at each retained iteration
the fitted mean is formed and normal noise with the current error variance
added, so the credible band covers both parameter and outcome uncertainty.
``inference.ci_lower`` and ``ci_upper`` are quantiles of the ATT draws;
``inference.details`` carries per-period bounds.

``additional_outputs["omega_gamma_spectrum"]`` is the sorted vector of mean
absolute loading scales, and reading where it falls away is how you see how
many factors the data supported. Only the sorted spectrum is interpretable: the
sampler flips the sign of each scale and its factor together at every
iteration, which leaves the fit unchanged but makes any individual signed
loading meaningless.

Two diagnostics carry warnings. ``method_details.parameters["short_pre_period"]``
is true below twenty pre-treatment periods. And the chains mix slowly enough
that a single run is not a point estimate: across seeds the reference itself
spans -1639 to -1509 on the German panel, so report a mean over several seeds
when the effect is close to the spread.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import DMLFM

   df = pd.read_stata("basedata/repgermany.dta", convert_categoricals=False)
   df = df.sort_values(["index", "year"])
   num = df.select_dtypes("number").columns
   df[num] = df[num].astype("float64")
   df["D"] = ((df["index"] == 7) & (df.year >= 1990)).astype(int)

   # the paper's covariates: unit means over the whole sample
   src = df.copy()
   for i in df["index"].unique():
       m, s = df["index"] == i, src[src["index"] == i]
       df.loc[m, "pgdp"] = s.gdp.mean()
       df.loc[m, "trade"] = s.trade.mean()
       df.loc[m, "inflation"] = s.infrate.mean()
       df.loc[m, "industry"] = s.industry.mean()
       df.loc[m, "schooling"] = s.schooling.mean()
       df.loc[m, "invest"] = np.nanmean(
           s[["invest60", "invest70", "invest80"]].to_numpy())

   res = DMLFM({
       "df": df, "outcome": "gdp", "unitid": "index", "time": "year",
       "treat": "D",
       "covariates": ["pgdp", "trade", "inflation", "industry",
                      "schooling", "invest"],
       "re": "time", "r": 10, "niter": 25000, "burn": 5000,
       "seed": 1234, "display_graphs": False,
   }).fit()

   print(res.effects.att)                    # about -1500 to -1600
   print(res.inference.ci_lower, res.inference.ci_upper)
   print(res.additional_outputs["omega_gamma_spectrum"][:4])

Choosing against the alternatives
---------------------------------

On the authors' own simulations, DMLFM and :doc:`gsynth` both dominate a plain
synthetic control by a wide margin -- RMSE of 1.8 to 3.7 against 3.4 to 6.2
across eighteen designs. Between the two the picture is narrow: DMLFM has the
lower RMSE in six of those eighteen cells and the higher in twelve, and its
coverage is closer to nominal in seven while gsynth is closer in ten. The cells
DMLFM wins are the ones with eight latent factors, which matches the paper's
own statement that its advantage appears when the factors are many and each is
weak. It runs eleven to eighty seconds against gsynth's under two.

So the reason to reach for DMLFM is what it gives you that gsynth does not:
covariate coefficients that vary by unit and by time, factor selection without
a cross-validation step, and a posterior you can quote directly. It is not a
more accurate estimator of the same object.

Verification
------------

Cross-validated against ``pblasso`` 1.0.8, the implementation behind the
paper's figures, on the German reunification panel. See
:doc:`replications/dmlfm` and the benchmark case
`benchmarks/cases/dmlfm_germany.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/dmlfm_germany.py>`_.

The paper's simulation half is
`benchmarks/cases/pang_liu_xu_sims.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/pang_liu_xu_sims.py>`_,
which runs the single-treated-unit designs of Appendix Tables A6 and A7 against
their published cells and cross-validates both arms against ``gsynth`` 1.0 and
``pblasso`` 1.0.8 on shared panels. Those designs generate no treatment effect,
so their bias column is the mean estimate and their coverage column is coverage
of zero.

The horseshoe is cross-validated separately, against the authors' own
implementation of it -- the ``HBSCM`` function in the replication archive of Ma
et al. (2026) -- and the lasso against ``bpCausal`` 0.0.1, the maintained rename
of ``pblasso``. Three empirical panels, twenty seeds per arm, 5000 draws with a
2500 burn-in, matched specification on both sides:

.. list-table::
   :header-rows: 1
   :widths: 24 10 20 20 22

   * - Panel
     - Prior
     - Reference
     - mlsynth
     - Difference
   * - German reunification
     - lasso
     - :math:`-1532.7\ (62.2)`
     - :math:`-1579.6\ (85.9)`
     - :math:`-46.9,\ t=-1.98`
   * - German reunification
     - horseshoe
     - :math:`-1532.6\ (103.2)`
     - :math:`-1580.7\ (60.7)`
     - :math:`-48.1,\ t=-1.80`
   * - Hong Kong, CEPA
     - lasso
     - :math:`+0.0271\ (0.0007)`
     - :math:`+0.0272\ (0.0006)`
     - :math:`+0.0002,\ t=+0.86`
   * - Hong Kong, CEPA
     - horseshoe
     - :math:`+0.0272\ (0.0007)`
     - :math:`+0.0268\ (0.0006)`
     - :math:`-0.0004,\ t=-1.80`
   * - Proposition 99
     - lasso
     - :math:`-16.66\ (6.14)`
     - :math:`-17.51\ (6.52)`
     - :math:`-0.85,\ t=-0.42`
   * - Proposition 99
     - horseshoe
     - :math:`-15.95\ (3.62)`
     - :math:`-17.88\ (5.89)`
     - :math:`-1.93,\ t=-1.25`

Parentheses are seed-to-seed standard deviations over the twenty runs, and the
:math:`t` statistics are Welch tests of the implementation difference.

Hong Kong and Proposition 99 agree. On Hong Kong the comparison can resolve a
difference of a few ten-thousandths and finds none in the lasso arm; the
horseshoe arm's :math:`-0.0004` is 1.5 percent of the effect. On Proposition 99
both arms are well inside their own noise.

German reunification does not fully agree. Both priors put mlsynth about 47 below
the reference, which is three percent of the estimate, and the two arms land
within one of each other despite different sampler internals. One marginal
:math:`p` among six comparisons is unremarkable; the same sign and magnitude
under two priors is less so. The offset is small against the estimator's own seed
spread on that panel, it is not attributed to any step, and it is recorded here
so that a later change to the ingestion or the design blocks has a number to
move.

Proposition 99 needs the largest tolerance because the estimator's standard
deviation there is about six under every implementation, against an effect near
seventeen. At three seeds the same four arms disagreed by as
much as eight, and three of them moved by between 3.7 and 5.1 when the seed count
rose to twenty, so a Proposition 99 figure from a single run carries no
information. Twenty seeds is the floor for that panel.

These figures come from direct runs of both reference implementations against
mlsynth on the same panels. The durable benchmark case covering them is not
landed yet: it waits on staggered adoption, so that one case can cover the 2026
paper's Case 3 (election-day registration, Xu 2017) alongside the three above
instead of being written twice.

One trap for anyone re-running the references. The ``est.avg`` field is a vector
of posterior draws in the replication package's ``effSummary``, and a
:math:`1 \times 3` summary matrix of ``(mean, ci_l, ci_u)`` in both
``bpCausal``'s ``effSummary`` and the 2026 archive's ``heffSummary``. Calling
``mean()`` on it gives the posterior mean in the first case and the average of a
point estimate with its own two credible bounds in the other two. Extract by
column name.

Core API
--------

.. autoclass:: mlsynth.DMLFM
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.utils.dmlfm_helpers.config.DMLFMConfig
   :members:
   :undoc-members:
