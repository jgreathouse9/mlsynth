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

5. A single treated unit, at one adoption date.

   Remark. The method is defined for staggered adoption with several treated
   units; this implementation is validated only for the one-unit case and
   enforces it.

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

Core API
--------

.. autoclass:: mlsynth.DMLFM
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.utils.dmlfm_helpers.config.DMLFMConfig
   :members:
   :undoc-members:
