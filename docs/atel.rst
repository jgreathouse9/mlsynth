Average Treatment Effect Localization (ATEL)
============================================

.. currentmodule:: mlsynth

When to Use This Method
-----------------------

A policy is adopted, and you are asked what it did. The usual answer averages the
gap between the treated unit and its counterfactual over every post-treatment
period. That average describes the policy only if the treated unit stood still
while the policy acted on it, and units do not stand still. An organisation
facing a funding cut fundraises, reorganises, and lays people off; a state
facing a new firearms law sees policing and migration respond. Average over ten
years and you measure the policy and every response to it, added together, with
no way to read one off the other.

ATEL (Lee, [ATEL]_) reports a different number: an average that puts most of its
weight on the periods just after adoption, and little on the ones far from it.
The estimand is still an average treatment effect on the treated, and it is
still a single number, but it is localized at the moment the policy took effect,
where the response has had least time to accumulate.

Two consequences follow, and they pull in opposite directions. What ATEL
estimates is closer to the policy's own effect, since it is measured before the
unit has adapted. What it estimates is also a narrower quantity: it says nothing
about the long run, and a policy whose effect grows over years will look small.

Reach for ATEL when
^^^^^^^^^^^^^^^^^^^

* the post-period is long enough that adaptation is plausible, and you want the
  policy's effect and not the policy plus the response;
* you have to report inside a budget or election cycle, so the near-term effect
  is the decision-relevant one;
* the treated unit's relationship to the donor pool moves over the sample. ATEL
  lets the factor loading vary with time, so a unit whose comparison group drifts
  is still usable;
* you have time-varying covariates that plausibly drive the unit's exposure to
  common shocks. These give the strongest version of the method, since the sieve
  basis in them is what lets the loading move with observables. They are not
  required -- see `Where the weights come from`_.

Do not use ATEL when
^^^^^^^^^^^^^^^^^^^^

* the question is about the long run. The kernel discards exactly the periods a
  long-run question is about; use :doc:`fma`, :doc:`gsynth` or a standard
  synthetic control;
* you have one post-period, or two with a small bandwidth. The localization
  window is :math:`\lfloor T_1 h \rfloor` and an empty window has no estimand;
* nothing in the panel plausibly correlates with the loadings. The projection
  needs weights that do, and neither a covariate nor an outcome transformation
  will help if the relationship is absent -- though no diagnostic can tell you
  this, so it is a judgement about the setting;
* the effect is expected to switch sign over the post-period. A weighted average
  of a sign-changing path is a number whose interpretation depends on the
  weights, and the kernel's weights are not a policy choice.

Localization against donor weighting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ATEL differs from the estimators around it on two axes at once, and the second
is easy to miss.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Estimator
     - Factors from
     - Loading
     - Post-period aggregation
   * - Synthetic control
     - donor outcomes, convex hull
     - fixed weights
     - unweighted mean
   * - :doc:`fma`
     - principal components
     - fixed over time
     - unweighted mean
   * - :doc:`gsynth`
     - principal components
     - fixed per unit
     - unweighted mean
   * - ATEL
     - diversified projection
     - varies with time
     - kernel, localized at :math:`T_0`

The loading row is the substantive disagreement with :doc:`fma`. Taking factors
by principal components and holding the loading fixed recovers, when the true
loading moves, a time-average of it; the paper makes this point directly
([ATEL]_, p. 12). ATEL estimates the movement instead of averaging over it.

Overview
--------

ATEL implements Lee, R.-C. (2026), *"Average Treatment Effect Localization:
Projection Methods in Synthetic Control"*, Econometric Theory.

The counterfactual comes from a factor model in which the treated unit's
loading changes over time. Three ideas make that estimable.

The first is diversified projection (Fan and Liao, [FanLiao]_). A factor is
usually extracted as an eigenvector of the donor panel's covariance, which needs
that covariance to have a clean spectrum. Diversified projection instead takes a
weighted average across donors,
:math:`\widehat F_{tj} = N^{-1} \sum_i Y_{it} W_{it}^{(j)}`, with weights chosen
so that they correlate with the loadings. No eigendecomposition is taken, so the
donor panel's own spectrum never enters.

The second is the sieve. The weights have to correlate with the loadings, and
the loadings are unobserved, so they are built from the observed covariates: if
the loading at time t is a smooth function of :math:`X_{it}`, a finite basis in
:math:`X_{it}` spans that function approximately, and the basis values serve as
the weights.

The third is local linear smoothing. With the factors in hand, the treated
unit's loading is fit on the pre-period by a regression that weights nearby
periods more than distant ones, giving a level and a slope at :math:`T_0`, and
the slope carries the loading into the post-period.

Notation
--------

Let :math:`i = 1` denote the treated unit and :math:`i = 2, \ldots, N + 1` the
donors, observed over :math:`t = 1, \ldots, T`. Treatment begins at
:math:`T_0 + 1`, giving :math:`T_1 = T - T_0` post-treatment periods. Following
Abadie's potential-outcome notation, :math:`Y_{it}^N` is the outcome without
intervention and :math:`Y_{it}^I` the outcome with it, so the observed outcome is
:math:`Y_{1t} = Y_{1t}^N` for :math:`t \le T_0` and :math:`Y_{1t} = Y_{1t}^I`
after.

:math:`\mathbf F_t \in \mathbb R^J` is the vector of :math:`J` latent factors at
time t and :math:`\boldsymbol\beta_{it} \in \mathbb R^J` the loading of unit i at
time t, so the no-intervention outcome is
:math:`Y_{it}^N = \boldsymbol\beta_{it}^\top \mathbf F_t + u_{it}`. The loading
carries a time subscript, which is the model's departure from a conventional
factor structure.

:math:`X_{it} \in \mathbb R^P` are the observed covariates and
:math:`W_{it}^{(j)}` the j-th diversified weight built from them.
:math:`K(\cdot)` is the Epanechnikov kernel, :math:`h` the bandwidth as a
fraction of the sample, and :math:`K_h(\cdot) = h^{-1} K(\cdot / h)`.

The estimand is

.. math::

   \alpha = \frac{1}{T_1} \sum_{t = T_0 + 1}^{T}
            \mathbb E \bigl( Y_{1t}^I - Y_{1t}^N \bigr)
            K_h\!\left( \frac{t - T_0}{T_1} \right).

Setting :math:`K` to the constant 1 recovers the ATT, so ATEL nests it and
differs from it only in the weights.

Assumptions
-----------

*Assumption 1 (time-varying factor structure).* The no-intervention outcomes
follow :math:`Y_{it}^N = \boldsymbol\beta_{it}^\top \mathbf F_t + u_{it}` with a
fixed number :math:`J` of factors, and the loading path
:math:`\boldsymbol\beta_{it} = \boldsymbol\beta_i(t/T)` is smooth in rescaled
time.

*Remark.* The smoothness is what makes a local fit informative: within a window
of width :math:`\lfloor T_0 h \rfloor` the loading is approximately constant, so
the periods in that window speak to the same quantity. A loading that jumps --
a merger, a redefinition of the series -- violates this at the jump, and the fit
near it averages two different regimes.

*Assumption 2 (the covariates span the loadings).* The loading is a smooth
function of the observed covariates, well approximated by the chosen sieve, and
the resulting weights satisfy Fan and Liao's rank condition: the matrix
:math:`\mathbb E [ W_{it} \boldsymbol\beta_{it}^\top ]` has full rank J.

*Remark.* This assumption has two halves and only one of them is checkable.
Fan and Liao's Assumption 2.1 asks that the weights be bounded and that
:math:`\lambda_{\min}(W'W / N)` stay away from zero, so the weight series are
not carrying the same information as each other. That is a number, and it comes
back on the result as ``diagnostics["weight_lambda_min"]``. Their rank
condition, :math:`\mathrm{rank}(W'B/N) = r`, involves the unobserved loadings
and cannot be checked at all: if the weights carry no information about the
loadings, the factors are noise and nothing downstream reports a problem. So
read the conditioning diagnostic, choose weights that plausibly relate to the
unit's exposure to common shocks, and read the implied donor weights to see
which donors the answer rests on.

*Assumption 3 (no anticipation, untreated donors).* The treated unit is
unaffected before :math:`T_0 + 1`, and no donor is treated over the sample, so
both the factors and the pre-period loading are estimated from no-intervention
outcomes.

*Remark.* ATEL is more exposed to this than a long-horizon estimator. Its weight
is concentrated at the adoption date, so contamination in the periods either side
of :math:`T_0` lands where the estimate is most sensitive. A treated unit that
responds in anticipation biases exactly the periods the kernel weights most.

*Assumption 4 (the loading extrapolates).* The level and slope fit at
:math:`T_0` continue to describe the treated unit's no-intervention loading over
the post-period.

*Remark.* This is a linear extrapolation of a smooth path, so its error grows
with distance from :math:`T_0`. The localization and this assumption support each
other: the kernel puts its weight where the extrapolation is most reliable, and
the extrapolation is only asked to travel as far as the kernel's weight does.

*Assumption 5 (regularity for inference).* The idiosyncratic errors satisfy the
moment and weak-dependence conditions of [ATEL]_ Theorem 1, and
:math:`N, T_0, T_1` are large enough for the normal approximation, with
:math:`h \to 0` and :math:`T_0 h \to \infty`.

*Remark.* The two bandwidth conditions conflict at any finite sample size: a
small h localizes the estimand but leaves few effective observations. The
cross-validated default resolves it by fit, and the estimator says when that
resolution lands at an endpoint of the search grid, which means the criterion
was still improving where the grid stopped.

Mathematical Formulation
------------------------

Step 1, the weights. For each covariate p, evaluate a sieve basis of width J at
every :math:`X_{it,p}` pooled over all units and periods, giving
:math:`J \times P` weight series. The B-spline basis takes its order from the
width: linear at :math:`J = 2`, quadratic at :math:`J = 3`, cubic with
:math:`J - 2` breakpoints above that.

Step 2, the factors. For :math:`j = 1, \ldots, J`,

.. math::

   \widehat F_{tj} = \frac{1}{N} \sum_{i = 2}^{N + 1} Y_{it} W_{it}^{(j)} .

Step 3, the loading. With
:math:`\mathbf Z_t = [\widehat{\mathbf F}_t,\ \widehat{\mathbf F}_t (t/T_0 - 1)]`
and kernel weights :math:`k_t` one-sided at :math:`T_0`,

.. math::

   (\widehat{\mathbf b}_0, \widehat{\mathbf b}_1)
   = \arg\min_{\mathbf b} \sum_{t = 1}^{T_0}
     k_t \bigl( Y_{1t} - \mathbf Z_t^\top \mathbf b \bigr)^2 ,

and the post-period loading is
:math:`\widehat{\boldsymbol\beta}_{1t} = \widehat{\mathbf b}_0 + \widehat{\mathbf b}_1 (T_0 - t)/T_1`.

Step 4, the counterfactual.
:math:`\widehat Y_{1t}^N = \widehat{\boldsymbol\beta}_{1t}^\top \widehat{\mathbf F}_t`
for :math:`t > T_0`.

Step 5, the estimate. With
:math:`\widetilde T_1 = \lfloor T_1 h \rfloor` and
:math:`\kappa_t = 2 K((t - T_0)/\widetilde T_1)`,

.. math::

   \widehat\alpha = \frac{1}{\widetilde T_1} \sum_{t = T_0 + 1}^{T}
                    \bigl( Y_{1t} - \widehat Y_{1t}^N \bigr) \kappa_t .

The kernel is one-sided with its mass doubled, since :math:`T_0` is an endpoint
of the sample and only half the Epanechnikov support is available.

Inference and diagnostics
-------------------------

The interval on :math:`\widehat\alpha` comes from [ATEL]_ Theorem 1, whose
variance has two parts: the sampling error in the estimated loading, propagated
through the kernel-weighted aggregation, and the treated unit's own
idiosyncratic errors over the effective window. Both are built from pre-period
residuals refit period by period with the Su and Wang ([SuWang]_) boundary
kernel. The p-value uses :math:`t_{\widetilde T_1 - 1}`.

Five diagnostics come back on the result, and each answers a question the point
estimate alone does not.

``kernel_weights`` is the weight on each post-period. They do not sum to one, so
a constant effect d gives an estimate of d times their mass, available as
``kernel_mass``. Reading these tells you which periods the number is made of.

``implied_donor_weights`` is the weight each donor carries at each post-period.
Substituting the projection into the counterfactual,

.. math::

   \widehat Y_{1t}^N = \sum_{i} Y_{it}
       \Bigl[ \frac{1}{N} \sum_j \widehat\beta_{1t,j} W_{it}^{(j)} \Bigr],

so ATEL is a donor-weighting estimator whose weights move over time. They come
from a projection, so they are unconstrained in sign and do not sum to one. The
standardized ``weights`` slot carries each donor's path averaged against the
kernel, which is the weight it carries in the reported estimate.

``diagnostics["bandwidth_at_grid_edge"]`` records whether the cross-validated
bandwidth landed on an endpoint of the search grid, and the estimator warns when
it does. An endpoint means the criterion was still improving where the grid ran
out, so the bandwidth is a corner solution and not an interior optimum. On the
paper's own Arizona panel this happens at every factor count tried: the selected
h is 0.95, the top of the grid, and the smoothing window then spans 16 of 17
pre-periods, so the local fit is close to a global one.

``diagnostics["weight_lambda_min"]`` is
:math:`\min_t \lambda_{\min}(W_t' W_t / N)` over the periods, the checkable
half of Fan and Liao's Assumption 2.1 evaluated on the cross-section actually
projected. Near zero means two weight series carry the same information and the
projection has fewer usable directions than its factor count suggests.
``diagnostics["weight_source"]`` records which construction produced them.

``pointwise_standard_errors`` supports a band around the counterfactual path.
This needs the treated unit's loading path to be independent and normal across
periods, which is stronger than Theorem 1 requires. The interval on
:math:`\widehat\alpha` is the defensible output; treat the band as indicative.

Where the weights come from
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``weight_source`` picks the construction, following Fan and Liao's Section 4.
Only the first of the three needs covariates.

.. list-table::
   :header-rows: 1
   :widths: 18 14 34 34

   * - ``weight_source``
     - Fan-Liao
     - What builds the weights
     - What it assumes
   * - ``"covariates"``
     - 4.1
     - a sieve basis in the observed covariates at every unit-period
     - the loadings are driven by those covariates
   * - ``"initial"``
     - 4.3
     - a sieve basis in each unit's first outcome, :math:`\phi_k(x_{i0})`
     - :math:`(f_0, u_0)` independent of the later errors
   * - ``"hadamard"``
     - 4.4
     - deterministic sign columns, no data at all
     - nothing about the data; the rank condition holds only generically

``"covariates"`` is the default and is what the paper uses. The other two exist
because the projection does not need covariates -- a panel carrying only an
outcome and a treatment indicator is still usable.

``"initial"`` holds out the first period, since that is what the weights are
built from, and leaving it in would return the outcome the weights came from to
the fit they are used for. The pre-period count therefore drops by one.

What these constructions must not be is a function of the errors they are meant
to diversify away. A tempting choice is each unit's pre-period mean outcome,
which keeps post-treatment information out and still violates the assumption,
because the mean is a function of the pre-period errors the loading is fit on.
On the HCW panel that choice moves the estimate from 0.026-0.034 across factor
counts to 0.032-0.046 drifting upward, and away from what the rest of the
library reports. The initial observation is the legitimate version of the same
idea.

Fan and Liao's Section 4.2, weights from trimmed principal-component loadings on
an earlier sample split, is not offered. It needs a split and serial
independence of the errors, which is a different assumption burden.

Which basis, and why it matters here
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``"initial"`` the choice of ``basis`` decides whether Assumption 2.1(ii)
survives. Fan and Liao's own simulations use the polynomial
:math:`\phi_k(z) = z^k`, which is fine for a covariate of order one. An outcome
in natural units is often far smaller, and then the monomials collapse onto each
other. On the HCW panel, where the initial growth rates run -0.03 to 0.14:

.. list-table::
   :header-rows: 1
   :widths: 16 12 26 22 22

   * - basis
     - J
     - :math:`\lambda_{\min}(W'W/N)`
     - :math:`\mathrm{cond}(W)`
     - :math:`\max|w|`
   * - polynomial
     - 2
     - 5.3e-06
     - 2.3e+01
     - 0.14
   * - polynomial
     - 6
     - 1.6e-17
     - 1.3e+07
     - 0.14
   * - bspline
     - 2
     - 1.1e-01
     - 2.2e+00
     - 1.00
   * - bspline
     - 6
     - 1.2e-03
     - 1.4e+01
     - 1.00

The B-spline basis satisfies both halves of Assumption 2.1 comfortably and
without rescaling: partition of unity bounds the entries at one exactly, and the
knots come from the data's own range, so the result does not depend on the units
the outcome is measured in. It is the default for every source.

The trigonometric basis is the one to avoid at an even factor count: an
intercept plus complete cos/sin pairs fills an odd number of columns, so the
last one is zero and :math:`\lambda_{\min}` is exactly zero. ATEL refuses that
configuration where the projection would consume the empty column.

The factor count and the covariate count
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``n_factors`` is required, and it must be a multiple of the number of
covariates. Both constraints come from measurement on the reference
implementation.

It is required because the information criterion that would choose it does not
choose. On the paper's Arizona panel the criterion is monotone over its whole
candidate range, so it returns an endpoint of that range under either extremum:
the residual it penalises is an in-sample projection onto :math:`2J` directions
against :math:`N = 14` donors, and it collapses to zero once :math:`2J` reaches
N, taking the criterion with it. See :doc:`replications/atel` for the full
decomposition.

Being required is less of an imposition than it sounds, because the theory says
what to do instead. Fan and Liao ([FanLiao]_) prove the projection valid for any
working number of factors :math:`R \ge r`, admitting even :math:`r = 0` with
:math:`R \ge 1`, and advise taking "a slightly large R so that
:math:`R \ge r` is likely to hold". Over-estimating is proved safe;
under-estimating is not. There is a second reason to be generous: their rank
condition fails when more than :math:`R - r` weight series are nearly orthogonal
to the loadings, and extra series make that less likely. So pick a count a little
above what you think the panel supports and report it, instead of hunting for a
number the data will not give you.

This is also why the paper can claim its results do not rely on accurate
estimation of J while its own selection criterion is broken. The claim rests on
Fan and Liao's theorem, not on the criterion, and the two are compatible: the
criterion is superfluous.

It must be a multiple of the covariate count because the projection consumes the
first J of the :math:`J \times P` weight blocks, which is a complete set of
(basis, covariate) pairs only in that case. Otherwise the slice takes some basis
for one covariate and not another, and the estimate depends on the order the
covariates are named -- on the Arizona panel, by 22.6 at :math:`J = 3` and 71.4
at :math:`J = 5`, against standard errors near 13. Such a configuration raises
:class:`~mlsynth.exceptions.MlsynthConfigError` at construction.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import ATEL

   # UK inward FDI against 29 OECD comparators, 1990-2017.
   # The Brexit referendum lands in 2016, giving 22 pre-periods and 6 after.
   df = pd.read_csv("basedata/fdi_oecd_brexit.csv")

   res = ATEL({
       "df": df, "outcome": "fdi", "treat": "treated",
       "unitid": "country", "time": "year",
       "covariates": ["log_gdp", "log_gdp_percap"],
       "n_factors": 2,
       "display_graphs": False,
   }).fit()

   print(res.atel)                      # the localized estimate
   print(res.inference.ci_lower, res.inference.ci_upper)
   print(res.att)                       # the unweighted post-period mean
   print(res.kernel_mass)               # what a constant effect would scale by
   print(res.diagnostics["bandwidth_at_grid_edge"])

A panel with no covariates at all works the same way, with the weights coming
from each unit's first outcome:

.. code-block:: python

   # Hsiao, Ching and Wan (2012): Hong Kong quarterly GDP growth against 24
   # comparator economies, integration with mainland China from quarter 44.
   # The file carries an outcome and a treatment indicator, nothing else.
   df = pd.read_csv("basedata/HongKong.csv")

   res = ATEL({
       "df": df, "outcome": "GDP", "treat": "Integration",
       "unitid": "Country", "time": "Time",
       "weight_source": "initial",
       "n_factors": 3,
       "display_graphs": False,
   }).fit()

   print(res.atel, res.att)                      # 0.0324, 0.0346
   print(res.diagnostics["weight_lambda_min"])   # 0.0301, the conditioning check

Two things this panel shows. The estimate sits beside what the library already
reports -- :doc:`fdid` gives 0.02540 and :doc:`fma` 0.02543 -- and the
pre-treatment fit is tighter than either, with an RMSE of 0.0143 at three factors
and 0.0132 at six against 0.0162 and 0.0177, on weights built from one
observation per unit.

And the asymmetry between over- and under-estimating the factor count is visible
directly. At three factors and above the estimate settles: 0.0324, 0.0317, 0.0309,
0.0312 for J of 3, 4, 5, 6. At two factors it is 0.0171, a different answer, which
is what taking R below r looks like. The cost of going the other way is not in the
estimate but in the conditioning, which falls from 1.1e-01 at two factors to
1.2e-03 at six: over-estimating stays valid and gradually spends the weights'
independence, so there is a practical ceiling even though there is no
correctness one.

What this panel does not show is why you would localize. Whether ``atel`` lands
above or below ``att`` here depends on the selected bandwidth -- above it at two
factors, below it at three -- because the kernel's window moves with h and the
post-period gap is not monotone. A panel whose effect has a clear direction over
the post-period is where the estimand earns its place, and this is not one.

``res.atel`` and ``res.att`` are different estimands and neither substitutes for
the other: ``atel`` is the kernel-weighted localized estimate the method is
about, and ``att`` is the unweighted post-period mean that the two outcome paths
imply. ``res.inference`` describes ``atel``.

Verification
------------

ATEL is cross-validated against the author's MATLAB toolbox
(`rueichilee/ATEL <https://github.com/rueichilee/ATEL>`_), run under Octave. The
paper's Section 5 application -- Arizona's 1994 right-to-carry law on violent
crime, 14 donor states, 1977-2006 -- reproduces every row of its Table 4 to four
decimals:

.. list-table::
   :header-rows: 1
   :widths: 12 22 22 18

   * - :math:`J`
     - ATEL
     - Standard error
     - Order-invariant
   * - 2
     - 49.7665
     - 15.0055
     - yes
   * - 3
     - 71.9248
     - 12.3949
     - no
   * - 4
     - 57.8420
     - 13.3495
     - yes
   * - 5
     - 99.0401
     - 13.7880
     - no

The final column is this implementation's finding, not the paper's. The two rows
marked no move by 22.6 and 71.4 when the two covariates are passed in the other
order, for the reason given above, and ``ATEL`` refuses those configurations.
Restricted to the two that are order-invariant the estimate is stable, 49.77 and
57.84, a spread well inside one standard error.

Two smaller fixtures are pinned in ``mlsynth/tests/test_atel.py`` against the
same toolbox: the bundled Brexit panel above and a deterministic synthetic panel,
each at :math:`J = 2` and :math:`J = 4` and at a fixed bandwidth. Agreement is
to nine significant figures on the estimate, the standard error and the p-value.
The B-spline basis is checked separately against ``scipy``'s
``BSpline.design_matrix``, which computes the same object as the toolbox's
``spcol``.

See :doc:`replications/atel` for the full replication record, including the
defects found in the reference implementation along the way.

Core API
--------

.. automodule:: mlsynth.estimators.atel
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.ATELConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.atel_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.atel_helpers.sieve
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.atel_helpers.factors
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.atel_helpers.loadings
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.atel_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.atel_helpers.numerics
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.atel_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.atel_helpers.plotter
   :members:
   :undoc-members:

.. note::

   ``ATEL.fit()`` returns an :class:`~mlsynth.config_models.EffectResult` on the
   standardized two-family contract, so ``res.att`` / ``res.att_ci`` /
   ``res.counterfactual`` / ``res.gap`` / ``res.pre_rmse`` /
   ``res.donor_weights`` all resolve. The localized estimate is ``res.atel``,
   also mirrored in ``res.effects.additional_effects["atel"]``, and
   ``res.inference`` describes it.

.. automodule:: mlsynth.utils.atel_helpers.structures
   :members:
   :undoc-members:

References
----------

.. [ATEL] Lee, R.-C. (2026). "Average Treatment Effect Localization:
   Projection Methods in Synthetic Control." *Econometric Theory*.

.. [FanLiao] Fan, J., & Liao, Y. (2022). "Learning Latent Factors from
   Diversified Projections and Its Applications to Over-Estimated and
   Weak Factors." *Journal of the American Statistical Association*
   117(538):909-924.

.. [SuWang] Su, L., & Wang, X. (2017). "On Time-Varying Factor Models:
   Estimation and Testing." *Journal of Econometrics* 198(1):84-101.
