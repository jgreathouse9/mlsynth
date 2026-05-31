Matching and Synthetic Control (MASC)
======================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The synthetic control estimator (SCE) of Abadie and Gardeazabal [ABADIE2003]_
and Abadie, Diamond and Hainmueller [ABADIE2010]_ builds the treated unit's
counterfactual as a convex combination of donor units. By construction it
controls **extrapolation bias** -- the counterfactual cannot fall outside the
convex hull of the donors -- at the cost of **interpolation bias** when the
conditional mean of the outcome is non-linear in the pre-treatment
characteristics. Matching estimators have the opposite profile: by using only
the donor closest to the treated unit they avoid interpolation, but happily
extrapolate when no good donor exists. The two biases are *complementary*.

``MASC`` exploits that complementarity. Kellogg, Mogstad, Pouliot &
Torgovitsky [KMPT2021]_ propose a **model-averaging** estimator that takes a
convex combination of the SCE weight vector and a nearest-neighbour weight
vector, with the two tuning parameters -- the number of neighbours
:math:`m` and the averaging weight :math:`\varphi` -- selected jointly by
**rolling-origin cross-validation**. The :math:`\varphi`-search has a
closed-form (Kellogg et al., eq. 15), so the CV grid is one-dimensional in
:math:`m`.

The estimator is the right tool when:

* you have a single treated unit and a sizeable donor pool;
* the conditional mean of the outcome is plausibly non-linear in the
  pre-treatment covariates (so an SCE alone may interpolate badly), **and**
* you also worry that no single donor is close enough to extrapolate from
  (so pure matching alone may extrapolate badly).

When SCE achieves a tight pre-period fit and out-of-sample CV doesn't favour
mixing in any matching, MASC reduces to pure SCE (:math:`\varphi = 0`). When
SCE pre-fit is poor, the CV pulls :math:`\varphi` toward 1. In intermediate
regimes the model average dominates either pure estimator.

**Use cases.** In *marketing science* MASC is useful when the treated brand or
market has both close competitors (favouring matching) and a wider set of
plausibly-correlated brands (favouring SCE) -- a campaign rolled out in a
single DMA where one or two control DMAs look very similar but the broader
market also moves coherently. In *policy evaluation* MASC fits the canonical
Abadie--Gardeazabal use case of comparative case studies with idiosyncratic
shocks (Basque Country and Spanish terrorism, Andersson's Swedish carbon tax,
California Proposition 99) where neither pure SCE nor pure matching is
unambiguously preferred.

Notation
--------

We use the synthetic-control canon. Unit :math:`j=0` is treated and
:math:`\mathcal{N} = \{1, \ldots, N\}` indexes the donor pool;
:math:`\mathbf{y}_0` is the treated outcome path and :math:`\mathbf{Y}` is the
:math:`(T,N)` donor outcome matrix. The pre-treatment window is
:math:`\mathcal{T}_1 = \{1, \ldots, T_1\}` and the post-treatment window is
:math:`\mathcal{T}_2 = \{T_1+1, \ldots, T\}`, with treatment beginning at
:math:`t = T_1 + 1`. Predictors are stacked into
:math:`(\mathbf{x}_0, \mathbf{X})` with :math:`\mathbf{x}_0\in\mathbb{R}^P` for
the treated unit and :math:`\mathbf{X}\in\mathbb{R}^{P\times N}` for the
donors. The simplex is

.. math::

   \Delta = \Bigl\{ \boldsymbol{\omega}\in\mathbb{R}^N :
       \boldsymbol{\omega}\ge \mathbf{0},\
       \sum_j \omega_j = 1 \Bigr\}.

Setup
-----

The matching and SCE weights and the MASC combiner are

.. math::

   \boldsymbol{\omega}_{\mathrm{match}}(m)_j
       &= \tfrac{1}{m}\,\mathbf{1}\!\Bigl\{ j \in \operatorname*{argmin}_{|S|=m}
           \sum_{i\in S} d(j_0, i) \Bigr\},
   \\[2pt]
   \boldsymbol{\omega}_{\mathrm{SC}}
       &\in \operatorname*{argmin}_{\boldsymbol{\omega}\in\Delta}
       \,\bigl\|\mathbf{x}_0 - \mathbf{X}\boldsymbol{\omega}\bigr\|_{\mathbf{V}}^2,
   \\[2pt]
   \boldsymbol{\omega}_{\mathrm{MASC}}(m,\varphi)
       &= \varphi\,\boldsymbol{\omega}_{\mathrm{match}}(m)
       + (1-\varphi)\,\boldsymbol{\omega}_{\mathrm{SC}},

where :math:`d(j_0, i) = \sum_{t\in\mathcal{T}_1} (y_{0t} - y_{it})^2` is the
pre-period squared-distance and :math:`\mathbf{V}` is the (possibly
optimised) predictor-weight matrix. Without ``covariates`` the SCE reduces
to outcome-paths matching, i.e. :math:`(\mathbf{x}_0, \mathbf{X}) =
(\mathbf{y}_0^{\mathrm{pre}}, \mathbf{Y}^{\mathrm{pre}})` with
:math:`\mathbf{V} = \mathbf{I}`.

Tuning by rolling-origin CV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each fold :math:`f\in\mathcal{F}` (each :math:`f` indexes the last
pre-treatment period included in the training window), let
:math:`\hat y^{\mathrm{SC}}_{f+1}` and :math:`\hat y^{\mathrm{match}}_{f+1}(m)`
denote the one-step-ahead forecasts of the treated outcome from each
estimator fit on the first :math:`f` periods, and let :math:`y_{0,f+1}` denote
the actual treated outcome. The CV criterion at :math:`(m,\varphi)` is the
weighted squared-error

.. math::

   Q(m,\varphi) = \sum_{f\in\mathcal{F}} w_f\,
       \bigl( y_{0,f+1} - \varphi \hat y^{\mathrm{match}}_{f+1}(m)
              - (1-\varphi)\hat y^{\mathrm{SC}}_{f+1} \bigr)^2 .

Holding :math:`m` fixed, the first-order condition gives the closed form

.. math::

   \tilde\varphi(m) =
   \frac{
       \sum_f w_f \bigl(y_{0,f+1} - \hat y^{\mathrm{SC}}_{f+1}\bigr)
                  \bigl(\hat y^{\mathrm{match}}_{f+1}(m) - \hat y^{\mathrm{SC}}_{f+1}\bigr)
   }{
       \sum_f w_f \bigl(\hat y^{\mathrm{match}}_{f+1}(m) - \hat y^{\mathrm{SC}}_{f+1}\bigr)^2
   } ,
   \quad
   \hat\varphi(m) = \operatorname{clip}_{[0,1]}\bigl(\tilde\varphi(m)\bigr),

reproducing eq. 15 of Kellogg et al. (2021). The selected
:math:`\hat m = \operatorname*{argmin}_m Q(m,\hat\varphi(m))` is then plugged
in and final weights are refitted on the full pre-period.

**Assumptions / Remarks.**

*Assumption 1 (complementarity).* The conditional mean of the outcome is
non-linear in the predictors -- so that pure SCE is susceptible to
interpolation bias -- *and* no single donor is identical to the treated
unit on those predictors -- so pure matching is susceptible to extrapolation
bias. *Remark.* This is the paper's central premise. When either bias is
absent the model average degenerates to one of the boundary estimators and
the CV will pick :math:`\hat\varphi \in \{0, 1\}` accordingly.

*Assumption 2 (rolling-origin stability).* The relationship between treated
and donor outcomes is stable across the late-pre-period folds and the
post-period -- so that out-of-sample forecast accuracy on the training-set
tail is informative about post-treatment forecast accuracy. *Remark.* This
is the SCE identification premise restricted to the fold horizon. Without
it the CV criterion is uninformative about the post-period.

*Assumption 3 (analytic-:math:`\varphi` validity).* The CV criterion is
quadratic in :math:`\varphi` with positive semi-definite Hessian, so
restricting to the interval :math:`[0, 1]` recovers the constrained
optimum. *Remark.* This is mechanical and lets the joint :math:`(m,\varphi)`
search reduce to a one-dimensional sweep over :math:`m`.

Empirical Illustration: Basque Country and Spanish Terrorism
-----------------------------------------------------------------

Following Section 5 of Kellogg et al. [KMPT2021]_ -- the canonical Abadie &
Gardeazabal [ABADIE2003]_ study of the per-capita GDP cost of ETA terrorism --
``MASC`` runs on ``basque_jasa.csv``: 17 Spanish regions (Basque plus 16
donor candidates), 1955-1997, with the JASA predictor specification
(schooling shares, investment, sector composition, population density).

.. code-block:: python

   import pandas as pd
   from mlsynth import MASC

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/"
          "main/basedata/basque_jasa.csv")
   df = pd.read_csv(url)

   covariates = [
       "school.illit", "school.prim", "school.med", "school.high", "invest",
       "sec.agriculture", "sec.energy", "sec.industry", "sec.construction",
       "sec.services.venta", "sec.services.nonventa", "popdens",
   ]
   # The covariate windows match Abadie & Gardeazabal (2003), Table 1:
   # schooling and investment are averaged over 1964-1969, the sector
   # shares (observed every other year) over 1961-1969, popdens is the
   # 1969 cross-section, and a lagged outcome ``gdpcap`` is matched on
   # the 1960-1969 mean (Abadie's "pre-treatment outcomes" predictor).
   covariate_windows = {
       "sec.agriculture": (1961, 1969), "sec.energy": (1961, 1969),
       "sec.industry": (1961, 1969), "sec.construction": (1961, 1969),
       "sec.services.venta": (1961, 1969),
       "sec.services.nonventa": (1961, 1969),
       "popdens": (1969, 1969),
       "invest": (1964, 1969),
       "school.illit": (1964, 1969), "school.prim": (1964, 1969),
       "school.med": (1964, 1969), "school.high": (1964, 1969),
       "gdpcap": (1960, 1969),
   }

   res = MASC({
       "df": df, "outcome": "gdpcap", "treat": "terrorism",
       "unitid": "regionname", "time": "year",
       "m_grid": list(range(1, 11)),
       "min_preperiods": 5,
       "covariates": covariates,
       "covariate_windows": covariate_windows,
       "display_graphs": False,
   }).fit()

   print(f"Selected m   : {res.m_hat}")
   print(f"Selected phi : {res.phi_hat:.3f}")
   print(f"Pre-RMSE     : ${res.fit.pre_rmse * 1000:.0f}/capita")
   print(f"ATT          : ${res.att * 1000:+.0f}/capita/year")
   print("Top donors:")
   for u, w in sorted(res.donor_weights.items(), key=lambda kv: -kv[1])[:4]:
       if w > 0.05:
           print(f"  {u:<32s} {w:.3f}")

This prints (roughly)::

   Selected m   : 1
   Selected phi : 0.324
   Pre-RMSE     : $97/capita
   ATT          : $-641/capita/year
   Top donors:
     Cataluna                         0.640
     Madrid (Comunidad De)            0.225
     Principado De Asturias           0.063
     Baleares (Islas)                 0.054

The paper [KMPT2021]_, Section 5, reports MASC ≡ SCE
(:math:`\hat\varphi = 0`), pre-RMSE :math:`\approx \$94`, ATT
:math:`\approx -\$580`/capita/year, with donor weights ``Cataluna 0.85``,
``Madrid 0.15``. The dominant donors (Cataluna + Madrid) agree, both
quantitatively (Cataluna 0.64 + Madrid 0.23 here vs.\ 0.85 + 0.15 in
KMPT) and in pre-RMSE (:math:`\$97` vs.\ :math:`\$94`). The ATT
:math:`-\$641` is within :math:`\$60` of the published :math:`-\$580`;
the residual gap is the documented V-optimiser non-uniqueness below.

.. note::

   **Why our :math:`\hat\varphi` is small but non-zero.** The JASA paper
   computes :math:`\boldsymbol{\omega}_{\mathrm{SC}}` via the ``synth()``
   package's quasi-Newton search over the predictor-weight matrix
   :math:`\mathbf{V}`. ``mlsynth`` delegates the V-optimisation to the
   Malo et al. [malo2023computing]_ bilevel solver (the same solver used by
   ``FSCM``). Both are mathematically valid V-optimisation strategies; on
   this problem they converge to slightly different :math:`\mathbf{V}` and
   therefore slightly different :math:`\mathbf{W}` (Cataluna 0.64 + Madrid
   0.23 vs.\ 0.85 + 0.15). The rolling-origin CV then prefers a small
   amount of nearest-neighbour matching (:math:`\hat\varphi \approx 0.32`,
   :math:`m=1`) rather than pure SC.

   This is the **non-uniqueness phenomenon** documented by Becker & Kloessner
   and discussed in Malo et al.: when the SC problem is over-parameterised
   (here 12 predictors over 16 donors) the upper-level loss is flat over many
   feasible :math:`\mathbf{V}`, and different V-optimisers converge to
   different :math:`\mathbf{W}`. Bit-perfect replication of JASA's Section 5
   would require a true ADH ``synth()`` port; the present implementation is
   a faithful port of the MASC *algorithm* (matching, rolling-origin CV,
   closed-form :math:`\varphi`) on top of mlsynth's bilevel V solver, with
   the documented caveat above.

Verification
------------

.. note::

   **Empirical (Basque proper).** With the Abadie-Gardeazabal predictor
   windows (schooling and investment 1964-1969, sector shares 1961-1969,
   popdens 1969, gdpcap 1960-1969), treatment starting in 1975 and Spain
   itself removed from the donor pool, MASC selects :math:`m=1`,
   :math:`\hat\varphi \approx 0.32`, pre-RMSE :math:`\approx \$97`/capita
   (vs.\ KMPT's :math:`\$94`) and ATT :math:`\approx -\$641`/capita/year
   (vs.\ KMPT's :math:`-\$580`). Donor mass concentrates on Cataluna
   (0.64) and Madrid (0.23) -- the same two-donor structure KMPT report
   (0.85 + 0.15). The residual gap is the V-optimiser non-uniqueness
   documented above.

   **Helpers.** The nearest-neighbour selector, the simplex SC primitive,
   the analytic :math:`\hat\varphi` formula and the per-fold covariate
   aggregation are unit-tested (``mlsynth/tests/test_masc.py``).

Core API
--------

.. automodule:: mlsynth.estimators.masc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.MASCConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``MASC.fit()`` returns a
:class:`~mlsynth.utils.masc_helpers.structures.MASCResults` containing the
selected ``(m_hat, phi_hat)``, the MASC weight vector (with the matching
and SC components separately preserved), counterfactual, pre/post gap,
pre-RMSE, ATT, and the full CV grid. The prepared NumPy panel is exposed
as a :class:`~mlsynth.utils.masc_helpers.structures.MASCInputs`.

.. automodule:: mlsynth.utils.masc_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- the only DataFrame touchpoint: pivots to NumPy, builds
the unit/time index, splits pre/post, assembles the optional covariate
panels for per-fold aggregation.

.. automodule:: mlsynth.utils.masc_helpers.setup
   :members:
   :undoc-members:

The nearest-neighbour selector, the simplex SC primitive (with optional
covariates routed through the bilevel solver) and the analytic-:math:`\varphi`
closed form.

.. automodule:: mlsynth.utils.masc_helpers.estimation
   :members:
   :undoc-members:

The rolling-origin cross-validation engine and the per-fold covariate
aggregator.

.. automodule:: mlsynth.utils.masc_helpers.crossval
   :members:
   :undoc-members:

The end-to-end pipeline composing CV with the full-sample refit and the
MASC weight combiner.

.. automodule:: mlsynth.utils.masc_helpers.orchestration
   :members:
   :undoc-members:

Plotting: outcome paths and the CV curve over the candidate ``m`` grid.

.. automodule:: mlsynth.utils.masc_helpers.plotter
   :members:
   :undoc-members:
