Functional Synthetic Control (FSC)
==================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Almost every synthetic control method assumes that what you observe each period
is a number. FSC is for the case where it is a whole object: a fertility rate
recorded separately for every age, a distribution of ages at death, a covariance
matrix across a country's export categories.

You can always turn such an object into a number and run an ordinary synthetic
control on that. The question is what you lose. East Germany legalised abortion
in 1972, and the effect on fertility was concentrated among women aged roughly
20 to 30 and close to nil at either end of the childbearing range. Collapse the
age profile to a total fertility rate and that result becomes a single negative
number; the shape of the response, which is the part a demographer would want,
is gone. The same argument applies whenever the interesting variation is
*within* the object rather than in its total: a policy that compresses a wage
distribution without moving its mean, a shock that reallocates trade between
categories without changing the total.

Reach for FSC when the object per unit-period is a curve, a distribution, or a
matrix, and when you already have it in that form. If your rows are individual
people rather than points on a grid, :doc:`dsc` is the right tool for
unconditional quantile effects and :doc:`drsc` for conditional ones. If your
object is a vector of shares summing to one, use :doc:`compsc`.

Notation
--------

There are :math:`N` units observed over :math:`T` periods. Unit :math:`i = 1` is
treated from period :math:`T_0 + 1` onward; the rest are the donor pool. The
outcome :math:`\nu_{it}` is not a number but a point in a metric space
:math:`(\mathcal M, d)` -- the set of curves, of distributions, of positive
semidefinite matrices.

The difficulty is that :math:`\mathcal M` has no linear structure. Averaging two
probability distributions by averaging their density values does not generally
give a sensible "average distribution", so the weighted-average counterfactual
that defines synthetic control is not even well posed.

FSC removes the difficulty with an isometric embedding: a map
:math:`\Psi : \mathcal M \to \mathcal H` into a Hilbert space that preserves
distances exactly,

.. math::

   d(x, y) \;=\; \lVert \Psi(x) - \Psi(y) \rVert_{\mathcal H}
   \qquad \text{for all } x, y \in \mathcal M .

An isometry is a relabelling that gets the geometry right: distances measured
after the embedding are the same distances as before, so nothing about the
problem is distorted by moving into :math:`\mathcal H`. Inside :math:`\mathcal H`
addition and scalar multiplication are available again, and the ordinary
synthetic control construction goes through. Write
:math:`Y_{it} = \Psi(\nu_{it})` for the embedded outcomes and
:math:`\mathcal Y = \Psi(\mathcal M)` for their image.

Which map :math:`\Psi` to use depends on the space, and the ``space`` option
picks it:

.. list-table::
   :header-rows: 1
   :widths: 16 32 26 26

   * - ``space``
     - Objects
     - :math:`\Psi`
     - :math:`\mathcal Y`
   * - ``"function"``
     - curves in :math:`L^2`
     - the identity
     - all of :math:`\mathcal H`
   * - ``"distribution"``
     - distributions under the 2-Wasserstein metric
     - the quantile function
     - increasing functions
   * - ``"matrix"``
     - symmetric positive semidefinite matrices, Frobenius metric
     - the half-vectorisation
     - the PSD cone

The estimand is the counterfactual object :math:`\nu^N_{1t}` for the treated
unit after treatment. What you do with it is up to you: the difference of
embedded objects :math:`Y_{1t} - Y^N_{1t}` is a curve, reported here as
``curve.effect``, and its length

.. math::

   d_t \;=\; d(\nu^I_{1t}, \nu^N_{1t})
        \;=\; \lVert Y_{1t} - Y^N_{1t} \rVert_{\mathcal H}

is the scalar magnitude of the effect, reported as ``curve.magnitude``.

Assumptions
-----------

Assumption 1 (Embeddability). :math:`(\mathcal M, d)` admits an isometric
embedding into a Hilbert space, and the image :math:`\Psi(\mathcal M)` is closed
and convex.

Remark. This is a real restriction, not a formality. A metric space embeds
isometrically into a Hilbert space exactly when it has 2-negative type
(Schoenberg), and many do not -- notably the space of probability measures on
:math:`\mathbb{R}^k` under the 2-Wasserstein metric for :math:`k \ge 2`. So
"metric space-valued" here does not mean any metric space. The three spaces
this estimator supports all satisfy it. Convexity of the image is what makes the
plain estimator work at all: a convex combination of donor objects then lands
back in :math:`\mathcal Y` automatically, so it corresponds to a real object.

Assumption 2 (Common grid). Every unit-period cell is observed at the same
argument values.

Remark. The estimator checks this at ingestion and raises when it fails.
Objects observed on different grids are not comparable coordinate by
coordinate, and interpolating them onto a common grid is a modelling choice
that belongs to you. Do it yourself first if you need to.

Assumption 3 (Data-generating process). The embedded control outcomes follow
either a functional autoregression or a latent factor model, with independent
mean-zero errors bounded in :math:`\mathcal H`.

Remark. These are the same two processes the scalar synthetic control literature
assumes, lifted to :math:`\mathcal H`. They are what deliver the error bounds
below; the estimator itself does not need them, but the guarantee does.

Assumption 4 (Pre-treatment fit). The treated unit's pre-treatment objects are
approximately matched by the weighted donors.

Remark. This is the assumption that does the work, and it is checkable. The
finite-sample bound says the estimation error is controlled by the pre-treatment
fit and the norm of the weights and by nothing else, so a poor pre-fit is a
direct warning about the estimate. ``pre_treatment_fit`` on the result is that
quantity.

Estimation
----------

The donor weights match the pre-treatment objects in :math:`\mathcal H`:

.. math::

   \widehat\gamma^{\mathrm{scm}} \in
   \operatorname*{arg\,min}_{\gamma \in \Delta^{N-1}}
   \sum_{t=1}^{T_0} \Bigl\lVert Y_{1t} - \sum_{i=2}^{N} \gamma_i Y_{it}
   \Bigr\rVert_{\mathcal H}^{2} ,

over the simplex, and the counterfactual is
:math:`\widehat\nu^{N}_{1t} = \Psi^{-1}\bigl(\sum_i \widehat\gamma_i Y_{it}\bigr)`.

Once each object is sampled on a common grid this is the ordinary simplex
least-squares problem, with the :math:`(\text{period}, \text{grid point})` pairs
stacked into one long vector. So the base solve here is mlsynth's own exact
quadratic program, the same one :doc:`vanillasc` uses. Nothing about the
optimisation is special; the embedding is what makes it applicable.

When the pre-treatment fit is imperfect the estimator is biased, and section 3.2
of the paper corrects it the way augmented synthetic control corrects the scalar
case: fit a ridge regression of the post-period object on the pre-period ones,
and add the imbalance it predicts. Expanding each centered object in a cubic
B-spline basis :math:`\{\varphi_k\}_{k=1}^{K}` and writing
:math:`r_{i\cdot}` for the stacked coefficients gives a closed form,

.. math::

   \widehat\gamma^{\mathrm{aug}}_i = \widehat\gamma^{\mathrm{scm}}_i
     + \bigl(r_{1\cdot} - r_{0\cdot}' \widehat\gamma^{\mathrm{scm}}\bigr)'
       \bigl(r_{0\cdot}' r_{0\cdot} + \lambda I\bigr)^{-1} r_{i\cdot} ,

which is exactly the ridge-augmented synthetic control formula with the
pre-period axis replaced by the (period, basis coefficient) axis. Two properties
follow directly. The correction vanishes when the
simplex fit is already perfect, so augmentation never disturbs a good fit. And
the augmented weights still sum to one but may go negative, which means the
synthetic unit is allowed outside the donor hull -- extrapolation, bought
deliberately in exchange for balance.

Because negative weights can push the result out of :math:`\mathcal Y`, the
augmented estimate is projected back:

.. math::

   \widetilde Y^{N,\mathrm{aug}}_{1t}
     = \operatorname*{arg\,min}_{y \in \mathcal Y}
       \lVert y - \widehat Y^{N,\mathrm{aug}}_{1t} \rVert_{\mathcal H} .

Both projections have closed forms. For quantile functions it is the increasing
rearrangement -- sort the values -- and for matrices it is the eigenvalue clip:
symmetrise, set negative eigenvalues to zero, reassemble. For plain curves
:math:`\mathcal Y` is everything and there is nothing to do.

The penalty :math:`\lambda` is chosen by leave-one-pre-period-out
cross-validation unless you pass ``ridge_lambda``. One caution about reading it:
the cross-validation objective is typically very flat near its minimum. On the
paper's fertility data it varies by 0.03 percent across :math:`\lambda \in [5,7]`
while the pre-treatment fit moves in its fourth decimal. The penalty is weakly
identified and the estimate is insensitive to it, so do not read the selected
value as an estimated quantity.

Inference and diagnostics
-------------------------

Two procedures, selected by ``inference``.

The conformal band inverts the sharp null :math:`Y^N_{1t}(x) = y_0` pointwise in
the argument, comparing the post-treatment residual against the :math:`T_0`
pre-treatment ones. The inversion is closed-form: the accepted set is an
interval centred on the estimate whose half-width is a quantile of the absolute
pre-treatment residuals. It requires :math:`\alpha > 1/(T_0 + 1)`; below that
threshold nothing is ever excluded and the band is unbounded, which the
estimator reports rather than returning an infinite interval.

Two caveats belong with any band you report from this. The weights are held
fixed across candidate nulls rather than refit, unlike Chernozhukov, Wüthrich and
Zhu (2021); that is a deliberate choice which guarantees the band contains the
point estimate, at the cost of the exchangeability argument that would justify
it. And the paper conjectures asymptotic validity rather than proving it. Treat
the band as a descriptive uncertainty measure, not a calibrated confidence set.

The placebo test recomputes everything with each donor cast as the treated unit
and ranks the real treated unit's effect magnitude among them. It is honest and
assumption-light, and it costs :math:`N` times a full fit. Its resolution is set
by the donor pool: with :math:`N` units the smallest attainable p-value is
:math:`1/N`, so 21 units can never produce a p-value below 0.048.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import FSC

   # one row per (country, year, age): the fertility rate at that age
   df = pd.read_csv("basedata/okano_fsc_fertility.csv")

   res = FSC({
       "df": df,
       "outcome": "asfr",       # the value
       "argument": "age",       # where along the object it sits
       "treat": "treat",
       "unitid": "unit",
       "time": "time",
       "space": "function",
       "display_graphs": False,
   }).fit()

   res.pre_treatment_fit_fsc     # fit before augmentation
   res.pre_treatment_fit         # fit after
   res.effects.att               # grid-averaged post-treatment effect

   last = res.curves[-1]
   last.effect                   # the effect as a curve over age
   last.magnitude                # its length, the paper's d_t

For a distribution, pass the quantile function and set
``space="distribution"``; ``value_bounds`` truncates to the support before the
rearrangement. For a covariance matrix, pass the row-major lower triangle with
``space="matrix"`` -- the off-diagonal coordinates are rescaled internally so
that the half-vectorisation really is a Frobenius isometry.

Verification
------------

Reproduced against Okano and Kurisu (2026) on the authors' own data. The
fertility application matches the published pre-treatment fits and every donor
weight of Table 1; the divergences on the other two applications are measured
and explained on the replication page rather than smoothed over. See
:doc:`replications/fsc`, `benchmarks/cases/fsc_okano.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/fsc_okano.py>`_
and `benchmarks/cases/fsc_estimator.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/fsc_estimator.py>`_.

Not to be confused with
-----------------------

:doc:`fscm` is Forward-Selected synthetic control (Cerulli), an entirely
different method that happens to share three letters. :doc:`dsc` and :doc:`drsc`
work on individual-level microdata rather than on objects supplied per cell.
:doc:`compsc` covers compositions, which are the paper's Example 5.

Core API
--------

.. autoclass:: FSC
   :members: fit

.. autoclass:: mlsynth.utils.fsc_helpers.config.FSCConfig
   :members:

.. autoclass:: mlsynth.utils.fsc_helpers.structures.FSCResults
   :members:

.. autoclass:: mlsynth.utils.fsc_helpers.structures.FSCCurve
   :members:
