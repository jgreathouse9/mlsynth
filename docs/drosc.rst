Distributionally Robust Synthetic Control (DROSC)
=================================================

Overview
--------

Classical synthetic control estimates a treatment effect by matching the treated
unit's pre-treatment outcomes with a weighted average of control units, and then
assumes that the same weighting carries over to the post-treatment period. That
argument needs two separate things to be true, and Koo & Guo (2026) label them:

(E1) the pre-treatment weights are unique -- exactly one weighting on the
simplex minimises the pre-period fit;

(E2) there is no weight shift -- the weighting that describes the treated unit
before the intervention is the same one that describes it after.

Both fail routinely. (E1) fails when the control units are highly correlated:
many weight vectors then fit the pre-period almost indistinguishably, the solver
returns one of them, and the effect read off it inherits that arbitrariness. In
the paper's own Basque application most correlations between the selected and
the remaining controls sit near one. (E2) fails when the intervention alters the
treated-control relationship. It also fails for a subtler reason: with the
simplex constraint dropped, the weights are population least-squares projections,
and a projection depends on the distribution of the controls -- so a shift in
that distribution moves the weights even when the conditional relationship
:math:`\mathbb{E}[Y^{(0)}_{1t} \mid X_t]` is unchanged.

When (E1) fails the post-treatment weight is not pinned down; when (E2) fails it
is pinned down to the wrong thing. Either way the weight the counterfactual needs
is known only to lie somewhere in a set.

DROSC, the method of Koo & Guo (2026), replaces the single fitted weight vector
with a worst-case over an entire *set* of weights: every weight vector compatible
with the pre-treatment moments up to a radius :math:`\lambda`. At
:math:`\lambda = 0` the set is tightest and DROSC agrees with classical synthetic
control; as :math:`\lambda` grows the set widens and the estimate becomes a
deliberately conservative proxy for the effect that the data can no longer pin
down. Sweeping :math:`\lambda` is an honest sensitivity display: it shows how much
robustness the conclusion can absorb before the effect is no longer
distinguishable from zero.

When to use this estimator
--------------------------

Reach for DROSC when the donor pool is highly collinear -- many similar regions,
stores, or products, so the classical weights are non-unique and fragile -- or
when you suspect the treated-control relationship shifts across the intervention
and want an effect and interval that stay valid when the standard identification
does not. The robustness radius is the dial: report the effect as a function of
:math:`\lambda`, not a single number.

If your donors are well separated and the standard assumptions are credible, the
plain estimators (:doc:`vanillasc`) are more efficient -- DROSC's conservatism is
wasted. If the fragility is specifically pre-fit non-uniqueness and you only need
a stable tie-break, not a robust interval, :doc:`lexscm` resolves the
non-uniqueness lexicographically instead.

DROSC targets a different estimand, and the two agree only where classical
synthetic control is identified. Where they disagree, DROSC's number is the
compatible effect closest to zero: understated in magnitude by construction, and
correct in sign. Read it as "the effect is at least this large", not as an
estimate of :math:`\bar\tau`.

Notation
--------

The treated unit's outcome is :math:`Y_{1,t}`; the :math:`N` controls' outcomes
are stacked in :math:`X_t`. Time splits at :math:`T^\star = T_0 + 1` into a
pre-period :math:`t \le T_0` of length :math:`T_0` and a post-period of length
:math:`T_1`. Write :math:`\Delta^N` for the simplex, the set of weightings that
are non-negative and sum to one.

The model that makes the two failure modes visible carries a weight vector for
each regime:

.. math::

   Y_{1,t}^{(0)} =
   \begin{cases}
     X_t^\top \beta^{(0)} + u_t^{(0)}, & t = 1, \dots, T_0, \\
     X_t^\top \beta^{(1)} + u_t^{(1)}, & t = T_0 + 1, \dots, T,
   \end{cases}
   \qquad \beta^{(0)}, \beta^{(1)} \in \Delta^N .

:math:`Y^{(0)}` is the untreated potential outcome, so :math:`\beta^{(1)}` is the
weighting the counterfactual actually needs and :math:`\beta^{(0)}` is the one
the pre-period can speak to. (E1) says :math:`\beta^{(0)}` is unique; (E2) says
:math:`\beta^{(0)} = \beta^{(1)}`. Classical synthetic control estimates
:math:`\beta^{(0)}` and uses it as though it were :math:`\beta^{(1)}`.

Any candidate weighting implies a time-averaged effect,

.. math::

   \tau(\beta) = \mu_Y - \mu^\top \beta,
   \qquad
   \mu_Y = \frac{1}{T_1}\sum_{t > T_0} \mathbb{E}[Y_{1,t}],
   \qquad
   \mu = \frac{1}{T_1}\sum_{t > T_0} \mathbb{E}[X_t],

and the estimand classical synthetic control targets is
:math:`\bar\tau = \tau(\beta^{(1)})`. Identifying :math:`\beta^{(1)}` is
sufficient for identifying :math:`\bar\tau` but not necessary: if every plausible
weighting implies the same :math:`\tau(\beta)`, the effect is point identified
even though the weights are not.

The uncertainty class
~~~~~~~~~~~~~~~~~~~~~

The pre-period residual is uncorrelated with the controls, so :math:`\beta^{(0)}`
solves the moment condition
:math:`T_0^{-1}\sum_{t \le T_0}\mathbb{E}[X_t(Y_{1,t} - X_t^\top\beta)] = 0`.
A weight shift means :math:`\beta^{(1)}` need not satisfy it exactly, so DROSC
admits every simplex weighting that satisfies it to within a tolerance
:math:`\lambda \ge 0`:

.. math::

   \Omega(\lambda) = \Big\{ \beta \in \Delta^N :
     \Big\lVert \frac{1}{T_0}\sum_{t \le T_0}
       \mathbb{E}\big[X_t(Y_{1,t} - X_t^\top\beta)\big]
     \Big\rVert_\infty \le \lambda \Big\}.

Writing :math:`\Sigma = T_0^{-1}\sum_{t \le T_0}\mathbb{E}[X_t X_t^\top]` and
:math:`\gamma = T_0^{-1}\sum_{t \le T_0}\mathbb{E}[X_t Y_{1,t}]`, the moment
condition reads :math:`\gamma = \Sigma\beta^{(0)}` and the class has two
equivalent forms:

.. math::

   \Omega(\lambda)
   = \big\{ \beta \in \Delta^N : \lVert \gamma - \Sigma\beta \rVert_\infty
       \le \lambda \big\}
   = \big\{ \beta \in \Delta^N : \lVert \Sigma(\beta - \beta^{(0)})
       \rVert_\infty \le \lambda \big\} .

The second form reads as a covariance-scaled neighbourhood of the pre-treatment
weight, and the first is what makes it usable: the moment discrepancy
:math:`\gamma - \Sigma\beta` is well defined even when :math:`\Sigma` is singular
and :math:`\beta^{(0)}` is not identifiable. No inverse is taken and no unique
pre-period weight has to be selected, which is what lets the class survive the
perfectly-correlated case that breaks (E1).

Three details decide how the class behaves. The norm is the maximum over
coordinates, so a weighting must reproduce every control's moment condition to
within :math:`\lambda`, not do well on average. The class sits inside the
simplex, so widening :math:`\lambda` never buys extrapolation; it only admits
more points of the donor convex hull. And :math:`\lambda` is a tolerance for
weight shift, a statement about the world, with nothing to do with sample size --
the separate slack for estimation error appears further down, under Estimation.

The estimand
------------

Fix a candidate effect :math:`\tau` and a candidate weighting :math:`\beta`, and
measure how much the post-period fit improves when the effect is allowed:

.. math::

   R_\beta(\tau) = \frac{1}{T_1}\sum_{t > T_0} \mathbb{E}\Big[
     \big(Y_{1,t} - X_t^\top\beta\big)^2
     - \big(Y_{1,t} - X_t^\top\beta - \tau\big)^2 \Big]
   = 2\tau\,\tau(\beta) - \tau^2 .

The first term is the post-period prediction error under a null effect, the
second the error after allowing :math:`\tau`, so :math:`R_\beta` is the reward
for positing an effect of that size. With oracle knowledge of
:math:`\beta^{(1)}`, maximising :math:`R_{\beta^{(1)}}` over :math:`\tau` returns
:math:`\bar\tau`, which is classical synthetic control written as an
optimisation.

When :math:`\beta^{(1)}` is known only to lie in :math:`\Omega`, each candidate
effect has a different reward depending on which plausible weighting is used.
DROSC scores an effect by its worst case over the class and takes the best such
score. This is the weight-robust treatment effect:

.. math::

   \tau^*(\Omega) = \operatorname*{arg\,max}_{\tau \in \mathbb{R}}
     \Big[ \min_{\beta \in \Omega} R_\beta(\tau) \Big].

Read it as a game. You name an effect; nature picks, from the plausible
weightings, the one that makes your claim look worst; you name the effect whose
worst case is best. Nothing in it selects a weight vector.

What the game reduces to
~~~~~~~~~~~~~~~~~~~~~~~~

Theorem 1 of the paper solves it. The optimum is

.. math::

   \tau^*(\Omega) = \mu_Y - \mu^\top\beta^*(\Omega),
   \qquad
   \beta^*(\Omega) \in \operatorname*{arg\,min}_{\beta \in \Omega}
     \big[\mu_Y - \mu^\top\beta\big]^2 ,

an adversarial weighting that drives the time-averaged effect as close to zero as
the class allows. So the quadratic program the software solves is a consequence
of the max-min problem, not the definition of it.

Two things follow immediately. The program is degenerate, because
:math:`\mu\mu^\top` has rank one and the objective sees :math:`\beta` only
through the scalar :math:`\mu^\top\beta`: the set of minimisers can be an entire
face of the polytope while the value :math:`\mu^\top\beta^*` is the same across
all of them. So :math:`\tau^*` is unique even when :math:`\beta^*` is not, which
is the non-uniqueness DROSC exists to tolerate -- the method declines to report a
weighting and still reports an effect. And the characterisation never invokes
(E1) or (E2), so :math:`\tau^*` is defined whether or not classical synthetic
control is identified.

The sensitivity interval
~~~~~~~~~~~~~~~~~~~~~~~~

Each plausible weighting implies an effect, and together they form an interval

.. math::

   I(\Omega) = \{\tau(\beta) : \beta \in \Omega\}
   = \Big[\min_{\beta \in \Omega}\tau(\beta),\;
          \max_{\beta \in \Omega}\tau(\beta)\Big],

which collects every effect compatible with the stated weight uncertainty. If
:math:`\beta^{(1)} \in \Omega` then :math:`\bar\tau \in I(\Omega)`, so the
interval is an identified set for the classical estimand. When
:math:`\tau(\beta)` happens to be constant over :math:`\Omega` the interval is a
single point and :math:`\bar\tau` is point identified despite the weights not
being; when it varies, :math:`\bar\tau` is partially identified.

Theorem 2 gives the cleanest statement of what DROSC returns:

.. math::

   \tau^* = \begin{cases}
     \min_{\beta \in \Omega} \tau(\beta)
       & \text{if } \tau(\beta) > 0 \text{ for all } \beta \in \Omega, \\
     \max_{\beta \in \Omega} \tau(\beta)
       & \text{if } \tau(\beta) < 0 \text{ for all } \beta \in \Omega, \\
     0 & \text{if } \tau(\beta) = 0 \text{ for some } \beta \in \Omega .
   \end{cases}

:math:`\tau^*` is the projection of the origin onto the sensitivity interval --
the compatible effect closest to no effect. Sensitivity analysis and DROSC use
the same uncertainty class and report different objects: sensitivity analysis
hands back the whole interval and declines to choose within it, while DROSC
returns one point of it, the conservative endpoint.

How conservative
~~~~~~~~~~~~~~~~

Theorem 3 says that if :math:`\beta^{(1)} \in \Omega`, then

.. math::

   |\tau^*| \le |\bar\tau|,
   \qquad \text{and } \tau^* \text{ cannot have the opposite sign to }
   \bar\tau .

Both halves matter for reading the output. The magnitude is never overstated, so
:math:`|\tau^*|` is a lower bound on the size of the true effect: if it is large,
no admissible weighting explains the gap away. And the sign is trustworthy
whenever :math:`\tau^* \ne 0`, so a directional claim survives even though the
magnitude is deliberately understated.

How much the conservatism costs is bounded. When
:math:`\lambda_{\min}(\Sigma) > 0`,

.. math::

   \big|\tau^* - \bar\tau\big| \;\le\;
     2\,\lambda_{\min}(\Sigma)^{-1}\,\lVert\mu\rVert_1 \sqrt{N}\,\lambda ,

so the gap closes as :math:`\lambda \to 0`, and at :math:`\lambda = 0` under (E1)
and (E2) it closes exactly: :math:`\tau^* = \bar\tau`.

Reading the robustness sweep
----------------------------

:math:`\Omega(\lambda)` grows with :math:`\lambda`, so :math:`I(\Omega(\lambda))`
grows too, and the projection of the origin onto a growing interval moves toward
zero. Reporting :math:`\widehat\tau` at a single radius therefore says very
little; the object to report is the curve.

The radius at which :math:`\widehat\tau` first reaches zero is a breakdown point,
and Theorem 2 says exactly what happens there: it is the amount of weight shift
at which the sensitivity interval first covers zero, that is, the first
:math:`\lambda` admitting a weighting under which there is no effect at all. A
finding that survives to a large :math:`\lambda` tolerates a lot of
misspecification; one that collapses at a small :math:`\lambda` rests on (E1) and
(E2) holding closely.

Assumptions
-----------

1. Single treated unit, complete panel. Ingestion is the standard
   :func:`mlsynth.utils.datautils.dataprep` contract: one treated unit with a
   pre/post split and a fully observed outcome matrix.

   Remark. There is no covariate cube or spatial input; DROSC uses only the
   outcome panel. A ``dependent`` flag switches the moment covariances to a
   Newey-West HAC estimator when the outcomes are autocorrelated.

2. The uncertainty class contains the post-treatment weight. At the chosen
   radius, :math:`\beta^{(1)} \in \Omega(\lambda)`. This is what makes
   :math:`I(\Omega)` an identified set for :math:`\bar\tau` and what Theorem 3
   needs: without it, neither the magnitude bound nor the sign guarantee holds.

   Remark. Under (E1) and (E2) it holds at :math:`\lambda = 0`. When they fail,
   the assumption is a statement about how large a weight shift is credible, and
   it is the reason the radius is reported as a sweep instead of a setting.

3. Simplex weights. Donor weights are non-negative and sum to one, so the
   counterfactual is an interpolation of the donors, with no extrapolation.

   Remark. The moment band is layered on top of the simplex, so DROSC never
   leaves the convex hull; it only restricts which hull points are admissible.

4. A non-degenerate class where the theory asks for one. The conservatism bound
   and the :math:`\lambda = 0` theory assume
   :math:`\lambda_{\min}(\Sigma) > 0`.

   Remark. This has a consequence that is easy to read the wrong way round. When
   :math:`\Sigma` is invertible, :math:`\Omega(0)` is the single point
   :math:`\{\beta^{(0)}\}` -- so at :math:`\lambda = 0` the class collapses and
   DROSC returns the classical estimand. The non-uniqueness story at
   :math:`\lambda = 0` needs :math:`\Sigma` genuinely singular, that is,
   perfectly correlated controls. Highly-but-not-perfectly correlated controls
   are handled through :math:`\lambda`, and through the estimation slack below.

Estimation
----------

Everything above is a population statement. The estimator replaces
:math:`\Sigma, \gamma, \mu_Y, \mu` with their sample analogues, and widens the
band to absorb the error in doing so:

.. math::

   \widehat\Omega(\lambda) = \big\{ \beta \in \Delta^N :
     \lVert \widehat\gamma - \widehat\Sigma\beta \rVert_\infty
     \le \lambda + \rho \big\},
   \qquad
   \widehat\tau = \widehat\mu_Y - \widehat\mu^\top\widehat\beta,
   \quad
   \widehat\beta \in \operatorname*{arg\,min}_{\beta \in \widehat\Omega}
     \big[\widehat\mu_Y - \widehat\mu^\top\beta\big]^2 .

The two terms in the band do different jobs and should not be conflated.
:math:`\lambda` is the tolerance for weight shift, a modelling choice that does
not shrink with more data. :math:`\rho` is slack for the sampling error in
:math:`\widehat\Sigma` and :math:`\widehat\gamma`, and it vanishes as
:math:`T_0` grows -- it is proportional to
:math:`\log(\max\{T_0, N\})^{1/2}/\sqrt{T_0}`. Setting :math:`\lambda = 0` still
leaves :math:`\rho`, which is why the :math:`\lambda = 0` fit is not the same
object as a classical synthetic control fit on the same panel.

The constant in front of :math:`\rho` is not chosen by the analyst. It starts
small and is multiplied by 1.25 until the minimisation over
:math:`\widehat\Omega` becomes feasible, giving the smallest slack that admits a
solution. A band too tight to contain any simplex point has no answer to return,
so feasibility is the criterion. ``dependent=True`` switches the moment
covariances to a Newey-West HAC estimator for autocorrelated outcomes.

Inference
---------

Set ``inference=True`` for the perturbation-based confidence interval. Two
distinct problems rule out a normal interval, and the paper separates them.

Non-regularity. Decompose
:math:`\widehat\tau - \tau^* = (\widehat\mu_Y - \mu_Y)
- (\widehat\mu^\top\widehat\beta - \mu^\top\beta^*)`. The first term obeys a
central limit theorem. The second need not: :math:`\beta^*` typically sits on the
boundary of the class, and small sampling changes flip which constraints are
active, which produces a mixture limiting law of the kind familiar from
parameter-on-the-boundary problems.

Instability. Highly correlated controls make :math:`\Omega` nearly flat in some
directions, so small errors in :math:`\widehat\Sigma` or :math:`\widehat\gamma`
move :math:`\widehat\Omega` and :math:`\widehat\beta` a lot. This amplifies the
active-set flipping, so the two problems arrive together.

The remedy exploits the structure of the decomposition. Perturb
:math:`\widehat\Sigma, \widehat\gamma, \widehat\mu_Y, \widehat\mu` from their
estimated sampling distributions, re-solve the band problem for each of
``n_perturbations`` draws, and form a candidate effect from each. The paper shows
some draw :math:`m^\star` nearly recovers the population problem; for that draw
the remaining uncertainty is almost entirely in :math:`\widehat\mu_Y`, which is
asymptotically normal, so a normal interval around it is valid. Since
:math:`m^\star` cannot be identified, DROSC keeps the plausible draws and returns
the union of their intervals.

The union is the point. It is a confidence set, possibly a disjoint one, and it
reads as the effects the data cannot reject -- not as a point estimate plus
symmetric error. The enveloping hull is exposed as ``res.inference.ci_lower`` /
``ci_upper`` (and ``res.att_ci``); the pieces are in
``res.inference.details["ci_intervals"]``.

One practical caution. The procedure is heavy and mildly seed-sensitive: the
moment band starts far tighter than the perturbation scale, so an internal slack
is inflated until enough draws are feasible (dozens of rounds of
``n_perturbations`` solves), and the endpoints move with the seed. Fix ``seed``
for reproducibility, and leave inference off unless the interval is needed.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import DROSC

   basque = pd.read_csv("basedata/basque_jasa.csv")
   basque = basque[basque.regionname != "Spain (Espana)"].copy()
   treat_year = sorted(basque.year.unique())[15]          # T0 = 15
   basque["treat"] = ((basque.regionname == "Basque Country (Pais Vasco)")
                      & (basque.year >= treat_year)).astype(int)

   base = dict(df=basque, outcome="gdpcap", treat="treat",
               unitid="regionname", time="year")

   # a robustness sweep: how fast does the effect shrink toward zero?
   for lam in (0.0, 0.03, 0.06):
       res = DROSC({**base, "robustness_lambda": lam}).fit()
       print(lam, round(res.effects.att, 3))     # -0.742, -0.256, 0.000

   # perturbation union confidence interval at a fixed radius (slower)
   res = DROSC({**base, "robustness_lambda": 0.0, "inference": True,
                "seed": 1}).fit()
   print(res.att_ci)                             # enveloping hull, contains 0

Verification
------------

DROSC is cross-validated against the authors' own R implementation (``helpers.R``,
``limSolve::lsei``) -- sourced from their repository and run live via ``Rscript``
-- on the Basque study: the worst-case estimand
:math:`\widehat{\tau}(\lambda)` and the :math:`\lambda = 0` donor weights match
value-for-value (to :math:`\sim 10^{-7}`) across the robustness sweep. See the
durable case ``benchmarks/cases/drosc_basque.py`` and the replication page
:doc:`replications/drosc`.

Core API
--------

.. autoclass:: mlsynth.DROSC
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.config_models.DROSCConfig
   :members:
   :undoc-members:
   :show-inheritance:
