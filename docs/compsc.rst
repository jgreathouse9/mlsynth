COMPSC — Compositional Synthetic Controls in the Aitchison Geometry
===================================================================

When to use it
--------------

Reach for COMPSC when the outcome is a composition — a vector of shares that
sum to a whole — a single unit is treated, and what you care about is the
*relative* structure of the mix, not any one share on its own. An
electricity generation mix split across gas, coal and renewables; a marketing
budget split across channels; brand share within a product category; a modal
split across car, transit and bicycle; vote share across parties. The defining
feature is a sum constraint: if one share rises, others must fall.

Two things go wrong if you ignore the geometry of shares.

The first is multiplicity. Running a separate synthetic control per share gives
each share its own donor mix, so the fitted shares need not sum to one and no
single counterfactual unit exists. A common weight vector fixes this, and that
is what :doc:`propsc` does.

The second is scale dependence, and it is the reason COMPSC exists. A Euclidean
objective on raw shares penalises absolute deviations, not proportional ones.
Doubling a category that holds 2 percent of the total moves the objective by
roughly :math:`(0.02)^2`, while the same proportional error in a category at 50
percent moves it by :math:`(0.5)^2` — some six hundred times more. The fit is
therefore driven by the dominant categories, and a small emerging category is
almost invisible to the optimiser even when its relative evolution is the whole
question. Consider three regions splitting output across services, industry and
agriculture: A at (82.5, 15.0, 2.5) percent, B at (72.5, 25.0, 2.5), and C at
(92.5, 5.0, 2.5). A is the same Euclidean distance from B and from C. But the
services-to-industry ratio is 5.5 in A, 2.9 in B and 18.5 in C — structurally, B
is much the closer match, and only a log-ratio geometry sees that.

COMPSC works in log-ratios, where those two problems dissolve together. If your
composition has only large, comparably sized categories and you have a block of
treated units adopting simultaneously, prefer :doc:`propsc`, whose
synthetic-difference-in-differences machinery and jackknife inference are better
suited to that design. COMPSC is for one treated unit and a relative-scale
question.

Notation
--------

We observe :math:`J` units over :math:`T` periods. Unit :math:`j = 1` is treated
from a known date :math:`T_0`, with :math:`1 < T_0 < T`; units
:math:`j = 2, \dots, J` form the donor pool and are never treated. For each unit
and period we observe shares across :math:`p` mutually exclusive and exhaustive
categories,

.. math::

   \pi_{j,t} = (\pi_{1,j,t}, \dots, \pi_{p,j,t}) \in \mathcal{S}^{p-1},
   \qquad
   \mathcal{S}^{p-1} = \Big\{ \pi \in \mathbb{R}^{p}_{+} :
   \textstyle\sum_{k} \pi_{k} = 1 \Big\},

where :math:`\mathcal{S}^{p-1}` is the probability simplex — the set of all
valid compositions. Write :math:`\pi^{0}_{j,t}` and :math:`\pi^{1}_{j,t}` for
the share vectors absent and under treatment. The object of interest is the
treated unit's counterfactual composition :math:`\pi^{0}_{1,t}` for
:math:`t > T_0`.

The simplex is not a Euclidean space, and the transform that linearises it is
the log-ratio. Two versions matter here. The additive log-ratio takes ratios
against a nominated baseline category :math:`p`,

.. math::

   \operatorname{alr}(\pi)_{k} = \log \frac{\pi_{k}}{\pi_{p}},
   \qquad k = 1, \dots, p - 1,

and the centered log-ratio takes ratios against the geometric mean of all
categories,

.. math::

   \operatorname{clr}(\pi)_{k} = \log \pi_{k}
   - \frac{1}{p} \sum_{m=1}^{p} \log \pi_{m},
   \qquad k = 1, \dots, p.

Distance on the simplex is the Aitchison distance, which compares every pair of
categories by their log-ratio,

.. math::
   :label: aitchison

   \delta_{A}(\pi, \pi') = \Bigg[ \frac{1}{2p} \sum_{i=1}^{p} \sum_{j=1}^{p}
   \bigg( \log \frac{\pi_{i}}{\pi_{j}}
        - \log \frac{\pi'_{i}}{\pi'_{j}} \bigg)^{2} \Bigg]^{1/2}
   = \big\| \operatorname{clr}(\pi) - \operatorname{clr}(\pi') \big\|_{2}.

The second equality is the useful one: the centered log-ratio is an isometry, so
ordinary Euclidean distance between clr vectors *is* Aitchison distance on the
simplex. The additive log-ratio is not an isometry — a point we return to under
Assumption 5.

Where the log-ratios come from
------------------------------

The transform is not a convenience. Suppose the shares aggregate individual
discrete choices: each person in unit :math:`j` at time :math:`t` picks one of
the :math:`p` alternatives, with random utility
:math:`U_{ik,j,t} = V_{k,j,t} + \varepsilon_{ik,j,t}` and type-I extreme value
shocks. Then choice probabilities follow the multinomial logit, and the log-odds
of category :math:`k` against the baseline equal the relative systematic utility
exactly:

.. math::

   \log \frac{\pi^{0}_{k,j,t}}{\pi^{0}_{p,j,t}}
   = V^{0}_{k,j,t} - V^{0}_{p,j,t} =: \tilde{V}^{0}_{k,j,t}.

The observable log-ratios *are* the preference parameters governing choice. That
is why a factor model is imposed on them, not on the shares, and why the
donor weights read as preference similarity: a donor earns weight to the extent
that its population's relative preferences track the treated unit's.

Assumptions
-----------

1. Positivity. Every share is strictly positive:
   :math:`\pi^{0}_{k,j,t} > 0` for all :math:`k, j, t`.

   Remark. Log-ratios are undefined at zero. In practice a structural zero is
   repaired by adding a small constant and re-closing, the standard device in
   compositional data analysis; ``zero_replacement`` controls it and can be set
   to 0 to reject zeros instead. A category that is zero for a unit throughout
   the sample is better handled by dropping that unit from the donor pool, since
   the repaired value is arbitrary and its log-ratio is extreme.

2. Factor structure on relative utilities. For each category :math:`k`,

   .. math::

      \tilde{V}^{0}_{k,j,t} = \delta_{k,t} + \theta'_{k,t} Z_{j}
      + \lambda'_{k,t} \mu_{j} + \varepsilon_{k,j,t},

   where :math:`\mu_{j}` are unobserved unit loadings, common across all
   :math:`p - 1` relative-utility equations within a unit, and
   :math:`\varepsilon_{k,j,t}` is mean-zero given :math:`(Z_j, \mu_j)`.

   Remark. The loadings being shared across categories is what licenses fitting
   one weight vector to all of them, and what makes the stacking below an
   efficiency gain, not an assumption of convenience. The factor
   model sits on the log-ratios, not on the shares: because the transform is
   nonlinear, a factor model on one does not imply a factor model on the other,
   and the utilities are the economically primitive object.

3. Convex hull. There exist weights :math:`w^{*}_{j} \ge 0` summing to one with
   :math:`Z_{1} = \sum_{j \ge 2} w^{*}_{j} Z_{j}` and
   :math:`\mu_{1} = \sum_{j \ge 2} w^{*}_{j} \mu_{j}`.

   Remark. This is the familiar synthetic-control requirement, transposed to
   the space of preference parameters: the treated unit's persistent preference
   structure must be expressible as a mixture of the donors'. It fails in the
   usual way — when the treated unit is extreme, here meaning its category
   ratios lie outside the range the donors span. Check it by reading
   ``pre_treatment_aitchison_rmspe``, not by eyeballing the share levels.

4. Uniqueness of the weights. The donor log-ratio matrix has full column rank
   over the pre-period, so :math:`w^{*}` is the unique minimiser.

   Remark. Consistency needs only :math:`(p-1) T_0 \to \infty`, so it can come
   from a long pre-period or from many categories. But a perfect pre-treatment
   fit is not evidence the weights are identified: if the donors lie in a
   low-dimensional factor space with no idiosyncratic variation, many weight
   vectors fit equally well and the reported one is arbitrary. Treat a
   near-zero pre-treatment distance with a short pre-period as a warning.

5. Choice of geometry. The fitting objective is the pre-treatment distance in
   log-ratio coordinates, and which coordinates you choose is a modelling
   decision exposed as ``geometry``.

   Remark. The source paper's Algorithm 1 uses the additive log-ratio and
   describes the result as minimising Aitchison distance. Those are different
   objectives: alr is not an isometry for :eq:`aitchison`, so the alr fit
   depends on which category is nominated as the baseline — an arbitrary
   labelling choice. ``geometry="clr"`` is the default here and does minimise
   pre-treatment Aitchison distance, invariantly to relabelling.
   ``geometry="alr"`` with an explicit ``baseline`` reproduces the paper. The
   result carries ``baseline_sensitivity``, the largest change in the fitted
   weights across the possible baselines; when it is large the two routes give
   materially different synthetic units and the clr fit is the defensible one.

Identification and estimation
-----------------------------

Under Assumptions 1–3 the treated unit's counterfactual log-ratios are a convex
combination of the donors', and the counterfactual composition follows by
inverting the transform. The weights solve a convex quadratic program on the
simplex,

.. math::

   \hat{w} = \operatorname*{arg\,min}_{w \ge 0,\ \sum_{j} w_{j} = 1}
   \sum_{t=1}^{T_0} \sum_{k}
   \Big( \ell_{k}(\pi_{1,t}) - \sum_{j \ge 2} w_{j}\, \ell_{k}(\pi_{j,t}) \Big)^{2},

where :math:`\ell` is the configured log-ratio map. Because the coordinate
equations share the unit loadings :math:`\mu_j`, they stack into a single
regression with :math:`(p-1) T_0` scalar observations, not :math:`T_0`
vector-valued ones — the multi-outcome stacking of Sun, Ben-Michael and Feller
(2025), which :doc:`scmo` implements for general multiple outcomes.

Given the weights, the counterfactual is the Fréchet barycenter of the donor
compositions under the Aitchison metric, which works out to the closure of their
weighted geometric mean:

.. math::

   \hat{\pi}^{0}_{1,t}
   = \mathcal{C}\Big( \textstyle\prod_{j \ge 2} \pi_{k,j,t}^{\hat{w}_{j}} \Big)_{k=1}^{p},

with :math:`\mathcal{C}` the closure (renormalisation). Standard synthetic
control takes the arithmetic mean of the donors; COMPSC takes the geometric
mean. The result is a valid composition automatically — strictly positive,
summing to one — so the share effects sum to zero by construction, not by
correction.

Estimands
---------

Three read-outs, all on the result object.

``share_effects`` is the per-period effect on each share in share units,
:math:`\pi^{1}_{k,1,t} - \hat{\pi}^{0}_{k,1,t}`, summing to zero across
categories. This is the compositional analogue of the usual ATT.

``log_ratio_effects`` is the treatment-induced change in the odds of one
category against another,

.. math::

   \mathrm{LRTE}_{k\ell,t} = \log \frac{\pi^{1}_{k,1,t}}{\pi^{1}_{\ell,1,t}}
   - \log \frac{\hat{\pi}^{0}_{k,1,t}}{\hat{\pi}^{0}_{\ell,1,t}},

reported for every ordered pair. A value of 1 means the odds of :math:`k`
against :math:`\ell` are :math:`e \approx 2.7` times their counterfactual. This
is the natural estimand under the simplex geometry, and it is the one that makes
a small category legible: a category can gain ground relative to another while
losing share in absolute terms, and vice versa.

``aitchison_distance`` is the scalar summary :math:`\delta_A` of total
compositional displacement per period — how far the whole observed mix has moved
from its counterfactual.

Inference and diagnostics
-------------------------

Inference is the Abadie permutation test, with the Aitchison distance in place
of the usual root mean squared prediction error so the test statistic matches
the estimation objective. Each donor is treated in turn as if it had been
intervened on at :math:`T_0`, refitting against the others; the statistic is the
post-to-pre Aitchison RMSPE ratio. Donors whose pre-treatment fit is worse than
``placebo_screen`` times the treated unit's are dropped before the comparison
(the paper uses 5), and the p-value is the share of the retained set whose ratio
is at least the treated unit's. Set ``inference="none"`` to skip it.

The power of any such permutation test is bounded by the size of the retained
set: with 36 units retained the smallest attainable p-value is
:math:`1/36 \approx 0.028`, so a donor pool of 20 cannot produce evidence at the
1 percent level however large the effect. ``placebo.n_retained`` reports the
number actually used.

Two further diagnostics. ``pre_treatment_aitchison_rmspe`` is the quantity the
clr objective minimises, and is what to read when judging Assumption 3;
``pre_treatment_share_rmspe`` is the same fit measured on the level scale, for
comparison against a conventional synthetic control. And
``baseline_sensitivity`` is the geometry check described under Assumption 5.

Example
-------

A three-category compositional panel from a latent factor model, with one
treated unit and a shift toward the first category after period 8.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import COMPSC

   rng = np.random.default_rng(0)
   J, T, K, T0 = 12, 16, 3, 8
   factors = rng.standard_normal((T, 2, K - 1))
   loadings = rng.standard_normal((J, 2))
   V = np.einsum("tfk,jf->jtk", factors, loadings)
   V += 0.3 * rng.standard_normal(V.shape)
   V[0] = 0.5 * V[1] + 0.3 * V[2] + 0.2 * V[3]        # treated in the hull
   V[0] += 0.15 * rng.standard_normal(V[0].shape)     # its own transitory shocks
   V[0, T0:, 0] += 1.2                                 # the treatment

   logits = np.concatenate([V, np.zeros((J, T, 1))], axis=-1)
   P = np.exp(logits)
   P /= P.sum(axis=-1, keepdims=True)

   rows = []
   for j in range(J):
       for t in range(T):
           rows.append({"unit": f"u{j:02d}", "time": t,
                        "treat": int(j == 0 and t >= T0),
                        "gas": P[j, t, 0], "coal": P[j, t, 1],
                        "renew": P[j, t, 2]})
   df = pd.DataFrame(rows)

   res = COMPSC({"df": df, "outcomes": ["gas", "coal", "renew"],
                 "treat": "treat", "unitid": "unit", "time": "time",
                 "target": "gas", "display_graphs": False}).fit()

   print(res.att_vector)                  # per-category effects, summing to zero
   print(res.sum_constraint)              # ~0 to machine precision
   print(res.log_ratio_effects[-1, 0, 1]) # final-period gas-vs-coal log-odds effect
   print(res.placebo.p_value)

Verification
------------

COMPSC reproduces Boussim (2026) exactly on the paper's Pennsylvania
application. Under ``geometry="alr"`` with renewables as the baseline it returns
all nine donor weights of Table 1 to three decimals (Illinois 0.233, Wyoming
0.208, Iowa 0.168, Nebraska 0.149, Massachusetts 0.102, Connecticut 0.048, New
Jersey 0.043, Indiana 0.032, Montana 0.016), an Aitchison pre-treatment RMSPE of
0.187 and share RMSPE of 1.34 percentage points, every cell of Table 2 across
all eleven reported years, and the placebo result of section 6.4 — a treated
ratio of 9.18 against 36 retained units for a p-value of 0.111, with Michigan,
Louisiana and Florida the three exceeding placebos.

The correction of Assumption 5 is not cosmetic on that panel: switching to the
default clr geometry moves the weights by 0.85 in :math:`L_1`, changes the donor
set, improves the Aitchison pre-treatment fit from 0.187 to 0.177, and reduces
the 2022 natural-gas effect from 60.4 to 51.1 percentage points.

Since then the author's replication script has become available, so the fit is
also checked against the code that produced those tables and not only against
their printed form. On the same panel, mlsynth and the author's
``quadprog::solve.QP`` agree value for value: all ten donor weights to 3e-9,
the Aitchison and share pre-RMSPEs to 1e-9, every post-period effect across
twenty years and five columns to 1e-7, and the entire 42-donor placebo -- ratio,
retained count and p-value -- to 1e-8 or better.

That comparison settled one open item. Table 1 lists nine weights, mlsynth gives
Arkansas 0.000924, and the difference had been recorded as a solver artefact on
the assumption that ``quadprog`` drove that weight to zero. It does not: it
returns 0.000924 as well and reports ten active donors. The gap is between the
paper's table and the author's code.

The benchmark cases that pin all of this are
`benchmarks/cases/compsc_pennsylvania.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/compsc_pennsylvania.py>`_
against the published tables and
`benchmarks/cases/compsc_pennsylvania_r.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/compsc_pennsylvania_r.py>`_
against the replication script, both running on
``basedata/pa_aeps_generation.csv``. See
:doc:`replications/compsc` for the full validation, including the category
recipe recovered from the paper's balance table and the geometry comparison.

Core API
--------

.. autoclass:: mlsynth.COMPSC
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.config_models.COMPSCConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.utils.compsc_helpers.structures.COMPSCResults
   :members:
   :undoc-members:
   :show-inheritance:

References
----------

Boussim, O. (2026). "Compositional Synthetic Controls." arXiv:2607.16991.

Aitchison, J. (1982). "The Statistical Analysis of Compositional Data." Journal
of the Royal Statistical Society: Series B 44(2), 139–160.

Egozcue, J. J., V. Pawlowsky-Glahn, G. Mateu-Figueras, and C. Barceló-Vidal
(2003). "Isometric Logratio Transformations for Compositional Data Analysis."
Mathematical Geology 35(3), 279–300.

Sun, L., E. Ben-Michael, and A. Feller (2025). "Using Multiple Outcomes to
Improve the Synthetic Control Method." Review of Economics and Statistics.

Abadie, A., A. Diamond, and J. Hainmueller (2010). "Synthetic Control Methods
for Comparative Case Studies: Estimating the Effect of California's Tobacco
Control Program." Journal of the American Statistical Association 105(490),
493–505.
