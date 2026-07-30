DTWSC: Dynamic Synthetic Control
================================

Synthetic control lines a treated unit up against a weighted blend of donors,
period by period. That comparison quietly assumes something strong: that every
unit reacts to a common shock at the same *speed*. Regions, countries and firms
rarely do. If a donor's boom arrives two years after the treated unit's, then
matching them on the calendar compares a peak against a trough. The
pre-treatment fit degrades, and whatever the fit cannot explain is absorbed
into the estimated effect.

DTWSC (Cao and Chadefaux, 2025) removes that mismatch before fitting anything.
Dynamic time warping learns, from the pre-treatment window alone, how fast each
donor moves relative to the treated unit; the donor series is then resampled
onto the treated unit's clock and an ordinary synthetic control is fit to the
resampled donors.

When to use it
--------------

Reach for DTWSC when the donors look right in shape but wrong in timing: the
same cycle, the same turning points, arriving early or late. That shows up as a
pre-treatment fit that is poor in a structured way, with the synthetic path
consistently leading or lagging the treated one rather than scattering around
it.

It is the wrong tool when donors differ in *level* or *amplitude* rather than
timing (:doc:`vanillasc` with predictors, or :doc:`mcnnm`, handle that), and it
buys little when the donor pool is large and diverse enough that a convex blend
of early and late donors already recovers the treated unit's timing. The method
earns its keep when the mis-timing is systematic across the whole pool.

Do not confuse it with :doc:`dsc` (Gunsilius's *distributional* synthetic
control) or :doc:`dscar` (dynamic synthetic control for auto-regressive
processes). All three carry a "dynamic"-flavoured name; they are different
methods.

Notation
--------

Let :math:`y_{1t}` be the treated unit and :math:`y_{jt}`,
:math:`j = 2, \dots, J+1`, the donors, observed for
:math:`t = 1, \dots, T` with treatment beginning at :math:`T_0 + 1`.

A warping function :math:`g_j` maps the treated unit's clock onto donor
:math:`j`'s. Writing :math:`\omega_{js}` for the *speed* of donor :math:`j` at
its own period :math:`s` -- how much of the treated unit's clock that period
accounts for -- the warped donor is

.. math::

   y^{w}_{jt} \;=\; \big(\mathcal{W}_{\omega_j} \, y_j\big)_t ,

where :math:`\mathcal{W}_{\omega_j}` resamples :math:`y_j` so that period
:math:`s` occupies :math:`\omega_{js}` units of the new clock. Speeds above one
stretch a period out (the donor was running slow and must be sped up); below
one compress it. The synthetic control is then the usual simplex problem on the
warped donors,

.. math::

   \hat w \;=\; \arg\min_{w \in \Delta^{J}}
     \sum_{t \le T_0} \Big( y_{1t} - \sum_{j} w_j y^{w}_{jt} \Big)^2 ,
   \qquad
   \hat\tau_t \;=\; y_{1t} - \sum_j \hat w_j y^{w}_{jt} .

The speeds themselves come from a two-phase alignment. Phase one warps the
donor's pre-treatment window against the treated unit's and reads
:math:`\omega_j` off the resulting path. Phase two carries those speeds past
the intervention: each length-:math:`k` window of the donor's post-period is
matched against every window of its own pre-period, and inherits the speeds of
whichever pre-period stretch it most resembles.

Assumptions
-----------

1. Common shocks, heterogeneous speeds.
   Units respond to the same underlying process but at unit-specific rates, so
   the donor's untreated path is a time-reparameterised version of the treated
   unit's counterfactual rather than a scaled one.

   Remark. This is what separates DTWSC from a factor model. A factor model
   lets loadings differ; DTWSC lets the *clock* differ. If the real
   heterogeneity is in loadings, warping will not help and may hurt, because it
   spends degrees of freedom re-timing series that were already in time.

2. Speeds are learnable from the pre-period.
   The pre-treatment window is long enough, and varied enough, that the
   alignment identifies a stable speed profile rather than fitting noise.

   Remark. The method needs shape to align on. A donor whose pre-period is a
   straight line has no turning points, every alignment is equally good, and
   the learned speeds are arbitrary. Inspect ``pre_period_speeds`` before
   trusting a fit; speeds far from one on a featureless series are a warning,
   not a finding.

3. The intervention does not change the speeds learned from the pre-period.
   Phase two matches post-treatment windows to pre-treatment shapes, so the
   post-treatment outcome never sets its own speed.

   Remark. This is what preserves the effect. Speed differences the donor
   brought with it are removed; a change in speed caused by the intervention
   has nowhere to hide and shows up in the gap. The cost is that a donor whose
   post-period genuinely has no pre-period analogue gets whatever speeds its
   closest-looking window happened to have.

4. Standard synthetic-control conditions hold on the warped panel.
   No anticipation, no interference, and the treated unit lies in the convex
   hull of the warped donors.

   Remark. Warping does not relax any of these. It changes what the donors look
   like, not what the design requires of them.

Inference and diagnostics
-------------------------

Set ``inference="placebo"`` to run Cao and Chadefaux's procedure. It is off by
default because it costs one full warp per donor per perturbed pool.

The 95 percent band is a placebo distribution, not a sampling distribution.
Every donor is re-analysed as if it had been treated, with the real treated
unit removed from every placebo pool so no placebo counterfactual can borrow
from it; ``placebo_pairs`` additionally perturbs the pool by dropping two
donors at a time, in the paper's lexicographic order. ``res.placebo_band``
holds the pointwise ``alpha/2`` and ``1 - alpha/2`` quantiles of the resulting
gap paths for the warped fit, and ``lower_unwarped`` / ``upper_unwarped`` for
the unwarped one, because the paper's claim is comparative: warping should
narrow the band, so the same effect clears it more decisively.

``res.efficiency_test`` is the paper's second procedure, a t-test on
:math:`\log(\mathrm{MSE}_{\mathrm{DSC}} / \mathrm{MSE}_{\mathrm{SC}})`
across the placebo runs. The runs share units and donor pools, so they are not
independent; the degrees of freedom are deflated by a variance-inflation factor
:math:`1 + (k-1)\,\mathrm{ICC}` built from the intra-class correlation of a
two-way ANOVA over dataset and unit. A negative ``t`` means warping achieved the
lower post-treatment MSE. Note what this tests: whether warping helps, not
whether the treatment had an effect. ``mse_window`` sets how many post-treatment
periods enter the MSE, ten by default, as in the paper.

``res.inference`` mirrors the standardized slot, with ``ci_lower`` / ``ci_upper``
the post-period means of the band and ``p_value`` the efficiency test's. Both
are scalar summaries of pointwise objects; read ``placebo_band`` for what the
paper actually plots.

Two honest caveats. First, this is not the design the paper ran. The authors
give each placebo its own grid-optimal warping hyperparameters, drawn from a
sweep over seven DTW step patterns, and perturb the donor pool far more
aggressively than the default here; mlsynth runs every placebo at the
hyperparameters you configure. Their per-run choices are shipped in the
replication archive, so what separates the two is a matter of adopting their
design rather than of missing information -- :doc:`replications/dtwsc` sets out
exactly what differs. Second, on the
Basque panel the efficiency test reproduces the paper's direction and rough
magnitude (``t = -6.8`` against the paper's ``-7.91``, a 20 percent MSE
reduction), but the band does not narrow -- it comes out 0.3 percent wider than
the unwarped one.

Those are not in conflict. The efficiency test is a within-run paired
comparison, while the band is the cross-run spread, and reducing each run's
error need not shrink the dispersion across runs. Two explanations for the band
have been tested and ruled out: fitting the paper's 14-predictor specification
with ``sc_backend="malo"`` widens it further rather than tightening it, and the
warp itself is now bit-exact against the authors' R package. The per-run
hyperparameter optimisation remains untested, as does the paper's much larger
pool construction -- its quantiles are drawn over roughly 2000 runs against 256
here, and a 2.5 percent tail is not well determined at that count. Treat the
band as descriptive and the efficiency test as the reproducible claim;
:doc:`replications/dtwsc` has the measurements.

Choosing the step pattern
-------------------------

``step_pattern1`` and ``step_pattern2`` choose the recursion the alignment
follows, for the first-phase and second-phase passes respectively. Six are
available -- ``symmetricP1`` (the first-phase default), ``symmetricP2``,
``asymmetricP1``, ``asymmetricP2`` (the second-phase default), ``typeIc`` and
``mori2006`` -- each matching R ``dtw``'s definition of the same name.

They differ in how far a donor's clock may drift from the treated unit's before
the alignment refuses: a pattern permitting a steeper step tolerates more
speed difference, at the cost of admitting alignments that stretch the series
harder to achieve a fit. They also differ in what the reported distance is
divided by, which matters when comparing costs across candidate windows rather
than within one. Cao and Chadefaux do not fix a pattern; they select one per
run by grid search, which is a reasonable signal that the choice is empirical
rather than settled.

Putting the units on a common footing
-------------------------------------

``rescale_units=True`` maps every unit onto a common pre-treatment range before
anything is fitted, sending each unit's outcome through
:math:`(x - \min_i)\,\overline{r}/(\max_i - \min_i)`, where the minimum and
maximum are taken over the pre-treatment window and :math:`\overline{r}` is the
mean of those ranges across units.

The reason to want it is that a synthetic control minimises squared error. A
donor whose outcome swings widely contributes more of that error than a flat
one, so it gets fitted harder for reasons that have nothing to do with how much
its path resembles the treated unit's. Equalising the ranges first makes the fit
a comparison of shape.

Two consequences are worth knowing. The warp does not move: the alignment runs
on a normalised series, which an affine map leaves untouched, so only the
control changes. And the effect comes back in the rescaled units, not the
outcome's own -- divide by the treated unit's multiplier to read it as GDP per
capita, or whatever the outcome is.

This is ``dsc(rescale = TRUE)``, the reference implementation's default, and it
is required to compare mlsynth's synthetic-control half against the authors'
published numbers, which are reported on that scale. It defaults to False here
so the estimator does not quietly change the units its effects are reported in.

Beyond inference, the estimator exposes its warping diagnostics directly.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import DTWSC

   df = pd.read_csv("basedata/basque_data.csv")
   df = df[df.regionname != "Spain (Espana)"].dropna(subset=["gdpcap"])
   df["treated"] = ((df.regionname == "Basque Country (Pais Vasco)")
                    & (df.year >= 1975)).astype(int)

   res = DTWSC({
       "df": df, "outcome": "gdpcap", "treat": "treated",
       "unitid": "regionname", "time": "year", "display_graphs": False,
   }).fit()

   print(res.att, res.pre_rmse)

   # what the warp bought, on this panel
   plain = DTWSC({**cfg, "warp": False}).fit()
   print(plain.pre_rmse, "->", res.pre_rmse)

   # the paper's placebo band and efficiency test (slow: one warp per donor)
   inf = DTWSC({**cfg, "inference": "placebo"}).fit()
   print(inf.efficiency_test["t"], inf.efficiency_test["p"])
   lo, hi = inf.placebo_band["lower"], inf.placebo_band["upper"]

   # which donors were running slow
   {k: v for k, v in res.cutoffs.items() if v > res.inputs.n_pre}

Verification
------------

DTWSC is cross-validated against the authors' R package
(`conflictlab/dsc <https://github.com/conflictlab/dsc>`_) on the Basque panel.
See :doc:`replications/dtwsc` for what matches and to what tolerance, and
``benchmarks/reference/dtwsc_basque/`` for the reference generator.

Core API
--------

.. autoclass:: mlsynth.DTWSC
   :members:

.. autoclass:: mlsynth.config_models.DTWSCConfig
   :members:

.. autoclass:: mlsynth.utils.dtwsc_helpers.structures.DTWSCResults
   :members:
