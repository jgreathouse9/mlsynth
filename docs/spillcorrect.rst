Correcting Known Spillovers in a Design (SPILLCORRECT)
======================================================

When to Use This Estimator
--------------------------

A geo experiment compares the markets that received a campaign with a
synthetic control built from markets that did not. The comparison is valid only
if the campaign leaves the control markets alone. In practice it often does
not: shoppers cross a border to buy in a treated market, two markets share a
television signal, or the campaign's ad spend moves the auction prices every
market pays. A control market that feels the campaign is contaminated, and the
estimate inherits the contamination in proportion to that market's weight in
the synthetic control.

``SPILLCORRECT`` is a post-estimation step for that situation. It takes a
design that has already been fitted (a :doc:`marex` result), the realized
panel, and the control markets known to be contaminated. It rebuilds each
contaminated, weighted control from the clean controls, recomputes the
design's effect and inference on the corrected panel, and reports the
corrected estimate beside the uncorrected one. The design itself (which
markets are treated, and every weight) is kept.

Reach for it when a few named controls are contaminated and the design is
already in the field. Two situations call for something else:

* When a spillover is expected to reach every neighbour of the treated markets,
  exclude the neighbours when the design is chosen (MAREX's ``adjacency`` with
  ``exclude_bordering_donors``). In simulated geo panels, correcting eleven or
  twelve bordering controls cost more in accuracy than the spillover it
  removed, and the design that excluded them did better.
* When nobody knows which controls are contaminated, there is nothing to
  correct. SPILLCORRECT does not detect spillovers.

Notation
--------

The design covers :math:`N` units over :math:`T = T_0 + T_1` periods:
:math:`T_0` before treatment and :math:`T_1` after. Its fitting window is the
first :math:`T_f` periods, and the :math:`T_0 - T_f` periods between the end
of the fitting window and treatment are its blank window, held out of the fit
as a placebo period. Write :math:`y_{jt}` for unit :math:`j`'s outcome in
period :math:`t`.

The design has a treated composite with weights :math:`w_j` and a control
composite with weights :math:`v_i`, both on the simplex, and its estimate in
period :math:`t` is

.. math::

   \hat\tau_t = \sum_j w_j y_{jt} - \sum_i v_i y_{it} .

Let :math:`\mathcal{K}` be the contaminated controls with :math:`v_k > 0`, and
:math:`\gamma_{kt}` the spillover on control :math:`k` in period :math:`t`.
The clean controls :math:`\mathcal{C}` are the controls outside
:math:`\mathcal{K}` and outside the affected controls without weight.

How it works
------------

A spillover on control :math:`k` moves the estimate by exactly
:math:`-v_k \gamma_{kt}`. Each correction estimates the
:math:`\gamma_{kt}` and puts it back.

Iterative (default)
~~~~~~~~~~~~~~~~~~~

Following Melnychuk (2024), each contaminated control gets its own synthetic
control from the clean controls, fitted on the periods before the correction
starts, :math:`t_c`. Its spillover-free series replaces the control's outcomes
from :math:`t_c` on, and the spillover estimate is the difference,
:math:`\hat\gamma_{kt} = y_{kt} - \hat y_{kt}`.

Inclusive
~~~~~~~~~

Following Di Stefano and Mellace (2024), each contaminated control's synthetic
control may also draw on the treated composite (with weight :math:`l_k`) and
on the other contaminated controls (weights :math:`l_{ki}`). From :math:`t_c`
on, the gaps satisfy

.. math::

   g_{1t} = \theta_t - \sum_k v_k \gamma_{kt}, \qquad
   g_{kt} = \gamma_{kt} - l_k \theta_t - \sum_{i \neq k} l_{ki} \gamma_{it},

a linear system :math:`\Omega\,(\theta_t, \gamma_{\cdot t})' = g_t` solved for
the effect :math:`\theta_t` and every spillover at once. It has a solution
only when :math:`\det\Omega \neq 0`, and SPILLCORRECT raises
``MlsynthEstimationError`` when it does not. The two methods agreed in every
cell of the simulations behind #932, and only the inclusive one can fail, which
is why the iterative one is the default.

Restricted
~~~~~~~~~~

Given the configuration the design was fit with (``design_config``),
SPILLCORRECT also re-solves the design with the contaminated controls dropped
from its control pool, holding the treated markets fixed. This is the
comparison Di Stefano and Mellace recommend reporting beside a correction, and
``method="restricted"`` makes it the headline estimate.

Two choices in the rebuilding fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``scale`` (on by default)
   A simplex average of the clean controls can never exceed the largest of
   them. In a geo panel the control to be rebuilt is often larger than every
   clean one, and the simplex fit then misses it by a margin no weights can
   close. The scale-free fit divides each series by its mean over the fitting
   window, fits there, and takes the result back to the control's level:
   :math:`\hat y_{kt} = \sum_j (w_j m_k / m_j)\, y_{jt}` with :math:`m_j` the
   mean. Turn it off only for an outcome that does not grow with a unit's
   size.

``window`` (``"blank_and_post"`` by default)
   The published methods fit on the whole pre-period and correct the post
   period, :math:`t_c = T_0`. The default here fits on the design's fitting
   window and corrects from its end, :math:`t_c = T_f`, so the blank periods
   are corrected too. The design's band and test are calibrated on the blank
   periods, so they then carry the correction's own prediction error, as the
   post periods do. In simulated geo panels with one to three contaminated
   controls, the scale-free correction under this window covered the true
   effect about 97% of the time with about 3% false positives; the published
   corrections, fitted on the pre-period and applied to the post period,
   covered it 77-79% of the time and flagged an effect in 28% of the cases
   with none.

Assumptions
-----------

1. The contaminated controls are known.

   Remark: only the declared units are corrected. A contaminated control left
   off the list biases the estimate as before, and a clean control put on it
   costs a rebuilding fit's error and nothing more.

2. No spillover reaches the clean controls.

   Remark: the clean controls are the material every contaminated control is
   rebuilt from, so a spillover on them passes into every rebuilt series.

3. Each contaminated control is well rebuilt from the clean controls.

   Remark: the correction is as good as the rebuilding fits, and
   ``cleaning_rmspe`` reports each one's fit over the fitting window. A
   contaminated control unlike every clean one cannot be rebuilt, whatever its
   spillover.

4. No spillover reaches the contaminated controls before the correction
   starts.

   Remark: under the default window that is the end of the fitting window, so a
   spillover that begins during the blank window, through anticipation or an
   announcement, is still corrected.

5. For the inclusive method, :math:`\det\Omega \neq 0`.

   Remark: the system is singular when a contaminated control and the treated
   composite carry each other entirely, for instance a control that holds the
   whole control composite and whose own synthetic control is the treated
   composite.

Inference
---------

The design's inference is recomputed on the corrected panel with the design's
own settings (``inference_settings``): the same level, global test, number of
drawn subsets and seed, and the same cumulative band. With the panel the design
was fit on and nothing corrected, it reproduces the design's own p-value and
band exactly. ``global_test`` swaps in another test. A design fitted without
inference returns its corrected effect with no band and no p-value.

The estimate does not depend on the size or sign of the spillover: whatever
reaches the contaminated controls from :math:`t_c` on is replaced (iterative)
or solved for (inclusive), so the corrected effect, band and p-value are the
same for any spillover path.

Diagnostics
-----------

* ``uncorrected``: the design's estimate on the realized panel, spillover
  included.
* ``restricted``: the design re-solved without the contaminated controls,
  when ``design_config`` is given.
* ``spillover``: each corrected control's estimated spillover path from
  :math:`t_c` on.
* ``cleaning_rmspe``: each rebuilding fit's error over the fitting window.
* ``composite_weight`` and ``omega_det``: for the inclusive method, each
  control's load on the treated composite and the system's determinant.
* ``skipped_units``: affected controls without weight. They carry no bias, so
  they are not corrected, and they are kept out of the donors the others are
  rebuilt from. A warning names them.

A warning also fires when more than three controls are corrected, or when the
corrected controls hold more than half of the control composite: the
corrections were validated with one to three.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import MAREX, SPILLCORRECT

   rng = np.random.default_rng(0)
   J, T, T0 = 14, 60, 48
   t = np.arange(T)
   demand = 100 + 8 * np.sin(2 * np.pi * t / 13) + np.cumsum(rng.normal(0, 0.5, T))
   size = np.exp(rng.uniform(-0.7, 0.7, J))
   Y = size[:, None] * demand * (1 + 0.02 * rng.normal(size=(J, T)))
   panel = pd.DataFrame([{"region": f"r{j:02d}", "week": s, "sales": Y[j, s]}
                         for j in range(J) for s in range(T)])

   design_cfg = dict(T0=T0, blank_periods=20, T_post=12, m_eq=2,
                     backend="exact", inference=True)
   design = MAREX(dict(df=panel, outcome="sales", unitid="region",
                       time="week", **design_cfg)).fit()

   # After the test: a 5% lift on the treated regions, and a spillover onto
   # the heaviest control from the start of the blank window.
   realized = panel.copy()
   treated = realized.region.isin(list(design.composite.treated)) & (realized.week >= T0)
   realized.loc[treated, "sales"] *= 1.05
   k = max(design.composite.control, key=design.composite.control.get)
   hit = (realized.region == k) & (realized.week >= design.windows.fit_end)
   realized.loc[hit, "sales"] *= 1.4

   res = SPILLCORRECT(dict(df=realized, outcome="sales", unitid="region",
                           time="week", design=design, affected_units=[k],
                           design_config=design_cfg)).fit()

   res.uncorrected.att, res.att, res.restricted.att   # biased, corrected, re-solved
   res.corrected.p_value                               # the design's test, corrected
   res.spillover[k]                                    # the spillover taken out of k

Verification
------------

The tests in ``mlsynth/tests/test_spillcorrect.py`` pin:

* the uncorrected arm, which reproduces the design's own effect, band and
  p-value;
* the corrected arm, which is the design's inference on the corrected panel;
* the invariance of the estimate to the spillover's size and sign;
* the exact rebuild of a control inside the clean controls' span;
* the inclusive estimates, which solve the cross-weight system;
* the restricted arm, which equals MAREX re-solved without the contaminated
  controls.

The cross-validation against :doc:`spillsynth` and a benchmark case
reproducing the simulations' headline cells are step 4 of #932.

Core API
--------

.. automodule:: mlsynth.estimators.spillcorrect
   :members:

Configuration
-------------

.. autoclass:: mlsynth.utils.spillcorrect_helpers.config.SPILLCORRECTConfig
   :members:

Result Containers
-----------------

.. automodule:: mlsynth.utils.spillcorrect_helpers.structures
   :members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.spillcorrect_helpers.correct
   :members:

.. automodule:: mlsynth.utils.spillcorrect_helpers.setup
   :members:

.. automodule:: mlsynth.utils.spillcorrect_helpers.pipeline
   :members:

.. automodule:: mlsynth.utils.spillcorrect_helpers.plotter
   :members:
