ESC: short-panel prediction intervals (Prop 99, São Paulo, Monte Carlo)
=======================================================================

.. currentmodule:: mlsynth

This page documents the verification of mlsynth's :doc:`../esc` estimator
(Ensemble Synthetic Control, Lian & Pu 2026). ESC's contribution is inference
under a short pre-treatment panel, so it is validated on two empirical case
studies and on the paper's Monte Carlo coverage evidence.

Empirical -- California (Proposition 99)
----------------------------------------

On the canonical cigarette panel (California plus 38 donor states, treatment from
1989, :math:`T_0 = 19`), ESC reproduces the classic synthetic-control result
while its prediction interval remains informative. The ensemble counterfactual
correlates about :math:`0.998` with the convex synthetic California, and the
headline effect matches the literature:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Quantity
     - ESC
     - classic SC
     - Abadie et al.
   * - avg post-treatment gap (packs)
     - :math:`\approx -22.7`
     - :math:`-19.5`
     - :math:`\approx -19`
   * - 2000 gap (packs)
     - :math:`\approx -33.6`
     - :math:`-26.6`
     - :math:`\approx -26`
   * - pre-treatment RMSE
     - :math:`\approx 1.64`
     - :math:`1.66`
     - --

The ensemble path matches the convex synthetic California in direction and
magnitude at a comparable pre-treatment fit, and its :math:`90\%` pointwise band
places observed California below the lower bound in every post-treatment year, so
the effect reads as significant under ESC's predictive inference. See
`esc_prop99.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/esc_prop99.py>`_.

Empirical -- São Paulo homicides (short pre-window)
---------------------------------------------------

The São Paulo homicide study (Freire 2018; treated 1999, :math:`T_0 = 9` annual
pre-periods, 24 donor states) is the short-panel regime ESC is designed for.
Here the value of ESC is the honesty of its interval, not a point
estimate: the ensemble reports an average effect of about :math:`-7` per
:math:`100{,}000` (with the gap growing to roughly :math:`-20` by 2009) and a
wide :math:`90\%` interval on the number of lives saved --- on the order of
:math:`[14{,}000,\ 42{,}000]` around a central :math:`\approx 32{,}000`. That
width nearly spans the range that the individual synthetic-control variants
produce as point estimates, so ESC surfaces as one interval the model
uncertainty that a single specification hides. The observed series clears the
lower bound of the band only in the later post-treatment years, consistent with
the across-donor placebo evidence that the effect is large but marginal under
nine pre-periods. See `esc_saopaulo.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/esc_saopaulo.py>`_.

Monte Carlo -- interval coverage (open reproduction)
----------------------------------------------------

The paper's central claim is coverage: in short, high-dimensional panels the ESC
prediction interval attains close to its nominal level. This did not reproduce
from the equations as written. Drawing DGP-1 factor-model panels
(:math:`T_0 = 8`, :math:`J = 15`) and measuring pointwise coverage of the
realized counterfactual :math:`Y_{1t}(0)`, a faithful port of the equation-(7)
ensemble-quantile band covers only about :math:`0.5` to :math:`0.65` against a
:math:`0.90`--:math:`0.95` nominal level -- under the recommended Local +
:math:`\lambda_{1se}` rule and across block lengths and perturbation strengths.
The reason is structural: the ensemble spread reflects weight-estimation
variability, not the treated unit's irreducible disturbance :math:`u_t`, so the
band is an estimation-uncertainty band, not a full prediction interval,
and adding a resampled residual narrows but does not close the gap.

This is a reproduction gap, not necessarily a defect: the authors' ``esc``
package (Python/R/Stata) is not yet released, so the predictive component that
would lift coverage to nominal cannot be inspected. A calibrated coverage
benchmark is therefore deferred until the reference implementation is available;
the shipped estimator implements the ensemble band exactly as specified.
