SPOTSYNTH — O'Riordan & Gilligan-Lee (2025) spillover detection
===============================================================

.. currentmodule:: mlsynth

Reproduction of the empirical results in

   O'Riordan & Gilligan-Lee (2025). *Spillover detection for donor selection in
   synthetic control models.* Journal of Causal Inference 13:20240036.

SPOTSYNTH screens every candidate donor for spillover contamination -- a valid
donor's post-intervention value is forecastable from the other donors'
pre-intervention data (Theorem 3.1); a forecast failure flags an invalid donor --
then builds a simplex synthetic control on the donors judged valid. Two
screening rules are exposed: **S1** (keep the ``n`` best-forecast donors) and
**S2** (drop donors whose realised post value falls outside the forecast PPI).

The durable case ``benchmarks/cases/spotsynth_real_data.py`` reproduces three
figures of the paper.

Figure 6 — real-data screening (Path A, semi-synthetic)
-------------------------------------------------------

On the three canonical Abadie panels -- **German Reunification**
(``german_reunification.csv``), **California Tobacco Control**
(``smoking_data.csv``), and **Basque Country / ETA** (``basque_data.csv``) -- a
semi-synthetic invalid donor is planted: a noisy proxy of the target,
:math:`x_{\text{syn}} \sim \mathcal N(y, \sigma)`. Tracking the target, it earns
a large SC weight and biases the *unscreened* effect toward zero; **both** S1 and
S2 flag and exclude it, recovering the effect.

==================  ==============  ===================  ==================
panel               proxy excluded   screened ATT         unscreened ATT
==================  ==============  ===================  ==================
German Reunification  S1 ✓ / S2 ✓    −1458               −1084 (→ 0)
California            S1 ✓ / S2 ✓    −22.9 (≈ ADH)       −1.8 (→ 0)
Basque Country        S1 ✓ / S2 ✓    −1.20 (≈ ABG)       −0.12 (→ 0)
==================  ==============  ===================  ==================

Figure 2 — detection power (Path B)
-----------------------------------

Leave-one-out detection AUC (probability an invalid donor scores more anomalous
than a valid one): **0.97** sharp / **0.90** gradual under a *valid majority*
(30% invalid), and a documented **inversion** (≈0) under an *invalid majority*
(80%) -- the regime where the package pins the ``lag`` anchor instead.

Figure 4 — sensitivity / proximal debias (Path B)
-------------------------------------------------

When the kept donors are *noisy* proxies (errors-in-variables), even a perfect
valid-donor SC is attenuation-biased. The proximal two-stage debias (eq. 5),
using the screen-**excluded** donors as proximal controls, reduces that bias
(mean :math:`|\text{bias}|` 0.47 → 0.40 over the paper's EIV DGP).

.. note::

   The paper's §3.4 also gives analytical *bias bounds* for false-positive
   (eq. 7) and false-negative (eq. 8) screening errors. These are sensitivity
   *formulas* the analyst evaluates (the false-negative bound needs the unknown
   spillover :math:`\tau` as a sensitivity parameter); mlsynth implements the
   debias remedy (eq. 5) rather than the bounds, so the benchmark validates the
   debias, not the closed-form bounds.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py spotsynth_real_data
