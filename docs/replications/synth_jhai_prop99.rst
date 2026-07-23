.. _replication-synth-jhai-prop99:

VanillaSC vs j-hai/Synth — Prop 99 and the split-conformal band
===============================================================

:Estimator: :doc:`../vanillasc` — :class:`mlsynth.VanillaSC`
:Source: Abadie, Diamond & Hainmueller (2010), JASA 105(490); the maintained R
   ``Synth`` package with Hainmueller's ``synth_inference()``
   (`j-hai/Synth <https://github.com/j-hai/Synth>`_ 1.2.0).
:Replication type: cross-validation against the authors' reference package,
   point estimates and the split-conformal prediction band.
:Status: verified — weights/ATT cross-validate; the split-conformal construction
   matches value-for-value.

Why this case exists
--------------------

``synth_prop99`` already checks ``VanillaSC`` against the original ``Synth``
solver on the outcome-only fit. This case does two further things against the
*maintained* package (the one shipping Hainmueller's new ``synth_inference()``):
it uses the full canonical ADH (2010) predictor spec, and it cross-checks the
new split-conformal band that motivated ``inference="conformal_split"``.

The synthetic control
---------------------

Under the ADH spec (``loginc`` / ``p_cig`` / ``pct15-24`` averaged over 1980-1988,
``pc_beer`` over 1984-1988, and ``cigsale`` at 1975 / 1980 / 1988),
``VanillaSC(backend="mscmt")`` reproduces the package's ``synth()`` fit:

============  ===============  ================
Quantity      VanillaSC        j-hai/Synth
============  ===============  ================
Utah          0.335            0.343
Nevada        0.236            0.236
Montana       0.202            0.182
Colorado      0.160            0.175
Connecticut   0.068            0.062
ATT           −18.98           −18.72
pre-RMSPE     **1.754**        1.791
============  ===============  ================

The donor weights agree to about 0.02 (the Montana/Colorado split, two
interchangeable mountain-west donors, carries most of the difference) and the
ATT to a quarter of a pack. As in ``synth_prop99`` and ``masc_basque``, mlsynth
attains a *lower* pre-period RMSPE than the original nested V-search — the
MSCMT/Malo thesis that the data-driven predictor-weight optimizer can stop short
of the global optimum, shown here against ``Synth`` itself.

The split-conformal band
------------------------

``inference="conformal_split"`` is mlsynth's port of
``synth_inference(method = "conformal")``: a constant half-width :math:`q`, the
:math:`\lceil (n+1)(1-\alpha) \rceil`-th order statistic of the absolute
pre-period gaps, drawn as :math:`\widehat{y}^N_{1t} \pm q` over the whole
trajectory. On a shared set of gaps the two are the same estimator: feeding the
package's own pre-period gaps to
:func:`mlsynth.utils.inferutils.split_conformal_quantile` returns its
``conformal_q`` = 6.113436 exactly.

On its own synthetic control mlsynth's band is slightly tighter — :math:`q` =
5.90 against the package's 6.11 — a direct consequence of the lower pre-period
RMSPE: a better pre-fit shrinks the calibration residuals, so the conformal band
that reads its width off them narrows. Same construction, tighter input.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case synth_jhai_prop99

The mlsynth side reads ``basedata/augmented_cali_long.csv``. The R reference is
baked into ``benchmarks/reference/synth_jhai_prop99/`` from ``Synth`` 1.2.0;
regenerate it with

.. code-block:: bash

   # install route (CRAN firewalled; git clone the package):
   #   git clone --depth 1 https://github.com/j-hai/Synth && R CMD INSTALL Synth
   Rscript benchmarks/R/synth_jhai_prop99.R basedata/augmented_cali_long.csv \
       benchmarks/reference/synth_jhai_prop99
