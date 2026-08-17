.. _replication-ascm-mixtape:

Ridge ASCM on the Mixtape studies — Proposition 99 and Texas prisons
=====================================================================

:Estimator: :doc:`../vanillasc` with ``augment="ridge"`` and
   ``inference="jackknife_plus"``.
:Source: Ben-Michael, Feller & Rothstein (2021), *"The Augmented Synthetic
   Control Method,"* JASA 116(536); reference implementation: the ``augsynth``
   R package (``ebenmichael/augsynth``), commit ``7a90ea48``. The two
   applications are from Scott Cunningham's Mixtape teaching material.
:Replication type: Cross-validation — mlsynth matched value-for-value to a live
   ``augsynth`` run on two panels.
:Status: Verified — ATT, imbalance, jackknife+ interval, selected penalty and
   all 88 donor weights reproduced.

Why these two studies
---------------------

Both are one line of R:

.. code-block:: r

   augsynth(cigsale  ~ treated, unit = state,    time = year, data = smoking,
            progfunc = "ridge", scm = TRUE)
   augsynth(bmprison ~ treated, unit = statefip, time = year, data = texas,
            progfunc = "ridge", scm = TRUE)

and the augmentation does something completely different in each. That contrast
is the point of pairing them.

.. list-table::
   :header-rows: 1
   :widths: 14 14 8 8 14 22

   * - Study
     - Treated
     - :math:`T_0`
     - :math:`J`
     - :math:`\lambda`
     - Pre-fit RMSE, simplex → ridge
   * - Proposition 99
     - California, 1989
     - 19
     - 38
     - :math:`4.30\times 10^{2}`
     - :math:`1.656 \to 0.734`
   * - Texas prisons
     - Texas, 1993
     - 8
     - 50
     - :math:`1.74\times 10^{10}`
     - :math:`862.9 \to 861.3`

Proposition 99 is the method working as advertised. The simplex fit puts weight
on six donors and still misses California by 1.66 packs per capita. The ridge
augmentation halves that, spreading onto 36 donors with negative weights summing
to :math:`-0.28` — the extrapolation outside the convex hull that the
correction exists to permit.

Texas is where the method switches itself off. With only 8 pre-treatment years
the cross-validation cannot justify any augmentation: it selects a penalty of
:math:`1.74\times 10^{10}`, the fitted weights move by at most
:math:`3.2\times 10^{-4}` from the plain simplex solution, the support stays at
the same three donors (Florida 0.373, New York 0.356, Illinois 0.272), the
negative weights sum to :math:`-0.002`, and the pre-treatment fit improves by
0.2%. Augmented SCM has degenerated to SCM, and it did so on its own.

The penalty gap is not a units artefact. Prison counts run about 19,400 against
116 packs per capita, so an outcome-scale penalty would differ between the
panels by roughly :math:`2.8\times 10^{4}`; the selected penalties differ by
:math:`4.0\times 10^{7}`, three orders beyond that. The short pre-period is what
drives it.

What the case guards
--------------------

The Texas failure mode is invisible to inspection. An implementation whose
penalty grid was clamped, or whose cross-validation shrank differently at short
:math:`T_0`, would turn the augmentation back on — and because ridge moves the
pre-treatment fit by only 0.2% there, the fitted path would look the same in a
plot while the post-period counterfactual, and so the ATT, moved. Pinning the
ATT alone would catch it after the fact without saying why; the case pins the
mechanism as well, as ``ridge_rmse_gain`` and ``ridge_weight_shift``.

Cross-validation results
------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 20 20 16

   * - Quantity
     - mlsynth
     - ``augsynth`` 0.2.0
     - Relative gap
   * - ATT (Prop 99)
     - :math:`-15.952568`
     - :math:`-15.952578`
     - :math:`6.3\times 10^{-7}`
   * - jackknife+ interval (Prop 99)
     - :math:`[-22.13713, -12.26853]`
     - :math:`[-22.13712, -12.26854]`
     - :math:`\le 1.0\times 10^{-6}`
   * - Pre-fit :math:`L_2` (Prop 99)
     - :math:`3.197963`
     - :math:`3.197961`
     - :math:`6.9\times 10^{-7}`
   * - ATT (Texas)
     - :math:`20{,}972.161`
     - :math:`20{,}972.170`
     - :math:`4.1\times 10^{-7}`
   * - jackknife+ interval (Texas)
     - :math:`[18900.108, 23849.490]`
     - :math:`[18900.107, 23849.482]`
     - :math:`\le 3.4\times 10^{-7}`
   * - Pre-fit :math:`L_2` (Texas)
     - :math:`2436.2091`
     - :math:`2436.2089`
     - :math:`9.7\times 10^{-8}`

All 88 donor weights agree too, worst cell :math:`1.4\times 10^{-6}` on
Proposition 99 and :math:`4.3\times 10^{-6}` on Texas. The weights agree less
closely than the ATT does, which is what two exact solvers on a quadratic
program with near-collinear donors should do: mlsynth solves the base simplex
problem with its own active set and ``augsynth`` with ``osqp``, and the
difference between them falls in directions that barely move the fitted series.

The scaled :math:`L_2` imbalance — ``augsynth``'s ``scaled_l2_imbalance``, the
imbalance divided by what uniform donor weights would leave — is 0.0457 for
Proposition 99 and 0.0593 for Texas. Both fits close about 95% of the naive gap,
which is the one thing the two studies agree on.

Reproducing it
--------------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case ascm_mixtape

The reference values are committed, so the case runs without R. To regenerate
them against a live ``augsynth``:

.. code-block:: bash

   bash benchmarks/R/install_augsynth.sh
   python benchmarks/reference/generate.py ascm_mixtape

The bundle under ``benchmarks/reference/ascm_mixtape/`` holds the exact R
script, its verbatim output, the parsed values, and full provenance — R and
package versions, platform, git SHA, and a checksum of each input panel.

Data
----

``basedata/smoking_data.csv`` is the Mixtape's ``synth_smoking.dta`` with the
state labels decoded; California is state 3 there, which is what
``state == "3"`` selects in the original script. The two agree to
:math:`1.2\times 10^{-5}`, which is ``float32`` storage in the ``.dta``.

``basedata/texas_bmprison.csv`` is the ``bmprison`` column of the Mixtape's
``texas.dta`` — 51 states over 1985–2000, balanced, no missing values.

Both the R reference and the Python case read these CSVs, so the two sides see
byte-identical inputs and any disagreement is the estimator and not the
ingestion.

Durable cases and tests
-----------------------

* ``ascm_mixtape`` — this case (``benchmarks/cases/ascm_mixtape.py``), 21 pinned
  quantities across the two studies.
* Related: ``ascm_kansas`` (the four-spec ladder through the helper functions),
  ``ascm_jackknife_plus`` (the interval's per-drop construction), and
  ``augsynth_calibrated`` (the paper's Section-7 simulation). See
  :doc:`ascm_kansas` and :doc:`ascm_jackknife_plus`.
