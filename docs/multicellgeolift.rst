.. _multicellgeolift:

MULTICELLGEOLIFT — multi-cell GeoLift analysis
==============================================

When to use
-----------

A **multi-cell** geo experiment runs several treatments **at once** — different
channels, budgets, or creative strategies — each on its own group of geos
("cells" :math:`A, B, \dots`), all measured against a **shared** pool of control
geos over the same window. Use ``MULTICELLGEOLIFT`` to measure each cell's
incremental effect and to compare the cells. It is the analysis analogue of
GeoLift's ``GeoLiftMultiCell`` (single-cell measurement is :doc:`geolift`).

Data model
----------

A unit-level **cell-membership** column plus a treatment-window indicator:

* ``cell_column_name`` — each geo's cell label (``"A"``, ``"B"``, …); **blank /
  ``NaN``** (or an explicit ``control_label``) marks a **control** geo. The label
  is a property of the geo, so it is constant over that geo's rows.
* ``post_col`` — the (shared) ``0/1`` post-treatment window.

.. code-block:: python

   from mlsynth import MULTICELLGEOLIFT

   res = MULTICELLGEOLIFT({
       "df": panel, "outcome": "Y", "unitid": "location", "time": "date",
       "cell_column_name": "cell",   # "A"/"B"/... ; blank = control
       "post_col": "post",
       "fixed_effects": True,         # augsynth/GeoLift default
   }).fit()

   res.cells["A"].effects.att        # cell A's ATT (per unit)
   res.cells["A"].inference.p_value  # cell A's conformal p
   res.comparison                    # pairwise cross-cell rows
   res.winner                        # the cell that wins every comparison, or None

Method
------

**Per cell.** Each cell is measured against the **shared control** pool with the
fixed-effect Augmented SCM + conformal inference of :doc:`geolift` — *crucially,
the other cells' geos are excluded from the donor pool*, because they are treated
(with a different treatment) and so contaminated (GeoLift's
``filter(!location %in% other_cells)``). So cell :math:`A`'s synthetic control is
a combination of **control** geos only, never cell :math:`B`'s.

**Cross-cell winner.** For each pair, a cell wins when its ATT confidence
interval lies strictly above the other's (GeoLift's **non-overlapping-CI** rule;
the ATT interval is the per-period conformal band averaged). The overall
``winner`` wins every pairwise comparison, else ``None``. This is deliberately
conservative: measuring each cell cleanly is well-powered, but *separating* two
cells needs the difference to clear both intervals — so overlapping CIs (no
declared winner) is common and correct, not a failure.

Result
------

``MULTICELLGEOLIFT.fit`` returns a
:class:`~mlsynth.utils.geolift_helpers.multicell.structures.MultiCellResults` (a
:class:`~mlsynth.config_models.DesignResult`): ``cells`` maps each label to its
:class:`EffectResult`, ``comparison`` is the pairwise table, ``winner`` the
overall winner, and ``report`` is the representative (winner / largest) cell.

Verification
------------

Cross-validated against **augsynth** (the engine ``GeoLiftMultiCell`` wraps) on
the GeoLift_Test panel — cell A = {chicago, portland} (real effect), cell B =
{atlanta, boston} (placebo), the rest shared controls: the per-cell ATT matches
augsynth **to the decimal** (A ``156.84``, B ``119.38``), the conformal p-values
agree (A ``≈0.01``, B ``≈0.8``), and the donor-exclusion invariant holds (A never
uses B's markets). Durable case ``geolift_multicell``; unit tests
``mlsynth/tests/test_multicell.py``.

Core API
--------

.. autoclass:: mlsynth.MULTICELLGEOLIFT
   :members: fit

.. autoclass:: mlsynth.utils.geolift_helpers.multicell.config.MultiCellGeoLiftConfig
   :members:
