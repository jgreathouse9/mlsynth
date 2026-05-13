Synthetic Control Design Intervention (SCDI)
===========================================

.. currentmodule:: mlsynth

Overview
--------

Synthetic Control Design Intervention (SCDI) is an experimental-design style
module. Unlike estimators that consume an observed treatment indicator, SCDI
selects ``K`` treated units from a balanced panel and constructs synthetic
contrasts from pre-treatment outcomes.

The current implementation supports three formulations:

- ``global_2way``: a global mixed-integer synthetic treated/control contrast.
- ``global_equal_weights``: a global special case where selected treated units
  receive ``1 / K`` weight and controls receive ``1 / (N - K)`` weight.
  The helper solves this case by enumerating assignments instead of optimizing
  free weights.
- ``per_unit``: per-treated-unit synthetic twins.

Core API
--------

.. automodule:: mlsynth.estimators.scdi
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SCDIConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.scdi_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.scdi_helpers.formulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.scdi_helpers.optimization
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.scdi_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.scdi_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.scdi_helpers.structures
   :members:
   :undoc-members:

Example
-------

.. code-block:: python

   from mlsynth import SCDI
   from mlsynth.config_models import SCDIConfig

   config = SCDIConfig(
       df=df,
       outcome="outcome",
       unitid="unit",
       time="time",
       K=3,
       T0=50,
       mode="global_2way",
       display_graph=True,
   )

   results = SCDI(config).fit()
   print(results.selected_unit_labels)
   print(results.design.objective_value)

   # For global modes, q is the treated-side simplex and w - q is the
   # control-side simplex. In global_equal_weights these are exactly
   # 1 / K for treated units and 1 / (N - K) for controls.
   print(results.treated_weights_by_unit)
   print(results.control_weights_by_unit)
