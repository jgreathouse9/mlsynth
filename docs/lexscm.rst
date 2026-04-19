Lexicographic SCM
=====================

Estimator
---------

.. autoclass:: mlsynth.estimators.lexscm.LEXSCM
   :show-inheritance:
   :members:
   :undoc-members:
   :private-members:
   :special-members: __init__

Results & Containers
--------------------

.. autoclass:: mlsynth.estimators.lexscm.LEXSCMResults
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.fast_scm_helpers.power_helpers.PowerAnalysisResults
   :members:
   :undoc-members:

Core Helpers
------------

.. automodule:: mlsynth.utils.fast_scm_helpers.power_helpers
   :members: run_mde_analysis, rank_candidates, mde_summary_table
   :undoc-members:

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_setup
   :members: prepare_experiment_inputs, split_periods, build_X_tilde
   :undoc-members:

Search & Evaluation Engine
--------------------------

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_bb
   :members: branch_and_bound_topK
   :undoc-members:
   :noindex:

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_control
   :members: evaluate_candidates
   :undoc-members:
   :noindex:

Internal Scopes & Math
----------------------

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_setup
   :members: _prepare_working_df, build_candidate_mask, build_f_vector, build_X_tilde, build_Y_matrix, build_Z_matrix
   :undoc-members:
