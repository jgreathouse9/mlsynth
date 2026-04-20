Lexicographic SCM
=================

.. contents:: Table of Contents
   :depth: 3
   :local:

.. currentmodule:: mlsynth

All Classes and Data Structures
-------------------------------

.. automodule:: mlsynth.estimators.lexscm
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
   :special-members: __init__

.. automodule:: mlsynth.utils.fast_scm_helpers.power_helpers
   :members:
   :undoc-members:

Search & Evaluation Engine (Recursive)
--------------------------------------

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_bb
   :members:
   :undoc-members:
   :private-members:

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_control
   :members:
   :undoc-members:
   :private-members:

Matrix & Setup Utilities
------------------------

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_setup
   :members:
   :undoc-members:
   :private-members:   :members: prepare_experiment_inputs, split_periods, build_X_tilde
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
