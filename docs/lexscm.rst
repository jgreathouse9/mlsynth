Lexicographic SCM
=================

.. currentmodule:: mlsynth

Overview
--------

Lexicographic Synthetic Control (LexSCM) selects treated unit combinations
by jointly optimizing:

- Pre-treatment fit (NMSE)
- Statistical power (Minimum Detectable Effect)

The pipeline consists of:

1. Candidate generation (branch-and-bound)
2. Synthetic control fitting
3. Power analysis (MDE)
4. Final selection (validity-first rule)

Quick Start
-----------

.. code-block:: python

    from mlsynth.estimators import LexSCM

    model = LexSCM(...)
    results = model.fit(data)

    results.summary()

Core API
--------

.. automodule:: mlsynth.estimators.lexscm
   :members:
   :undoc-members:
   :show-inheritance:

Power Analysis (MDE)
-------------------

.. automodule:: mlsynth.utils.fast_scm_helpers.power_helpers
   :members:
   :undoc-members:

This module implements:

- Permutation-based inference
- Minimum Detectable Effect (MDE)
- Detectability curves across horizons

Search Engine (Candidate Generation)
------------------------------------

These modules implement the combinatorial search over treated unit subsets.

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_bb
   :members: branch_and_bound_topK
   :undoc-members:

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_control
   :members: evaluate_candidates
   :undoc-members:

Data Preparation & Matrix Construction
-------------------------------------

Utilities for constructing the synthetic control design matrices.

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_setup
   :members: prepare_experiment_inputs, split_periods, build_X_tilde
   :undoc-members:

Advanced / Internal Utilities
----------------------------

Low-level helpers used internally by the search and estimation routines.

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_setup
   :members:
       _prepare_working_df,
       build_candidate_mask,
       build_f_vector,
       build_Y_matrix,
       build_Z_matrix
   :undoc-members:
   :noindex:
