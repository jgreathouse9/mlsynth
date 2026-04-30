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





Example: Synthetic Sales Study with LEXSCM
===========================================

This example demonstrates a full end-to-end workflow.

Generate Synthetic Panel Data
-----------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd

    from mlsynth import LEXSCM


    def generate_synthetic_sales_panel(
        n_units: int = 100,
        n_time_periods: int = 100,
        n_candidates: int = 40,
        treatment_start: int = 80,
        seed: int = 42,
        sigma: float = 3.0,
        sales_scale: float = 100.0,
        budget: float = 4_000_000
    ) -> pd.DataFrame:

        np.random.seed(seed)

        unit_fe = np.random.normal(0, 8, size=n_units)
        unit_trend = np.random.normal(0.3, 0.15, size=n_units)
        unit_sensitivity = np.random.uniform(0.8, 1.2, size=n_units)

        t = np.arange(n_time_periods)

        common_factor = (
            1.5 * t + 120
            + 12 * np.sin(2 * np.pi * t / 52)
            + np.random.normal(0, 3.5, n_time_periods)
        )

        sales = np.zeros((n_time_periods, n_units))

        for j in range(n_units):
            sales[:, j] = (
                sales_scale
                + common_factor * unit_sensitivity[j]
                + unit_fe[j]
                + unit_trend[j] * t
                + np.random.normal(0, sigma, n_time_periods)
            )

        sales = np.maximum(sales, 5.0)

        base_cost = np.random.lognormal(mean=12, sigma=0.6, size=n_units) * 5
        size_factor = np.random.uniform(0.7, 1.4, n_units)
        treatment_cost = np.round((base_cost * size_factor) / 10) * 10

        unit_ids = np.repeat(np.arange(n_units), n_time_periods)
        time_ids = np.tile(np.arange(n_time_periods), n_units)
        sales_flat = sales.ravel(order="F")

        df = pd.DataFrame({
            "unitid": unit_ids,
            "time": time_ids,
            "sales": sales_flat,
            "treatment_cost": np.repeat(treatment_cost, n_time_periods)
        })

        candidate_mask = np.zeros(n_units, dtype=bool)
        candidate_idx = np.random.choice(n_units, size=n_candidates, replace=False)
        candidate_mask[candidate_idx] = True

        df["candidate"] = np.repeat(candidate_mask, n_time_periods)
        df["post"] = (df["time"] >= treatment_start).astype(int)

        df["avg_income"] = np.repeat(np.random.uniform(40, 90, n_units), n_time_periods)
        df["population"] = np.repeat(np.random.uniform(800, 2500, n_units), n_time_periods)

        return df


Run LEXSCM
-----------

.. code-block:: python

    df = generate_synthetic_sales_panel(
        n_units=200,
        n_candidates=60,
        n_time_periods=100,
        seed=4545,
        treatment_start=90,
        budget=4_000_000
    )

    config = {
        "df": df,
        "outcome": "sales",
        "unitid": "unitid",
        "time": "time",
        "m": 2,
        "top_K": 20,
        "candidate_col": "candidate",
        "lambda_penalty": 0.5,
        "post_col": "post"
    }

    model = LEXSCM(config)
    results = model.fit()


