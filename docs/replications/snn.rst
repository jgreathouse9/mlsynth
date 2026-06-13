SNN — ``deshen24/syntheticNN`` (Prop 99)
========================================

.. currentmodule:: mlsynth

Cross-validation against the reference implementation. ``SNN`` is mlsynth's port
of Synthetic Nearest Neighbors (Agarwal, Dahleh, Shah & Shen 2021, *Causal
Matrix Completion*); the canonical implementation is
`deshen24/syntheticNN <https://github.com/deshen24/syntheticNN>`_.

Why the two agree exactly
-------------------------

The reference finds each entry's anchor submatrix with a NetworkX **maximum
biclique** search; mlsynth uses a dependency-free **greedy** largest-fully-
observed-submatrix search. These are different algorithms in general — but under
**block missingness** (the synthetic-control setting, where the treated unit's
post-treatment cells are the *only* missing block) both return the same answer:
the full *control × pre-period* block. With the same Donoho-Gavish (2014)
universal rank and the same principal-component regression, the imputed
counterfactual is then identical.

Data
----

``basedata/smoking_data.csv`` — the Abadie, Diamond & Hainmueller (2010) Prop 99
panel: 39 states, 1970–2000, California treated from 1989, outcome ``cigsale``
(per-capita cigarette packs). 19 pre-periods.

Result
------

Masking California's 1989–2000 outcomes and imputing them with
``SyntheticNearestNeighbors(n_neighbors=1)`` (universal rank) gives a
counterfactual that mlsynth's :class:`SNN` reproduces to **< 1e-6**:

==========================  =====================  ===========================
Quantity                    SNN (mlsynth)          ``deshen24/syntheticNN``
==========================  =====================  ===========================
Average ATT (1989–2000)     −18.43 packs           −18.43 packs
Gap by 2000                 −29.33 packs           −29.33 packs
Counterfactual path         match to ``< 1e-6``    (reference)
==========================  =====================  ===========================

.. note::

   This case caught a real bug. mlsynth's Donoho-Gavish ``omega`` had its last
   two coefficients swapped (``1.43 β + 1.82`` instead of the published
   ``1.82 β + 1.43``), mis-selecting the rank and shifting the ATT by ~1
   pack/capita. The corrected formula matches the reference's rank — and hence
   its counterfactual — exactly.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py snn_prop99

The durable case is ``benchmarks/cases/snn_prop99.py`` (the reference
counterfactual is embedded from a live run, so no ``networkx``/``sklearn`` is
needed at benchmark time); unit regressions are in
``mlsynth/tests/test_snn.py`` (``TestReferenceProp99`` and the Donoho-Gavish
``omega`` guard ``test_universal_rank_matches_gavish_donoho``).
