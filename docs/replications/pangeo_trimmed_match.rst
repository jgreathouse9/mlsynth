.. _replication-pangeo-trimmed-match:

PANGEO pairs — Trimmed Match Design's pairing on its example data
=================================================================

:Estimator: :doc:`../pangeo` -- :class:`mlsynth.PANGEO`
:Source: Chen, A. & Au, T. C. (2022). "Robust causal inference for
   incremental return on ad spend with randomized paired geo experiments."
   *Annals of Applied Statistics* 16(1), 1-20. Chen, A., Longfils, M. &
   Remy, N. "Trimmed Match Design for randomized paired geo experiments."
   `arXiv:2105.07060 <https://arxiv.org/abs/2105.07060>`_. Reference code:
   `google/trimmed_match <https://github.com/google/trimmed_match>`_ at commit
   ``0be855234ee3a589eaddc8b945e2a0d924ad8791``.
:Replication type: cross-validation against the package's own pairing code,
   run live and unmodified in its own environment, its C++ core compiled and
   its own test suite passing there.
:Status: verified -- the same 87 pairs.
:Benchmark: ``benchmarks/cases/pangeo_trimmed_match_pairs.py``
   (`source <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/pangeo_trimmed_match_pairs.py>`__).

Why this case exists
--------------------

A paired geo experiment splits geos into pairs and randomises treatment within
each pair. The Supergeo paper (Chen, Doudchenko, Jiang, Stein and Ying 2023,
`arXiv:2301.12044 <https://arxiv.org/abs/2301.12044>`_) scores a pair by the
squared difference of its two geos' pretest totals and a design by the sum over
its pairs. PANGEO offers that score as ``objective="supergeo_total"`` and, with
``max_supergeo_size=1``, chooses the pairs by an exact-cover mixed-integer
program; this case runs the program through cvxpy (``mip_backend="cvxpy"``).

For pairs this objective has a known optimum: sort the geos by their totals and
pair each with its neighbour. For four sorted totals
:math:`a \le b \le c \le d`, pairing :math:`a` with :math:`c` and :math:`b` with
:math:`d` costs at least :math:`(b-a)^2 + (d-c)^2`, since each difference only
grows, and pairing :math:`a` with :math:`d` and :math:`b` with :math:`c` costs
:math:`(d-a)^2 + (c-b)^2 \ge (b-a)^2 + (d-c)^2`; exchanging pairs one at a
time extends the argument to any number of geos. Trimmed Match Design pairs
geos the same way: ``TrimmedMatchGeoXDesign.create_geo_pairs`` sorts them by a
weighted share of their pretest totals and pairs consecutive ones. With the
response as the only matching metric the two must agree, and this case checks
that they do on the data the package ships.

Data and windows
----------------

``example_data_for_design.csv`` from the pinned commit: 174 geos, daily
response and cost over 364 days of 2020. The windows are those of the package's
design colab: design 2020-01-01 to 2020-12-29, evaluation 2020-12-02 to
2020-12-29, with cross validation, so pairs are formed on the 336 days before
the evaluation window. PANGEO receives the same 364 days with
``frac_E = 336 / 364``, so its estimation window is those 336 days.

Results
-------

==========================================  ============  ===============
Quantity                                    mlsynth       trimmed_match
==========================================  ============  ===============
pairs                                       87            87
pairs in common                             87            87
cost / sorted pairing                       1.000000000   1.000000000
==========================================  ============  ===============

PANGEO's program has 15,051 candidate pairs and solved to optimality in 0.57
to 0.66 s through cvxpy (1.13 s through ``highspy``). With the package's
default matching metrics, which add each geo's share of spend with weight 0.01,
the reference's pairing keeps 81 of the 87 pairs and costs 1.000668 times the
optimum on the paper's objective: a pairing by response and spend together, not
a different method.

The reference environment
-------------------------

The package builds its C++ core (the Trimmed Match estimate and its interval)
with Bazel and was tested on Python 3.7. ``install.sh`` builds it without
Bazel: it clones the package at the pinned commit, makes a Python 3.11 virtual
environment with versions of the package's era (numpy 1.26.4, pandas 1.5.3,
scipy 1.11.4, matplotlib 3.8.4, seaborn 0.12.2, Jinja2 3.1.6, absl-py 2.5.1,
pybind11 2.13.6), and compiles the four core sources with the system C++
compiler. The core uses only glog's ``CHECK`` and absl's ``optional``; two
stand-in headers (``benchmarks/reference/trimmed_match/shim/``) provide them, a
failed check raising a Python ``RuntimeError`` where glog would abort. The
script then runs the package's own test suite in that environment: 156 tests
pass (the ``multi_cell`` tests are excluded, since at this commit they import a
module the package does not contain). The pairing gives the same 87 pairs with
the compiled core as with it absent, since pairing does not call it.

The estimate and the power analysis that the compiled core makes available are
the subject of later cases, once mlsynth carries a Trimmed Match estimator.

Reproducing
-----------

.. code-block:: bash

   bash benchmarks/reference/trimmed_match/install.sh python3.11
   python benchmarks/run_benchmarks.py --case pangeo_trimmed_match_pairs

``run.sh`` runs a reference script in the environment;
``MLSYNTH_TRIMMED_MATCH_PYTHON`` points it at another interpreter. Without
the environment the case skips.
