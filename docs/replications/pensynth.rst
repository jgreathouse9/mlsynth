.. _replication-pensynth:

Penalized Synthetic Control (Abadie & L'Hour 2021)
==================================================

:Estimator: :doc:`../vanillasc` — the ``penalized`` backend
   (:func:`mlsynth.utils.bilevel.penalized.penalized_weights`).
:Source: Abadie & L'Hour (2021), *"A Penalized Synthetic Control Estimator for
   Disaggregated Data,"* JASA 116(536), 1817–1834; reference implementation: the
   authors' ``pensynth`` repository (``jeremylhour/pensynth``), function
   ``wsoll1``.
:Replication type: **Cross-validation** — mlsynth's penalized solver matched to
   the authors' own ``wsoll1`` on identical inputs.
:Status: **Fully verified** — weights and ATT reproduced to solver precision.

Validation strategy
-------------------

The penalized estimator adds a pairwise matching penalty to the synthetic-control
objective. For treated predictors :math:`X_1`, donor predictors :math:`X_0` and
penalty :math:`\lambda \ge 0` it solves (the paper's eq. 5)

.. math::

   \min_{W}\; \lVert X_1 - X_0 W \rVert^2
     \;+\; \lambda \sum_j W_j \lVert X_1 - X_{0j} \rVert^2
   \quad\text{s.t.}\quad W \ge 0,\; \textstyle\sum_j W_j = 1 .

The penalty interpolates between the pure synthetic control (:math:`\lambda \to 0`)
and nearest-neighbour matching (large :math:`\lambda`); by the paper's Theorem 1,
for any :math:`\lambda > 0` the solution is unique and sparse. Because the program
is a strictly convex quadratic program in :math:`W` for :math:`\lambda > 0`, it has
a single optimum, which makes it an ideal target for a solver-level cross-check:
feed the same :math:`(X_0, X_1, \lambda)` to two independent solvers and they must
agree.

That is exactly what the benchmark does. mlsynth's implementation
(:func:`~mlsynth.utils.bilevel.penalized.penalized_weights`, the ``penalized``
backend of :class:`~mlsynth.estimators.vanillasc.VanillaSC`) solves the program by
projected-gradient FISTA; the reference ``wsoll1`` solves the identical program by
the interior-point routine ``LowRankQP``. We feed both the same predictor matrix
and compare across a regularisation path.

Cross-validation — the Prop 99 path
-----------------------------------

The predictor matrix is the canonical Abadie-Diamond-Hainmueller (2010) MLAB
vector: the four covariate averages — ln(personal income), retail cigarette price
and percent aged 15–24 over 1980–1988, beer consumption over 1984–1988 — plus the
three special lagged outcomes, cigarette sales in 1975, 1980 and 1988, matched
with :math:`V = I` (raw, no rescaling), the ``V = diag(ncol(X))`` convention of the
authors' California example (``examples/EXA_CaliforniaTobacco.R``). It is
constructed from mlsynth's vendored ``basedata/augmented_cali_long.csv`` through
:func:`mlsynth.utils.datautils.dataprep` and the covariate-mean helper
:class:`~mlsynth.estimators.vanillasc.VanillaSC` uses — no hand-pivoting.
California is the treated unit and the remaining 38 states are donors. The same
:math:`X_0` (:math:`7 \times 38`) and :math:`X_1` (:math:`7`) are sent to
``wsoll1`` and to ``penalized_weights`` over the grid
:math:`\lambda \in \{0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1\}`.

Across the whole path the two implementations agree to solver precision: the
largest donor-weight difference is :math:`\approx 3\times10^{-4}` and the largest
post-period ATT difference :math:`\approx 2\times10^{-3}` packs. At
:math:`\lambda = 0.1` the synthetic California loads :math:`\approx 0.54` on Idaho,
with a post-1989 ATT of :math:`-23.4` packs per capita matched to
:math:`\approx 4\times10^{-4}`. With :math:`V = I` (no nested :math:`V`
optimisation) the penalized fit is Idaho-led rather than ADH's published
Utah/Nevada pool — expected, since here the penalty, not a fitted :math:`V`,
resolves the donor weights. As :math:`\lambda` grows the weights concentrate
toward the nearest neighbour (Montana), reproducing the penalty's interpolation
property.

Durable benchmark
-----------------

The runnable case is ``pensynth_prop99`` in the durable suite
(``benchmarks/cases/pensynth_prop99.py``). It is a live cross-check: the reference
``wsoll1``/``TZero`` source is taken from a commit-pinned clone of
``jeremylhour/pensynth`` (``benchmarks/reference/clone_pensynth.py``) and
``LowRankQP`` is frozen by ``benchmarks/R/install_pensynth.sh``, so the same solver
runs every time. The case skips itself when ``Rscript``, ``LowRankQP`` or the clone
is unavailable, so it is a no-op where the reference toolchain is absent. Run it
with

.. code-block:: bash

   bash benchmarks/R/install_pensynth.sh        # one-time: R + LowRankQP
   python -m benchmarks.run_benchmarks --case pensynth_prop99
