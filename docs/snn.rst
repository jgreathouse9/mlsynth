Synthetic Nearest Neighbors / Causal Matrix Completion (SNN)
=============================================================

.. currentmodule:: mlsynth

Overview
--------

SNN (Agarwal, A., Dahleh, M., Shah, D. & Shen, D. (2021). *"Causal Matrix
Completion,"* arXiv:2109.15154) recovers the missing entries of a
partially observed matrix when the data are **missing not at random
(MNAR)** -- the probability that an entry is observed depends on the
underlying value. This selection bias is the norm in the two canonical
matrix-completion applications:

* **Recommender systems**: a user who dislikes horror films will almost
  never rate one, so the missingness pattern is informative about the
  ratings themselves.
* **Panel data / causal inference**: policy-makers adopt programs for
  reasons correlated with outcomes, and competing policies cannot be
  observed simultaneously, so the potential-outcome matrix is
  systematically (not randomly) missing.

Classical matrix completion assumes data are *missing completely at
random* (MCAR) and is biased under MNAR. SNN provides a causal framework
and an estimator with entry-wise (max-norm) finite-sample consistency and
asymptotic normality under MNAR.

The algorithm
^^^^^^^^^^^^^

To impute entry :math:`(i, j)`, SNN combines nearest neighbors
(collaborative filtering) with synthetic controls:

1. **Anchor rows and columns.** Find a fully observed submatrix
   :math:`S` whose rows are observed in column :math:`j` and whose
   columns are observed in row :math:`i` (paper Section 4.2). The
   reference implementation finds these via a maximum-biclique search;
   mlsynth uses a dependency-free greedy search for a large fully
   observed block.
2. **Principal component regression.** Truncate the SVD of :math:`S`,
   regress row :math:`i`'s anchor-column values :math:`q` on :math:`S` to
   learn weights :math:`\beta`, and apply them to column :math:`j`'s
   anchor-row values :math:`x`:
   :math:`\widehat A_{ij} = \langle x, \beta \rangle` (paper Algorithm 1).

SNN **generalises Synthetic Interventions** (Agarwal et al. 2021b, the
:class:`mlsynth.SI` estimator), which itself generalises classic
synthetic control: the same PCR machinery is applied, but the anchor
submatrix is found *per entry* rather than assuming a fixed treated/donor
block, so SNN handles arbitrary (block-structured) MNAR patterns.

Why panel data is a natural fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A fully observed *anchor block* is essential. Under independent MCAR no
large fully observed submatrix exists, but the **block-structured**
missingness of panel data -- a control block observed throughout, with
treated units missing their post-treatment :math:`Y(0)` -- *naturally
induces* anchor rows (controls) and columns (pre-periods). SNN is
therefore especially well suited to comparative case studies and
staggered-adoption designs, exactly the setting this estimator targets.

Causal use in mlsynth
^^^^^^^^^^^^^^^^^^^^^^

The :class:`SNN` estimator masks the treated post-treatment cells as
missing, imputes their untreated potential outcomes by SNN matrix
completion, and forms the treatment effect as observed minus imputed:

.. math::

   \widehat\tau_{it} = Y_{it} - \widehat Y_{it}(0), \qquad
   \widehat{\mathrm{ATT}} = \frac{1}{|\{(i,t): D_{it}=1\}|}
       \sum_{D_{it}=1} \widehat\tau_{it}.

The general matrix-completion engine is exposed directly as
:func:`mlsynth.utils.snn_helpers.snn_complete` for non-causal MNAR
completion (e.g. recommender systems): pass a matrix with ``NaN`` for
missing entries.

Core API
--------

.. automodule:: mlsynth.estimators.snn
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SNNConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.snn_helpers.completion
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.snn_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.snn_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.snn_helpers.structures
   :members:
   :undoc-members:

Example
-------

Proposition 99 -- California's 1988 tobacco-control program, the canonical
synthetic-control case study. SNN treats California's post-1988
per-capita cigarette sales as the missing entries, imputes the
counterfactual by matrix completion, and reports the ATT. With
``display_graphs=True`` it draws the observed-vs-counterfactual chart.

.. code-block:: python

   import pandas as pd

   from mlsynth import SNN

   # ------------------------------------------------------------------
   # Load the Prop 99 panel (39 states, 1970-2000; California treated 1989)
   # ------------------------------------------------------------------
   file = (
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/smoking_data.csv"
   )
   df = pd.read_csv(file)

   res = SNN({
       "df": df,
       "outcome": "cigsale",
       "treat": "Proposition 99",      # boolean treatment column
       "unitid": "state",
       "time": "year",
       "inference": True,              # leave-one-control jackknife
       "display_graphs": True,         # observed vs SNN counterfactual
   }).fit()

   print(f"ATT (avg 1989-2000) = {res.att:+.2f} packs/capita")
   lo, hi = res.inference.ci
   print(f"jackknife 95% CI    = [{lo:+.2f}, {hi:+.2f}]")
   print(f"gap by 2000         = {res.att_by_period[2000]:+.2f}")

The default ``universal_rank=True`` (Donoho-Gavish hard threshold) keeps
the rank well-calibrated for this small (39 x 31) low-rank panel; it
returns an average ATT of about ``-19`` packs/capita, widening to roughly
``-31`` by 2000 -- consistent with Abadie, Diamond & Hainmueller (2010).

The same SNN engine performs general (non-causal) matrix completion on
any matrix with ``NaN`` for the missing entries:

.. code-block:: python

   import numpy as np
   from mlsynth.utils.snn_helpers import snn_complete

   X = np.array([[1.0, 2.0, np.nan],
                 [2.0, 4.0, 6.0],
                 [3.0, np.nan, 9.0]])
   completed, feasible = snn_complete(X)

References
----------

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.

Agarwal, A., Shah, D., Shen, D., & Song, D. (2021b). "On Robustness of
Principal Component Regression." *Journal of the American Statistical
Association*.

Agarwal, A., Dahleh, M., Shah, D., & Shen, D. (2021). "Causal Matrix
Completion." arXiv:2109.15154.

Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021).
"Matrix Completion Methods for Causal Panel Data Models." *Journal of the
American Statistical Association* 116(536):1716-1730.

Gavish, M., & Donoho, D. L. (2014). "The Optimal Hard Threshold for
Singular Values is 4/sqrt(3)." *IEEE Transactions on Information Theory*
60(8):5040-5053.

Ma, W., & Chen, G. H. (2019). "Missing Not at Random in Matrix Completion:
The Effectiveness of Estimating Missingness Probabilities Under a Low
Nuclear Norm Assumption." *NeurIPS*.
