Local Principal Component Analysis (LPCA)
=========================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

``LPCA`` implements the local principal component analysis of Feng [LPCA]_. It
answers a question the rest of the factor-model family cannot: what if the
outcome responds to the latent drivers *nonlinearly*?

Every factor estimator in ``mlsynth`` -- :doc:`gsynth`, :doc:`cfm`,
:doc:`fma`, :doc:`mcnnm`, :doc:`rmsi` -- writes the untreated outcome as
:math:`y_{jt}(0) = \mathbf{f}_t^{\top}\boldsymbol{\alpha}_j + u_{jt}`. Each
unit's path is a linear combination of a few common factors, so the outcome
matrix is low rank and can be recovered by principal components or by shrinking
its singular values. Drop the linearity and that machinery stops working. If

.. math::

   y_{jt}(0) = \eta_t(\boldsymbol{\alpha}_j) + u_{jt},

with :math:`\boldsymbol{\alpha}_j` low-dimensional but :math:`\eta_t` an
unknown, possibly nonlinear function, then the matrix is generally *full* rank
even though the structure driving it has only a few dimensions. Global
principal components and nuclear-norm completion both misread such a panel.

The way through is that a smooth surface looks linear up close. Two units with
similar :math:`\boldsymbol{\alpha}` have outcomes that a first-order expansion
makes an approximately linear factor model, so principal components apply
*within a neighbourhood* even where they fail globally. Since
:math:`\boldsymbol{\alpha}_j` is never observed, the neighbourhood has to be
built by matching on the outcomes themselves.

Reach for LPCA when the panel is wide in both dimensions -- many units and many
periods -- and the response of outcomes to the latent drivers is plausibly
curved: saturating demand, threshold effects, bounded shares, anything where
doubling a latent input does not double the outcome. It is a matrix-completion
estimator in the same family as :doc:`mcnnm` and :doc:`snn`, and like them it
returns an imputation, not a donor-weight story.

Do not use LPCA when
~~~~~~~~~~~~~~~~~~~~~

* The panel is small. The neighbourhood defaults to
  :math:`K = \operatorname{round}(N^{2/3})` and half the periods are spent on
  matching, so a classic thirty-donor, thirty-period study leaves nine
  neighbours and fifteen fitting periods. Use :doc:`vanillasc`, :doc:`fdid` or
  :doc:`mcnnm` there.
* A linear factor structure is credible. LPCA pays for its generality in
  variance; when the matrix really is low rank, :doc:`gsynth` or :doc:`mcnnm`
  estimate the same object more efficiently.
* Inference is the deliverable. The paper reports no standard errors,
  confidence intervals or p-values -- see *Inference* below. Use :doc:`fma` or
  :doc:`cscipca` when a band is required.
* The post-treatment window is long. The theory assumes the number of treated
  periods is fixed while the panel grows; see Assumption 4.
* Interpretable weights are the result. Like other completion estimators, LPCA
  hands back a fitted path. Use :doc:`tssc` or :doc:`scmo`.

Notation
--------

The outcome panel is the :math:`N \times T` matrix
:math:`\mathbf{Y} = (y_{jt})` over units
:math:`j \in \mathcal{N} \coloneqq \{1, \ldots, N\}` and periods
:math:`t \in \mathcal{T} \coloneqq \{1, \ldots, T\}`. Unit 1 is treated from
:math:`T_0 + 1`, splitting :math:`\mathcal{T}` into
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` and
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`, with
:math:`T_2 \coloneqq |\mathcal{T}_2|`.

Each unit carries a latent :math:`\boldsymbol{\alpha}_j \in \mathbb{R}^r` with
:math:`r` small, and each period a latent response function :math:`\eta_t`. The
untreated outcome is :math:`y_{jt}(0) = \eta_t(\boldsymbol{\alpha}_j) + u_{jt}`
with mean-zero noise :math:`u_{jt}`.

LPCA needs a second split, this time of the periods, and it is not the
treatment split. Write
:math:`\mathcal{T} = \mathcal{M} \cup \mathcal{P}` where
:math:`\mathcal{M} \coloneqq \{1, \ldots, M\}` is the *matching block* and
:math:`\mathcal{P} \coloneqq \mathcal{T} \setminus \mathcal{M}` the *PCA
block*. Matching happens on :math:`\mathcal{M}`, fitting on :math:`\mathcal{P}`,
and the intervention must fall inside :math:`\mathcal{P}` -- that is,
:math:`M < T_0` -- so the fit has pre-treatment periods to be judged on.

The neighbourhood of unit :math:`j` is
:math:`\mathcal{K}_j \subset \mathcal{N}` with
:math:`|\mathcal{K}_j| = K`, containing :math:`j` itself.

The estimator
~~~~~~~~~~~~~

Step one, matching. On the matching block, measure how far apart two units are
by the pseudo-max distance of Zhang, Levina and Zhu [ZLZ2017]_,

.. math::

   \rho(\mathbf{y}_i, \mathbf{y}_j) \coloneqq \max_{l \neq i, j}
     \Bigl| \frac{1}{M} \sum_{t \in \mathcal{M}}
       (y_{it} - y_{jt})\, y_{lt} \Bigr| ,

and let :math:`\mathcal{K}_j` collect the :math:`K` smallest. Averaging across
the other units is what does the work: the idiosyncratic noise washes out as
:math:`M` grows, so the distance reflects the noise-free structure
:math:`\eta_t(\boldsymbol{\alpha})` and units close in it are close in
:math:`\boldsymbol{\alpha}`. The Euclidean distance does not have this property
under heteroskedastic errors, which is why it is not the default.

Step two, local principal components. Stack the neighbourhood's rows over the
PCA block into :math:`\mathbf{Y}_{\langle j \rangle} \in \mathbb{R}^{K \times |\mathcal{P}|}`
and take its rank-:math:`d` truncated singular value decomposition,

.. math::

   (\widehat{\mathbf{F}}_{\langle j \rangle},
    \widehat{\boldsymbol{\Lambda}}_{\langle j \rangle})
     \coloneqq \operatorname*{argmin}_{\mathbf{F}, \boldsymbol{\Lambda}}
       \bigl\| \mathbf{Y}_{\langle j \rangle}
         - \mathbf{F} \boldsymbol{\Lambda}^{\top} \bigr\|_F^2 ,

and read the counterfactual off unit :math:`j`'s row of
:math:`\widehat{\mathbf{F}}_{\langle j \rangle} \widehat{\boldsymbol{\Lambda}}_{\langle j \rangle}^{\top}`.
The treatment effect is
:math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}(0)` for
:math:`t \in \mathcal{T}_2` and the ATT is
:math:`\widehat{\tau} \coloneqq T_2^{-1} \sum_{t \in \mathcal{T}_2} \tau_t`.

Assumptions and remarks
~~~~~~~~~~~~~~~~~~~~~~~~

*Assumption 1 (smooth low-dimensional latent structure).* The latent variables
:math:`\boldsymbol{\alpha}_j` are i.i.d. on a compact support and each
:math:`\eta_t` is smooth, with bounded derivatives up to some order (paper
Assumption 2.1). *Remark.* Smoothness is what makes the tangent-plane
approximation good; the dimension :math:`r` being small is what makes
neighbours exist. Neither requires the matrix to be low rank, which is the
whole point.

*Assumption 2 (informative matching).* The chosen distance denoises the data,
and distances in the noise-free structure translate into distances in
:math:`\boldsymbol{\alpha}` (paper Assumption 4.1). *Remark.* Two things can
break here and they are different. If the panel is short, averaging does not
kill the noise and the neighbours are wrong. If two distinct
:math:`\boldsymbol{\alpha}` values generate near-identical outcome paths, the
observables are uninformative and no distance recovers them. The second is a
completeness condition, not a sample-size problem.

*Assumption 3 (independent blocks).* Matching and fitting use disjoint periods,
and the noise is independent (or weakly dependent) across periods (paper
Section 3). *Remark.* This is the reason for the split, and it is not
cosmetic. Choosing neighbours uses the realised noise; fitting on those same
periods would correlate the estimated factors with the errors and standard PCA
would no longer apply.

*Assumption 4 (short treated window).* The number of treated periods is fixed
as the panel grows (paper Theorem 6.1). *Remark.* The treated cells are set to
their period mean before fitting, and the theorem bounds the damage that does.
The bound is only meaningful when few cells are affected: Feng's Kansas
application zeroes 16 cells of 104. A long post-period puts the treated unit's
own imputed values into the decomposition that is supposed to predict them.

Preprocessing is yours to do
----------------------------

The estimator takes the outcome column as given. Feng's application works with
first-differenced log GDP per capita, not levels, and that transformation is
the user's to apply -- transform the column before passing it in. The
counterfactual comes back on the scale of whatever ``outcome`` names.

Internally each period is centred across units before matching, and the period
means are added back to the counterfactual. That round trip has to close: a
counterfactual left in centred space is offset from the observed series by the
average of the period means, which is a difference of the same order as the
effect being measured. It is pinned by a test.

Inference
---------

There is none. That is the paper's position, not an omission here.
Theorem 6.1 gives a uniform max-norm convergence rate for the fitted surface;
Section 6.1 reports point predictions, and its figures carry no bands. ``LPCA``
therefore returns an empty ``inference`` slot and ``res.att_ci`` is ``None``.

Four diagnostics stand in for it, all on the result. ``neighbourhood_size`` is
the realised neighbourhood, which can exceed the requested :math:`K` because
the selection rule is a threshold and ties are all kept -- discrete or binary
panels tie routinely. ``local_rank`` is the number of components the
singular-value ratio rule retained.

``neighbour_weights`` gives the weight each neighbour receives. The
reconstruction is literally that weighted combination of the neighbourhood's
rows: the weights are a column of the rank-:math:`d` projector
:math:`\mathbf{U}_d \mathbf{U}_d^{\top}`, so unlike a synthetic control's they
neither sum to one nor stay non-negative. They are still readable as a
comparison set.

``self_weight`` is the one to check. The treated unit belongs to its own
neighbourhood, so its own row enters the decomposition that produces its
counterfactual -- and for the post-treatment periods that row holds the values
the estimator overwrote with the period mean. This number says how much the
counterfactual leans on them, and it is the practical face of Assumption 4. It
lies in :math:`[0, 1]`, and across a neighbourhood the self-weights sum to the
rank, so :math:`d / K` is the natural benchmark. Well above that, on a long
post-period, is the configuration Theorem 6.1 stops covering.

.. warning::

   The rank rule is inert for :math:`K \le 15`. It keeps components while
   consecutive singular values satisfy
   :math:`\sigma_i / \sigma_{i+1} < \log \log K`. Ratios of a descending
   spectrum are at least 1, and :math:`\log \log K < 1` for
   :math:`K \le 15` (the threshold crosses 1 at :math:`e^e \approx 15.15`), so
   below sixteen neighbours the comparison never fires and the rank is pinned
   at ``max_components - 1`` whatever the data says. Feng's Kansas application
   uses :math:`K = 14`, so its rank of 2 is mechanical. Check
   ``res.metadata["rank_rule_active"]``. Paper Remark 4.3 leaves formal rank
   selection to future research.

The reported window
-------------------

Local PCA predicts only the periods held out from matching, so
``res.counterfactual`` and ``res.time_series`` cover :math:`\mathcal{P}`, not
all of :math:`\mathcal{T}`. The first :math:`M` periods are dropped from the
reported series instead of being padded, and ``time_series.time_periods``
carries the labels that remain. ``pre_rmse`` is therefore computed on
:math:`\mathcal{P} \cap \mathcal{T}_1`.

Example
-------

The 2012 Kansas tax cuts. Feng analyses quarterly log GDP per capita growth for
the 50 states, with Kansas treated from 2012Q2, so the outcome is
first-differenced before it reaches the estimator.

.. code-block:: python

   import pandas as pd
   from mlsynth import LPCA

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "refs/heads/main/basedata/kansas_taxcut.csv")
   df = pd.read_csv(url).sort_values(["state", "year_qtr"])

   # The paper works in growth rates, in percent.
   df["growth"] = df.groupby("state")["lngdpcapita"].diff() * 100.0
   df = df.dropna(subset=["growth"])

   res = LPCA({
       "df": df, "outcome": "growth", "treat": "treated",
       "unitid": "state", "time": "year_qtr",
       "match_periods": 40,        # the paper's split
       "display_graphs": True,
   }).fit()

   print(f"ATT (2012Q2-2016Q1) = {res.att:+.4f} pp")
   print(f"K = {res.n_neighbours}, realised = {res.neighbourhood_size}")
   print(f"local rank = {res.local_rank} "
         f"(rule active: {res.metadata['rank_rule_active']})")
   print(f"neighbours: {', '.join(res.neighbours)}")
   print(f"self weight = {res.self_weight:.3f} "
         f"(benchmark {res.local_rank / res.n_neighbours:.3f})")

The counterfactual sits about 0.53 points above observed Kansas growth: the
paper's estimate that the tax cut cost growth.

Verification
------------

.. note::

   Empirical (Kansas). The ATT reproduces Feng's Section 6.1 to four decimals
   (:math:`-0.5306` against the reported :math:`-0.53`), and the observed
   series falls below the LPCA path in 9 of 16 post-treatment quarters, as
   reported. Pinned in ``mlsynth/tests/test_lpca.py``.

   Monte Carlo. All 48 cells of the paper's Table 1 reproduce at 500
   replications, median disagreement 0.83 Monte Carlo standard errors, with
   local PCA beating global PCA on the two nonlinear designs and the advantage
   widening with the severity of the nonlinearity.

   Both live in ``benchmarks/reference/lpca_kansas/``, which also records that
   the synthetic-control comparison in the paper's November 2023 version was a
   centring defect the author corrected in July 2024. The docs above quote the
   corrected version.

Core API
--------

.. automodule:: mlsynth.estimators.lpca
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.LPCAConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``LPCA.fit()`` returns a
:class:`~mlsynth.utils.lpca_helpers.structures.LPCAResults`: the ATT, the
counterfactual path over the PCA block, the matched ``neighbours`` and their
realised count, the ``neighbour_weights`` and the treated unit's
``self_weight``, the ``local_rank`` and the local ``singular_values``, the
``period_means`` removed and restored, and the standardized sub-models.

.. autoclass:: mlsynth.utils.lpca_helpers.structures.LPCAResults
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

The two numerical steps: the matching distances, the neighbourhood rule, the
rank rule, and the local reconstruction.

.. automodule:: mlsynth.utils.lpca_helpers.core
   :members:
   :undoc-members:

Data preparation -- the DataFrame touchpoint: pivots to the outcome matrix,
masks the treated cells, and enforces the block split.

.. automodule:: mlsynth.utils.lpca_helpers.setup
   :members:
   :undoc-members:

Run loop: centring, matching, the local fits, and the re-centred counterfactual.

.. automodule:: mlsynth.utils.lpca_helpers.pipeline
   :members:
   :undoc-members:
