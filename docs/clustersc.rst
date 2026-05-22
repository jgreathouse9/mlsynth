Cluster Synthetic Controls
===========================


With Synthetic Control Methods, we may be unsure if the control group we use will be a good countrefactual for the treated unit-- even if we are relying on a weighted average of the controls instead of using all of them. Outcome trajectories may be noisy, have missing data, or have an extremely large number of control units, epsecially in modern settings where data are frequently disaggregated. To address this, [Amjad2018]_ developed a SVD version of the synthetic control method, Robust Synthetic Control. This is what the :class:`CLUSTERSC` class implements, and more. I begin with notations.

Notations
----------------

Here, we have :math:`\mathcal{N} \operatorname*{:=} \lbrace{1 \ldots N \rbrace}` units across 
:math:`t \in \left(1, T\right) \cap \mathbb{N}` time periods, where :math:`j=1` is our sole treated unit. 
This leaves us with :math:`\mathcal{N}_0 \operatorname*{:=} \lbrace{2 \ldots N \rbrace}` control units, 
with its cardinality being :math:`|N_0|`. We have two sets of time series 
:math:`\mathcal{T} \operatorname*{:=}\mathcal{T}_1 \cup \mathcal{T}_2`, where 
:math:`\mathcal{T}_1 \operatorname*{:=} \lbrace{1 \ldots T_0 \rbrace}` is the pre-intervention period and 
:math:`\mathcal{T}_2 \operatorname*{:=}\lbrace{T_0+1 \ldots T \rbrace}` denotes the post-intervention period, 
each with their respective cardinalities :math:`|T_1|` and :math:`|T_2|`. 

We represent the outcomes of all units over time as matrices and vectors. Let :math:`\mathbf{Y} \in \mathbb{R}^{N \times T}` denote the outcome matrix for all units across all time periods, where each row corresponds to a unit and each column corresponds to a time period. Specifically, let: :math:`\mathbf{Y}_{\mathcal{N}_0} \in \mathbb{R}^{(N-1) \times T}` denote the outcome matrix for the control units, :math:`\mathbf{y}_1 \in \mathbb{R}^{1 \times T}` denote the outcome vector for the treated unit.

Let :math:`\mathbf{w} \operatorname*{:=}\lbrace{w_2 \ldots w_N \rbrace}` 
be a weight vector learnt by regression. The basic problem with causal inference is that we see our units as being treated or untreated, never both.

.. math::
    y_{jt} = 
    \begin{cases}
        y^{0}_{jt} & \forall \: j \in \mathcal{N}_0 \\
        y^{0}_{1t} & \text{if } j = 1 \text{ and } t \in \mathcal{T}_1 \\
        y^{1}_{1t} & \text{if } j = 1 \text{ and } t \in \mathcal{T}_2
    \end{cases}

In the synthetic control method, we (typically) have a single treated unit which, along with the donors, follows a certain data generating process for all time periods until :math:`T_0`. 
After the final pre-treatment period, the control units follow the same process because they are unaffected by the intervention. However for unit :math:`j=1`, the outcomes we see are that pre-intervention DGP plus some treatment effect. To this end, we are concerned with :math:`\hat{y}_{1t}`, or the out of sample values we would have observed for the treated unit absent treatment. The average treatment effect on the treated

.. math::
    ATT = \frac{1}{T_2 - T_1} \sum_{T_0 +1}^{T} (y_{1t} - \hat{y}_{1t})

is our main statistic of interest, where :math:`(y_{1t} - \hat{y}_{1t})` is the treatment effect at some given time point. 

Estimation
-----------

PCR
~~~~~~~~~~~

In SCM, we exploit the linear relation 
between untreated and the treated unit to estimate its counterfactual. Typically, this is done like

.. math::
    \begin{align}
        \underset{w}{\operatorname*{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{Y}_{\mathcal{N}_{0}} w_j||_{2}^2 \\
        \text{s.t.} & \quad \mathbf{w}: w_{j} \in \mathbb{I}, \quad  \|\mathbf{w}\|_{1} = 1
    \end{align}
where the weights are constrained to lie on the unit interval and add up to 1, which we refer to as the *convex hull constraint*. Geometrically, and practically, this has some good properties; it means that we will never extrapolate beyond the support of the control group and allows for an interpretable solution. However, it also suffers from computational issues stemming from the (often) bilevel optimization ([BECKER20181]_ , [albalate2021decoupling]_, [malo2023computing]_). Instead, [Amjad2018]_ proposes a different solution, using low-rank matrix  techniques. Given the donor pool outcome matrix :math:`\mathbf{Y}_{\mathcal{N}_0} \in \mathbb{R}^{(N-1) \times T}`, we seek a low-rank approximation :math:`\widehat{\mathbf{Y}}_{\mathcal{N}_0} = \mathbf{U} \mathbf{S} \mathbf{V}^\top` of our control group. Here, :math:`\mathbf{U} \in \mathbb{R}^{(N-1) \times k}`, :math:`\mathbf{S} \in \mathbb{R}^{k \times k}`, and :math:`\mathbf{V} \in \mathbb{R}^{T \times k}` with rank :math:`k \ll \min(N-1, T)`. We learn this low-rank approximation, :math:`\mathbf{L}`,  by minimizing the reconstruction error *in the preintervention period*. Here is the objective function

.. math::
   \mathbf{L}=\underset{\mathbf{U}, \mathbf{S}, \mathbf{V}}{\text{argmin}} \quad \|\mathbf{Y}_{\mathcal{N}_0} - \mathbf{U} \mathbf{S} \mathbf{V}^\top\|_F^2

where :math:`\|\cdot\|_F` denotes the Frobenius norm. When we do this, we are left with the singular values, and how much of the total variance they explain. Selecting too few singular values/principal components means our synthetic control will underfit the pre-intervention time series of the treated unit. Selecting too many singular values means we will overft the pre-intervention time series. The benefits of this approach, as [Amjad2018]_ and [Agarwal2021]_ show, is that it implicitly performs regularization our donor pool, and has a denoising effect. PCR (also called Robust Synthetic Control) uses Universal Singular Value Thresholding to choose the optimal numver of principal componenets to retain [Chatterjee2015]_. We take the same approach, but other methods, such as SCREENOT by Donoho et al. [Donoho2023]_, may at some point be applied as an option. The final objective function is

.. math::
   \begin{align}
       \underset{w}{\text{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{L}\mathbf{w}^{\top}||_{2}^2 \\
       \text{s.t.} \: & \mathbf{w} \in \mathbb{R}
   \end{align}

where we simply use the reconstructed, denoised version of the control group to learn the values of the treated unit in the preintervention period via OLS. Then, we take the dot product of our control group using the learnt weights. This returns our in and out of sample predictions so that we may compute ATTs. Note as of January 11, the Bayesian version of this method is also an option, when the ``Frequentist`` option is set to False (for both the clustered and unclustered RSCs). The mathematical details of the Bayesian RSC/PCR will be laid out here, soon.

Robust PCA SYNTH
~~~~~~~~~~~~~~~~~

The next method :class:`CLUSTERSC` implements is the Robust PCA SC method by [Bayani2021]_. [Bayani2021]_ argues that the PCR/Robust Synthetic Control method described above is senstive to gross data corruptions and noisy outcomes. Furthermore, [Amjad2018]_ notes that even with the denoisign procedure, there still typically needs to be an expert in the field to determine an appropriate donor pool. To solve this issue, Robust PCA SYNTH  begins with a donor selection step. [Bayani2021]_ advocates for applying functional PCA to the fully observed outcome matrix in the pre-intervention period and applying k-means clustering. Given the set of outcome trajectories for all units during the pre-intervention period, denote the outcome matrix as :math:`\mathbf{Y} \in \mathbb{R}^{N \times T_0}`, where :math:`N` is the number of units and :math:`T_0` is the length of the pre-intervention period. Each unit's trajectory, :math:`\mathbf{y}_j(t)` for :math:`j \in \{1, \ldots, N\}`, can be modeled as a smooth function :math:`f_j(t)` by projecting onto a set of functional principal components:

.. math::
   f_j(t) \approx \mu(t) + \sum_{k=1}^{K} x_{jk} \phi_k(t),

where :math:`\mu(t)` is the mean function, :math:`\phi_k(t)` are the eigenfunctions, :math:`x_{jk}` are the corresponding functional principal component scores for unit :math:`j`. After, we can either apply SCREENOT as described above to provide us with the number of functional PC scores to use, or use the elbow method to select the number of scores that explain at least 90% of the preintervention data. At present, we use the latter method.

After retaining the top few FPC scores, :class:`CLUSTERSC` uses the k-means algorithm to group units with similar temporal patterns during the pre-intervention period. The idea is that units in the same cluster based on their low dimensional representation will be superior controls to these that are not. [Bayani2021]_ advocates the silhouette score to choose the number of clustes For each unit :math:`j`, the silhouette score :math:`s(j)` is defined as :math:`s(j) = \frac{b(j) - a(j)}{\max\{a(j), b(j)\}}`, where :math:`a(j)` is the average distance from unit :math:`j` to all other units in its own cluster and :math:`b(j)` is the minimum average distance from unit :math:`j` to units in any other cluster (the closest neighboring cluster). To determine the best number of clusters, we compute the silhouette coefficient for each potential number of clusters :math:`K`. The silhouette coefficient, denoted by :math:`SC(K)`, is the average silhouette score across all units in the dataset for a given value of :math:`K`: :math:`SC(K) = \frac{1}{N} \sum_{j=1}^{N} s(j)`. We then select the number of clusters :math:`K^\ast` that maximizes the silhouette coefficient:

.. math::
   K^\ast = \underset{K}{\mathrm{argmax}} \; SC(K).

Now that we know the number of clusters, we now can apply the k-means method. The objective of K-Means clustering is to partition the units into :math:`K` clusters by minimizing the within-cluster variance:

.. math::
   \underset{\mathcal{C}}{\mathrm{argmin}} \sum_{k=1}^{K} \sum_{j \in \mathcal{C}_k} \|\boldsymbol{x}_j - \mathbf{c}_k\|_2^2,

where :math:`\mathcal{C} = \{\mathcal{C}_1, \ldots, \mathcal{C}_K\}` is the set of clusters and :math:`\mathbf{c}_k` is the centroid of cluster :math:`k`. The cluster containing the treated unit is selected as the donor pool for constructing the synthetic control

.. math::

    \mathbf{1}\left( j \in \mathcal{C}_k \right) =
    \begin{cases}
    1, & \text{if } 1 \in \mathcal{C}_k \\
    0, & \text{otherwise}
    \end{cases}

With this brand new donor pool, we now can get to the estimation of the synthetic control weights, which we do via Robust PCA.


Robust PCA is advocated by [Bayani2021]_ because it is less sensitive to noise, corruption in the data, and outliers than standard PCA. Robust PCA asks the user to accept the very simple premise that the observed outcomes are byproducts of a low-rank structure with occasional/sparse outliers, :math:`\mathbf{L} + \mathbf{S}`, where both matrices respectively are of :math:`N \times T` dimensions. As before with PCR/Robust SC, if we can extract this low-rank component for our donor pool, we can use it to learn which combination of donors matters most for the construction of our counterfactual. This problem is written as:

.. math::

   \begin{align*}
   &\mathop {{\mathrm{minimize}}}\limits _{{\mathbf{L}},{\mathbf{S}}} ~{\mathrm{rank}}({\mathbf{L}}) + \lambda {\left \|{ {\mathbf{S}} }\right \|_{0}} \\
   &\textrm {subject to } ~~{\mathbf{Y}} = {\mathbf{L}} + {\mathbf{S}},
   \end{align*}

However, this program is NP-hard due to the rank portion of the objective function. Instead, we use the nuclear norm and :math:`\ell_1` norm on the low-rank matrix and sparse matrix, respectively:

.. math::

   \begin{align*}
   &\mathop {{\mathrm{minimize}}}\limits _{{\mathbf{L}},{\mathbf{S}}} ~{\left \|{ {\mathbf{L}} }\right \|_{*}} + \lambda {\left \|{ {\mathbf{S}} }\right \|_{1}} \\
   &\textrm {subject to } ~~{\mathbf{Y}} = {\mathbf{L}} + {\mathbf{S}},
   \end{align*}

This is done via taking the augmented Lagrangian, solved with proximal gradient descent

.. math::

   \begin{aligned}
   \mathbf{L}_{k+1} &= \mathrm{SVT}_{1/\rho}\left(\mathbf{X} - \mathbf{S}_{k} + \frac{1}{\rho} \mathbf{Y}_{k}\right) \\
   \mathbf{S}_{k+1} &= \mathcal{S}_{\lambda/\rho}\left(\mathbf{X} - \mathbf{L}_{k+1} + \frac{1}{\rho} \mathbf{Y}_{k}\right) \\
   \mathbf{Y}_{k+1} &= \mathbf{Y}_{k} + \rho\left(\mathbf{X} - \mathbf{L}_{k+1} - \mathbf{S}^{k+1}\right)
   \end{aligned}


In the above, all this means is that we iteratively estimate the rank of the donor matrix via the SVT operator, and we use the :math:`\ell_1` norm to extract to the noise component, and the :math:`\rho` (the proximal gradient operator) encourages updates. With this low-rank structure, we estimate our weights by solving the following optimization problem:

.. math::

   \begin{align}
       \underset{w}{\text{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{L} \mathbf{w}^{\top}||_{2}^2 \\
       \text{s.t.} \: & \mathbf{w}: w_{j} \in \mathbb{R}_{\geq 0}
   \end{align}

our weights are here onlt constrained to be positive, since the donor selection provess attempts to ensure that the donors are already more similar to the treated unit.

Estimating ``CLUSTERSC``
------------------------

.. autoclass:: mlsynth.CLUSTERSC
   :show-inheritance:
   :members:
   :undoc-members:
   :private-members:
   :special-members: __init__

PCR variants
~~~~~~~~~~~~

The PCR family is implemented in
``mlsynth/utils/clustersc_helpers/pcr/`` as a subpackage, organised by
algorithmic role:

* ``hsvt.py`` -- Hard Singular Value Thresholding (Algorithm 2 Step 2 of
  Rho et al. 2025). Three rank-selection rules are exposed:
  ``"cumvar"`` (paper default, smallest :math:`r` with cumulative
  spectral energy at least ``cumvar_threshold``), ``"fixed"`` (caller
  supplies ``rank``), and ``"usvt"`` (Chatterjee 2015 / Donoho-Gavish
  universal threshold).
* ``clustering.py`` -- Algorithm 3 (silhouette-driven :math:`k`-means on
  rows of :math:`\\widetilde{U} = U\\Sigma_r`) plus Algorithm 4 Step 2
  target matching via :math:`\\tilde{u} = V_r^\\top x_0^-`.
* ``frequentist.py`` -- the paper's OLS weight solver
  :math:`\\widehat{f} = \\arg\\min_f \\| \\widetilde{M}^- f - x_0^- \\|_2`
  (Algorithm 2 Step 3), with optional elastic-net regularisation
  (Appendix E).
* ``bayesian.py`` -- mlsynth extension: replace OLS with the Bayani
  (2022) Gaussian posterior over the weight vector, propagate samples
  through the HSVT-denoised donor matrix in both pre and post periods
  (Algorithm 4 Step 5) to produce per-period credible bands.
* ``convex.py`` -- mlsynth extension: keep HSVT denoising but swap OLS
  for the classical Abadie et al. (2010) simplex-constrained program.
* ``pipeline.py`` -- the public dispatcher :func:`run_pcr` composing
  the steps above into Algorithms 2 and 4.

Configuration
~~~~~~~~~~~~~

The modernised :class:`mlsynth.config_models.CLUSTERSCConfig` exposes
the paper-aligned knobs below.

PCR family
^^^^^^^^^^

* ``method``: ``"pcr"``, ``"rpca"``, or ``"both"`` (case-insensitive).
* ``primary``: when ``method="both"``, selects which family drives the
  convenience aliases (``att``, ``counterfactual``, ``gap``,
  ``donor_weights``). Default ``"pcr"``.
* ``pcr_objective``: ``"OLS"`` (paper's Algorithm 2) or ``"SIMPLEX"``
  (the convex mlsynth extension). Legacy alias: ``objective``.
* ``estimator``: ``"frequentist"`` (Algorithm 2) or ``"bayesian"``
  (Bayani 2022 posterior, Ch. 1). Legacy alias: ``Frequentist``
  (bool).
* ``clustering``: enable Algorithm 4 donor clustering + target
  matching. Legacy alias: ``cluster``.
* ``rank``: explicit HSVT truncation rank :math:`r`. ``None`` (default)
  defers to ``rank_method``.
* ``rank_method``: ``"cumvar"`` (default; paper Section 6.1's 95%
  rule), ``"fixed"`` (use ``rank``), or ``"usvt"``.
* ``cumvar_threshold``: cumulative-variance target when
  ``rank_method="cumvar"``. Default ``0.95``.
* ``k_clusters``: number of clusters for Algorithm 3. ``None`` lets
  the silhouette coefficient pick :math:`k \\in [2, k_{\\max}]`.
* ``k_max``: upper bound for the silhouette search. Default ``8``.
* ``alpha``: nominal level for the Bayesian credible band
  (default ``0.05``).
* ``n_bayes_samples``: posterior sample count for the Bayesian path.
  Default ``1000``.
* ``random_state``: seed forwarded to k-means and the Bayesian sampler.
  Default ``0``.
* ``lambda_penalty``, ``p``, ``q``: optional elastic-net knobs for the
  frequentist OLS path (Appendix E of Rho et al. 2025).

RPCA family
^^^^^^^^^^^

* ``rpca_method``: ``"PCP"`` (Candes et al. 2011) or ``"HQF"`` (Wang
  et al. 2023). Legacy alias: ``ROB``.

Self-contained Monte Carlo example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A copy-paste, dependency-free synthetic-panel illustration. We draw a
two-factor panel, plant an ATT of ``1.0`` on the treated unit's
post-period, fit CLUSTERSC under ``method="both"``, and inspect the
PCR and RPCA fits side by side via the frozen results container.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mlsynth import CLUSTERSC

    rng = np.random.default_rng(0)
    J, T_pre, T_post, r = 12, 14, 6, 2
    T = T_pre + T_post
    F = rng.standard_normal((T, r))
    lam = rng.standard_normal((J + 1, r))
    eps = rng.standard_normal((T, J + 1)) * 0.4
    Y = F @ lam.T + eps
    Y[T_pre:, 0] += 1.0  # planted ATT on unit 0

    df = pd.DataFrame(
        [{"unit": j, "time": t, "y": float(Y[t, j]),
          "D": int(j == 0 and t >= T_pre)}
         for j in range(J + 1) for t in range(T)]
    )

    res = CLUSTERSC({
        "df": df, "outcome": "y", "treat": "D",
        "unitid": "unit", "time": "time",
        "method": "both", "primary": "pcr",
        "pcr_objective": "OLS", "estimator": "bayesian",
        "rpca_method": "HQF", "alpha": 0.10,
    }).fit()

    print(f"selected family : {res.selected_variant}")
    print(f"primary ATT     : {res.att:+.3f}")
    print(f"PCR  ATT        : {res.pcr.att:+.3f}")
    print(f"RPCA ATT        : {res.rpca.att:+.3f}")
    print(f"PCR  pre-RMSE   : {res.pcr.pre_rmse:.3f}")
    print(f"RPCA pre-RMSE   : {res.rpca.pre_rmse:.3f}")
    if res.inference.method == "bayesian_credible":
        lo, hi = res.inference.credible_interval
        print(f"Bayesian {int((1 - 0.10) * 100)}% CrI : [{lo:+.3f}, {hi:+.3f}]")

The PCR fit's posterior credible interval is exposed via
``res.inference.credible_interval`` only when ``estimator="bayesian"``;
otherwise ``res.inference.method == "none"``.

Empirical example: German reunification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from mlsynth import CLUSTERSC
    import pandas as pd

    file = (
        "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
        "refs/heads/main/basedata/german_reunification.csv"
    )
    df = pd.read_csv(file)

    res = CLUSTERSC({
        "df": df,
        "outcome": "gdp",
        "treat": "Reunification",
        "unitid": "country",
        "time": "year",
        "method": "both",
        "primary": "rpca",
        "display_graphs": True,
    }).fit()






.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/examples/clustersc/German.png
   :alt: CLUSTERSC Plot
   :align: center
   :width: 600px


Here we plot the West Germany Synthetic Control predictions. Here are the weights as produced by RPCA-SC

.. list-table:: Weights
   :header-rows: 1

   * - Country
     - Value
   * - UK
     - 0.0
   * - Austria
     - 0.023
   * - Belgium
     - 0.0
   * - Denmark
     - 0.0
   * - France
     - 0.354
   * - Italy
     - 0.0
   * - Netherlands
     - 0.0
   * - Norway
     - 0.485
   * - Japan
     - 0.0
   * - Australia
     - 0.0
   * - New Zealand
     - 0.296


These are the same results Mani gets in his dissertation. The Root Mean Squared Error for RPCA-SC is 88.60, which is not quite as tight as the original SCM, but better than the RSC (98.69).


To-Do List
----------------

- Implement Bayesian Inference from Amjad's Paper
- Implement Cross Validation to choose lambda for ADMM
- Maybe, allow users to choose between USVT and Spectral Rank
