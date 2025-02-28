Factor Model Approach  
=====================  

Oftentimes we assume that our truly observed data are a byproduct of a latent factor model, where there are a set of common factors across all units that influence the outcome of each unit differently, along perhaps with some set of covariates, :math:`\mathbf{Y} = \mathbf{Z} \boldsymbol{\beta} + \mathbf{F} \boldsymbol{\Lambda}^\top + \mathbf{E}`, where :math:`\mathbf{Y} \in \mathbb{R}^{T \times N}` represents the outcome matrix, with :math:`T` time periods and :math:`N` units. The covariates matrix is given by :math:`\mathbf{Z} \in \mathbb{R}^{T \times p}`, where :math:`p` is the number of covariates. The latent factor matrix is denoted by :math:`\mathbf{F} \in \mathbb{R}^{T \times r}`, where each column corresponds to a latent factor over time. The factor loadings are contained in :math:`\boldsymbol{\Lambda} \in \mathbb{R}^{N \times r}`, where each row corresponds to the factor loadings for a specific unit. Here, :math:`\boldsymbol{\beta}` is a vector of coefficients, and :math:`\mathbf{E}` denotes the error matrix. Motivated by this idea, Li and Sonnier [li2023statistical]_ argue that if we may estimate the number of factors that generate our outcomes in the pre-intervention period, we may use these factors to predict our counterfactual outcomes.

As ususal, we denote our units as indexed by :math:`j`, we observe :math:`\mathcal{N} \operatorname*{:=} \{1, 2, \ldots, N\}` units where the set of all units, :math:`\mathcal{N}`, has cardinality :math:`N = |\mathcal{N}|`. :math:`j = 1` is the treated unit with the controls being :math:`\mathcal{N}_0 \operatorname*{:=} \mathcal{N} \setminus \{1\}` whose cardinality is :math:`N_0 = |\mathcal{N}_0|`. Time periods are indexed by :math:`t`. Let :math:`\mathcal{T}_1 \operatorname*{:=} \{1, 2, \ldots, T_0\}` represent the pre-intervention periods, where :math:`T_0` is the final pre-intervention period, and :math:`\mathcal{T}_2 \operatorname*{:=} \{T_0 + 1, \ldots, T\}` represents the post-intervention periods. Both of these sets have cardinalities :math:`T_1 = |\mathcal{T}_1|` and :math:`T_2 = |\mathcal{T}_2|`. Let :math:`\mathcal{T} \operatorname*{:=} \mathcal{T}_1 \cup \mathcal{T}_2` represent the full time series, with cardinality :math:`T = |\mathcal{T}|`. Let :math:`\mathbf{y}_1 \in \mathbb{R}^T` be the vector for the treated unit and :math:`\mathbf{Y}_0 \in \mathbb{R}^{T \times N_0}` be the matrix for the control units that were unexposed.

Li and Sonnier [li2023statistical]_ advocate for using PCA upon the control group to estimate the latent factor matrix directly (as opposed to, say, Robust Synthetic Control by Amjad et. al. [Amjad2018]_ who seeks only to approximate the rank of the matrix instead of directly estimating the number of factors. Li and Sonnier [li2023statistical]_ advocate  describe two ways of selecting the number of factors. The first is a modified information criteron from Bai and Ng [BaiNg2002]_ and the second is a leave-one-out cross validation procedure by Xu [Xu2017]_.

To begin, modified factor selection by Bai and Ng basically adds a penalty term to the originally proposed method (see page 457 and 470 of [li2023statistical]_ for more). It penalizes the number of factors selected based on the number of units and time periods in the data.

**Modified Bai and Ng Procedure**

.. math::

   \text{MBN}(r) = \frac{1}{N_0 T} \left\| \mathbf{Y}_0 - \hat{\mathbf{F}}^{(r)} \hat{\boldsymbol{\Lambda}}^{(r)\top} \right\|_F^2 
   + c_{N,T} r \hat{\sigma}^2 \left( \frac{N_0 + T}{N_0 T} \right) 
   \log \left( \frac{N_0 + T}{N_0 T} \right)

Li and Sonnier [li2023statistical]_ justify this method on the basis of the original criterion selecting too many factors, thus leading to poor out of sample performance.


**Xu's Cross-Validation Procedure**

Xu's method is derivied from an iterative cross validation algorithm. It proceeds along these steps:

1. **De-Mean the Data**: Naturally, we subtract the mean of the outcome matrix across time to remove unit-fixed effects.

   .. math::

      \tilde{\mathbf{Y}}_0 = \mathbf{Y}_0 - \frac{1}{T_0} \mathbf{1}_{T_0} \mathbf{1}_{T_0}' \mathbf{Y}_0

2. **SVD**: Perform SVD upon the donor pool

.. math::

   \tilde{\mathbf{Y}}_0 = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}'

Select the first :math:`r` columns of :math:`\mathbf{U}` to form the estimated factor matrix. The corresponding rows of :math:`\mathbf{V}'` represent the factor loadings, and :math:`\boldsymbol{\Sigma}` contains the singular values. We now are concerned with the factor matrix so that way may select the number of them.


3. **Cross-Validation**: For each candidate number of factors  :math:`r`, and each pre-intervention time period :math:`s`, estimate the number of factors. Then, use OLS to predict one-step ahead out of sample into the validation period, up until the end of the training/pre-intervention period

.. math::

   r^\ast = \operatorname*{argmin}_{r \in \{1, 2, \ldots, r_{\max}\}} \sum_{s \in \mathcal{T}_1} \left( \mathbf{y}_1 - \hat{\mathbf{F}}_s^{(r)'} \hat{\boldsymbol{\lambda}}^{(r)} \right)^2


The optimal number of factors in this case is the number that minimizes the one-step out of sample validation error. Both of these methods are computed underneath the hood. ``mlsynth`` choses whichever method selects the least number of factors to avoid overfitting.

Estimating FMA in ``mlsynth``
-----------------------------


.. autoclass:: mlsynth.mlsynth.FMA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__



.. code-block:: python
   :linenos:

   from mlsynth.mlsynth import FMA
   import pandas as pd
   
   file = 'https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/HongKong.csv'
   # Load the CSV file using pandas
   df = pd.read_csv(file)
   
   treat = "Integration"
   outcome = "GDP"
   unitid = "Country"
   time = "Time"
   
   config = {
       "df": df,
       "treat": treat,
       "time": time,
       "outcome": outcome,
       "unitid": unitid,
       "display_graphs": True
   }
   
   model = FMA(config)
   
   arco = model.fit()



After we run this, we get this plot saved.

.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/examples/FMA/HK_Integration.png
   :alt: Factor Model Plot
   :align: center
   :width: 600px




To-Do List
----------------

- Staggered Adoption
