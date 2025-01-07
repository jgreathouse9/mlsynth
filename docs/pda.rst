PDA Explained
==================

Many recent developments in the causal inference literature focus on developing some linear/convex combination of untreated units to approximate the pre-intervention values of a treated unit, such  that we may learn their our of sample counterfactuals had the unit never been treated. However, as with any causal study, the results we get depend on how we choose or give weight to units in the control group.  The basic idea od the panel data approach, essentially, is that we may use regression based methods such as OLS, LASSO, or other unconstrained regression methods to construct the counterfactual using (mainly) pre-intervention data.   ``mlsynth`` implements three such implementations: the LASSO method by [LASSOPDA]_, the forward selection method by [fsPDA]_, and the :math:`\ell_2`-relaxation method by [l2relax]_. As the names suggest, all methods exploit the pre-intervention correlations between the units across time in order to predict the counterfactual for a treated unit, and their main difference is in how they select or weigh the control units.

Formally, I lay out the notations. Indexed by :math:`j`, we observe :math:`\mathcal{N} \operatorname*{:=} \{1, 2, \ldots, N\}` units where the set :math:`\mathcal{N}` has cardinality :math:`N = |\mathcal{N}|`. :math:`j = 1` is the treated unit with the controls being :math:`\mathcal{N}_0 \operatorname*{:=} \mathcal{N} \setminus \{1\}` whose cardinality is :math:`N_0 = |\mathcal{N}_0|`. Time periods are indexed by :math:`t`. Let :math:`\mathcal{T}_1 \operatorname*{:=} \{1, 2, \ldots, T_0\}` represent the pre-intervention periods, where :math:`T_0` is the final pre-intervention period, and :math:`\mathcal{T}_2 \operatorname*{:=} \{T_0 + 1, \ldots, T\}` represents the post-intervention periods. Both of these sets have cardinalities :math:`T_1 = |\mathcal{T}_1|` and :math:`T_2 = |\mathcal{T}_2|`. Let :math:`\mathcal{T} \operatorname*{:=} \mathcal{T}_1 \cup \mathcal{T}_2` represent the full time series, with cardinality :math:`T = |\mathcal{T}|`. 

Let :math:`\mathbf{y}_1 \in \mathbb{R}^T` be the vector for the treated unit and :math:`\mathbf{Y}_0 \in \mathbb{R}^{T \times N_0}` be the matrix for the control units that were unexposed. Furthermore, let :math:`\widehat{\boldsymbol{\beta}} \in \mathbb{R}^{N_0}` be the coefficients for some controls.


:math:`\ell_2` approach
-----------------------

.. math::

   \begin{aligned}
   &\min_{\boldsymbol{\beta}} \quad & \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 \\
   &\text{subject to} \quad & \|\boldsymbol{\eta} - \boldsymbol{\Sigma} \boldsymbol{\beta}\|_\infty \leq \tau \\
   &\boldsymbol{\eta} = \frac{1}{T_1} \mathbf{Y}_0^\top \mathbf{y}_1 \\
   &\boldsymbol{\Sigma} = \frac{1}{T_1} \mathbf{Y}_0^\top \mathbf{Y}_0 \in \mathbb{R}^{N_0 \times N_0}
   \end{aligned}

Forward Selected Approach
-------------------------

LASSO Approach
--------------
