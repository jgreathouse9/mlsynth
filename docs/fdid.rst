Forward Difference-in-Differences
==================================

With Difference-in-Differences (DID) designs, we may be unsure if the parallel trends assumption holds. Under parallel-trends, we posit the average difference between the treated unit and control group would be constant if the treatment did not happen, holds. In practice, parallel trends may not hold due to poor controls, "poor" in the sense that the pre-intervention outcome trends for some controls may be too dissimilar to the treated unit of interest.

Formally, I lay out the notations. Indexed by :math:`j`, we observe :math:`\mathcal{N} \operatorname*{:=} \{1, 2, \ldots, N\}` units where the set :math:`\mathcal{N}` has cardinality :math:`N = |\mathcal{N}|`. :math:`j = 1` is the treated unit with the controls being :math:`\mathcal{N}_0 \operatorname*{:=} \mathcal{N} \setminus \{1\}` whose cardinality is :math:`N_0 = |\mathcal{N}_0|`. Let :math:`\widehat{U} \subset \mathcal{N}_0` be a subset of controls, with cardinality :math:`U = |\widehat{U}|`. Time periods are indexed by :math:`t`. Let :math:`\mathcal{T}_1 \operatorname*{:=} \{1, 2, \ldots, T_0\}` represent the pre-intervention periods, where :math:`T_0` is the final pre-intervention period, and :math:`\mathcal{T}_2 \operatorname*{:=} \{T_0 + 1, \ldots, T\}` represents the post-intervention periods. Both of these sets have cardinalities :math:`T_1 = |\mathcal{T}_1|` and :math:`T_2 = |\mathcal{T}_2|`. Let :math:`\mathcal{T} \operatorname*{:=} \mathcal{T}_1 \cup \mathcal{T}_2` represent the full time series, with cardinality :math:`T = |\mathcal{T}|`.


With this in mind, we may describe the algorithm. By design, we are agnostic as to which sub-matrix of units we should use or how even many controls we should use. The Forward DID method is an iterative, data driven algorithm which chooses the controls. The Forward DID method updates the selected control set :math:`\widehat{U}_k` iteratively over :math:`k = 1, 2, \ldots, N_0`. The process begins with an empty control set :math:`\widehat{U}_0 = \emptyset` and an empty candidate set :math:`\mathcal{U} = \emptyset`. For :math:`k = 1`, we estimate :math:`N_0` DID models using the outcome vector of each control unit as a predictor, computing the :math:`R^2` statistic for each control unit. The optimization is


.. math::
    \operatorname*{argmin}_{\boldsymbol{\beta}_{\widehat{U}_{k-1} \cup \{i\}}} 
    \left\| 
        \mathbf{y}_1 - \mathbf{Y}_{\widehat{U}_{k-1} \cup \{i\}} \, 
        \boldsymbol{\beta}_{\widehat{U}_{k-1} \cup \{i\}} 
        - \beta_{\text{cons}} 
    \right\|_2^2,
    \quad 
    \text{subject to } 
    \boldsymbol{\beta}_{\widehat{U}_{k-1} \cup \{i\}} = \frac{1}{|\widehat{U}_{k-1}| + 1}.




where we have a :math:`T_1 \times 1` pre-intervention vector of outcomes for the treated unit, :math:`\mathbf{y}_1`, and a :math:`T_1 \times N_0` matrix of outcomes for our control group, :math:`\mathbf{Y}_0` and an intercept :math`\beta}_{\text{cons}`. After iterating over all controls, we then select the control unit that maximizes the pre-intervention :math:`R^2`:  

.. math::

    i^\ast_1 = \operatorname*{argmax}_{i \in \mathcal{N}_0} R^2_i, 
    \quad \widehat{U}_1 = \{i^\ast_1\}, 
    \quad \mathcal{U} = \{\widehat{U}_1\}.

For subsequent iterations :math:`k = 2, 3, \ldots, N_0`, we construct :math:`N_0 - k + 1` DID models by combining the previously selected set :math:`\widehat{U}_{k-1}` with each remaining control unit :math:`i \notin \widehat{U}_{k-1}`. For each k-th iteration, we select the submodel/control unit with the highest :math:`R^2`:

.. math::

    i^\ast_k = \operatorname*{argmax}_{i \in \mathcal{N}_0 \setminus \widehat{U}_{k-1}} R^2_{\widehat{U}_{k-1} \cup \{i\}}, 
    \quad \widehat{U}_k = \widehat{U}_{k-1} \cup \{i^\ast_k\}, 
    \quad \mathcal{U} = \mathcal{U} \cup \{\widehat{U}_k\}.

This process continues until iteration :math:`k = N_0`. The control group ultimately returned by FDID is :math:`\widehat{U} \operatorname*{:=} \operatorname*{argmax}_{\widehat{U}_k \in \mathcal{U}} R^2(\widehat{U}_k)`, or the candidate set of control units that has the highest R-squared statistic of all the candidate sets.

Implementing FDID via mlsynth
-----------------------------

Here is the input FDID accepts:

.. autoclass:: mlsynth.FDID
   :show-inheritance:
   :members:
   :special-members: __init__

