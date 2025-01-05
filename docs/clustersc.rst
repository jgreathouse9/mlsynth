Cluster SC Explained
==================

With Synthetic Control Methods, we may be unsure if the control group we use will be a good countrefactual for the treated unit-- even if we are relying on a weighted average of the controls instead of using all of them. Outcome trajectories may be noisy, have missing data, or have an extremely large number of control units, epsecially in modern settings where data are frequently disaggregated. To address this, [Amjad2018]_ developed a SVD version of the synthetic control method, Robust Synthetic Control. This is what the :class:`CLUSTERSC` class implements, and more. I begin with notations.

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
        y^{0}_{1t} & \text{if } j = 1 \text{ and } t \in \mathcal{T}_0 \\
        y^{1}_{1t} & \text{if } j = 1 \text{ and } t \in \mathcal{T}_1
    \end{cases}

In the synthetic control method, we (typically) have a single treated unit which, along with the donors, follows a certain data generating process for all time periods until :math:`T_0`. 
After the final pre-treatment period, the control units follow the same process because they are unaffected by the intervention. However for unit :math:`j=1`, the outcomes we see are that pre-intervention DGP plus some treatment effect. To this end, we are concerned with :math:`\hat{y}_{1t}`, or the out of sample values we would have observed for the treated unit absent treatment. The average treatment effect on the treated

.. math::
    ATT = \frac{1}{T_1 - T_0} \sum_{T_0 +1}^{T} (y_{1t} - \hat{y}_{1t})

is our main statistic of interest, where :math:`(y_{1t} - \hat{y}_{1t})` is the treatment effect at some given time point. In SCM, we exploit the linear relation 
between untreated and the treated unit to estimate its counterfactual.

SCM and SVD
-----------

Normal SCM is estimated like

.. math::
    \begin{align}
        \underset{w}{\operatorname*{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{Y}_{\mathcal{N}_{0}} w_j||_{2}^2 \\
        \text{s.t.} & \quad \mathbf{w}: w_{j} \in \mathbb{I}, \quad  \|\mathbf{w}\|_{1} = 1
    \end{align}

