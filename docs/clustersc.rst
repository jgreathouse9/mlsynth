Cluster SC Explained
==================

With Synthetic Control Methods, we may be unsure if the control group we use will be a good countrefactual for the treated unit-- even if we are relying on a weighted average of the controls instead of using all of them. Outcome trajectories may be noisy, have missing data, or have an extremely large number of control units, epsecially in modern settings where data are frequently disaggregated. [Amjad2018]_ developed a SVD version of the synthetic control method, Robust Synthetic Control  this is what the :class:`CLUSTERSC` class implements, and more.

Here, we have :math:`\mathcal{N} \coloneqq \lbrace{1 \ldots N \rbrace}` units across 
:math:`t \in \left(1, T\right) \cap \mathbb{N}` time periods, where :math:`j=1` is our sole treated unit. 
This leaves us with :math:`\mathcal{N}_0 \coloneqq \lbrace{2 \ldots N \rbrace}` control units, 
with the cardinality of this set being the number of controls. We have two sets of time series 
:math:`\mathcal{T} \coloneqq \mathcal{T}_0 \cup \mathcal{T}_1`, where 
:math:`\mathcal{T}_0 \coloneqq \lbrace{1 \ldots T_0 \rbrace}` is the pre-intervention period and 
:math:`\mathcal{T}_1 \coloneqq \lbrace{T_0+1 \ldots T \rbrace}` denotes the post-intervention period, 
each with their respective cardinalities. Let :math:`\mathbf{w} \coloneqq \lbrace{w_2 \ldots w_N \rbrace}` 
be a generic weight vector we assign to untreated units. We observe

.. math::
    y_{jt} = 
    \begin{cases}
        y^{0}_{jt} & \forall \: j \in \mathcal{N}_0 \\
        y^{0}_{1t} & \text{if } j = 1 \text{ and } t \in \mathcal{T}_0 \\
        y^{1}_{1t} & \text{if } j = 1 \text{ and } t \in \mathcal{T}_1
    \end{cases}

We have a single treated unit which, along with the donors (the set of untreated units), 
follows a certain data generating process for all time periods until :math:`T_0`. 
Afterwards, the control units follow the same process. The change of the outcomes 
:math:`j=1, \forall t \in \mathcal{T}_1` is whatever that process was, plus some treatment effect. 
To this end, we are concerned with :math:`\hat{y}_{1t}`, or the values we would have observed 
absent treatment. The statistic we are concerned with is the average treatment effect on the treated

.. math::
    ATT = \frac{1}{T_1 - T_0} \sum_{T_0 +1}^{T} (y_{1t} - \hat{y}_{1t})

where :math:`(y_{1t} - \hat{y}_{1t})` is the treatment effect. In SCM, we exploit the linear relation 
between untreated units and the treated unit to estimate its counterfactual.

SCM and SVD
-----------

Normal SCM is estimated like

.. math::
    \begin{align}
        \underset{w}{\text{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{Y}_{\mathcal{N}_{0}} w_j||_{2}^2 \\
        \text{s.t.} & \quad \mathbf{w}: w_{j} \in \mathbb{I}, \quad  \|\mathbf{w}\|_{1} = 1
    \end{align}

