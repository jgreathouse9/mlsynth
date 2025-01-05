Cluster SC Explained
==================

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
----------------

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

where :math:`\|\cdot\|_F` denotes the Frobenius norm. When we do this, we are left with the singular values, and how much of the total variance they explain. Selecting too few singular values/principal components means our synthetic control will underfit the pre-intervention time series of the treated unit. Selecting too many singular values means we will overft the pre-intervention time series. The benefits of this approach, as [Amjad2018]_ and [Agarwal2021]_ show, is that it implicitly performs regularization our donor pool, and has a denoising effect. Original PCR used Universal Singular Value Thresholding to choose the optimal numver of principal componenets to retain [Chatterjee2015]_. However, :class:`CLUSTERSC` instead uses the SCREENOT method by Donoho et al. [Donoho2023]_.

The final objective function is

.. math::
   \begin{align}
       \underset{w}{\text{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{L} w_j||_{2}^2 \\
       \text{s.t.} \: & \mathbf{w}: w_{j} \in \mathbb{R}
   \end{align}

where we simply use the reconstructed, denoised version of the control group to learn the values of the treated unit in the preintervention period. Then, we take the dot product of our control group to estimate the post intervention counterfactual.

Robust PCA
----------------

The next method :class:`CLUSTERSC` implements is the Robust PCA SC method by [Bayani2021]_. Robust PCA asks the user to accept the very simple premise that our outcomes that we observe are byproducts of a low-rank structure with occasional/sparse outliers, :math:`\mathbf{L} + \mathbf{S}`, where both matrices respectively are of :math:`N \times T` dimensions. As before with PCR/Robust SC, if we can extract this low-rank component for our donor pool, we can use this to learn which combination of donors matters most for the construction of our counterfactual. This problem is written as:

.. math::
   \begin{align*}
   &\mathop {{\mathrm{minimize}}}\limits _{{\mathbf{L}},{\mathbf{S}}} ~{\mathrm{rank}}({\mathbf{L}}) + \lambda {\left \|{ {\mathbf{S}} }\right \|_{0}} \\
   &\textrm {subject to } ~~{\mathbf{Y}} = {\mathbf{L}} + {\mathbf{S}},
   \end{align*}

However, this program is NP-hard due to the rank portion of the objective function. Instead, we use the nuclear norm and :math:`\ell_1` norm on the low-rank matrix and sparse matrix respectively:

.. math::
   \begin{align*}
   &\mathop {{\mathrm{minimize}}}\limits _{{\mathbf{L}},{\mathbf{S}}} ~{\left \|{ {\mathbf{L}} }\right \|_{*}} + \lambda {\left \|{ {\mathbf{S}} }\right \|_{1}} \\
   &\textrm {subject to } ~~{\mathbf{Y}} = {\mathbf{L}} + {\mathbf{S}},
   \end{align*}

With this low rank structure, we estimate our weights

.. math::
   \begin{align}
       \underset{w}{\text{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{L} w_{j}||_{2}^2 \\
       \text{s.t.} \: & \mathbf{w}: w_{j} \in \mathbb{R}_{\geq 0}
   \end{align}


