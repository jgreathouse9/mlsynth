Two-Step Synthetic Control
==========================

Sometimes with syntehtic control methods, treated units are a little too extreme relative to their donor pool. As a result, analysts amy sometimes need to use intecepts or relax the nonnegativiety constraint, but it's unclear, a priori, in which situations this is necessary or beneficial. The Two Step SCM is actually the only method in ``mlsynth`` that does not rely on machine-learning methods; it does, however, allow analysts to choose between these different modifications to SCM where it may be warranted. This is the benefit of the two-step synthetic control method, as detailed by Li and Shankar [TSSC]_

Here, we have :math:`\mathcal{N} \operatorname*{:=} \lbrace{1 \ldots N \rbrace}` units across :math:`t \in \left(1, T\right) \cap \mathbb{N}` time periods, where :math:`j=1` is our sole treated unit. This leaves us with :math:`\mathcal{N}_{0} \operatorname*{:=} \lbrace{2 \ldots N\rbrace}` control units, with the cardinality of this set being the number of controls. We have two sets of time series :math:`\mathcal{T} \operatorname*{:=} \mathcal{T}_{0} \cup \mathcal{T}_{1}`, where :math:`\mathcal{T}_{0} \operatorname*{:=}  \lbrace{1\ldots T_0 \rbrace}` is the pre-intervention period and :math:`\mathcal{T}_{1} \operatorname*{:=} \lbrace{T_0+1\ldots T \rbrace}` denotes the post-intervention period, each with their respective cardinalities. Let :math:`\mathbf{w} \operatorname*{:=} \lbrace{w_2 \ldots w_N  \rbrace}` be a generic weight vector we assign to untreated units. We observe

.. math::
    y_{jt} = 
    \begin{cases}
        y^{0}_{jt} & \forall \: j\in \mathcal{N}_0\\
        y^{0}_{1t} & \text{if } j = 1 \text{ and } t \in \mathcal{T}_0 \\
        y^{1}_{1t} & \text{if } j = 1 \text{ and } t \in \mathcal{T}_1
    \end{cases}

We have a single treated unit which, along with the donors, follows a certain data generating process for all time periods until :math:`T_0`. Afterwards, the control units follow the same process. The change of the outcomes :math:`j=1,  \forall t \in \mathcal{T}_1` is whatever that process was, plus some treatment effect. To this end, we are concerned with :math:`\hat{y}_{j1}`, or the values we would have observed absent treatment. The statistic we are concerned with is the average treatment effect on the treated

.. math::
    ATT = \frac{1}{T_1 - T_0} \sum_{T_0 +1}^{T} (y_{1t} - \hat{y}_{1t})

where :math:`(y_{1t} - \hat{y}_{1t})` is the treatment effect. Next, we can think about how to model this.

Consider

.. math::
    \mathbf{M}^{\ast}_{jt} = \sum_{k=1}^{r} \boldsymbol{\lambda}_{jk}\boldsymbol{\gamma}_{tk},

a model known as a factor model in the econometrics literature. The DID method assumes the outcomes we observe are byproduct of a time specific effect and a unit specific effect, :math:`\mathbf{M}^{\ast}_{jt} = a_j + b_t`. This means, from the above, that :math:`r=2` and :math:`\boldsymbol{\lambda}_{j1}=a_j, \boldsymbol{\lambda}_{j2}=1, \boldsymbol{\gamma}_{t1}=1, \boldsymbol{\gamma}_{t2}=b_t`. This means we may use the average of controls as a proxy for the unit-specific coefficient (in the pre and post period). The time-specific coefficient must be estimated. Accordingly, think about DID as a weighting estimator which solves

.. math::
    (\hat{\mu},\hat{w}) = \underset{\mu,w}{\operatorname*{argmin}} \quad (\mathbf{y}_{1} - \mu - \mathbf{w}^\top - \mathbf{Y}_{\mathcal{N}_{0}})^\top (\mathbf{y}_{1} - \mu - \mathbf{w}^\top - \mathbf{Y}_{\mathcal{N}_{0}})

.. math::
    \text{s.t.} \quad \mathbf{w}= N^{-1}_{0}

.. math::
    \mu = \frac{1}{T_0}\sum_{t=1}^{T_0}y_{1t} - \frac{1}{N_{0} \cdot T_0} \sum_{t=1}^{T_0}\sum_{j=2}^{N_0}y_{j \in \mathcal{N}_{0}.}

Here, we seek the line that minimizes the differences between the treated vector :math:`\mathbf{y}_{1}` and the uniformly weighted average of controls, :math:`\mathbf{w}= N^{-1}_{0}`. We don't typically think of DID as a weighting estimator, but this makes sense; in our intro to causal inference courses, we learn that DID posits that our counterfactual for the treated unit would be the pure average of our control units plus some intercept, :math:`\mu`. Well, the only way we may have this is if we are implicitly giving every control unit as much as weight as all of the other control units. Any imbalance in the preintervention period comes from poor control units, anticipatory effects, or omitted variable biases. This means that for objective causal inference in DID, we must compare the treated unit to a set of controls that are as similar to the treated unit in every way but for the treatment. This leaves analysts with a few paths to take: we either discard dissimilar donors, or we adjust the weighting scheme for our units. In the case of DID, we may discard units. SCM, however, has a different weighting scheme. In the SCM world, one of the primary innovations is that we are explicitly saying that the weights are not meant to be constant. Generically, we may express classic SCM as

.. math::
    \underset{w}{\operatorname*{argmin}} \quad ||\mathbf{y}_{1} - \mathbf{Y}_{\mathcal{N}_{0}}w_j||_{2}^2

.. math::
    \text{s.t.} \quad \mathbf{w}: w_{j} \in \mathbb{I} \quad  {\| \mathbf{w} \|_{1} = 1}

Where :math:`\mathbb{I}` is the unit interval. Much as with DID, the unit weights must also add up to 1. In fact, we can plug these exact weights into standard regression software like Stata and get the exact synthetic control we predict using standard DID regression techniques! However, in SCM, the weights may vary. SCM simply asks us to assume that some units matter more than others, and that in doing so (in absence of an intercept), our counterfactual would be the weighted average of controls. We use convex hull constraints to build our synthetic control.

Consider three modifications, call them Modified Synthetic Controls (MSC).

MSCa
-----
.. math::

    \underset{w}{\text{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{Y}_{\mathcal{N}_{0}}w_{j}, - \mathbf{\beta}||_{2}^2 \\
    \text{s.t.} \: & \mathbf{w}: w_{j} \in \mathbb{I}, \quad  {\| \mathbf{w} \|_{1} = 1}, \mathbf{\beta} \neq 0.

MSCa shifts the counterfactual within the convex hull using the :math:`T \times 1` vector of 1s with its corresponding coefficient being :math:`\mathbf{\beta}` (which is unconstrained). It is the intercept.

MSCb
-----
.. math::

    \underset{w}{\text{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{Y}_{\mathcal{N}_{0}}w_{j}||_{2}^2 \\
    \text{s.t.} \: & \mathbf{w}: w_{j} \in \mathbb{R}_{\geq 0}

MSCb gets rid of the intercept and forces the weights to only be positive.

MSCc
-----
.. math::

    \underset{w}{\text{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{Y}_{\mathcal{N}_{0}}w_{j}-\mathbf{\beta}]||_{2}^2 \\
    \text{s.t.} \: & \mathbf{w}: w_{j} \in \mathbb{R}_{\geq 0}

MSCc allows for both an intercept and unrestricted positive weights. We now are projecting the treated unit (as with MSCb) onto a convex cone, instead of the convex hull. We typically would want to use these latter estimators if the treated unit has a particularly higher slope or trend compared to the donor units. Given these different options, it makes sense for analysts to care about which set of restrictions are the most plausible. If a convex combination is enough, then we simply use SC as it was originally formulated. If not, we must select the proper set of constraints to use.

Step 1: Testing the Relevant Hypotheses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The point of TSSC is to first test the viability of the parallel pre-trends assumption for vanilla SCM, which chooses between the original model and the other three presented. Precisely, we make a null hypothesis

.. math::

    H_0 : w_{j} \in \mathbb{I}, \quad  {\| \mathbf{w} \|_{1} = 1}

or, that we've violated the pre-intervention trend convex hull restriction. In order to test this null hypothesis, we use subsampling (see Kathy's original paper for details) to test the convex SCM's pre-intervention fit against MSCc's. The reason MSCc is the benchmark is because if the intercept is 0 (even though we've constrained it not to be) and the unit weights add up to 1 (even though they need not), MSCc reduces to vanilla SCM.

We first test a joint null hypothesis. We may write our null hypothesis as :math:`H_0 : \mu = 0, \quad   {||\mathbf{w}||_{1} = 1}`, or, :math:`\mathbf R\beta_0 - \mathbf q=\mathbf{0}_{2}`, or in words, that the intercept is 0 and the unit weights should add to 1. :math:`\mathbf R` is a matrix where the first and second rows of the first column are 0 and 1 respectively with the latter columns being 1 and 0 respectively.

.. math::

    \left(
    \begin{bmatrix}
        0 & \mathbf{1}^{\top}_{N-1} \\
        1 & \mathbf{0}^{\top}_{N-1}.
    \end{bmatrix}
    \right)

:math:`\mathbf q` is a vector to be used in the joint null hypothesis test

.. math::

    \begin{bmatrix}
        1 \\
        0
    \end{bmatrix}

The top row of :math:`\mathbf{R}` corresponds to the summation to one constraint and the lower row of :math:`\mathbf{R}` corresponds to the zero intercept. We can use :math:`\mathbf{R}` and :math:`\mathbf{q}` to define a vector, :math:`\mathbf{d}`, to test the null hypothesis: :math:`\mathbf{d} = \mathbf{R} \mathbf{w} - \mathbf{q}`. :math:`\mathbf{d}` is a 2 by 1 vector, with the first element corresponding to the sum of the weights and the second element being the value of the intercept generated by MSCc. We then use :math:`\mathbf{d} T_0` to derive the test statistic

.. math::

    \tilde{S}_{T_1}= (\sqrt{T_0}\mathbf{d} )\hat{V}^{-1} (\sqrt{T_0}\mathbf{d})

where :math:`V` is the asymptotic variance of :math:`\sqrt{T_1}\mathbf{R}(\hat{\mathbf{w}}_{T_0}^{\text{MSC}}-\mathbf{w}_{T_0}^{\text{MSC}})`. The natural issue, then, is how to estimate the variance. To do this, we can use a subsampling routine. We begin by taking random draws of the pre-intervention period :math:`$m=1 \ldots T_0` for both the treated and control units and estimate the synthetic control :math:`\hat{\mathbf{w}}_{T_0}^{\text{MSC}}`, checking its differences versus the original weights :math:`\mathbf{w}_{T_0}^{\text{MSC}}`. We repeat this process many times (10000 in this case). We can then get a consistent estimator of the variance

.. math::

    \hat{V} = \mathbf{R}\sigma^{\ast}(\sqrt{T_0}\hat{\mathbf{w}}_{T_0}^{\text{MSC}})\mathbf{R}^{\top}

where :math:`\sigma^{\ast}(\sqrt{T_0}\hat{\mathbf{w}}_{T_0}^{\text{MSC}})` is

.. math::

    \frac{m}{B} \sum_{b=1}^{B}(\hat{\mathbf{w}}_{T_0}^{\text{MSC,m,b}}-\mathbf{w}_{T_0}^{\text{MSC}})(\hat{\mathbf{w}}_{T_0}^{\text{MSC,m,b}}-\mathbf{w}_{T_0}^{\text{MSC}})^{\top}

with :math:`b` being the number of draws. The sub-sampling statistic itself is

.. math::

    S^{\ast}_{m,b} = (\sqrt{m}(\hat{\mathbf{w}}_{T_0}^{\text{MSC,m,b}}-\mathbf{w}_{T_0}^{\text{MSC}})^{\top})\mathbf{V}^{-1} (\sqrt{m}(\hat{\mathbf{w}}_{T_0}^{\text{MSC,m,b}}-\mathbf{w}_{T_0}^{\text{MSC}}))

and, after sorting these in ascending order, the confidence interval is :math:`[\hat{S}_{m,(\alpha B/2)}, \hat{S}_{m,((1-\alpha/2)B)}]`. Should :math:`\tilde{S}_{T_1}` fall within the confidence interval, we reject the joint null hypothesis. If this fails, we then proceed to test the summing to 1 and intercept constraints individually. For each, we make null hypotheses using the row vectors of :math:`\mathbf{R}` above, with the respective nulls for summation and the zero intercept being :math:`H_{0_{a}} : \| \mathbf{w} \| = 1` and :math:`H_{0_{b}} : \mu = 0`. For each, we can write them as :math:`\mathbf{R}_{a} \mathbf{w}^{\text{MSC}} - \mathbf{q}_{a}` and :math:`\mathbf{R}_{b} \mathbf{w}^{\text{MSC}} - \mathbf{q}_{b}`. For summation and the intercept respectively, :math:`\mathbf{q}_a = 1` and :math:`\mathbf{q}_b = 0`, with :math:`\mathbf{R}_a = (0, \mathbf{1}_{N-1}^{\top})` and :math:`\mathbf{R}_b = (0, \mathbf{0}_{N-1}^{\top})`. For each null hypothesis, our test statistic is :math:`(\sqrt{T_0} \mathbf{d})^{2}`, where for nulls :math:`s = a, b` we compute :math:`\mathbf{d} = \mathbf{R}_s \hat{\mathbf{w}}_{T_0}^{\text{MSC}} - \mathbf{q}_s`. We also use the subsampling procedure I just described to calculate the subsampling statistics and confidence intervals. TSSC proceeds sequentially. If the joint null is violated, we then first test the summation constraint. If we fail to reject summation, we use MSCa and include the intercept. If we reject, we then test the intercept constraint. If we fail to reject, we use MSCb since it does not impose the summation constraint and does not use the intercept. If the intercept null is also invalid, we use MSCc, the most flexible SCM listed here as it uses both an intercept and unconstrained positive donor weights.

Step 2: Estimation
~~~~~~~~~~~~~~~~~~

After we choose the correct set of constraints, we then estimate the counterfactual. Below, I use TSSC in the provided empirical application.

