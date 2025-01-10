Two-Step Synthetic Control
==========================

# Model Primitives
Here, we have :math:`\mathcal{N} \coloneqq \lbrace{1 \ldots N \rbrace}` units across :math:`t \in \left(1, T\right) \cap \mathbb{N}` time periods, where :math:`j=1` is our sole treated unit. This leaves us with :math:`\mathcal{N}_{0} \coloneqq \lbrace{2 \ldots N\rbrace}` control units, with the cardinality of this set being the number of controls. We have two sets of time series :math:`\mathcal{T} \coloneqq \mathcal{T}_{0} \cup \mathcal{T}_{1}`, where :math:`\mathcal{T}_{0} \coloneqq  \lbrace{1\ldots T_0 \rbrace}` is the pre-intervention period and :math:`\mathcal{T}_{1} \coloneqq \lbrace{T_0+1\ldots T \rbrace}` denotes the post-intervention period, each with their respective cardinalities. Let :math:`\mathbf{w} \coloneqq \lbrace{w_2 \ldots w_N  \rbrace}` be a generic weight vector we assign to untreated units. We observe

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
    (\hat{\mu},\hat{w}) = \underset{\mu,w}{\text{arg\,min}} \quad (\mathbf{y}_{1} - \mu - \mathbf{w}^\top - \mathbf{Y}_{\mathcal{N}_{0}})^\top (\mathbf{y}_{1} - \mu - \mathbf{w}^\top - \mathbf{Y}_{\mathcal{N}_{0}})

.. math::
    \text{s.t.} \quad \mathbf{w}= N^{-1}_{0}

.. math::
    \mu = \frac{1}{T_0}\sum_{t=1}^{T_0}y_{1t} - \frac{1}{N_{0} \cdot T_0} \sum_{t=1}^{T_0}\sum_{j=2}^{N_0}y_{j \in \mathcal{N}_{0}.}

Here, we seek the line that minimizes the differences between the treated vector :math:`\mathbf{y}_{1}` and the uniformly weighted average of controls, :math:`\mathbf{w}= N^{-1}_{0}`. We don't typically think of DID as a weighting estimator, but this makes sense; in our intro to causal inference courses, we learn that DID posits that our counterfactual for the treated unit would be the pure average of our control units plus some intercept, :math:`\mu`. Well, the only way we may have this is if we are implicitly giving every control unit as much as weight as all of the other control units. Any imbalance in the preintervention period comes from poor control units, anticipatory effects, or omitted variable biases. This means that for objective causal inference in DID, we must compare the treated unit to a set of controls that are as similar to the treated unit in every way but for the treatment. This leaves analysts with a few paths to take: we either discard dissimilar donors, or we adjust the weighting scheme for our units. As I've mentioned `elsewhere <https://github.com/jgreathouse9/FDIDTutorial/blob/main/Vignette.md>`, we may use methods such as forward selection to obtain the correct donor pool, under certain instances. In the case of DID, we may discard units.

SCM, however, has a different weighting scheme. In the SCM world, one of the primary innovations is that we are explicitly saying that the weights are not meant to be constant. Generically, we may express classic SCM as

.. math::
    \underset{w}{\text{argmin}} \quad ||\mathbf{y}_{1} - \mathbf{Y}_{\mathcal{N}_{0}}w_j||_{2}^2

.. math::
    \text{s.t.} \quad \mathbf{w}: w_{j} \in \mathbb{I} \quad  {\| \mathbf{w} \|_{1} = 1}

Where :math:`\mathbb{I}` is the unit interval. Much as with DID, the unit weights must also add up to 1. In fact, we can plug these exact weights into standard regression software like Stata and get the exact synthetic control we predict using standard DID regression techniques! However, in SCM, the weights may vary. SCM simply asks us to assume that some units matter more than others, and that in doing so (in absence of an intercept), our counterfactual would be the weighted average of controls. We use convex hull constraints to build our synthetic control.
