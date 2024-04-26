A Tutorial on the [Two-Step](https://doi.org/10.1287/mnsc.2023.4878) Synthetic Control Method 
==============

***A Marketing Application***

**Author:** *Jared Greathouse*

# Introduction
This tutorial uses publicly available data to demonstrate the utility of the [Two-Step](https://doi.org/10.1287/mnsc.2023.4878) Synthetic Control Method (TSSCM). The Python code is based on MATLAB code by [Kathleen Li](https://sites.utexas.edu/kathleenli/). The tutorial is also intended to give social scientists a more precise idea of the parallel trends assumptions underlying difference-in-differences (DID) and SCM, as these designs are increasingly popular for policy analysts, economists, marketers, and other fields. As a sort of prerequisite, I presumse that the reader is familiar with the basics of causal inference as well as the estimation of these designs. I begin with the mathematical preliminaries:
# Model Primitives
Here, we have $\mathcal{N} \coloneqq \lbrace{1 \ldots N \rbrace}$ units across $t \in \left(1, T\right) \cap \mathbb{N}$ time periods, where $j=1$ is our sole treated unit. This leaves us with $\mathcal{N}\_{0} \coloneqq \lbrace{2 \ldots N\rbrace}$ control units, with the cardinality of this set being the number of controls. We have two sets of time series $\mathcal{T}\coloneqq \mathcal{T}\_{0} \cup \mathcal{T}\_{1}$, where $\mathcal{T}\_{0}\coloneqq  \lbrace{1\ldots T_0 \rbrace}$ is the pre-intervention period and $\mathcal{T}\_{1}\coloneqq \lbrace{T_0+1\ldots T \rbrace}$ denotes the post-intervention period. Let $\mathbf{w}$ be a generic weight vector that is assigned to some donor units. We observe
```math
\begin{equation*}
y_{jt} = 
\begin{cases}
    y^{0}_{jt} & \forall \: j\in \mathcal{N}_0\\
    y^{0}_{1t} & \text{if } j = 1 \text{ and } t \in \mathcal{T}_0 \\
    y^{1}_{1t} & \text{if } j = 1 \text{ and } t \in \mathcal{T}_1
\end{cases}

\end{equation*}
```
We have a single treated unit which along with the donors follows a certain data generating process for all time periods before $T_0$. Afterwards, the control units follow the same process (assuming no spillovers), and the change of the outcomes for the treated unit is whatever that process was, plus some treatment effect. Accordingly, we may estimate DID as
```math
\begin{align}
    (\hat{\mu},\hat{w}) = \underset{\mu,w}{\text{arg\,min}} & \quad (\mathbf{y}_{1} - \mu -\mathbf{w}^\top -\mathbf{Y}_{\mathcal{N}_{0}})^\top (\mathbf{y}_{1} - \mu -\mathbf{w}^\top -\mathbf{Y}_{\mathcal{N}_{0}}) \\
    \text{s.t.} & \quad \mathbf{w}= N^{-1}_{0} \\
    & \mu = \frac{1}{T_0}\sum_{t=1}^{T_0}y_{1t} - \frac{1}{N_{0} \cdot T_0} \sum_{t=1}^{T_0}\sum_{j=2}^{N_0}y_{j \in \mathcal{N}_{0}}
\end{align}
```
While this may seem complicated, it is simple OLS. Here, we seek the line that minimizes the differences between the treated vector $\mathbf{y}\_{1}$ and the weighted average of controls $\mathbf{w}^\top -\mathbf{Y}\_{\mathcal{N}_{0}}$. There are constraints placed on the weigths however, for DID. Here, they must be constant and add up to 1, or proportionality $\mathbf{w}= N^{-1}\_{0}$. This makes sense; in our intro to causal inference courses, we learn that DID posits that our counterfactual to the treated unit would be the average of our control units plus some intercept, $\mu$. Well, the only way we may have this is if we are implicitly giving every control unit as much as weight as all of the other control units. This means that for objective causal inference, we must compare the treated unit to a set of controls that are as similar to the treated unit in every way but for the treatment. This leaves analysts with a few paths to take: we either discard dissimilar donors, or we adjust the weighting scheme for our units. As I've mentioned [elsewhere](https://github.com/jgreathouse9/FDIDTutorial/blob/main/Vignette.md), we may use methods such as forward selection to obtain the correct donor pool, under certain instances. In the case of DID, we may discard units.

SCM however has a different weighting scheme
