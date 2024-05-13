A Tutorial on [Principal Componenet Regression](https://doi.org/10.1080/01621459.2021.1928513) for Synthetic Controls 
==============

**Author:** *Jared Greathouse*

> **Note**
>
> **mlsynth** is an ongoing project; any feedback or comments are most welcome! See the Cali.py file for my replication code of the empirical example.

# Introduction
This tutorial uses publicly available data to demonstrate the utility of Principal Component Regression (PCR). As a sort of prerequisite, I presume that the reader is familiar with synthetic control methods (SCM). Precisely, I show how we may use the syntax to estimate the causal impact of interventions.
# Model Primitives
Here, we have $\mathcal{N} \coloneqq \lbrace{1 \ldots N \rbrace}$ units across $t \in \left(1, T\right) \cap \mathbb{N}$ time periods, where $j=1$ is our sole treated unit. This leaves us with $\mathcal{N}\_{0} \coloneqq \lbrace{2 \ldots N\rbrace}$ control units, with the cardinality of this set being the number of controls. We have two sets of time series $\mathcal{T}\coloneqq \mathcal{T}\_{0} \cup \mathcal{T}\_{1}$, where $\mathcal{T}\_{0}\coloneqq  \lbrace{1\ldots T_0 \rbrace}$ is the pre-intervention period and $\mathcal{T}\_{1}\coloneqq \lbrace{T_0+1\ldots T \rbrace}$ denotes the post-intervention period, each with their respective cardinalities. Let $\mathbf{w} \coloneqq \lbrace{w_2 \ldots w\_N  \rbrace}$ be a generic weight vector we assign to untreated units. We observe
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
We have a single treated unit which, along with the donors, follows a certain data generating process for all time periods until $T_0$. Afterwards, the control units follow the same process. The change of the outcomes $j=1,  \forall t \in \mathcal{T}_1$ is whatever that process was, plus some treatment effect.  To this end, we are concerned with $\hat{y}\_{j1}$, or the values we would have observed absent treatment. The statistic we are concerned with is the average treatment effect on the treated

$$ ATT = \frac{1}{T\_1 - T\_0} \sum_{T\_0 +1}^{T} (y_{1t} - \hat{y}_{1t}) $$

where $(y_{1t} - \hat{y}_{1t})$ is the treatment effect. Next, we can think about how to model this.

Consider $$\mathbf{M}^{\ast}\_{jt} = \sum\_{k=1}^{r} \boldsymbol{\lambda}\_{jk}\boldsymbol{\gamma}\_{tk},$$ a model known as a factor model in the econometrics literature. 

# Conclusion

The reason this matters, from a very practical standpoint, has significance beyond econometrics. Policies and business decisions made by governments and companies are usually costly and have the capability to affect thousands, sometimes tens of millions of people (depending on the government or corporation). If a government wishes to try a new policy intervention or a company decides to do a new kind of advertising in one of their higher sales volume cities, presumably these interventions will be logistically and financially costly. In order to know if our policies are having the effect sizes we desire, we need to use estimators of the effect size that are robust to different kinds of circumstances. Otherwise, we'll sometimes find that our policies are twice as effective as they in fact are, as we did with the simulated dataset here. Particularly if we are rolling out a policy to other areas of a business or policies to other areas if a state or city or nation, we can end up taking actions that are ineffective at the very best, or harmful at worst, because our priors are founded on wrong or at elast inaccurate information. The benefit of newer, cutting edge estimators in econometrics is not simply that they are advanced. They also improve the accuracy of our estimates. TSSC is a one of the ways to do this. It would be nice to see how the method plays in an experimental setting. Recently, such methods [were derived](https://doi.org/10.48550/arXiv.2108.02196) for penalized SCMs; to extend them to the world of A/B testing (in the industry setup) would be quite helpful for future researchers using SCM.

# Contact
- Jared Greathouse: <jgreathouse3@student.gsu.edu>
- Kathleen Li: <kathleen.li@mccombs.utexas.edu>
- Venkatesh Shankar: <vshankar@mays.tamu.edu>

