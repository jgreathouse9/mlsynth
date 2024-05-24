A Short Tutorial on [Robust Principal Componenet Synthetic Control](https://academicworks.cuny.edu/cgi/viewcontent.cgi?article=6069&context=gc_etds) 
==============

**Author:** *Jared Greathouse*

> [!IMPORTANT]
> **mlsynth** is an ongoing project; any feedback or comments are most welcome!

#### Sections
+ [Introduction](#introduction)
+ [Model Primitives](#model-primitives)
+ [On Donor Selection](#on-donor-selection)
+ [RobustPCA](#robustpca)
+ [West Germany](#west-germany)
+ [Conclusion](#conclusion)

# Introduction
This tutorial uses publicly available data to demonstrate the utility of Robust PCA Synthetic Controls, as an extension to [my PCR tutorial](https://github.com/jgreathouse9/mlsynth/blob/main/Vignettes/PCR/PCRVignette.md). As a sort of prerequisite, I presume that the reader is familiar with synthetic control methods (SCM) and PCA. Precisely, I show how we may use the syntax to estimate the causal impact of interventions, replicating the case study of West Germany as my friend Mani did in his dissertation.
# Model Primitives
In the interest of making this vignette self contained, I introduce the standard notations I use in all of these tutorials. Here, we have $\mathcal{N} \coloneqq \lbrace{1 \ldots N \rbrace}$ units across $t \in \left(1, T\right) \cap \mathbb{N}$ time periods, where $j=1$ is our sole treated unit. This leaves us with $\mathcal{N}\_{0} \coloneqq \lbrace{2 \ldots N\rbrace}$ control units, with the cardinality of this set being the number of controls. I denote a subset of all controls as $\widetilde{\mathcal{N}}\_{0}$. We have two sets of time series $\mathcal{T}\coloneqq \mathcal{T}\_{0} \cup \mathcal{T}\_{1}$, where $\mathcal{T}\_{0}\coloneqq  \lbrace{1\ldots T_0 \rbrace}$ is the pre-intervention period and $\mathcal{T}\_{1}\coloneqq \lbrace{T_0+1\ldots T \rbrace}$ denotes the post-intervention period, each with their respective cardinalities. Let $\mathbf{w} \coloneqq \lbrace{w_2 \ldots w\_N  \rbrace}$ be a generic weight vector we assign to untreated units. We observe
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
We have a single treated unit which, along with the donors (the set of untreated units), follows a certain data generating process for all time periods until $T_0$. Afterwards, the control units (generally) follow the same process. The change of the outcomes $j=1,  \forall t \in \mathcal{T}_1$ is whatever that process was, plus some treatment effect.  To this end, we are concerned with $\hat{y}\_{1t}$, or the values we would have observed absent treatment. The statistic we are concerned with is the average treatment effect on the treated

$$ ATT = \frac{1}{T\_1 - T\_0} \sum_{T\_0 +1}^{T} (y_{1t} - \hat{y}_{1t}) $$

where $(y_{1t} - \hat{y}_{1t})$ is the treatment effect. In SCM, we exploit the linear relation between untreated units and the treated unit to estimate its counterfactual.

# On Donor Selection
One of the main aspects of SCM that users have lots of control over is the donors included in the estimation. The donors directly influence the counterfactual we get as a result of our estimation. This of course is not new. A randomized controlled trial, for example, randomly assigns units to being treated or not. Assuming a large enough sample (among other things), the randomization balances our treated and control units in terms of covairates because in the experimental setting, only a random process determines treatment, not other covariates. A natural principle of quasi-experimental design, then, is that our treatment assignment mechanism is *as good as random* conditional on some set of metrics (parallel trends, in our case). For SCM, the parallel trends assumption is that the counterfactual would be the weighted average of donors in the post-intervention period absent treatment.

A practical consequence of this is that the donors should be as similar to the treated unit in the pre-intervention period as possible. As Abadie [says](https://dspace.mit.edu/bitstream/handle/1721.1/144417/jel.20191450.pdf?sequence=2&isAllowed=y),

>  each of the units in the donor pool have to be chosen judiciously to provide a reasonable control for the treated unit. Including in the donor pool units that are regarded by the analyst to be unsuitable controls (because of large discrepancies in the values of their observed attributes or because of suspected large differences in the values of the unobserved attributes relative to the treated unit) is a recipe for bias.
