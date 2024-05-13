A Tutorial on [Principal Component Regression](https://doi.org/10.1080/01621459.2021.1928513) for Synthetic Controls 
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

where $(y_{1t} - \hat{y}_{1t})$ is the treatment effect. In SCM, we exploit the cross-sectional linear relation between untreated units and the target/treated unit to estimate the counterfactual for a given intervention. That is, as some [researchers](https://doi.org/10.3982/ECTA21248) have succinctly said, "similar units behave similarly".
## SCM and SVD
Normal SCM, therefore, is estimated like
```math
\begin{align}
    \underset{w}{\text{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{Y}_{\mathcal{N}_{0}}w_j||_{2}^2 \\
    \text{s.t.} & \mathbf{w}: w_{j} \in \mathbb{I} \quad  {\| \mathbf{w} \|_{1} = 1}
\end{align}
```
where the treated unit is projected on to the convex hull of the donor pool. [Elsewhere](https://github.com/jgreathouse9/mlsynth/blob/main/Vignettes/TSSC/TSSCVignette.md), I've discusssed the convex hull idea and why it matters to SCM. However, this may break down in settings where we have noisy outcome trends; in such settings, the units we match to may be idiosyncratically similar to the treated unit instead of being actually similar on latent factors. One key feature of real datasets in economics, policy, marketing, and statistics is that datasets are most often noisily observed. We mean "noisily" observed in the sense that we do not see the real data generating process. Oftentimes, a variety of factors will influence the values GDP will take on for a given time point. Thus, researchers benefit from having a causal estimator that can adjust for noise and debias our estimates. Another appealin property of this is that we do not always need additional covairates in order to estimate causal effects, as discussed in [Amjad, Shah, and Shen, 2018](https://www.jmlr.org/papers/volume19/17-777/17-777.pdf). Precisely, we use a linear algebra technique called singular value decomposition (SVD) to denoise our donor matrix of outcomes
```math
\mathbf{L} \coloneqq \mathbf{Y}=\mathbf{U}\mathbf{S}\mathbf{V}^{\top}
```
where $\mathbf{L}$ is our low-dimensional representation of our pre-intervention outcomes. $\mathbf L$ has as many columns as we have donors. I do not wish to reinvent wheels, so to those who do not rememeber/haven't learned SVD/PCA, [others](https://www.youtube.com/watch?v=FgakZw6K1QQ) have explained SVD a lot simpler than they do in class/textbooks. The basic idea is that we wish to find a simplified, low-dimensional representation of our donor matrix and use that to explain the pre-intervention variation of our treatment unit. By using this low-dimensional (also called low rank) representation of our donors, we may then use this low-dimensional representation in simple OLS regression to predict the post-intervention treatment effect for a treated unit. In order to select the optimal amount of singular values for regression, I employ [Universal Singular Value Thresholding](https://doi.org/10.1214/14-AOS1272) to shave off the irrelevant singular values. In other words, if we use too many principal components/singular values, we will overfit the pre-intervention time series of the treated unit, much as OLS is apt to do. If we use too few, we will underfit (and thus, have worse our of sample predictions). The final regression model we use is
```math
\begin{align}
    \underset{w}{\text{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{L} w_j||_{2}^2 \\
    \text{s.t.} \: & \mathbf{w}: w_{j} \in \mathbb{R}
\end{align}
```
where the weights may be any value on the real line. The $\mathbf L$ matrix in this optimization, as I mentioned above, is truncated by the USVT method.
# Empirical Example: Proposition 99
Let's replicate the Proposition 99 example, the classic SCM case study.

<details>
    
  <summary>We begin with the Python code to import the data.</summary>
  
```python
import pandas as pd
from mlsynth.mlsynth import PCR
import matplotlib
# matplotlib theme
jared_theme = {'axes.grid': True,
              'grid.linestyle': '-',
              'legend.framealpha': 1,
              'legend.facecolor': 'white',
              'legend.shadow': True,
              'legend.fontsize': 14,
              'legend.title_fontsize': 16,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'axes.labelsize': 16,
              'axes.titlesize': 20,
              'figure.dpi': 100,
               'axes.facecolor': 'white',
               'figure.figsize': (10, 8)}

matplotlib.rcParams.update(jared_theme)

unitid = "State"
time = "Year"
outcome = "Cigarette Consumption per Capita"
treat = "Prop 99"

file = 'https://raw.githubusercontent.com/synth-inference/synthdid/master/data/california_prop99.csv'
df = pd.read_csv(file, sep=';')

df = df.rename(columns={'year': 'Year', 'treated': 'Prop 99', 'PacksPerCapita': 'Cigarette Consumption per Capita'})

df.sort_values(by=["State", "Year"], ascending=True)
```

</details>

Here we import the data from the synthdid github repo. I used it because it already has the requirements that any **mlsynth** package needs, that is, a unit, time, treatment dummy, and outcome variable. We define them accordingly and pass them to the PCR class. Now we can estimate the model.

```python

model = PCR(
    df=df,
    treat=treat,
    time=time,
    outcome=outcome,
    unitid=unitid,
    figsize=(10, 8),
    grid=True,
    treated_color='black',
    counterfactual_color='blue',
    display_graphs=True, objective="OLS")

# Run the PCR analysis
autores = model.fit()

```

And here is the plot of our results.

<p align="center">
  <img src="PCRCalifornia.png" alt="PCR Analysis" width="90%">
</p>

The plot shows the observed values of the treated unit, California, and the counterfactual estimates for California. We see a high fidelity between the counterfactual predictions before 1989 (the first full year that Prop 99 was active for) and the real California. In the post-intervention period, we see Prop 99 decreased the amount of cigarettes per pack smoked between California and its counterfactual, an average difference of 18.23. Notice how we used unconstrained OLS to compute the unit weights. The practical effect of this is that every donor gets weight. However, we can also add Convex Hull constraints to the objective function, for a more sparse/interpretable solution. We do this by changing OLS to "SC".

<p align="center">
  <img src="ConvexPCRCalifornia.png" alt="Convex PCR Analysis" width="90%">
</p>

# Conclusion

The reason this matters, from a very practical standpoint, has significance beyond econometrics. Policies and business decisions made by governments and companies are usually costly and have the capability to affect thousands, sometimes tens of millions of people (depending on the government or corporation). If a government wishes to try a new policy intervention or a company decides to do a new kind of advertising in one of their higher sales volume cities, presumably these interventions will be logistically and financially costly. In order to know if our policies are having the effect sizes we desire, we need to use estimators of the effect size that are robust to different kinds of circumstances. Otherwise, we'll sometimes find that our policies are twice as effective as they in fact are, as we did with the simulated dataset here. Particularly if we are rolling out a policy to other areas of a business or policies to other areas if a state or city or nation, we can end up taking actions that are ineffective at the very best, or harmful at worst, because our priors are founded on wrong or at elast inaccurate information. The benefit of newer, cutting edge estimators in econometrics is not simply that they are advanced. They also improve the accuracy of our estimates. TSSC is a one of the ways to do this. It would be nice to see how the method plays in an experimental setting. Recently, such methods [were derived](https://doi.org/10.48550/arXiv.2108.02196) for penalized SCMs; to extend them to the world of A/B testing (in the industry setup) would be quite helpful for future researchers using SCM.

# Contact
- Jared Greathouse: <jgreathouse3@student.gsu.edu>
- Kathleen Li: <kathleen.li@mccombs.utexas.edu>
- Venkatesh Shankar: <vshankar@mays.tamu.edu>

