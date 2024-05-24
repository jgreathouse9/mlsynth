A Short Tutorial on [Robust Principal Component Synthetic Control](https://academicworks.cuny.edu/cgi/viewcontent.cgi?article=6069&context=gc_etds) 
==============

**Author:** *Jared Greathouse*

> [!IMPORTANT]
> **mlsynth** is an ongoing project; any feedback or comments are most welcome!

#### Sections
+ [Introduction](#introduction)
+ [Model Primitives](#model-primitives)
+ [On Donor Selection](##on-donor-selection)
+ [Why PCR Will Not Work](##why-pcr-will-not-work)
+ [Tandem Clustering](##tandem-clustering)
+ [RobustPCA](#robustpca)
+ [West Germany](#west-germany)
+ [Conclusion](#conclusion)

# Introduction
This tutorial uses publicly available data to demonstrate the utility of Robust PCA Synthetic Control, as formulated by my coworker and friend Mani Bayani in [his dissertation](https://academicworks.cuny.edu/cgi/viewcontent.cgi?article=6069&context=gc_etds). As a sort of prerequisite, I presume that the reader is familiar with synthetic control methods (SCM) and PCA. Precisely, I show how we may use the syntax to estimate the causal impact of interventions, replicating the case study of West Germany as Mani did in his dissertation.
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

## On Donor Selection
One of the main aspects of SCM that users have lots of control over is the donors included in the estimation. The donors directly influence the counterfactual we get as a result of our estimation. This of course is not new. A randomized controlled trial, for example, randomly assigns units to being treated or not. Assuming a large enough sample (among other things), the randomization balances our treated and control units in terms of covairates because in the experimental setting, only a random process determines treatment, not other covariates. A natural principle of quasi-experimental design, then, is that our treatment assignment mechanism is *as good as random* conditional on some set of metrics (parallel trends, in our case). For SCM, the parallel trends assumption is that the counterfactual would be the weighted average of donors in the post-intervention period absent treatment.

A practical consequence of this is that the donors should be as similar to the treated unit in the pre-intervention period as possible. As Abadie [says](https://dspace.mit.edu/bitstream/handle/1721.1/144417/jel.20191450.pdf?sequence=2&isAllowed=y),

>  each of the units in the donor pool have to be chosen judiciously to provide a reasonable control for the treated unit. Including in the donor pool units that are regarded by the analyst to be unsuitable controls (because of large discrepancies in the values of their observed attributes or because of suspected large differences in the values of the unobserved attributes relative to the treated unit) is a recipe for bias.

In settings with very few controls (say, under 10), this is readily satisfied, and there's [pretty good work](https://www.urban.org/research/publication/update-synthetic-control-method-tool-understand-state-policy) on how we might trim down our pool of controls assuming one or two are obviously irrelevant. However, this breaks down in high-dimensions. Suppose we have 90 controls and one treated unit-- how do we know which of these 90 are relevant? What about in cases [where we don't know](https://doi.org/10.1080/13504851.2021.1927958) the names of our donors in our dataset for privacy reasons? A donor pool of many controls and few pre-intervention periods can lead to overfitting and/or the weighing of donors that are not actually reflective of the *unobserved* factors which drive our outcome. Thus, some regularization method is needed to choose the donors before we estimate any weights at all.

## Why PCR Will Not Work

To those who read my PCR tutorial, one may ask why not simply use PCR. After all, it exrtacts the low-rank structure of the donor matrix and projects this on to the treated unit, shaving off the irreelvant singular values. Well... It does, however it is sensitive to outliers. The reason for this is because of the optimization of PCA: PCA [uses](https://onefishy.github.io/ML_notes/the-math-of-principal-component-analysis.html) the covariance matrix to calculate our principal componenets. As we know from Bill Gates walking into a bar, the presence of outliers will skew the mean and covariance matrix, meaning outliers can dominate the influence of principal componenets. Thus, PCA will not always be a viable solution to the donor selection problem in the case of outlier donors.

## Tandem Clustering

As a solution to this problem, Mani used an approach called "tandem clustering" ([also](https://www.youtube.com/watch?v=ISD8OvuQasY&t=75) called partitional clustering in machine learning). Tandem clustering is based on the idea that first we find a low dimensional representation of our dataset before we perform clustering upon it. In this case, we use functional PCA and k-means to select the donors. This post would be very, very long if I went into fPCA and k-means, so I refer the interested readers to [these](https://doi.org/10.1016/j.jbiomech.2020.110106) [sources](https://bradleyboehmke.github.io/HOML/kmeans.html) on the details for both, or Mani's dissertation. The central pitch for fPCA is that we cluster over the functional PCA scores. The main difference between fPCA and normal PCA for our purposes [is](https://www.tandfonline.com/doi/full/10.1080/14763141.2017.1392594) "In PCA ... the data points on each curve are assumed to be independent of each other, but in reality it is known that any point on a continuous time-series is correlated with the data points that precede and follow that point." After we have our low-dimensional representation of our pre-intervention time series $\forall j \in \mathcal N$, we then apply the k-means algorithm to select our donor pool. The idea is that the cluster that contains our treated unit will be much more similar to the donors in its own cluster than donors outside of its cluster. To select the number of clusters, we use a method called [the Silhouette method](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html). Our new control group, after clustering, is $\widetilde{\mathcal{N}}\_{0}$. As an aside, there are other ways we could've done this (such as [this way](https://link.springer.com/article/10.1007/s00357-017-9232-z) which clusters AND reduces dimensionality all in one objective function), however this would demand more formal theoretical justification.

# RobustPCA

Now that we have our new donor pool, we may finally estimate our counterfactual. To do this, we use Robust PCA, [which is](https://freshprinceofstandarderror.com/ai/robust-principal-component-analysis/) a form of PCA that is robust against outliers. Formally, we can think of our observed outcomes matrix ${\mathbf{Y}}$ as a low-rank component plus outlier observations, ${\mathbf{L}} + {\mathbf{S}}$. As before, if we can extract this low-rank component, we can use this to learn which donors matter most for the construction of our counterfactual. A natural formulation of this is

```math
\begin{align*}&\mathop {{\mathrm{minimize}}}\limits _{{\mathbf{L}},{\mathbf{S}}} ~{\mathrm{ rank}}({\mathbf{L}}) + \lambda {\left \|{ {\mathbf{S}} }\right \|_{0}} \\&\textrm {subject to } ~~{\mathbf{Y}} = {\mathbf{L}} + {\mathbf{S}},\end{align*}
```
however this program is NP-hard due to the rank portion of the objective function. As a workaround, we use the nuclear norm and $\ell_1$ norm on the low-rank matrix and sparse matrix respectively

```math
\begin{align*}&\mathop {{\mathrm{minimize}}}\limits _{{\mathbf{L}},{\mathbf{S}}} ~{\left \|{ {\mathbf{L}} }\right \|_{*}} + \lambda {\left \|{ {\mathbf{S}} }\right \|_{1}} \\&\textrm {subject to } ~~{\mathbf{Y}} = {\mathbf{L}} + {\mathbf{S}},\end{align*}
```
using [the singular value thresholding operator](https://soulpageit.com/ai-glossary/singular-value-thresholding-explained/) to extract the singular values and the shrinkage operator on the $\mathbf{S}$ outlier matrix (because the shrink operator reduces the influence of the outliers closer to 0, fitting with the $\ell_1$ norm. With the $\mathbf{L}$ matrix computed (which again simply represents the "pattern" portion of our donor matrix in the pre-intervention period), we may now use the pre-intervention portion of this matrix to learn our donor weights

```math
\begin{align}
    \underset{w}{\text{argmin}} & \quad ||\mathbf{y}_{1} - \mathbf{L} w_{\widetilde{\mathcal{N}}_{0}}||_{2}^2 \\\
    \text{s.t.} \: & \mathbf{w}: w_{j} \in \mathbb{R}_{\geq 0}
\end{align}
```
The counterfactual is the dot product of our low-rank matrix and the computed donor weights. We can then calculate the ATT and other relevant statistics as we usually would.

# West Germany

In what is by now one of the classical SCM papers, Abadie, Diamond, and Hainmuller [sought](https://economics.mit.edu/sites/default/files/publications/Comparative%20Politics%20and%20the%20Synthetic%20Control.pdf) to investigate the casual impact of West German reunification on the GDP per Capita of West Germany. To do this, they compare West Germany to 16 other donor nations, employing a list of five covariate measures to construct the synthetic control for West Germany. The intervention happened in 1990, and the dataset they use ranges from 1960 to 2003.

<details>
    
  <summary>Let's replicate their results using the RPCA estimator.</summary>

<table>
  <tr>
    <th>Importing</th>
    <th>Estimation</th>
  </tr>
  <tr>
    <td>
    <pre><code>
from mlsynth.mlsynth import PCASC
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
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


def get_edited_frames(stub_url, urls, base_dict):
    edited_frames = []

    for url, (key, params) in zip(urls, base_dict.items()):
        subdf = pd.read_csv(stub_url + url)

        # Keep only the specified columns
        subdf = subdf[params['Columns']]

        # Ensure the time column is of integer type
        subdf[params['Time']] = subdf[params['Time']].astype(int)

        # Generate the treatment variable
        subdf[params["Treatment Name"]] = (subdf[params["Panel"]].str.contains(params["Treated Unit"])) & \
                                          (subdf[params["Time"]] >= params["Treatment Time"])

        # Handle specific case for Basque dataset
        if key == "Basque" and "Spain (Espana)" in subdf[params["Panel"]].values:
            subdf = subdf[~subdf[params["Panel"]].str.contains("Spain \(Espana\)")]
            subdf.loc[subdf['regionname'].str.contains('Vasco'), 'regionname'] = 'Basque'

        # Append the edited DataFrame to the list
        edited_frames.append(subdf)

    return edited_frames

stub_url = 'https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/master/examples/datasets/'

base_dict = {
    "Basque": {
        "Columns": ['regionname', 'year', 'gdpcap'],
        "Treatment Time": 1975,
        "Treatment Name": "Terrorism",
        "Treated Unit": "Vasco",
        "Time": "year",
        "Panel": 'regionname',
        "Outcome": "gdpcap"
    },
    "Germany": {
        "Columns": ['country', 'year', 'gdp'],
        "Treatment Time": 1990,
        "Treatment Name": "Reunification",
        "Treated Unit": "Germany",
        "Time": "year",
        "Panel": 'country',
        "Outcome": "gdp"
    },
    "Smoking": {
        "Columns": ['state', 'year', 'cigsale'],
        "Treatment Time": 1989,
        "Treatment Name": "Proposition 99",
        "Treated Unit": "California",
        "Time": "year",
        "Panel": 'state',
        "Outcome": "cigsale"
    }
}

edited_frames = get_edited_frames(stub_url, ['basque_data.csv', 'german_reunification.csv', 'smoking_data.csv'], base_dict)


df = edited_frames[1]

def plot_gdp_outcomes(df, countries, time_column, gdp_column, save=False, filename="gdp_outcomes.png"):
    """
    Plot GDP outcomes for specified countries over time.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    countries (list of str): List of country names to plot.
    time_column (str): Name of the column representing time (e.g., 'year').
    gdp_column (str): Name of the column representing GDP.
    save (bool): Whether to save the plot as a file. Default is False.
    filename (str): The filename to save the plot. Default is 'gdp_outcomes.png'.
    """
    plt.figure(figsize=(10, 6))

    for country in countries:
        country_data = df[df['country'] == country]
        if country in ["USA", "Switzerland"]:
            plt.plot(country_data[time_column], country_data[gdp_column], label=country, color='grey', linewidth=3)
        elif country in ["Spain", "Portugal", "Greece"]:
            plt.plot(country_data[time_column], country_data[gdp_column], label=country, color='#7DF9FF', linewidth=2)
        else:
            plt.plot(country_data[time_column], country_data[gdp_column], label=country, color="black", linewidth=3)

    plt.xlabel(time_column)
    plt.ylabel(gdp_column)
    plt.title("Germany versus Excluded Donors")
    plt.legend()
    plt.grid(True)

    plt.show()

stub_url = 'https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/master/examples/datasets/'
german_reunification_url = 'german_reunification.csv'

countries_to_plot = ["West Germany", "USA", "Spain", "Portugal", "Switzerland", "Greece"]
    </pre></code>
    </td>
    <td>
    <pre><code>

unitid = "country"
time = "year"
outcome = "gdp"
treat = "Reunification"

model = PCASC(
    df=df,
    treat=treat,
    time=time,
    outcome=outcome,
    unitid=unitid,
    figsize=(10, 6),
    grid=True,
    treated_color='black',
    counterfactual_color='red',
    display_graphs=True, save=False)


autores = model.fit()
    </pre></code>
    </td>
  </tr>
</table>

</details>

The left panel of the table imports the West Germany dataset. In accordance with the requirements of **mlsynth**, it defines the treatment variable "Reunification" to be equal to 1 if the unit is West Germany and the year is greter than 1990, else 0. We then pass these values off to the ```PCASC``` class which uses the method described above to select our donors, conduct RPCA, and estimate our counterfactual.
