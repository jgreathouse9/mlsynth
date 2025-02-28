Proximal Synthetic Control
==========================

Next, I discuss the proximal SCM method. In synthetic control methods, we oftentimes make the presumption that our outcomes are generated by some latent variable model, :math:`y_{jt} = \delta_t + \lambda_t\mu_j+\epsilon_{jt}`. This data-generating process may include covariates, but the basic idea is that there are a set of common time effects which impact all units the same and unit-specific effects, which interact with the former, which generates the outcomes that we see. Abadie and co [ABADIE2010]_ and others use this linear factor model to justify SCM. They also derive a bias bound for the SCM, which is (with qualifications) unbiased as the the number of pre-treatment periods grows without bound. This, naturally, poses a problem in settings where we have few pre-treatment periods. Additionally, it may not always be possible to obtain good pre-treatment fit, especially with a short pre-intervention time series. As a result, the Proximal Inference SCM was developed. In the paper, Shi, Li, Miao, Hu, and Tchetgen Tchetgen [ProxSCM]_



   argue that the outcomes of control units are essentially proxies of the latent factors. Rather than directly regressing the outcome on all such proxies, one can split the set of  proxies into two, thus leveraging one set of proxies to assist the construction of a SC defined in terms of the other set of proxies.


In other words, if we agree that the outcomes of the control units are time variant proxies for our control units, we may be able to still employ the outcomes of other control units, especially if we believe they are correlated with common time factors. This page describes, broadly, the three methods they propose, and applies it to an empirical example.

Notations
----------

Formally, let us define the notation. We observe a set of units indexed by :math:`j`, where 
:math:`\mathcal{N} \operatorname*{:=} \{1, 2, \ldots, N\}` is the set of all units, with cardinality 
:math:`N = |\mathcal{N}|`. Let :math:`j = 1` represent the treated unit, and 
:math:`\mathcal{N}_0 \operatorname*{:=} \mathcal{N} \setminus \{1\}` denote the set of control units, 
with cardinality :math:`N_0 = |\mathcal{N}_0|`. The time periods are indexed by :math:`t`, with 
:math:`\mathcal{T}_1 \operatorname*{:=} \{1, 2, \ldots, T_0\}` representing the pre-intervention periods 
and :math:`\mathcal{T}_2 \operatorname*{:=} \{T_0 + 1, \ldots, T\}` representing the post-intervention periods. 
We denote the full time series as :math:`\mathcal{T} \operatorname*{:=} \mathcal{T}_1 \cup \mathcal{T}_2`, 
with :math:`T = |\mathcal{T}|` representing the total number of time periods.

Let :math:`\mathbf{y}_1 \in \mathbb{R}^T` be the vector for the treated unit's outcomes, and 
:math:`\mathbf{Y}_0 \in \mathbb{R}^{T \times N_0}` be the matrix containing the outcomes of the control units. 
Let :math:`\mathbf{P}_t \in \mathbb{R}^{k \times T}` be a matrix of proxy variables that are assumed to be 
correlated with the common factors driving the outcomes of the treated unit. Here, :math:`k` represents the 
number of proxy variables. Let :math:`\mathbf{w} \in \mathbb{R}^{N_0}` represent the vector of weights for the synthetic control.

Shi, Li, Miao, Hu, and Tchetgen Tchetgen [ProxSCM]_ advocate for a GMM approach, defining the estimating function for the proximal synthetic control method as:

.. math::

    U_t(\mathbf{w}) = g(\mathbf{P}_t) \cdot \left( \mathbf{y}_1 - (\mathbf{Y}_0 \mathbf{w})_t \right),

where :math:`g(\mathbf{P}_t)` is a function applied to the proxy variables :math:`\mathbf{P}_t` at time :math:`t`, 
and :math:`\mathbf{y}_1` and :math:`(\mathbf{Y}_0 \mathbf{w})_t` are the observed and predicted outcomes at time :math:`t`, respectively. In this setup, the :math:`\mathbf{P}_t` matrix may be comprised of anything we believe to be correlated with the time variant common factors. Per HCW [HCW]_, the outcomes of other donor units is one example of a proxy. We may also use other covariates that are unaffected by the treatment but are correlated with the time latent factors.

We define a set of moment conditions over the pre-intervention period such that :math:`\mathbf{y} - \mathbf{Y}_0 \mathbf{w}` should be orthogonal to the proxies :math:`\mathbf{P}`:
   
.. math::

    \mathbb{E}\left[ \mathbf{y}_1- \tau - \mathbf{Y}_0 \mathbf{w} \right] = 0.

We combine these conditions into a single stacked moment vector :math:`U(\theta)`, where :math:`\theta = (\mathbf{w}, \tau)` contains the weights and the treatment effect. For :math:`t \leq T_0`, the residuals are:

.. math::

    U_0(t, \mathbf{w}) = \mathbf{P}^\top \left( \mathbf{y}_1 - (\mathbf{Y}_0 \mathbf{w})_t \right).

and for :math:`t > T_0`, we have:

.. math::

    U_1(t, \tau, \mathbf{w}) = \mathbf{y}_1 - \tau - (\mathbf{Y}_0 \mathbf{w})_t.

The complete moment vector is:

.. math::

    U(\theta) =
    \begin{bmatrix}
    U_0(t, \mathbf{w}) \\
    U_1(t, \tau, \mathbf{w})
    \end{bmatrix} =
    \begin{bmatrix}
    \mathbf{P}^\top \left( \mathbf{y}_1 - \mathbf{Y}_0 \mathbf{w} \right) \\ 
    \mathbf{y}_1 - \tau - \mathbf{Y}_0 \mathbf{w}
    \end{bmatrix}.

The goal is to estimate the weights :math:`\mathbf{w}` by solving a quadratic programming problem that minimizes the moment vector:

.. math::

    \mathbf{w} = \arg\min_{\mathbf{w}} \sum_{t \in \mathcal{T}_1} U_t(\mathbf{w})^\top \Omega^{-1} U_t(\mathbf{w}),

where :math:`\Omega` is the covariance matrix. When we estimate our weights, we now can estimate the treatment effect like

.. math::

    \tau = \mathbf{y}_1 - \mathbf{Y}_0 \mathbf{w}.

The sample average of the treatment effect in the post-intervention period is the ATT. For inference, we estimate the variance-covariance matrix of the moment conditions, denoted by 
:math:`\boldsymbol{\Omega}`. This is done using a HAC estimator. The matrix :math:`\boldsymbol{\Omega}` is computed as:

.. math::

    \boldsymbol{\Omega} = \frac{1}{T} \sum_{j=-J}^{J} k(j, J) \sum_{t=1}^{T - |j|} \mathbf{g}_t \mathbf{g}_{t+j}^\top,

where :math:`k(j, J)` is the Bartlett kernel, :math:`J` is the bandwidth, and 
:math:`\mathbf{g}_t` is the vector of moment conditions at time :math:`t`. The outer summation runs over all lags 
within the valid range, while the inner summation computes the covariance contribution for each lag. For our purposes, the number of lags we choose to use is :math:`h = \lfloor 4 \cdot \left( \frac{T_2}{100} \right)^{\frac{2}{9}} \rfloor`.

We now calculate the covariance matrix 
of the parameters as:

.. math::

    \text{Cov} = \mathbf{G}^{-1} \boldsymbol{\Omega} \left(\mathbf{G}^{-1}\right)^\top,

The variance of the ATT estimate :math:`\tau` is then extracted from the covariance matrix as:

.. math::

    \text{Var}(\tau) = \text{Cov}[-1, -1],

where :math:`\text{Cov}[-1, -1]` refers to the bottom-right entry of the covariance matrix. Finally, the standard error 
of :math:`\tau` is computed as:

.. math::

    \text{SE}(\tau) = \sqrt{\frac{\text{Var}(\tau)}{T}}.

So, to summarize quickly, we employ these proxies to assist in the matching on the pre-intervention common factors, using other donor unit outcomes or auxiliary predictors.

Surrogate Approach 
--------------------

We may also employ surrogate variables which capture latent factors that drive outcome. This appears in the extension paper [LiuTchetgenVar]_, adding to the above. Let :math:`\mathbf{X}_t \in \mathbb{R}^H` represent a vector of observed surrogates for the treated unit, where :math:`H` is the number of surrogate variables. These surrogates are chosen because they are highly predictive of the treatment effects and driven by the same latent factors as the treated unit. The surrogates are unique because we exploit their post-intervention period data only. The treatment effect is decomposed into two components, 

.. math::

    \mathbf{y}_1(1) - \mathbf{y}_1(0) = \boldsymbol{\rho}_t^\top \boldsymbol{\theta} + \delta_t,

where :math:`\boldsymbol{\rho}_t \in \mathbb{R}^K` is a vector of latent factors driving the causal effect, :math:`\boldsymbol{\theta} \in \mathbb{R}^K` is a vector of factor loadings for the causal effect, and :math:`\delta_t` represents an error term uncorrelated with the latent factors. The observed surrogates :math:`\mathbf{X}_t` also follow a similar factor model,

.. math::

    \mathbf{X}_t^\top = \boldsymbol{\rho}_t^\top \mathbf{\Phi} + \boldsymbol{\epsilon}_{X,t}^\top,

where :math:`\mathbf{\Phi} \in \mathbb{R}^{K \times H}` is a matrix of factor loadings for the surrogates, and :math:`\boldsymbol{\epsilon}_{X,t}` is an error term for the surrogates. In this framework, proxies are introduced for donors and surrogates. Let :math:`\mathbf{P}_{0t}` represent proxy variables for donor outcomes, assumed to capture latent factors :math:`\boldsymbol{\lambda}_t`, and let :math:`\mathbf{P}_{1t}` represent proxy variables for surrogates, capturing both both donor latent factors :math:`\boldsymbol{\lambda}_t` and surrogate latent factors :math:`\boldsymbol{\rho}_t`.


The surrogate framework introduces two sets of moment conditions, one for the pre-treatment period and one for the post-treatment period. In the pre-treatment period, the residuals are orthogonal to the proxies for donor units:

.. math::

   \mathbb{E}[\mathbf{y}_1 - \mathbf{Y}_0^\top \mathbf{w} \mid \mathbf{P}_{0t}, t \leq T_0] = 0.

As above, the weights match on the treated unit's outcomes to the donor pool via the proxies :math:`\mathbf{P}_{0,t}` for the common time factors. In the post-treatment period, the residuals must also account for the surrogate variables via the moment condition:

.. math::

   \mathbb{E}[\mathbf{y}_1 - \mathbf{Y}_0^\top \mathbf{w} - \mathbf{X}_t^\top \boldsymbol{\gamma} \mid \mathbf{P}_{0t}, \mathbf{P}_{1t}, t > T_0] = 0.

Here, :math:`\mathbf{X}_t` is the matrix of surrogate variables, and :math:`\boldsymbol{\gamma}` is the vector of coefficients relating the surrogates to the latent unit specific factors.

As above, we combine these into a function of moment conditions, :math:`\mathbf{U}_t(\mathbf{w}, \boldsymbol{\gamma})`. For :math:`t \leq T_0`, the moment vector is:

.. math::

   U_0(t, \mathbf{w}) = \mathbf{g}_0(\mathbf{P}_{0t}) \left(\mathbf{y}_1- \mathbf{Y}_0^\top \mathbf{w} \right),

where :math:`\mathbf{g}_0(\mathbf{P}_{0t})` is a user-specified function of the proxies :math:`\mathbf{P}_{0t}`. For :math:`t > T_0`, the moment vector is:

.. math::

   U_1(t, \mathbf{w}, \boldsymbol{\gamma}) = \mathbf{g}_1(\mathbf{P}_{0t}, \mathbf{P}_{1t}) \left(\mathbf{y}_1 - \mathbf{Y}_0^\top \mathbf{w} - \mathbf{X}_t^\top \boldsymbol{\gamma} \right),

where :math:`\mathbf{g}_1(\mathbf{P}_{0t}, \mathbf{P}_{1t})` is a user-specified function of the proxies :math:`\mathbf{P}_{0t}` and :math:`\mathbf{P}_{1t}`. The complete stacked moment vector is:

.. math::

   \mathbf{U}_t(\mathbf{w}, \boldsymbol{\gamma}) =
   \begin{cases}
   \mathbf{g}_0(\mathbf{P}_{0,t}) \left( \mathbf{y}_1 - \mathbf{Y}_0^\top \mathbf{w} \right), & t \leq T_0, \\
   \mathbf{g}_1(\mathbf{P}_{0,t}, \mathbf{P}_{1,t}) \left( \mathbf{y}_1 - \mathbf{Y}_0^\top \mathbf{w} - \mathbf{X}_t^\top \boldsymbol{\gamma} \right), & t > T_0.
   \end{cases}

In matrix form, the stacked moment vector can be written as:

.. math::

   \mathbf{U}(\mathbf{w}, \boldsymbol{\gamma}) =
   \begin{bmatrix}
   \mathbf{g}_0(\mathbf{P}_{0t}) \left( \mathbf{y}_1- \mathbf{Y}_0 \mathbf{w} \right) \\ 
   \mathbf{g}_1(\mathbf{P}_{0,t}, \mathbf{P}_{1t}) \left( \mathbf{y}_1 - \mathbf{Y}_0 \mathbf{w} - \mathbf{X} \boldsymbol{\gamma} \right)
   \end{bmatrix}.

The goal is to estimate the weights :math:`\mathbf{w}` and the surrogate coefficients :math:`\boldsymbol{\gamma}` by minimizing the quadratic form of the moment residuals:

.. math::

   \arg \min_{\mathbf{w}, \boldsymbol{\gamma}} \sum_{t \in \mathcal{T}} \mathbf{U}_t(\mathbf{w}, \boldsymbol{\gamma})^\top \Omega^{-1} \mathbf{U}_t(\mathbf{w}, \boldsymbol{\gamma}),

where :math:`\Omega` is the covariance matrix of the moment conditions, accounting for heteroskedasticity and autocorrelation.

The Average Treatment Effect on the Treated (ATT) is expressed with the surrogates information in mind, as

.. math::

    \tau = \frac{1}{T - T_0} \sum_{t > T_0} \mathbf{X}_t^\top \boldsymbol{\gamma},

where the surrogate coefficients :math:`\boldsymbol{\gamma}` are estimated from the post-treatment period.

We can also do the surrogate approach with post intervention data only. The first moment condition is the treated unit's post-treatment outcomes are consistent with the donors (the ones where the weights come from) via the proxy variables:

.. math::

   \mathbb{E}[\mathbf{P}_1 \cdot (\mathbf{y}_1 - \mathbf{Y}_0^\top \mathbf{w} - \mathbf{X}^\top \boldsymbol{\gamma})] = 0.

The surrogate moment condition does the same with respect to the proxy variables of the surrogates:

.. math::

   \mathbb{E}[\mathbf{P}_2 \cdot (\mathbf{y}_1 - \mathbf{Y}_0^\top \mathbf{w} - \mathbf{X}^\top \boldsymbol{\gamma})] = 0.

The surrogate treatment effect condition links the surrogates directly to the causal effect:

.. math::

   \mathbb{E}[\mathbf{X}^\top \boldsymbol{\gamma} - \tau] = 0.

The combined moment vector is given by:

.. math::

   \mathbf{U}(\mathbf{w}, \boldsymbol{\gamma}, \tau) =
   \begin{bmatrix}
   \mathbf{P}_1 \cdot (\mathbf{y}_1 - \mathbf{Y}_0^\top \mathbf{w} - \mathbf{X}^\top \boldsymbol{\gamma}) \\
   \mathbf{P}_2 \cdot (\mathbf{y}_1 - \mathbf{Y}_0^\top \mathbf{w} - \mathbf{X}^\top \boldsymbol{\gamma}) \\
   \mathbf{X}^\top \boldsymbol{\gamma} - \tau
   \end{bmatrix}.

Our GMM optimization becomes:

.. math::

   \arg \min_{\mathbf{w}, \boldsymbol{\gamma}, \tau} \mathbf{U}(\mathbf{w}, \boldsymbol{\gamma}, \tau)^\top \Omega^{-1} \mathbf{U}(\mathbf{w}, \boldsymbol{\gamma}, \tau),

.. tip::

   Let's take a breath and understand what's really going on here. If we're interested in the treatment effect for a treated unit, proxies will help capture time specific effects that are common across all units. But what about surrogates? Surrogates are post-intervention metrics we think will be informative of the causal effect that are similar on unit specific latent factors. This means we can even include other units we think are affected by the treatment, or entities that are within the exact same geography (if spillovers are a concern). In standard SCM studies, we would throw out these metrics. But here, we repurpose them. We remove them from the donor pool and use them to adjust our effect size based on their correlation with the treatment effect. 



.. autoclass:: mlsynth.mlsynth.PROXIMAL
   :show-inheritance:
   :special-members: __init__




Estimating Proximal Inference SCM via ``mlsynth``
----------------------------------------------------

In the paper by Liu, Tchetgen and Varjão [LiuTchetgenVar]_, the authors give an example of the Proximal Causal Inference SCM. The authors exploit the Panic of 1907, using data from [fohlin2021]_ as an intervention. Basically, this financial crisis led to the downfall of the Knickerbocker Trust, a major bank in New York City. We have price data on the stock for 59 major trusts, where Knickerbocker is the treated unit. Of the remaining 58, there are two more trusts which also experienced devastating bank runs. 7 trusts were connected to major firms. Finally, of the 58 trusts, one trust was dropped because it was missing one time periods. This leaves us with the logged price for 49 potential control units, with Knickerbocker being the treated unit. With the Proximal Inference SCM, we can use the logged bid price for the 49 unit control group as a proxy for the log-stock price of Knickerbocker. And this makes sense, since the bid price may reasonably be correlated with other macroeconomic factors which affect the overall stock price of the treated unit. Here is the code for both approaches I discuss


.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from mlsynth.mlsynth import PROXIMAL
    import matplotlib
    import os
    from theme import jared_theme

    matplotlib.rcParams.update(jared_theme)

    file_path = os.path.join(os.path.dirname(__file__), '..', 'basedata', 'trust.dta')

    # Load the CSV file using pandas
    df = pd.read_stata(file_path)

    df = df[df["ID"] != 1]  # Dropping the unbalanced unit

    surrogates = df[df['introuble'] == 1]['ID'].unique().tolist()  # Our list of surrogates
    donors = df[df['type'] == "normal"]['ID'].unique().tolist()  # Our pure controls

    vars = ["bid_itp", "ask_itp"]

    df[vars] = df[vars].apply(np.log)  # We take the log of these, per the paper.
    df['Panic'] = np.where((df['time'] > 229) & (df['ID'] == 34), 1, 0)

    # Here is when our treatment began, on the 229th tri-week.
    treat = "Panic"
    outcome = "prc_log"
    unitid = "ID"
    time = "date"

    var_dict = {
        "donorproxies": ["bid_itp"],
        "surrogatevars": ["ask_itp"]
    }

    new_directory = os.path.join(os.getcwd(), "examples")
    os.chdir(new_directory)

    # Define the 'PROXIMAL' directory
    save_directory = os.path.join(os.getcwd(), "PROXIMAL")

    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # First run
    save_1 = {
        "filename": "PanicProx",
        "extension": "png",
        "directory": save_directory,
    }

    config_1 = {
        "df": df,
        "treat": treat,
        "time": time,
        "outcome": outcome,
        "unitid": unitid,
        "treated_color": "black",
        "counterfactual_color": ["blue"],
        "display_graphs": True,
        "vars": var_dict,
        "donors": donors,
        "save": save_1
    }

    model_1 = PROXIMAL(config_1)
    SC_1 = model_1.fit()

    plt.clf()

    # Second run with surrogates and new filename
    save_2 = {
        "filename": "PanicSurrogates",
        "extension": "png",
        "directory": save_directory,
    }

    config_2 = {
        "df": df,
        "treat": treat,
        "time": time,
        "outcome": outcome,
        "unitid": unitid,
        "treated_color": "black",
        "counterfactual_color": ["blue", "red", "lime"],
        "display_graphs": True,
        "vars": var_dict,
        "donors": donors,
        "surrogates": surrogates,  # Here are our surrogates!!
        "save": save_2
    }

    model_2 = PROXIMAL(config_2)
    SC_2 = model_2.fit()


This is the plot we get when we estimate the causal impact.


.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/examples/PROXIMAL/PanicProx.png
   :alt: Proximal Synthetic Control Estimation
   :align: center
   :width: 600px


As we can see, the use of the bidding price (a proxy for supply) as a proxy causes the synthetic control to fit the pre-intervention time series well. What if we just used the post-intervention data, or added in the surrogates? In the original code, the authors used the bidding price of the two other affected trusts as surrogates. They also included the bidding price of Knickerbocker itself as a surrogate. The proxies for the surrogates was the weekly asking price for the trust. Keep in mind, the surrogates were also affected substantially by their own bank runs. But in our case that's okay because they do not contribute to the synthetic control, they are effectively used as instruments for the latent unit effects in the post-intervention period since it's reasonable to assume they are correlated with the treatment effect. Here is the plot we get when we add in the surrogates

.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/examples/PROXIMAL/PanicSurrogates.png
   :alt: Surrogate Synthetic Control Estimation
   :align: center
   :width: 600px


As we can see, even when we use only post-intervention data to estimate the causal impact, the result largely agrees with the original Proximal Inference estimates.

The authors list prediction intervals and inference for the setting of contaminated surrogates as useful directions for future work, and I agree with them. However, I think another extension may also be warranted, namely the selection of proxies and surrogates. In other words, suppose we have a high dimensional setup where we observe very many potentially relevant proxies for common time factors; which proxies should we use and why? How many should we use? If we believe many valid surrogates exist, which ones should we employ, how many, and why? Coupled with inference, I believe there are exciting future topics for this setting which policy analysts and other researchers may use.
