Panel Data Approach Explained
=============================

As with any causal study, the results we get depend on how we choose or give weight to units in the control group. Many recent developments in the causal inference literature focus on developing some linear/convex combination of untreated units to approximate the pre-intervention values of a treated unit, such  that we may learn their our of sample counterfactuals had the unit never been treated. Synthetic control methods are probably the most prevalent formulation of that idea, ususually taking the form of a covex combination of donor/control units. The basic idea of the panel data approach, essentially, is that we may use unconstrained regression methods such as OLS, LASSO, or others to construct the counterfactual using (mainly) pre-intervention data. Part of the upshoot of this, as Li and van den Bulte [ADID]_ document in their paper, is that panel data approaches are a lot more flexible than synthetic control methods, both in terms of their estimation and (occasionally) inference.  ``mlsynth`` implements three such implementations of the panel data approach, and there are certainly more to choose from. ``mlsynth`` estimates the LASSO method [LASSOPDA]_, the forward selection method [fsPDA]_, and the :math:`\ell_2`-relaxation method by [l2relax]_. The main difference between these methods is in how they select or weigh the control units. The LASSO, for example, assumes that a sparse set of coefficients driven by some common factors will be a good proxy for the out-of-sample counterfactual. Forward selection presumes the same, using a forward selection algorithm to choose the control units for the treated unit with a modified Bayesian Information Criterion to stop selection. the :math:`\ell_2` relaxation method, instead, assumes a dense data generating process, instead minimizing the upper bound of the deviation between the treated unit and the donor pool/control group.

Formally, I lay out the notations. Indexed by :math:`j`, we observe :math:`\mathcal{N} \operatorname*{:=} \{1, 2, \ldots, N\}` units where the set :math:`\mathcal{N}` has cardinality :math:`N = |\mathcal{N}|`. :math:`j = 1` is the treated unit with the controls being :math:`\mathcal{N}_0 \operatorname*{:=} \mathcal{N} \setminus \{1\}` whose cardinality is :math:`N_0 = |\mathcal{N}_0|`. Time periods are indexed by :math:`t`. Let :math:`\mathcal{T}_1 \operatorname*{:=} \{1, 2, \ldots, T_0\}` represent the pre-intervention periods, where :math:`T_0` is the final pre-intervention period, and :math:`\mathcal{T}_2 \operatorname*{:=} \{T_0 + 1, \ldots, T\}` represents the post-intervention periods. Both of these sets have cardinalities :math:`T_1 = |\mathcal{T}_1|` and :math:`T_2 = |\mathcal{T}_2|`. Let :math:`\mathcal{T} \operatorname*{:=} \mathcal{T}_1 \cup \mathcal{T}_2` represent the full time series, with cardinality :math:`T = |\mathcal{T}|`.  Let :math:`\mathbf{y}_1 \in \mathbb{R}^T` be the vector for the treated unit and :math:`\mathbf{Y}_0 \in \mathbb{R}^{T \times N_0}` be the matrix for the control units that were unexposed. Furthermore, let :math:`\boldsymbol{\beta} \in \mathbb{R}^{N_0}` be the coefficients for some controls.


:math:`\ell_2` Relaxation
-----------------------

The first approach we discuss is the :math:`\ell_2`-relaxation PDA. Much criticism has been made of sparse-models such as the LASSO or other methods designed to induce sparsity, especially in situations where the true underlying model may actually be dense [l2relax]_. In such situations, the LASSO will perform poorly for variable selection tasks because it will assign zeros to informative predictors, or even select the wrong variables altogether. As an answer to this, econometricians increasingly have experimented with the Ridge regression estimator as a tool of interest, say as Amjad et. al [Amjad2018] does with their Robust Synthetic Control estimator. In paticular, they argue that the primary thing of interest should be our out of sample predictions; so, imposing sparsity or nonnegative weights, while easily interpretable, may not be the best in practice. 

Panel data approaches generally assume that there are a common set of factors that generate our outcomes of interest, which interact with units differently over time. We wish to exploit the pre-intervention correlations between the treated unit and the control units across time in order to generate our counterfactual predicitons. In this scenario, we literally do this, by first estimating the sample covariance vector between the treated unit and donor matrix, while taking into account the inter-donor correlations. Essentially, we are computing two Gram matrices. With this in mind, our optimization problem becomes:

.. math::

   \begin{aligned}
   &\min_{\boldsymbol{\beta}} \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 \quad \text{subject to } \|\boldsymbol{\eta} - \boldsymbol{\Sigma}  \boldsymbol{\beta}\|_\infty \leq \tau \\
   &\boldsymbol{\eta} = \frac{1}{T_1} \mathbf{Y}_0^\top \mathbf{y}_1, \: \boldsymbol{\Sigma} = \frac{1}{T_1} \mathbf{Y}_0^\top \mathbf{Y}_0 \in \mathbb{R}^{N_0 \times N_0}
   \end{aligned}

where we seek the coefficients which minimize the predictions between the sample covariance vector and the covariance matrix of the control units. In other words, we are projecting the treated unit on to the control units. We are doing this with the sep-norm, which in this case simply places an upper bound on the deviation from the covariance vector, using the constant tau. There is a key problem here, though: as tau shrinks to 0, we approach the OLS estimator, which seeks to totally minimize the discrepancies. Of course, this will result in overfitting. So to mitigate this, Shi and Wang employ cross validation to select tau [l2relax]_, and I follow them here. I divide the data into the training period and the out-of-sample validation periods (that is, the post-intervention period). I further divide the training period into :math:`\mathcal{T}_1^{\text{train}} = \{1, 2, \ldots, \left\lfloor \frac{T_1}{2} \right\rfloor\}` for training, and the validation period :math:`\mathcal{T}_2^{\text{val}} = \{\left\lfloor \frac{T_1}{2} \right\rfloor\ +1, \ldots, T_0\}` for out-of-sample testing. For our purposes, we are concerned with the value for tau that minimizes the MSE for the valiation period



.. math::

    \tau^{\ast} = \operatorname*{argmin}_{\tau} \left( \frac{1}{|\mathcal{T}_2^{\text{val}}|} \| \mathbf{y}^{\ell_2} - \mathbf{y}_1 \|_F^2 \right)



The code below replicates [l2relax]_ who themselves are replicating [HCW]_. The goal is to see how the economic year over year growth rate of Hong King would have evolved had it not become economically integrated with the mainland of China.

.. code-block:: python

    import pandas as pd
    from mlsynth.mlsynth import PDA
    import matplotlib
    import os

    jared_theme = {'axes.grid': True,
                  'grid.linestyle': '-',
                  'legend.framealpha': 1,
                  'legend.facecolor': 'white',
                  'legend.shadow': True,
                  'legend.fontsize': 12,
                  'legend.title_fontsize': 14,
                  'xtick.labelsize': 12,
                  'ytick.labelsize': 12,
                  'axes.labelsize': 12,
                  'axes.titlesize': 20,
                  'figure.dpi': 100,
                   'axes.facecolor': 'white',
                   'figure.figsize': (11, 6)}

    matplotlib.rcParams.update(jared_theme)


    def load_and_process_data():
        """
        Loads the GDP data, processes it, and returns the DataFrame with additional columns.

        Returns:
            pd.DataFrame: Processed DataFrame with columns 'Country', 'GDP', 'Time', and 'Integration'.
        """
        # Define column names
        column_names = [
            "Hong Kong", "Australia", "Austria", "Canada", "Denmark", "Finland",
            "France", "Germany", "Italy", "Japan", "Korea", "Mexico", "Netherlands",
            "New Zealand", "Norway", "Switzerland", "United Kingdom", "United States",
            "Singapore", "Philippines", "Indonesia", "Malaysia", "Thailand", "Taiwan", "China"
        ]

        # Load the dataset
        df = pd.read_csv(
            "https://raw.githubusercontent.com/leoyyang/rhcw/master/other/hcw-data.txt",
            header=None,
            delim_whitespace=True,
        )

        # Assign column names
        df.columns = column_names

        # Melt the dataframe
        df = pd.melt(df, var_name="Country", value_name="GDP", ignore_index=False)

        # Add 'Time' column ranging from 0 to 60
        df["Time"] = df.index

        # Create 'Integration' column based on conditions
        df["Integration"] = (df["Country"].str.contains("Hong") & (df["Time"] >= 44)).astype(int)

        return df

    df = load_and_process_data()

    treat = "Integration"
    outcome = "GDP"
    unitid = "Country"
    time = "Time"


    new_directory = os.path.join(os.getcwd(), "examples")
    os.chdir(new_directory)

    save_directory = os.path.join(os.getcwd(), "l2relax")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save={
            "filename": "HK_Integration", # The title of the plot
            "extension": "png",
            "directory": save_directory
    }

    config = {
        "df": df,
        "treat": treat,
        "time": time,
        "outcome": outcome,
        "unitid": unitid,
        "counterfactual_color": "blue",
        "treated_color": "black",
        "display_graphs": True,
        "save": save,
        "method": "l2" # Or, "LASSO" or "fs"
    }

    model = PDA(config)

    autores = model.fit()

When we estimate the counterfactual, we get


.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/examples/l2relax/HK_Integration.png
   :alt: Counterfactual Hong Kong
   :align: center
   :width: 600px

Forward Selected Approach
-------------------------

LASSO Approach
--------------
