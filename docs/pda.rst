Panel Data Approach
=====================

.. autoclass:: mlsynth.mlsynth.PDA
   :show-inheritance:
   :special-members: __init__

As with any causal study, the results we get depend on how we choose or give weight to units in the control group. Many recent developments in the causal inference literature focus on developing some linear/convex combination of untreated units to approximate the pre-intervention values of a treated unit, such  that we may learn their out of sample counterfactuals had the unit never been treated. Synthetic control methods are probably the most prevalent formulation of that idea, typically taking the form of a convex combination of donor/control units. The basic idea of the panel data approach, essentially, is that we may use unconstrained regression methods such as OLS, LASSO, or others to construct the counterfactual using (mainly) pre-intervention data. Part of the upshoot of this, as Li and van den Bulte [ADID]_ document in their paper, is that panel data approaches are a lot more flexible than synthetic control methods, both in terms of their estimation and (occasionally) inference.  ``mlsynth`` implements three such implementations of the panel data approach, and there are certainly more to choose from. ``mlsynth`` estimates the LASSO method [LASSOPDA]_, the forward selection method [fsPDA]_, and the :math:`\ell_2`-relaxation method by [l2relax]_. The main difference between these methods is in how they select or weigh the control units. The LASSO, for example, assumes that a sparse set of coefficients driven by some common factors will be a good proxy for the out-of-sample counterfactual. Forward selection presumes the same, using a forward selection algorithm to choose the control units for the treated unit with a modified Bayesian Information Criterion to stop selection. the :math:`\ell_2` relaxation method, instead, assumes a dense data generating process, instead minimizing the upper bound of the deviation between the treated unit and the donor pool/control group.

Formally, I lay out the notations. Indexed by :math:`j`, we observe :math:`\mathcal{N} \operatorname*{:=} \{1, 2, \ldots, N\}` units where the set :math:`\mathcal{N}` has cardinality :math:`N = |\mathcal{N}|`. :math:`j = 1` is the treated unit with the controls being :math:`\mathcal{N}_0 \operatorname*{:=} \mathcal{N} \setminus \{1\}` whose cardinality is :math:`N_0 = |\mathcal{N}_0|`. Time periods are indexed by :math:`t`. Let :math:`\mathcal{T}_1 \operatorname*{:=} \{1, 2, \ldots, T_0\}` represent the pre-intervention periods, where :math:`T_0` is the final pre-intervention period, and :math:`\mathcal{T}_2 \operatorname*{:=} \{T_0 + 1, \ldots, T\}` represents the post-intervention periods. Both of these sets have cardinalities :math:`T_1 = |\mathcal{T}_1|` and :math:`T_2 = |\mathcal{T}_2|`. Let :math:`\mathcal{T} \operatorname*{:=} \mathcal{T}_1 \cup \mathcal{T}_2` represent the full time series, with cardinality :math:`T = |\mathcal{T}|`. Let :math:`\mathbf{y}_1 \in \mathbb{R}^T` be the vector for the treated unit and :math:`\mathbf{Y}_0 \in \mathbb{R}^{T \times N_0}` be the matrix for the control units that were unexposed. Furthermore, let :math:`\boldsymbol{\beta} \in \mathbb{R}^{N_0}` be the coefficients for some controls. The sup norm of a vector :math:`\mathbf{y} \in \mathbb{R}^N` is defined as the maximum absolute value of its components, :math:`\|\mathbf{y}\|_\infty = \max_{j = 1, \ldots, N} |y_j|`. The floor function of a real number :math:`x \in \mathbb{R}`, denoted :math:`\lfloor x \rfloor`, returns :math:`\lfloor x \rfloor = \max \{k \in \mathbb{N}_+ : k \leq x\}`.


:math:`\ell_2` Relaxation
-----------------------

The first approach we discuss is the :math:`\ell_2`-relaxation PDA. Much criticism has been made of sparse-models such as the LASSO or other methods designed to induce sparsity, especially in situations where the true underlying model may actually be dense [l2relax]_. In such situations, the LASSO will perform poorly for variable selection tasks because it will assign zeros to informative predictors, or even select the wrong variables altogether. As an answer to this, econometricians increasingly have experimented with the Ridge regression estimator as a tool of interest, say as Amjad et. al [Amjad2018]_ does with their Robust Synthetic Control estimator. In paticular, they argue that the primary thing of interest should be our out of sample predictions; so, imposing sparsity or nonnegative weights, while easily interpretable, may not be the best in practice. 

Panel data approaches generally assume that there are a common set of factors that generate our outcomes of interest, which interact with units differently over time. We wish to exploit the pre-intervention correlations between the treated unit and the control units across time in order to generate our counterfactual predicitons. :math:`\ell_2` PDA does this, per Equation 10 of the paper, with the regression model :math:`\mathbf{y}_1 = \boldsymbol{\alpha}+ \mathbf{Y}_0\boldsymbol{\beta}+\mathbf{\epsilon}`, which as we can see includes an intercept. We first estimate the sample covariance vector between the treated unit and donor matrix and the covariance matrix of the control group (two Gram matrices). With this in mind, our optimization problem becomes:

.. math::

   \begin{aligned}
   &\min_{\boldsymbol{\beta}} \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 \: \forall t \in \, \mathcal{T}_1, \quad \text{subject to } \|\boldsymbol{\eta} - \boldsymbol{\Sigma}  \boldsymbol{\beta}\|_\infty \leq \tau \\
   &\boldsymbol{\eta} = \frac{1}{T_1} \mathbf{Y}_0^\top \mathbf{y}_1 \\
   &\boldsymbol{\Sigma} = \frac{1}{T_1} \mathbf{Y}_0^\top \mathbf{Y}_0 \in \mathbb{R}^{N_0 \times N_0}
   \end{aligned}

where we seek the coefficients which minimize the predictions between the sample covariance vector and the covariance matrix of the control units. In other words, we are projecting the treated unit on to the control units. We are doing this with the sup-norm, which in this case simply places an upper bound on the deviation from the covariance vector, using the constant tau. There is a key problem here, though: as tau shrinks to 0, we approach the OLS estimator, which seeks to totally minimize the discrepancies. Of course, this will result in overfitting. So to mitigate this, Shi and Wang employ cross validation to select tau [l2relax]_, and I follow them here. I divide the data into the training period and the out-of-sample validation periods (that is, the post-intervention period). I further divide the training period into :math:`\mathcal{T}_1^{\text{train}} = \{1, 2, \ldots, \left\lfloor \frac{T_1}{2} \right\rfloor\}` for training, and the validation period :math:`\mathcal{T}_2^{\text{val}} = \{\left\lfloor \frac{T_1}{2} \right\rfloor\ +1, \ldots, T_0\}` for out-of-sample testing. For our purposes, we are concerned with the value for tau that minimizes the MSE for the valiation period

.. math::

    \tau^{\ast} = \operatorname*{argmin}_{\tau} \left( \frac{1}{|\mathcal{T}_2^{\text{val}}|} \| \mathbf{y}^{\ell_2} - \mathbf{y}_1 \|_2^2 \right)

The code below replicates [l2relax]_ who themselves are replicating [HCW]_. The goal is to see how the year over year growth rate of Hong Kong's GDP would have evolved had it not become economically integrated with the mainland of China.

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

Next I describe the forward-selection PDA implemented by ``mlsynth``. In fsPDA, the control group is selected using forward selection. The selection method iteratively chooses control units to maximize the model's explanatory power based on the :math:`R^2` statistic.

We begin with an empty set of selected control units :math:`\hat{U}_0 = \emptyset`. Our iterations span :math:`r = {1, 2, \ldots, R}`. For the first iteration, we estimate a single OLS regression model per each control unit :math:`j \in \mathcal{N}_0 \setminus \hat{U}_{r-1}`. In this instance,  :math:`\mathbf{y}_1` is predicted by the set of previously selected controls :math:`\mathbf{Y}_{\hat{U}_{r-1}}` plus the candidate control unit :math:`\mathbf{y}_j`. After our first iteration, we select the control unit :math:`j_r` that maximizes the :math:`R^2` of the regression. We then update the selected set: :math:`\hat{U}_r = \hat{U}_{r-1} \cup \{j_r\}`. The next model proceeds the same way, including the originally selected unit. The process stops after :math:`R` iterations, where :math:`R` is chosen by a modified Bayesian Information Criterion as described by Shi and Huang (2023) [fsPDA]_.

After selecting the control group, the counterfactual for the treated unit is predicted using the following regression model:

.. math::

    (\hat{\alpha}, \hat{\boldsymbol{\beta}}_{\hat{U}_r}) = \operatorname*{argmin}_{\alpha, \boldsymbol{\beta}_{\hat{U}_r}} 
    \|\mathbf{y}_1 - \mathbf{Y}_{\hat{U}_r} \boldsymbol{\beta}_{\hat{U}_r}- \alpha\|_2^2.

Here is the example repeated above, except with the forward selection PDA. Note that all we needed to do is just change around a few parameters, instead of needing to switch softwares or learn a new syntax.


.. code-block:: python

    # Update the method
    config["method"] = "fs"

    # Create the 'fsPDA' directory
    save_directory = os.path.join(os.getcwd(), "fsPDA")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Update the save file
    config["save"] = {
        "filename": "HK_Integration_fs",
        "extension": "png",
        "directory": save_directory
    }

    # Initialize the model with forward selection
    model_fs = PDA(config)

    # Fit the fsPDA model
    autores_fs = model_fs.fit()


.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/examples/fsPDA/HK_Integration_fs.png
   :alt: Counterfactual Hong Kong
   :align: center
   :width: 600px


LASSO PDA
--------------

The objective function for LASSO's PDA is given by:

.. math::

    \hat{\boldsymbol{\beta}} = \operatorname*{argmin}_{\boldsymbol{\beta}} \|\mathbf{y}_1 - \mathbf{Y}_0 \boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1,

In this PDA, we simply use the LASSO to choose the control units, as implemented by ``sklearn``.

This code, implemented like

.. code-block:: python

    # Update the method to "LASSO" for LASSO-based PDA
    config["method"] = "LASSO"

    # Create the 'L1PDA' directory for saving results
    save_directory = os.path.join(os.getcwd(), "L1PDA")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Update the save configuration
    config["save"] = {
        "filename": "HK_Integration",
        "extension": "png",
        "directory": save_directory
    }

    # Initialize the model with LASSO-based PDA
    model_fs = PDA(config)

    # Fit the model using LASSO-based PDA
    LASSO_est = model_fs.fit()



returns this plot


.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/examples/L1PDA/HK_Integration.png
   :alt: Counterfactual Hong Kong
   :align: center
   :width: 600px


To-Do List
----------------

- Multiple treated units for the :math:`\ell_2` approach
- Maybe implement HCW for comparison's sake.
- Maybe implement more PDAs, since quite a few exist...




