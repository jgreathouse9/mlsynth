Factor Model Approach
=====================

Oftentimes we assume that our truly observed data are a byproduct of a latent factor model, where there are a set of common factors across all units that influence the outcome of each unit differently, along perhaps with some set of covariates, :math:`\mathbf{Y} = \mathbf{Z} \boldsymbol{\beta} + \mathbf{F} \boldsymbol{\Lambda}^\top + \mathbf{E}`. Motivated by this apporach, Li and Sonnier [li2023statistical]_ argue that if we may estimate the number of factors that generate our outcomes in the pre-intervention period, we may use these factors to predict our counterfactual outcomes.

As ususal, we denote our units as indexed by :math:`j`, we observe :math:`\mathcal{N} \operatorname*{:=} \{1, 2, \ldots, N\}` units where the set of all units, :math:`\mathcal{N}`, has cardinality :math:`N = |\mathcal{N}|`. :math:`j = 1` is the treated unit with the controls being :math:`\mathcal{N}_0 \operatorname*{:=} \mathcal{N} \setminus \{1\}` whose cardinality is :math:`N_0 = |\mathcal{N}_0|`. Time periods are indexed by :math:`t`. Let :math:`\mathcal{T}_1 \operatorname*{:=} \{1, 2, \ldots, T_0\}` represent the pre-intervention periods, where :math:`T_0` is the final pre-intervention period, and :math:`\mathcal{T}_2 \operatorname*{:=} \{T_0 + 1, \ldots, T\}` represents the post-intervention periods. Both of these sets have cardinalities :math:`T_1 = |\mathcal{T}_1|` and :math:`T_2 = |\mathcal{T}_2|`. Let :math:`\mathcal{T} \operatorname*{:=} \mathcal{T}_1 \cup \mathcal{T}_2` represent the full time series, with cardinality :math:`T = |\mathcal{T}|`. Let :math:`\mathbf{y}_1 \in \mathbb{R}^T` be the vector for the treated unit and :math:`\mathbf{Y}_0 \in \mathbb{R}^{T \times N_0}` be the matrix for the control units that were unexposed.

Li and Sonnier [li2023statistical]_ advocate for using PCA upon the control group to estimate the latent factor matrix directly (as opposed to, say, Robust Synthetic Control by Amjad et. al. [Amjad2018]_ who seeks only to approximate the rank of the matrix instead of directly estimating the number of factors. Li and Sonnier [li2023statistical]_ advocate  describe two ways of selecting the number of factors. The first is a modified information criteron from Bai and Ng [BaiNg2002]_ and the second is a leave-one-out cross validation procedure by Xu [Xu2017]_.

To begin, modified factor selection by Bai and Ng basically adds a penalty term to the originally proposed method (see page 457 and 470 of [li2023statistical]_ for more). It penalizes the number of factors selected based on the number of units and time periods in the data.

**Modified Bai and Ng Procedure**

.. math::

   \text{MBN}(r) = \frac{1}{N_{0} T} \sum_{i=2}^{N} \sum_{t=1}^{T} 
   \left( y_{it} - \hat{\lambda}_{i}^{(r)\prime} \hat{F}_t^{(r)} \right)^2 
   + c_{N,T}r \hat{\sigma}^2 \left( \frac{N_{0} + T}{N_{0} T} \right) 
   \log \left( \frac{N_{0} + T}{N_{0} T} \right)

Li and Sonnier [li2023statistical]_ justify this method on the basis of the original criterion selecting too many factors, thus leading to poor out of sample performance.


**Xu's Cross-Validation Procedure**

Xu's method is derivied from an iterative cross validation algorithm. It proceeds along these steps:

1. **De-Mean the Data**: Naturally, we subtract the mean of the outcome matrix across time to remove unit-fixed effects.

   .. math::

      \tilde{\mathbf{Y}}_0 = \mathbf{Y}_0 - \frac{1}{T_0} \mathbf{1}_{T_0} \mathbf{1}_{T_0}' \mathbf{Y}_0

2. **SVD**: Perform SVD upon the donor pool

   .. math::

      \tilde{\mathbf{Y}}_0 \tilde{\mathbf{Y}}_0' = \mathbf{U} \boldsymbol{\Sigma} \mathbf{U}'.

   Select the first :math:`r` columns of :math:`\mathbf{U}` to form the factor matrix.

3. **Cross-Validation**: For each candidate number of factors  :math:`r`, and each pre-intervention time period :math:`s`, estimate the number of factors. Then, use OLS to predict one-step ahead out of sample into the validation period, up until the end of the training/pre-intervention period

.. math::

   r^\ast = \operatorname*{argmin}_{r \in \{1, 2, \ldots, r_{\max}\}} \sum_{s \in \mathcal{T}_1} \left( \mathbf{y}_1 - \hat{\mathbf{F}}_s^{(r)'} \hat{\boldsymbol{\lambda}}^{(r)} \right)^2


The optimal number of factors in this case is the number that minimizes the one-step out of sample validation error. Both of these methods are computed underneath the hood. ``mlsynth`` choses whichever method selects the least number of factors to avoid overfitting.

Estimating FMA in ``mlsynth``
-----------------------------


.. autoclass:: mlsynth.FMA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__



.. code-block:: python
   :linenos:

   import pandas as pd
   from mlsynth.mlsynth import FMA
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

   # Define the 'FMA' directory
   save_directory = os.path.join(os.getcwd(), "FMA")

   # Create the directory if it doesn't exist
   if not os.path.exists(save_directory):
       os.makedirs(save_directory)

   save={
           "filename": "HK_Integration",
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
       "criti": 10,
       "DEMEAN": 1
   }

   model = FMA(config)

   FMAest = model.fit()


After we run this, we get this plot saved.

.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/examples/FMA/HK_Integration.png
   :alt: Factor Model Plot
   :align: center
   :width: 600px
