Forward SCM
=====================

.. autoclass:: mlsynth.FSCM
   :show-inheritance:
   :special-members: __init__

Uses Forward Selection to choose the donor pool for the vanilla SCM.

.. code-block:: python

    import pandas as pd  # To work with panel data
    from IPython.display import display, Markdown  # To create the table
    from mlsynth.mlsynth import FSCM  # The method of interest

    url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/smoking_data.csv"

    data = pd.read_csv(url)

    config = {
        "df": data,
        "outcome": data.columns[2],
        "treat": data.columns[-1],
        "unitid": data.columns[0],
        "time": data.columns[1],
        "display_graphs": True,
        "save": False,
        "counterfactual_color": "red"
    }

    arco = FSCM(config).fit()


fSCM function
=====================

.. autofunction:: mlsynth.utils.estutils.fSCM
   :noindex:

Performs Forward Selection Synthetic Control (FSCM) estimation.  
If ``augmented=True``, post-selection refinement is performed using affine-hull regularization.

Returns:
- Augmented weights (if applicable)
- Original SCM weights


fit_affine_hull_scm
=====================

.. autofunction:: mlsynth.utils.estutils.fit_affine_hull_scm
   :noindex:

Refines synthetic control weights using affine-hull constrained ridge regression.  
The ridge penalty is tuned via Bayesian optimization with a training-validation split on pre-treatment data.

Returns:
- Refined weight vector
- Optimal regularization parameter (beta)
