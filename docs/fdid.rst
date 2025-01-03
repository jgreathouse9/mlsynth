Forward DID Method
==================

This is the documentation for the Forward DID method.

.. autoclass:: mlsynth.mlsynth.FDID
    :members: __init__, fit
    :undoc-members:  # Optional: Use if you want undocumented parts of __init__ or fit
    :show-inheritance:


FDID proceeds iteratively over :math:`k` total candidate iterations. The main regression specification is

.. math::
    \mathbf{y}_{1} = \widehat{\boldsymbol{\beta}}_{0} + \mathbf{Y}^{\prime}_{\widehat{U}_{k-1} \cup \{i\}} \widehat{\boldsymbol{\beta}}_{\widehat{U}_{k-1} \cup \{i\}} 
            \quad \text{s.t. } \widehat{\boldsymbol{\beta}}_{\widehat{U}_{k-1}} = \frac{1}{U_{k-1}}, \quad \forall t \in \mathcal{T}_1


We begin with an empty control group. Forward DID proceeds in the manner of forward-selection. Let :math:`\mathcal{U} \operatorname*{:=} \{\widehat{U}_1, \widehat{U}_2, \ldots, \widehat{U}_{N_0}\}` represent the set of candidate control groups. For :math:`k=1`, we estimate :math:`N_0` one unit DID models, using the pre-intervention outcomes of our treaated unit as the dependent variable and the outcomes of the control units as predictors. Of these, we select the control unit which produces the highest R-squared statistic,

.. math::

   i^\ast_1 = \operatorname*{argmax}_{i \in \mathcal{N}_0} R^2_i, \quad \widehat{U}_1 = \{i^\ast_1\}

We add this one unit set to :math:`\{\mathcal{U} \operatorname*{:=} U_1 \}` as our first selected control unit. For :math:`k=2`, we repeat this again, except now we include the first selected control one of the remaining control units, estimating :math:`N_0 - 1` DID models in total. After this, we choose the optimal two control unit combination (optimal in the sense that the DID model maximizes R-squared in the pre-intervention period)

.. math::

   i^\ast_2 = \operatorname*{argmax}_{i \in \mathcal{N}_0 \setminus \{i^\ast_1\}} R^2_{\{i^\ast_1, i\}}, 
   \quad \widehat{U}_2 = \{i^\ast_1, i^\ast_2\}.

We then repeat this a third time. For each iteration :math:`k`, we loop over the remaining set of control units and add the first one, first two, and first three best control untis, selecting the optimal set of control units per model:

.. math::

   i^\ast_k = \operatorname*{argmax}_{i \in \mathcal{N}_0 \setminus \widehat{U}_{k-1}} R^2_{\widehat{U}_{k-1} \cup \{i\}}, 
   \quad \widehat{U}_k = \widehat{U}_{k-1} \cup \{i^\ast_k\}.

These candidate sets of optimal controls are added to :math:`\mathcal{U}` until :math:`k = N_0`. The control group ultimately returned by FDID is :math:`\widehat{U} \operatorname*{:=} \operatorname*{argmax}_{\widehat{U}_k \in \mathcal{U}} R^2(\widehat{U}_k)`, or the control group that has the highest R-squared statistic of them all.


.. code-block:: python

    from mlsynth.mlsynth import FDID
    import matplotlib
    import pandas as pd
    # matplotlib theme
    jared_theme = {
        'axes.grid': True,
        'grid.linestyle': '-',
        'grid.color': 'black',
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
        'axes.facecolor': '#b2beb5',
        'figure.figsize': (10, 6)
    }

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
                subdf = subdf[~subdf[params["Panel"]].str.contains("Spain \\(Espana\\)")]
                subdf.loc[subdf['regionname'].str.contains('Vasco'), 'regionname'] = 'Basque'

            # Append the edited DataFrame to the list
            edited_frames.append(subdf)

        return edited_frames


    # Example usage
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
            "Treatment Time": 1978,
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

    number = 0
    df = edited_frames[number]

    # Get the keys as a list
    keys_list = list(base_dict.keys())

    # Match based on position
    position = number  # For "Basque"
    selected_key = keys_list[position]

    # Access the corresponding dictionary
    selected_dict = base_dict[selected_key]

    # Example: Accessing specific values
    columns = selected_dict["Columns"]
    treatment_name = selected_dict["Treatment Name"]

    # Example usage
    unitid = df.columns[0]
    time = df.columns[1]
    outcome = df.columns[2]
    treat =  selected_dict["Treatment Name"]

    config = {
        "df": df,
        "treat": treat,
        "time": time,
        "outcome": outcome,
        "unitid": unitid,
        "counterfactual_color": "#7DF9FF",  # Optional, defaults to "red"
        "treated_color": "red",  # Optional, defaults to "black"
        "display_graphs": True  # Optional, defaults to True
    }

    model = FDID(config)

    # Run the FDID analysis
    autores = model.fit()


Next.
