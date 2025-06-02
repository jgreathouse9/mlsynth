Synthetic Regression Control Method
=====================

.. autoclass:: mlsynth.SRC
   :show-inheritance:
   :special-members: __init__


Uses Synthetic Regressing Control Method for the Basque data


.. code-block:: python

  import pandas as pd
  from mlsynth.mlsynth import SRC
  import matplotlib.pyplot as plt
  import matplotlib
  
  # URL to fetch the dataset
  url = 'https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/basque_data.csv'
  df = pd.read_csv(url)
  
  treat = df.columns[-1]
  time = df.columns[1]
  outcome = df.columns[2]
  unitid = df.columns[0]
  
  config = {
      "df": df,
      "treat": treat,
      "time": time,
      "outcome": outcome,
      "unitid": unitid,
      "display_graphs": True,
      "counterfactual_color": "blue"
  }
  
  result = SRC(config).fit()
