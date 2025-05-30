Synthetic Interventions
=====================

.. autoclass:: mlsynth SI
   :show-inheritance:
   :special-members: __init__


Uses Synthetic Interventions to estimate how Louisiana's tobacco trends would have evolved had they done taxes on tobacco, or had they done a state-wide tobacco program.


.. code-block:: python

  import pandas as pd

  from mlsynth import SI
  
   # Load the data
   url = "https://raw.githubusercontent.com/jehangiramjad/tslib/refs/heads/master/tests/testdata/prop99.csv"

   df = pd.read_csv(url)
   
   # Filter for rows where the last column equals 2
   df = df[df.iloc[:, -1] == 2]
   
   # Keep selected columns: first 4 and the 8th (index 7)
   df = df.iloc[:, list(range(4)) + [7]]
   
   # Rename columns for clarity
   df = df.rename(columns={"Data_Value": "cigsale", "LocationDesc": "State"})
   
   # Filter for years before 2000 and sort by Year and State
   df = df[df.iloc[:, 2] < 2000].sort_values(by=[df.columns[1], df.columns[2]])
   
   # Add policy indicators
   df['SynthInter'] = ((df['State'] == 'LA') & (df.iloc[:, 2] >= 1992)).astype(int)
   df['Taxes'] = df['State'].isin(['MA', 'AZ', 'OR', 'FL']).astype(int)
   df['Program'] = df['State'].isin(['AK', 'HI', 'MD', 'MI', 'NJ', 'NY', 'WA', 'CA']).astype(int)

  
  config = {
      "df": df,
      "outcome": 'cigsale',
      "treat": 'SynthInter',
      "unitid": 'State',
      "time": 'Year',
      "display_graphs": True,
      "save": False,
      "counterfactual_color": ["red", "blue"],
      "inters": ["Taxes", "Program"],
      "objective": "OLS"
  }
  
  arco = SI(config).fit()
