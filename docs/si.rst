Synthetic Interventions
=====================

.. autoclass:: mlsynth.mlsynth.SI
   :show-inheritance:
   :special-members: __init__


Uses Synthetic Interventions to estimate how Louisiana's tobacco trends would have evolved had they done taxes on tobacco, or had they done a state-wide tobacco program.


.. code-block:: python

  import pandas as pd
  from mlsynth.mlsynth import SI
  
  # Load the data
  url = "https://raw.githubusercontent.com/jehangiramjad/tslib/refs/heads/master/tests/testdata/prop99.csv"
  df = pd.read_csv(url)
  last_column = df.columns[-1]
  df_filtered = df[df[last_column] == 2]
  
  columns_to_keep = list(df.columns[0:4]) + [df.columns[7]]
  df_filtered = df_filtered[columns_to_keep]
  
  sort_columns = [df_filtered.columns[1], df_filtered.columns[2]]
  df_filtered = df_filtered.sort_values(by=sort_columns)
  
  df_filtered = df_filtered[df_filtered[df_filtered.columns[2]] < 2000]
  
  df_filtered['SynthInter'] = ((df_filtered[df_filtered.columns[0]] == 'LA') &
                           (df_filtered[df_filtered.columns[2]] >= 1992)).astype(int)
  
  
  tax_states = ['MA', 'AZ', 'OR', 'FL']  # Massachusetts, Arizona, Oregon, Florida abbreviations
  df_filtered['Taxes'] = (df_filtered[df_filtered.columns[0]].isin(tax_states)).astype(int)
  
  program_states = ['AK', 'HI', 'MD', 'MI', 'NJ', 'NY', 'WA', 'CA']
  df_filtered['Program'] = (df_filtered[df_filtered.columns[0]].isin(program_states)).astype(int)
  
  df_filtered = df_filtered.rename(columns={"Data_Value": "cigsale", "LocationDesc": "State"})
  
  df_filtered = df_filtered.rename(columns={"Data_Value": "cigsale", "LocationDesc": "State"})
  
  config = {
      "df": df_filtered,
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
