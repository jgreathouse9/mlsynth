Example: Estimating the Impact of the Asian Financial Crisis on the Malaysian Ringgit
-------------------------------------------------------------------------------------

This example uses the **Synthetic Historical Control (SHC)** estimator from
``mlsynth`` to measure the Year-over-Year (YoY) growth rate impact of the
1997–1998 Asian Financial Crisis on the Malaysian Ringgit.

.. autoclass:: mlsynth.SHC
   :show-inheritance:
   :special-members: __init__

.. code-block:: python

    from mlsynth import SHC
    import pandas as pd

    # Load monthly exchange rate data from FRED
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        "bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&"
        "graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&"
        "txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&"
        "show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=EXMAUS&"
        "scale=left&cosd=1971-01-01&coed=2025-06-01&line_color=%230073e6&"
        "link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&"
        "ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&"
        "fgsnd=2020-02-01&line_index=1&transformation=lin&"
        "vintage_date=2025-07-17&revision_date=2025-07-17&nd=1971-01-01"
    )

    df = pd.read_csv(url)

    # Rename and parse date column
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])

    # Compute Year-over-Year growth rate
    value_col = df.columns[1]
    df["YoY Growth Rate"] = df[value_col].pct_change(periods=12)
    df = df.dropna(subset=["YoY Growth Rate"])

    # Drop original exchange rate column
    df = df.drop(columns=df.columns[1])

    # Keep only data prior to 1999
    df = df[df["Date"].dt.year < 1999]

    # Assign unit identifier
    df["Unit"] = "Ringgit"

    # Define treatment: post-July 1997 = 1
    df["Asian Financial Crisis"] = (
        df["Date"] >= pd.Timestamp("1997-07-01")
    ).astype(int)

    # Configuration for SHC
    config = {
        "df": df,
        "outcome": df.columns[1],       # "YoY Growth Rate"
        "treat": df.columns[-1],        # "Asian Financial Crisis"
        "unitid": "Unit",
        "time": df.columns[0],          # "Date"
        "display_graphs": True,
        "save": False,
        "counterfactual_color": ["blue"],
        "m": 12 * 3                      # Segment Length
    }







Developer API
=============

These functions are part of the internal optimization and tuning routines
for SHC and ASHC. When  :class:`use_augmented` is true, we pass the original SHC weights chosen by forward selection to the ASHC for bias correction, if necessary.

.. autofunction:: mlsynth.utils.estutils._solve_SHC_QP
.. autofunction:: mlsynth.utils.estutils.tune_lambda_ashc
