import requests
import pandas as pd
import json
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

def get_smoothed_series(df, state_name, period=12, robust=True):
    """
    Returns a smoothed (trend) homicide rate series for a given state using STL decomposition.

    Parameters
    ----------
    df : pd.DataFrame
        Output from get_elcrimen_data(), must include 'State', 'date', 'Homicide Rate'.
    state_name : str
        Name of the state to process (e.g., 'Chihuahua').
    period : int, optional
        Seasonal period (12 for monthly data).
    robust : bool, optional
        Whether to use robust fitting in STL.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['date', 'Homicide Rate', 'Trend', 'Seasonal', 'Residual'].
    """

    # --- Subset and ensure monthly continuity ---
    state_df = df[df['State'] == state_name].copy()
    if state_df.empty:
        raise ValueError(f"State '{state_name}' not found in dataset.")

    state_df = state_df.set_index('date').asfreq('MS')  # monthly start frequency
    state_df['Homicide Rate'] = state_df['Homicide Rate'].interpolate()

    # --- Apply STL decomposition ---
    stl = STL(state_df['Homicide Rate'], period=period, robust=robust)
    result = stl.fit()

    # --- Add components ---
    state_df['Trend'] = result.trend
    state_df['Seasonal'] = result.seasonal
    state_df['Residual'] = result.resid
    state_df['Seasonally Adjusted'] = state_df['Homicide Rate'] - state_df['Seasonal']

    # --- Reset index for convenience ---
    return state_df.reset_index()

def get_elcrimen_data(year_start=None, year_end=None):
    """
    Fetches EL CRiMEN homicide data and maps region codes to state names.

    Parameters:
        year_start (int, optional): Filter data starting from this year.
        year_end (int, optional): Filter data up to this year.

    Returns:
        pd.DataFrame: Cleaned DataFrame with columns ['region_name', 'date', 'r']
    """

    # --- Step 1: Fetch homicide JSON ---
    cookies = {
        '_ga': 'GA1.1.608703246.1760279437',
        'FCNEC': '%5B%5B%22AKsRol-Nr5UfqJuHY_Zvt18cukk6BTKrcwEiGwmumu86KhcIpWZm_mPzTGfUlO8Kebqa_r_5_eIcrfy52uiJjLW81ZfHeZZr2MIjZrGRJREiNh2lLo_pvnokKwK7PXN9ntfdfZhOPcxn3fDFcn4jkS_PexG4USKU1Q%3D%3D%22%5D%5D',
        '_ga_SMLSV8EVFV': 'GS2.1.s1760279437$o1$g1$t1760280490$j11$l0$h0',
        'ph_phc_362TiSyKgdQYtDzSZNEHI9tvena7mln2IkZaVfZN3bK_posthog': '%7B%22distinct_id%22%3A%220199d8d4-973b-7426-8099-1f3e6d5ba949%22%2C%22%24sesid%22%3A%5B1760280778651%2C%220199d8d4-9779-79b3-8c5a-517a8ae2aa92%22%2C1760279435087%5D%2C%22%24initial_person_info%22%3A%7B%22r%22%3A%22%24direct%22%2C%22u%22%3A%22https%3A%2F%2Felcri.men%2Fen%2F%22%7D%7D',
    }

    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'priority': 'u=1, i',
        'referer': 'https://elcri.men/en/',
        'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
    }

    # Fetch JSON
    response = requests.get('https://elcri.men/elcrimen-json/national_1990.json', cookies=cookies, headers=headers)
    data = json.loads(response.text)

    # --- Step 2: Flatten JSON into DataFrame ---
    rows = []
    for region, region_data_list in data.items():
        # Only take the second list (index 1)
        full_series = region_data_list[1]
        for obs in full_series:
            rows.append({
                "region": region,
                "date": obs["d"],
                "Rate": obs["r"],
                "Count": obs["c"],
                "Population": obs["p"]
            })

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])

    # --- Step 3: Replace 'national' and map region codes to state names ---
    df.loc[df['region'] == 'national', 'region'] = '33'

    # Load population data for state mapping
    pop_url = "https://raw.githubusercontent.com/diegovalle/elcrimen-trends/refs/heads/master/data/states_pop.csv"
    pop_df = pd.read_csv(pop_url)
    code_to_state = pop_df[['state_code', 'state_name']].drop_duplicates().set_index('state_code')[
        'state_name'].to_dict()

    # Convert to int and map
    df['region'] = df['region'].astype(int)
    df['region_name'] = df['region'].map(code_to_state)
    df = df.drop(columns=['region'])

    df = df.set_index('date')

    # Aggregate quarterly by sum of counts and average population
    quarterly = df.groupby(['region_name']).resample('QE').agg({
        'Count': 'sum',  # total homicides per quarter
        'Population': 'mean'  # average population
    }).reset_index()

    # Compute quarterly rate per 100k
    quarterly['Homicide Rate'] = quarterly['Count'] / quarterly['Population'] * 100000

    # --- Step 4: Optional filtering by year ---
    if year_start is not None:
        quarterly = quarterly[quarterly['date'].dt.year >= year_start]
    if year_end is not None:
        quarterly = quarterly[quarterly['date'].dt.year <= year_end]

    # Sort for clarity
    quarterly = quarterly.sort_values(by=['region_name', 'date']).reset_index(drop=True)

    quarterly = quarterly.rename(columns={'region_name': 'State'})

    quarterly = quarterly.drop_duplicates(subset=['State', 'date'], keep='first').reset_index(drop=True)

    quarterly = quarterly.drop(columns=['Population', 'Count'])

    return quarterly

# --- Get base dataset ---
df = get_elcrimen_data(year_start=1990, year_end=2025)

df.to_csv("MexicanHomicideData.csv", index=False)



