# tests/conftest.py
import pytest
import numpy as np
from sim_functions import simulate_districts, sim_to_long_df  # Import from your stashed file

@pytest.fixture
def curacao_sim_data():
    """Fixture providing simulated Curaçao tourism spending data for MAREX tests."""
    sim_data = simulate_districts(districts, seed=42, T0=104, T1=24)
    df = sim_to_long_df(sim_data, T0=104)
    df['Region'] = (df['district'] == "Willemstad").astype(int)
    clusters = df['Region'].iloc[::128].values  # First value per town
    costs = np.random.uniform(1, 5, 21).repeat(128)
    return {"df": df, "clusters": clusters, "costs": costs, "sim_data": sim_data}
