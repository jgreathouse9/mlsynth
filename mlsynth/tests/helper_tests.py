# tests/helper_tests.py

import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def incrementality_synth_panel(
    pre_start="2024-11-01",
    pre_end="2025-01-31",   # 60 pre-period days
    treated_geos=("GEO_18"),
    n_geos_total=20,
    n_clusters=4,
    seed=42
):
    """
    Synthetic SCM panel for smoke and parameterized tests.

    Returns:
        y: np.ndarray, target vector (pre-treatment outcomes for one treated unit)
        Y0: np.ndarray, donor matrix (pre-treatment outcomes for all other geos)
        T0: int, number of pre-treatment periods
    """
    rng = np.random.default_rng(seed)

    pre_start = pd.Timestamp(pre_start)
    pre_end   = pd.Timestamp(pre_end)
    dates_pre = pd.date_range(pre_start, pre_end, freq="D")
    T0 = len(dates_pre)

    # Build donor + treated list
    donors = [f"GEO_{i:02d}" for i in range(n_geos_total - len(treated_geos))]
    geos = donors + list(treated_geos)

    T = T0+20
    weekly = 10 * np.sin(2*np.pi*np.arange(T)/7.0)             # weekly seasonality
    trend = np.linspace(0, 15, T)                              # mild trend

    # AR(1) global disturbance
    ar = np.zeros(T); phi=0.85; eps = rng.normal(0, 2.0, T)
    for t in range(1, T):
        ar[t] = phi*ar[t-1] + eps[t]

    # Cluster structure
    cluster_ids = rng.integers(0, n_clusters, size=len(geos))
    cluster_map = dict(zip(geos, cluster_ids))
    cluster_factors = {c: (rng.normal(0, 1.0, T).cumsum()/20.0) for c in range(n_clusters)}

    # Geo size multipliers
    size_scale = rng.uniform(0.6, 1.6, size=len(geos))
    size_map = dict(zip(geos, size_scale))

    # --- Generate complete pre-period panel
    panel = []
    for g in geos:
        c = cluster_map[g]
        sz = size_map[g]
        geo_noise = rng.normal(0, 4.0, T)
        latent = trend + weekly + ar + 8.0*cluster_factors[c] + geo_noise
        sales = np.maximum(1.0, latent * sz + rng.normal(0, 3.0, T))
        panel.append(sales)

    panel = np.column_stack(panel)  # shape: (T0, n_geos_total)

    # Select one treated geo for target vector y
    treated_idx = n_geos_total - 1
    y = panel[:, treated_idx]

    # Donor matrix (all other geos)
    donor_idx = [i for i in range(n_geos_total) if i != treated_idx]
    Y0 = panel[:, donor_idx]

    return y, Y0, T0
