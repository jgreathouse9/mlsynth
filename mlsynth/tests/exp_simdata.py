# mlsynth/tests/exp_simdata.py
import numpy as np
import pandas as pd

# -------------------- Population Data --------------------
populations = {
    # Willemstad neighborhoods
    "Brievengat": 4695, "Groot Kwartier": 2611, "Groot Piscadera": 2822,
    "Hato": 45, "Koraal Partir": 3958, "Otrobanda": 1369, "Pietermaai": 99,
    "Piscadera Bay": 787, "Saliña": 2538, "Scharloo": 523,
    "Sint Michiel": 5732, "Steenrijk": 3752,
    # Regional totals
    "Willemstad": 136660, "Bandabou": 13125, "Bandariba": 20838,
    # Bandabou towns
    "Barber": 2412, "Lagún": 321, "Sint Willibrordus": 588, "Soto": 2233,
    "Tera Corá": 4388, "Westpunt": 738,
    # Bandariba towns
    "Oostpunt": 555, "Santa Rosa": 5198, "Spaanse Water": 3119
}

densities = {
    "Brievengat": 950, "Groot Kwartier": 1100, "Groot Piscadera": 820,
    "Piscadera Bay": 220, "Sint Michiel": 320, "Steenrijk": 3600,
    "Barber": 270, "Lagún": 28, "Sint Willibrordus": 25, "Soto": 120,
    "Tera Corá": 230, "Westpunt": 54, "Oostpunt": 10,
    "Santa Rosa": 1200, "Spaanse Water": 150
}

# -------------------- Simulation --------------------
def simulate_districts(districts, R_town=6, F_town=3,
                      R_dist=4, F_dist=1,
                      T0=104, T1=24, sigma2=.25,
                      phi=0.95, treated_units=None, seed=None):
    """Simulate panel data of weekly tourism spending across districts and towns."""
    if seed is not None:
        np.random.seed(seed)

    T = T0 + T1
    district_labels, town_labels = [], []
    Y_N_full, Y_I_full = [], []

    t_idx = np.arange(T)
    tourism_spending = np.random.uniform(100, 500, T)

    for district, towns in districts.items():
        Jd = len(towns)

        # District-level persistent factors
        Delta = np.cumsum(np.random.normal(0, 0.05 if district != "Willemstad" else 0.1, T))
        Theta = np.random.normal(0, 0.05 if district != "Willemstad" else 0.1, (T, R_dist))
        U = np.random.uniform(0, 1, R_dist)

        # Seasonal amplitudes
        if district == "Willemstad":
            annual_amp, semi_amp, quarterly_amp, weekly_amp = 12, 6, 3, 2
        else:
            annual_amp, semi_amp, quarterly_amp, weekly_amp = 6, 3, 1.5, 1

        # Random base phases per district
        base_annual_phase = np.random.uniform(0, 2 * np.pi)
        base_semi_phase = np.random.uniform(0, 2 * np.pi)
        base_quarter_phase = np.random.uniform(0, 2 * np.pi)
        base_weekly_phase = np.random.uniform(0, 2 * np.pi)

        # Town-level latent structure
        Z = np.random.uniform(0, 1, (Jd, R_town))
        mu = np.random.uniform(0, 1, (Jd, F_town))

        Y_N, Y_I = np.zeros((Jd, T)), np.zeros((Jd, T))

        tourism_weight = 0.1 if district != "Willemstad" else 0.2

        for j, town_name in enumerate(towns):
            # Population scaling
            pop = populations.get(town_name, 1000)
            pop_scale = np.log1p(pop) / 10
            dens = densities.get(town_name, 200)
            dens_scale = np.log1p(dens) / 7

            # Seasonal with town-specific shift
            phase_shift = np.random.uniform(-0.2, 0.2, 4)
            seasonal_component = (
                annual_amp * np.sin(2 * np.pi * t_idx / 52 + base_annual_phase + phase_shift[0]) +
                semi_amp * np.sin(4 * np.pi * t_idx / 52 + base_semi_phase + phase_shift[1]) +
                quarterly_amp * np.sin(12 * np.pi * t_idx / 52 + base_quarter_phase + phase_shift[2]) +
                weekly_amp * np.sin(2 * np.pi * t_idx + base_weekly_phase + phase_shift[3])
            )
            seasonal_component *= pop_scale

            # Town factor
            town_factor = pop_scale * (Z[j] @ np.ones(R_town) * 0.1 + mu[j] @ np.ones(F_town) * 0.1)

            # Noise
            eps = np.random.normal(0, np.sqrt(sigma2) * dens_scale, T)
            xi = np.random.normal(0, np.sqrt(sigma2) * dens_scale, T)

            # Initialize
            district_component = Delta[0] + Theta[0] @ U
            Y_N[j, 0] = district_component + town_factor +\
                        tourism_weight * tourism_spending[0] * pop_scale +\
                        seasonal_component[0] + eps[0]
            Y_I[j, 0] = Y_N[j, 0] + 2 + xi[0]

            for t in range(1, T):
                district_component = Delta[t] + Theta[t] @ U
                Y_N[j, t] = (phi * Y_N[j, t - 1] +
                             (1 - phi) * (district_component + town_factor +
                                          tourism_weight * tourism_spending[t] * pop_scale +
                                          seasonal_component[t]) +
                             eps[t])
                Y_I[j, t] = Y_N[j, t] + 2 + xi[t]

        Y_N_full.append(Y_N)
        Y_I_full.append(Y_I)
        district_labels.extend([district] * Jd)
        town_labels.extend(towns)

    Y_N_full = np.vstack(Y_N_full)
    Y_I_full = np.vstack(Y_I_full)
    Y_obs_full = np.copy(Y_N_full)

    if treated_units:
        for j in treated_units:
            Y_obs_full[j, T0:] = Y_I_full[j, T0:]

    return {
        "Y_N": Y_N_full,
        "Y_I": Y_I_full,
        "Y_obs": Y_obs_full,
        "district_labels": district_labels,
        "town_labels": town_labels,
        "treated_units": treated_units or [],
        "tourism_spending": tourism_spending
    }

def sim_to_long_df(sim_data, T0=104):
    """Convert simulation output to long-format DataFrame."""
    district_labels = sim_data["district_labels"]
    town_labels = sim_data["town_labels"]
    Y_N = sim_data["Y_N"]
    Y_I = sim_data["Y_I"]
    Y_obs = sim_data["Y_obs"]
    treated_units = sim_data["treated_units"]
    tourism_spending = sim_data["tourism_spending"]
    T = Y_obs.shape[1]

    rows = []
    for i, (district, town) in enumerate(zip(district_labels, town_labels)):
        treated_flag = i in treated_units
        pop = populations.get(town, 1000)
        for t in range(T):
            rows.append({
                "district": district,
                "town": town,
                "time": t + 1,  # 1-indexed week
                "Y_N": Y_N[i, t],
                "Y_I": Y_I[i, t],
                "Y_obs": Y_obs[i, t],
                "treated": treated_flag and t >= T0,
                "tourism_spending": tourism_spending[t],
                "population": pop
            })

    return pd.DataFrame(rows)

# -------------------- District Setup --------------------
districts = {
    "Bandabou": ["Barber", "Lagún", "Sint Willibrordus", "Soto", "Tera Corá", "Westpunt"],
    "Bandariba": ["Oostpunt", "Santa Rosa", "Spaanse Water"],
    "Willemstad": ["Brievengat", "Groot Kwartier", "Groot Piscadera",
                   "Hato", "Koraal Partir", "Otrobanda", "Pietermaai",
                   "Piscadera Bay", "Saliña", "Scharloo", "Sint Michiel", "Steenrijk"]
}
