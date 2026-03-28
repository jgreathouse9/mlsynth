import numpy as np
import matplotlib.pyplot as plt

# Example data
np.random.seed(0)
T0, N = 52, 160
X_pre = np.random.randint(8,20,(T0,N))
x_bar = X_pre.mean(axis=1)

# R^2 batch function
def r2_batch(y_c, ss_tot, X_pre):
    X_mean = X_pre.mean(axis=0)
    ss_X = np.sum((X_pre - X_mean)**2, axis=0)
    cross = np.dot(y_c, X_pre - X_mean)
    ss_res = ss_tot + ss_X - 2.0 * cross
    return 1.0 - ss_res / ss_tot

# Greedy treated/control selection
def greedy_treated_control(X_pre, x_bar, min_treated=1, max_treated=None):
    T0, N = X_pre.shape
    remaining_units = list(range(N))
    treated_units = []
    control_units = []

    y_c = x_bar - x_bar.mean()
    ss_tot = np.sum(y_c**2)

    treated_mean = np.zeros(T0)
    control_mean = np.zeros(T0)

    k_treated = 0
    k_control = 0
    if max_treated is None:
        max_treated = N-1

    while remaining_units:
        best_r2 = -np.inf
        best_unit = None
        best_group = None

        for idx in remaining_units:
            # Treated candidate
            if k_treated < max_treated:
                new_mean = (treated_mean * k_treated + X_pre[:, idx]) / (k_treated + 1)
                r2 = r2_batch(y_c, ss_tot, new_mean.reshape(-1,1))[0]
                if r2 > best_r2:
                    best_r2 = r2
                    best_unit = idx
                    best_group = "treated"

            # Control candidate
            if len(remaining_units) - 1 >= min_treated - k_treated:
                new_mean = (control_mean * k_control + X_pre[:, idx]) / (k_control + 1)
                r2 = r2_batch(y_c, ss_tot, new_mean.reshape(-1,1))[0]
                if r2 > best_r2:
                    best_r2 = r2
                    best_unit = idx
                    best_group = "control"

        # Assign the best unit
        if best_group == "treated":
            treated_units.append(best_unit)
            treated_mean = (treated_mean * k_treated + X_pre[:, best_unit]) / (k_treated + 1)
            k_treated += 1
        else:
            control_units.append(best_unit)
            control_mean = (control_mean * k_control + X_pre[:, best_unit]) / (k_control + 1)
            k_control += 1

        remaining_units.remove(best_unit)

    return treated_units, control_units, treated_mean, control_mean

# Step 1: Loop over possible treated group sizes
best_score = np.inf
best_result = None
min_treated_units = 1
max_treated_units = N-1

for max_treated in range(min_treated_units, N):
    treated, control, t_mean, c_mean = greedy_treated_control(X_pre, x_bar, max_treated=max_treated)
    # Compute combined error (L2 norm)
    score = np.linalg.norm(t_mean - x_bar) + np.linalg.norm(c_mean - x_bar)
    if score < best_score:
        best_score = score
        best_result = {
            "treated_units": treated,
            "control_units": control,
            "treated_mean": t_mean,
            "control_mean": c_mean,
            "n_treated": max_treated,
            "score": score
        }

# Extract best
treated_units = best_result["treated_units"]
control_units = best_result["control_units"]
treated_mean = best_result["treated_mean"]
control_mean = best_result["control_mean"]

print(f"Optimal number of treated units: {best_result['n_treated']}")
print("Treated units:", treated_units)
print("Control units:", control_units)

# Plot
plt.figure(figsize=(12,6))
plt.plot(x_bar, marker='o', linestyle='-', color='black', label='National mean')
plt.plot(treated_mean, marker='s', linestyle='--', color='red', label='Treated mean')
plt.plot(control_mean, marker='^', linestyle='-.', color='blue', label='Control mean')
plt.xlabel('Pre-treatment period')
plt.ylabel('Outcome')
plt.title('National Mean vs Treated and Control Means (Optimal Selection)')
plt.legend()
plt.grid(True)
plt.show()
