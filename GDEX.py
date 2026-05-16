import numpy as np
from typing import List, Optional, Dict, Any

class GDEX:
    """
    Greedy Design of Experiments (Extended & Corrected)

    Supports:
    - unconstrained selection (fixed k)
    - optional budget-constrained selection
    - mathematically exact FDID parallel trend tracking via OLS R²
    - vectorized matrix expansions (no iterative loops or matrix reconstructions)
    """

    def __init__(
        self,
        pre_treatment_matrix: np.ndarray,
        market_names: List[str],
        costs: Optional[np.ndarray] = None
    ):
        if pre_treatment_matrix.ndim != 2:
            raise ValueError("pre_treatment_matrix must be 2D.")

        self.X = np.asarray(pre_treatment_matrix, dtype=float)
        self.market_names = market_names

        self.T, self.N = self.X.shape
        self.all_indices = np.arange(self.N)

        # Optional cost structures
        self.costs = (
            np.asarray(costs, dtype=float)
            if costs is not None
            else np.ones(self.N)
        )

        if len(self.costs) != self.N:
            raise ValueError("Costs array length must match number of units.")

        # Crucial for True FDID: Pre-center the data over the TIME axis (rows)
        # This allows us to drop the OLS intercept alpha out of the batch loops entirely.
        self.X_mean_time = self.X.mean(axis=0) # Shape: (N,)
        self.X_tilde = self.X - self.X_mean_time # Time-centered matrix, Shape: (T, N)

        # Global totals for fast complementary control derivations
        self.total_sum = self.X.sum(axis=1) # Shape: (T,)

    def select_treatment_group(
        self,
        k: Optional[int] = None,
        budget: Optional[float] = None,
        verbose: bool = True
    ) -> List[str]:
        """
        Selects optimal treatment units based on fixed-k cardinality
        or an explicit financial budget constraint.
        """
        if k is None and budget is None:
            raise ValueError("Provide either k or budget.")

        if k is not None and budget is not None:
            raise ValueError("Provide only one of k or budget.")

        available_mask = np.ones(self.N, dtype=bool)
        treatment_indices = []
        remaining_budget = budget
        step = 0

        # Sufficient statistics trackers for the treated cluster (uncentered values)
        A_t = np.zeros(self.T)  # Rolling time-series sum of treated units

        while True:
            remaining = self.all_indices[available_mask]

            if len(remaining) == 0:
                break

            # Budget check
            if budget is not None:
                feasible = self.costs[remaining] <= remaining_budget
                remaining = remaining[feasible]
                if len(remaining) == 0:
                    break

            # Cardinality constraint check
            if k is not None and len(treatment_indices) >= k:
                break

            # ====================================================
            # Correct Vectorized Candidate Trajectory Evaluation
            # ====================================================
            X_cand = self.X[:, remaining]        # Uncentered candidate slices (T, C)
            X_tilde_cand = self.X_tilde[:, remaining]  # Centered candidate slices (T, C)
            c_cand = self.costs[remaining]

            n_new = len(treatment_indices) + 1
            n_c = self.N - n_new

            # 1. Map out what the Synthetic Treated series would look like
            A_new = A_t[:, None] + X_cand        # Rolling uncentered sum matrix (T, C)
            y_T_cand = A_new / n_new             # Proposed Treatment averages (T, C)
            
            # Center the proposed treatment averages over time
            y_T_tilde = y_T_cand - y_T_cand.mean(axis=0)
            
            # 2. Derive the complementary Control series matrix
            A_c = self.total_sum[:, None] - A_new # Total matrix complement (T, C)
            y_S_cand = A_c / n_c                  # Proposed Control averages (T, C)
            
            # Center the proposed control averages over time
            y_S_tilde = y_S_cand - y_S_cand.mean(axis=0)

            # 3. Calculate exact OLS Parallel Trend Components
            # SSR = Sum((y_T_tilde - y_S_tilde)^2)
            ssr = np.sum((y_T_tilde - y_S_tilde) ** 2, axis=0)
            
            # SST = Sum(y_T_tilde^2)
            sst = np.sum(y_T_tilde ** 2, axis=0)
            sst = np.where(sst <= 1e-12, 1e-12, sst) # Protect denominator

            # Exact Time-Series R² Fitness Metric
            r2 = 1.0 - (ssr / sst)

            # Cost Efficiency Allocation Modifier
            if budget is not None:
                score = r2 / (c_cand + 1e-12)
            else:
                score = r2

            # Isolate the index maximizing parallel trend trajectory alignment
            best_idx = np.argmax(score)
            best = remaining[best_idx]

            # ====================================================
            # Commit the Selection
            # ====================================================
            treatment_indices.append(best)
            available_mask[best] = False

            # Update sufficient statistics
            A_t += self.X[:, best]

            if budget is not None:
                remaining_budget -= self.costs[best]

            step += 1

            if verbose:
                msg = f"Step {step}: {self.market_names[best]} (Step R²: {r2[best_idx]:.4f})"
                if budget is not None:
                    msg += f" | Budget Remaining={remaining_budget:.2f}"
                print(msg)

        self.treatment_indices = np.array(treatment_indices)
        self.control_indices = self.all_indices[available_mask]

        return [self.market_names[i] for i in treatment_indices]

    def build_group_means(self):
        if not hasattr(self, "treatment_indices"):
            raise RuntimeError("Run selection first.")

        treated = self.X[:, self.treatment_indices].mean(axis=1)
        control = self.X[:, self.control_indices].mean(axis=1)

        return treated, control
