"""
supergeo_solver.py
==================
Implements the Supergeo Design from:
    Chen, Doudchenko, Jiang, Stein, Ying (2023).
    "Supergeo Design: Generalized Matching for Geographic Experiments."
    arXiv:2301.12044
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Optional, List, Tuple, Dict, Any

import cvxpy as cp
import numpy as np
import pandas as pd


def _aggregate(Z: np.ndarray, indices: Tuple[int, ...]) -> float:
    """
    Calculate the aggregate response for a set of units.

    Parameters
    ----------
    Z : np.ndarray
        Array of pre-period characteristics for each unit.
    indices : tuple of int
        The indices of the units to aggregate.

    Returns
    -------
    float
        The sum of Z for the specified indices.
    """
    return float(Z[list(indices)].sum())


def _optimal_split(Z: np.ndarray, indices: Tuple[int, ...]) -> Tuple[List[int], List[int], float]:
    """
    Find the bipartition of a group that minimizes the squared difference.

    This function implements the inner minimization of the score(G) calculation.

    Parameters
    ----------
    Z : np.ndarray
        Array of pre-period characteristics for each unit.
    indices : tuple of int
        The indices of the units in group G.

    Returns
    -------
    best_plus : list of int
        Indices assigned to the treatment side (G+).
    best_minus : list of int
        Indices assigned to the control side (G-).
    best_score : float
        The minimum squared difference (Z_{G+} - Z_{G-})².
    """
    idx = list(indices)
    n = len(idx)
    Z_group = Z[idx]

    best_score = np.inf
    best_plus: List[int] = []
    best_minus: List[int] = []

    if n < 2:
        return list(indices), [], np.inf

    # Enumerate all non-empty proper subsets for G+
    for r in range(1, n):
        for plus_local in combinations(range(n), r):
            minus_local = [i for i in range(n) if i not in plus_local]
            z_plus = Z_group[list(plus_local)].sum()
            z_minus = Z_group[list(minus_local)].sum()
            score = (z_plus - z_minus) ** 2
            if score < best_score:
                best_score = score
                best_plus = [idx[i] for i in plus_local]
                best_minus = [idx[i] for i in minus_local]

    return best_plus, best_minus, best_score


def _candidates_from_index_pool(
    Z: np.ndarray,
    index_pool: List[List[int]],
    min_size: int,
    max_size: int,
) -> List[Dict[str, Any]]:
    """
    Build candidate supergeo pairs from a specified pool of unit indices.

    Parameters
    ----------
    Z : np.ndarray
        Array of pre-period characteristics for each unit.
    index_pool : list of list of int
        A list containing subsets of unit indices to consider for combinations.
    min_size : int
        Minimum size of a supergeo pair.
    max_size : int
        Maximum size of a supergeo pair.

    Returns
    -------
    list of dict
        A list of candidate dictionaries containing 'indices', 'g_plus', 
        'g_minus', and 'score'.
    """
    candidates: List[Dict[str, Any]] = []
    seen: set[Tuple[int, ...]] = set()

    for pool in index_pool:
        for size in range(min_size, max_size + 1):
            for combo in combinations(sorted(pool), size):
                if combo in seen:
                    continue
                seen.add(combo)
                g_plus, g_minus, score = _optimal_split(Z, combo)
                candidates.append(
                    dict(indices=combo, g_plus=g_plus, g_minus=g_minus, score=score)
                )

    return candidates


class SupergeoSolver:
    """
    Solver for Supergeo experimental design using Mixed-Integer Programming.

    Parameters
    ----------
    unit_names : list of str
        Names/Labels for the N experimental units.
    Y0 : np.ndarray
        Pre-period outcome matrix of shape (T, N), where T is time points 
        and N is units.
    min_size : int, optional
        Minimum size (l) of a supergeo pair, by default 2.
    max_size : int, optional
        Maximum size (u) of a supergeo pair, by default 4.
    kappa : int, optional
        Minimum number of supergeo pairs (k) required, by default 1.

    Attributes
    ----------
    Z : np.ndarray
        Aggregated pre-test response for each unit (proxy for uninfluenced response).
    N : int
        Total number of experimental units.
    """

    def __init__(
        self,
        unit_names: List[str],
        Y0: np.ndarray,
        min_size: int = 2,
        max_size: int = 4,
        kappa: int = 1,
    ) -> None:
        if Y0.ndim != 2:
            raise ValueError("Y0 must be a 2-D array of shape (T, N).")
        if Y0.shape[1] != len(unit_names):
            raise ValueError("Y0.shape[1] must equal len(unit_names).")

        self.unit_names = list(unit_names)
        self.Y0 = Y0
        self.N = len(unit_names)
        self.min_size = min_size
        self.max_size = max_size
        self.kappa = kappa
        self.Z: np.ndarray = Y0.sum(axis=0)
        self._candidates: List[Dict[str, Any]] = []

    def generate_candidates_exhaustive(self) -> None:
        """
        Enumerate all possible unit subsets within the specified size bounds.

        Warning
        -------
        This method is exponential in complexity. It is recommended only 
        for small unit counts (N < 50).
        """
        self._candidates = _candidates_from_index_pool(
            self.Z,
            [list(range(self.N))],
            self.min_size,
            self.max_size,
        )

    def generate_candidates_partition(self, n_partitions: int = 10, seed: int = 0) -> None:
        """
        Partition units randomly into buckets to generate candidates.

        This heuristic reduces the number of MIP variables by only matching 
        units within the same random partition.

        Parameters
        ----------
        n_partitions : int, optional
            Number of partitions to divide units into, by default 10.
        seed : int, optional
            Random seed for reproducibility, by default 0.
        """
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.N).tolist()
        partitions = [perm[i::n_partitions] for i in range(n_partitions)]

        self._candidates = _candidates_from_index_pool(
            self.Z, partitions, self.min_size, self.max_size
        )

    def generate_candidates_per_geo(
        self,
        beta: Optional[int] = None,
        alpha: float = 0.05,
    ) -> None:
        """
        Generate candidates using the per-geo pruning heuristic.

        Identifies the largest units and keeps only the best-scoring 
        candidate subsets containing them.

        Parameters
        ----------
        beta : int, optional
            Number of largest units to apply pruning to. Defaults to N // 2.
        alpha : float, optional
            The top fraction of best-scoring subsets to retain per unit, by default 0.05.
        """
        if beta is None:
            beta = max(1, self.N // 2)

        order = np.argsort(self.Z)[::-1]
        large_geos = set(order[:beta].tolist())
        small_geos = order[beta:].tolist()

        large_pool = list(large_geos)
        all_large = _candidates_from_index_pool(self.Z, [large_pool], self.min_size, self.max_size)

        kept_indices: set[Tuple[int, ...]] = set()
        for geo in large_geos:
            geo_cands = [c for c in all_large if geo in c["indices"]]
            geo_cands.sort(key=lambda c: c["score"])
            n_keep = max(1, int(np.ceil(alpha * len(geo_cands))))
            for c in geo_cands[:n_keep]:
                kept_indices.add(c["indices"])

        large_candidates = [c for c in all_large if c["indices"] in kept_indices]
        small_candidates = _candidates_from_index_pool(self.Z, [small_geos], self.min_size, self.max_size)

        self._candidates = large_candidates + small_candidates

    def solve(self, solver: str = "CBC") -> pd.DataFrame:
        """
        Solve the covering Mixed-Integer Program to find the optimal design.

        Parameters
        ----------
        solver : str, optional
            The name of the CVXPY-compatible solver, by default "CBC".

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the Pair_ID, Treatment units, Control units, 
            aggregated Z values, and individual pair scores.

        Raises
        ------
        RuntimeError
            If a unit is not covered by any candidate or the solver fails 
             to find a feasible solution.
        """
        if not self._candidates:
            warnings.warn("No candidates found. Running exhaustive generation.", stacklevel=2)
            self.generate_candidates_exhaustive()

        m = len(self._candidates)
        scores = np.array([c["score"] for c in self._candidates], dtype=float)
        x = cp.Variable(m, boolean=True)

        cover_constraints = []
        for i in range(self.N):
            covering_j = [j for j, c in enumerate(self._candidates) if i in c["indices"]]
            if not covering_j:
                raise RuntimeError(f"Unit {i} is not covered by any candidate.")
            cover_constraints.append(cp.sum(x[covering_j]) == 1)

        constraints = cover_constraints + [cp.sum(x) >= self.kappa]
        prob = cp.Problem(cp.Minimize(scores @ x), constraints)
        prob.solve(solver=solver)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MIP solver failed with status '{prob.status}'.")

        return self._format_solution(x.value)

    def _format_solution(self, x_val: np.ndarray) -> pd.DataFrame:
        """
        Convert the solver's boolean vector into a readable DataFrame.

        Parameters
        ----------
        x_val : np.ndarray
            The optimal boolean vector from the solver.

        Returns
        -------
        pd.DataFrame
            Formatted experimental design.
        """
        chosen = np.where(x_val > 0.5)[0]
        rows = []
        for pair_id, j in enumerate(chosen, start=1):
            c = self._candidates[j]
            rows.append(dict(
                Pair_ID=pair_id,
                Treatment=[self.unit_names[i] for i in c["g_plus"]],
                Control=[self.unit_names[i] for i in c["g_minus"]],
                Z_plus=_aggregate(self.Z, tuple(c["g_plus"])),
                Z_minus=_aggregate(self.Z, tuple(c["g_minus"])),
                Score=float(c["score"]),
            ))
        df = pd.DataFrame(rows)
        if not df.empty:
            df["Loss"] = df["Score"].sum()
        return df

    def training_loss(self, design: pd.DataFrame) -> float:
        """
        Calculate the total pre-test loss for a given design.

        Parameters
        ----------
        design : pd.DataFrame
            The design DataFrame returned by the solve() method.

        Returns
        -------
        float
            Total squared difference across all supergeo pairs.
        """
        return float(design["Score"].sum())

    def test_loss(self, design: pd.DataFrame, Y_test: np.ndarray) -> float:
        """
        Evaluate the loss of a design on a held-out test dataset.

        Parameters
        ----------
        design : pd.DataFrame
            The design DataFrame returned by the solve() method.
        Y_test : np.ndarray
            Post-period or held-out pre-period outcome matrix.

        Returns
        -------
        float
            Total squared difference calculated on the test data.
        """
        Z_test = Y_test.sum(axis=0)
        loss = 0.0
        name_to_idx = {name: i for i, name in enumerate(self.unit_names)}

        for _, row in design.iterrows():
            z_plus = sum(Z_test[name_to_idx[n]] for n in row["Treatment"])
            z_minus = sum(Z_test[name_to_idx[n]] for n in row["Control"])
            loss += (z_plus - z_minus) ** 2

        return loss
