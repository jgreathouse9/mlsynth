"""BiqCrunch as an external program, on the two-way design's combinatorial core.

The two-way exact backend chooses the treated set by maximising
``sigma'R sigma`` over sign vectors with ``K`` entries ``+1``
(:mod:`mlsynth.utils.syndes_helpers.gram`): the design's value at a set is at
least ``4 alpha / sigma'R sigma``, with equality where the set's closed form
holds. That core is a cardinality-constrained binary quadratic program, the
class BiqCrunch solves (Krislock, Malick and Roupin, ACM TOMS 43(4), 2017).

BiqCrunch reads 0/1 variables and integer coefficients (its sec. 1.4): with
``sigma = 2z - 1``,

    sigma'R sigma = sum_{i<j} 8 R_ij z_i z_j + sum_i 4 (R_ii - (R1)_i) z_i + 1'R1,

and the coefficients are multiplied by ``scale`` and rounded. BiqCrunch prunes a
node whose bound is under the incumbent plus one, which is exact for integer
values, so it solves the rounded instance exactly; a set's rounded value is
within ``rounding_error()`` of its ``sigma'R sigma``. The cardinality
constraint is given with its products by each variable,
``z_i (sum_j z_j - K) = 0`` (the paper's sec. 5.3), which the lift's ``X r = 0``
already holds, so both searches bound the same relaxation family.

BiqCrunch is GPL v3. It is built and run from outside the repository; nothing
of it is copied here. The BC grammar is the user's guide's (sec. 2.2) and the
summary lines are the ones its ``printFinalOutput`` writes.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

#: Bound on a set's integer value: BiqCrunch keeps values in a C ``int``.
VALUE_CEILING = 2 ** 28


def binary_form(R: np.ndarray):
    """``(Q, lin, const)`` with ``sigma'R sigma = sum_{i<j} Q_ij z_i z_j +
    lin'z + const`` at ``sigma = 2z - 1``."""
    R = np.asarray(R, dtype=float)
    Q = 8.0 * R
    np.fill_diagonal(Q, 0.0)
    lin = 4.0 * (np.diag(R) - R.sum(axis=1))
    return Q, lin, float(R.sum())


@dataclass(frozen=True)
class Instance:
    """The core at size ``K`` in BiqCrunch's integers: pair coefficients ``C``
    (upper triangle read), linear ``d``, and the map back to ``sigma'R sigma``."""

    C: np.ndarray
    d: np.ndarray
    K: int
    scale: float
    const: float

    @classmethod
    def build(cls, R: np.ndarray, K: int, scale: Optional[float] = None) -> "Instance":
        """Round the 0/1 form at ``scale``; by default the largest scale whose
        sets of size ``K`` all stay under ``VALUE_CEILING``."""
        Q, lin, const = binary_form(R)
        terms = K * (K - 1) / 2 + K
        top = max(np.abs(np.triu(Q, 1)).max(initial=0.0), np.abs(lin).max(initial=0.0))
        if scale is None:
            # half a unit under, so the rounded largest coefficient stays in
            scale = (VALUE_CEILING / terms - 0.5) / top if top > 0 else 1.0
        C = np.rint(scale * np.triu(Q, 1)).astype(np.int64)
        d = np.rint(scale * lin).astype(np.int64)
        return cls(C=C, d=d, K=int(K), scale=float(scale), const=const)

    @property
    def n(self) -> int:
        return len(self.d)

    def integer_value(self, z) -> int:
        z = np.asarray(z, dtype=np.int64)
        return int(z @ self.C @ z + self.d @ z)

    def core_value(self, value: float) -> float:
        """``sigma'R sigma`` of a set from its integer value, to within
        :meth:`rounding_error`."""
        return value / self.scale + self.const

    def rounding_error(self) -> float:
        """Half a unit on each of a set's ``K(K-1)/2`` pairs and ``K`` linear
        terms, in ``sigma'R sigma``."""
        return (self.K * (self.K - 1) / 2 + self.K) * 0.5 / self.scale

    def bc(self, products: bool = True) -> str:
        """The instance in BC format, maximised, all constraints equalities."""
        n, K = self.n, self.K
        m = 1 + (n if products else 0)
        last = n + 1
        rows = [f"# two-way core: {n} units, {K} treated, scale {self.scale!r}",
                "1 = max problem", f"{m} = number of constraints",
                "1 = number of blocks", f"{last}",
                " ".join([f"{K}.0"] + ["0.0"] * (m - 1))]
        iu, ju = np.nonzero(self.C)
        rows += [f"0 1 {i + 1} {j + 1} {self.C[i, j] / 2:.1f}" for i, j in zip(iu, ju)]
        rows += [f"0 1 {i + 1} {last} {v / 2:.1f}" for i, v in enumerate(self.d) if v]
        rows += [f"1 1 {i + 1} {last} 0.5" for i in range(n)]
        if products:
            for p in range(n):
                k = 2 + p
                rows += [f"{k} 1 {min(p, j) + 1} {max(p, j) + 1} 0.5"
                         for j in range(n) if j != p]
                if K > 1:
                    rows.append(f"{k} 1 {p + 1} {last} {-(K - 1) / 2:.1f}")
        return "\n".join(rows) + "\n"


def params(*, time_limit: Optional[float] = None, soln_value: Optional[int] = None,
           root: bool = False) -> str:
    """A parameter file setting only what is asked; the rest are BiqCrunch's
    defaults (Table I), which the lift's schedule follows."""
    rows = []
    if time_limit is not None:
        rows.append(f"time_limit = {time_limit:g}")
    if soln_value is not None:
        rows += ["soln_value_provided = 1", f"soln_value = {int(soln_value)}"]
    if root:
        rows.append("root = 1")
    return "\n".join(rows) + ("\n" if rows else "")


_NUM = r"(-?[0-9.]+|inf)"


def parse(text: str) -> dict:
    """The summary BiqCrunch prints at the end of a run. A run its time limit
    stopped reports its best value, its bound and the gap, and no solution."""
    def field(name):
        m = re.search(rf"^{name} = {_NUM}", text, flags=re.M)
        return None if m is None or m.group(1) == "inf" else float(m.group(1))

    nodes, cpu = field("Nodes"), field("CPU time")
    if nodes is None or cpu is None:
        raise ValueError("no BiqCrunch summary in the output")
    stopped = re.search(r"^Best value = ", text, flags=re.M) is not None
    value = field("Best value") if stopped else field("(?:Maximum|Minimum) value")
    sol = re.search(r"^Solution = \{([0-9 ]*)\}", text, flags=re.M)
    current = field("Current bound")
    gap = field("Gap") if stopped else 0.0
    if stopped:
        gap = float(re.search(r"^Gap = ([0-9.]+)%", text, flags=re.M).group(1))
    return {"nodes": int(nodes),
            "bound": current if current is not None else field("Root node bound"),
            "value": None if value is None else int(value),
            "solution": (None if sol is None
                         else [int(t) - 1 for t in sol.group(1).split()]),
            "cpu": cpu, "stopped": stopped, "gap_percent": gap}


def run(inst: Instance, *, binary, workdir, products: bool = True,
        time_limit: Optional[float] = None, soln_value: Optional[int] = None,
        root: bool = False) -> dict:
    """Write the instance and its parameters to ``workdir``, run ``binary``
    (BiqCrunch's ``problems/generic/biqcrunch``) on them, and read its summary.
    ``wall`` is the run's clock time; ``cpu`` is BiqCrunch's own."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    inst_path, param_path = workdir / "core.bc", workdir / "core.param"
    inst_path.write_text(inst.bc(products=products))
    param_path.write_text(params(time_limit=time_limit, soln_value=soln_value, root=root))
    t0 = time.perf_counter()
    done = subprocess.run([str(binary), inst_path.name, param_path.name], cwd=workdir,
                          capture_output=True, text=True, check=False)
    out = parse(done.stdout)
    out["wall"] = time.perf_counter() - t0
    return out
