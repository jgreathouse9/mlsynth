"""Kerman, Wang and Vaver (2017) section 5.2, run through ``mlsynth.TBR``.

``simulation.py`` beside this established the paper's coverage result against
``tbr.py``, a port written from sections 3.2 and 9.1 before the estimator
existed. That port and the shipped estimator are different bodies of code:
``mlsynth`` reaches the same posterior through ``dataprep``, ``fit_pretest``,
``cumulative_posterior`` and ``interval``, with its own validation and its own
aggregation. Agreement with ``google/matched_markets`` on one panel says the
two implementations match; it does not say either attains nominal coverage.
This harness measures that for the shipped estimator.

The design is the paper's and the generator is ``dgp.py`` unchanged, so there
is one section 5.1 in the repository and not two.

Two assignment schemes
----------------------

The paper does not say how geos are split. ``simulation.py`` draws a free
permutation. The authors' own R package, released in section 6 as
``google/GeoexperimentsResearch``, uses ``GeoStrata(n.groups=2,
group.ratios=c(1, 1))`` with ``Randomize``: geos are sorted by volume, divided
into strata of size two, and one geo of each pair goes to each group. Both are
run here. Stratified pairing balances volume between the groups, which raises
the between-group correlation and narrows the intervals, so the same nominal
coverage under both is a stronger statement than under either alone.

How the effect is injected
--------------------------

Section 5.2 takes the incremental cost to be a known constant, so the true
response is a constant too and :func:`injected_truth` takes no panel. Deriving
it from realised treatment volume correlates the truth with test-period noise,
which ``results/monte_carlo_floor.txt`` measured at twice the bias floor.

How the effect is spread across the treated geos does not matter: TBR sums them
before regressing, so only the per-period total reaches the fit. The reference
spreads spend in proportion to baseline volume; for TBR that is the same
experiment.
"""
from __future__ import annotations

import zlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from mlsynth import TBR
from mlsynth.config_models import TBRConfig

from . import dgp

#: Section 5.1's grid, and section 5.2's design.
RHOS: Tuple[float, ...] = (0.0, 0.5, 0.8)
CS: Tuple[float, ...] = (0.15, 0.25, 0.5)
PRES: Tuple[int, ...] = (10, 20, 30, 40)
N_GEOS = 20
N_TEST = 4
TRUE_IROAS = 2.0
#: Lift as a share of the expected treatment volume over the test window.
LIFT = 0.10
#: The two nominal levels Figure 7 reports.
LEVELS = (0.90, 0.50)


def injected_truth(n_test: int) -> Tuple[float, float]:
    """``(true_response, true_cost)`` for a test window of ``n_test`` periods.

    The geo shares sum to one and half the geos are treated, so the expected
    treatment volume over the window is ``0.5 * n_test``. Both quantities are
    constants of the design: nothing about a realised panel enters.
    """
    response = LIFT * 0.5 * n_test
    return response, response / TRUE_IROAS


def assign(y_pre: np.ndarray, n_geos: int, rng: np.random.Generator,
           scheme: str) -> np.ndarray:
    """Indices of the treated geos under ``scheme``."""
    if scheme == "permutation":
        return np.sort(rng.permutation(n_geos)[: n_geos // 2])
    if scheme == "stratified":
        if n_geos % 2:
            raise ValueError(
                f"the stratified scheme pairs geos by volume, so it needs an "
                f"even geo count; got {n_geos}")
        order = np.argsort(-y_pre.sum(axis=0))
        pairs = order.reshape(-1, 2)
        pick = rng.integers(0, 2, size=len(pairs))
        return np.sort(np.array([int(p[k]) for p, k in zip(pairs, pick)]))
    raise ValueError(
        f"unknown assignment scheme {scheme!r}; expected 'permutation' or "
        f"'stratified'")


def _frame(y: np.ndarray, treated: Sequence[int], n_pre: int,
           n_geos: int) -> pd.DataFrame:
    treated = set(int(j) for j in treated)
    n_periods = y.shape[0]
    return pd.DataFrame([
        {"geo": f"g{j:02d}", "t": t, "sales": float(y[t, j]),
         "post": int(t >= n_pre), "is_treat": int(j in treated),
         "is_ctrl": int(j not in treated)}
        for j in range(n_geos) for t in range(n_periods)])


def one_replication(rng: np.random.Generator, n_pre: int, rho: float, c: float,
                    *, scheme: str = "permutation", n_geos: int = N_GEOS,
                    n_test: int = N_TEST) -> Tuple[bool, bool, float]:
    """One simulated experiment: ``(in the 90%, in the 50%, iROAS estimate)``."""
    y = dgp.panel(n_geos, n_pre, n_test, rho, c, rng)
    treated = assign(y[:n_pre], n_geos, rng, scheme)
    response, cost = injected_truth(n_test)

    y = y.copy()
    y[n_pre:, treated] += response / n_test / len(treated)

    report = TBR(TBRConfig(
        df=_frame(y, treated, n_pre, n_geos), outcome="sales", unitid="geo",
        time="t", treatment_col="is_treat", control_col="is_ctrl",
        post_col="post", level=LEVELS[0])).fit().report

    cum = report.cumulative
    loc, scale, df = cum.estimate[-1], cum.scale[-1], cum.df
    hits = []
    for level in LEVELS:
        tail = (1.0 - level) / 2.0
        lo, hi = stats.t.ppf([tail, 1.0 - tail], df, loc=loc, scale=scale)
        hits.append(bool(lo <= response <= hi))
    return hits[0], hits[1], float(loc / cost)


def cell_rng(seed: int, rho: float, c: float, n_pre: int,
             scheme: str) -> np.random.Generator:
    """A generator belonging to one cell, and to nothing else.

    A single generator consumed across the grid makes a cell's draws depend on
    which cells preceded it, so the same cell answers differently when it is
    requested alone and when it is requested inside the full grid -- measured
    at a 13-hit swing on 200 replications, which is larger than any effect this
    harness is looking for. Seeding per cell makes a cell's result a function
    of its own identity: the same `(seed, rho, c, n_pre, scheme)` is the same
    draws whatever else was asked for.

    ``zlib.crc32`` and not ``hash`` because the latter is salted per process.
    """
    return np.random.default_rng([
        int(seed), int(round(rho * 1_000_000)), int(round(c * 1_000_000)),
        int(n_pre), int(zlib.crc32(scheme.encode("utf-8"))),
    ])


def run_grid(reps: int, *, rhos: Sequence[float] = RHOS,
             cs: Sequence[float] = CS, pres: Sequence[int] = PRES,
             scheme: str = "permutation", seed: int = 11,
             n_geos: int = N_GEOS, n_test: int = N_TEST) -> List[Dict]:
    """One row per ``(rho, c, n_pre)`` cell."""
    if reps < 1:
        raise ValueError(f"a cell needs at least one replication; got {reps}")
    rows: List[Dict] = []
    for rho in rhos:
        for c in cs:
            for n_pre in pres:
                rng = cell_rng(seed, rho, c, n_pre, scheme)
                hits90 = hits50 = 0
                est = np.empty(reps)
                for r in range(reps):
                    a, b, iroas = one_replication(
                        rng, n_pre, rho, c, scheme=scheme, n_geos=n_geos,
                        n_test=n_test)
                    hits90 += a
                    hits50 += b
                    est[r] = iroas
                rows.append({
                    "rho": rho, "c": c, "n_pre": n_pre, "reps": reps,
                    "scheme": scheme, "hits90": hits90, "hits50": hits50,
                    "cov90": hits90 / reps, "cov50": hits50 / reps,
                    "median_iroas": float(np.median(est)),
                    "mean_iroas": float(np.mean(est)),
                    "mse_iroas": float(np.mean((est - TRUE_IROAS) ** 2)),
                })
    return rows


def beta_interval(hits: int, trials: int, level: float = 0.95
                  ) -> Tuple[float, float]:
    """Section 5.2's own criterion for a coverage rate.

    The posterior of the rate under Kerman (2011)'s neutral prior,
    ``Beta(1/3 + y, 1/3 + n - y)``, reported at its central ``level``. The
    paper judges coverage by whether the nominal rate sits inside this, which
    is a criterion that tightens with the replication count instead of a
    tolerance chosen by hand.
    """
    if not 0 <= hits <= trials:
        raise ValueError(
            f"hits must be between 0 and the number of trials; got "
            f"{hits} of {trials}")
    tail = (1.0 - level) / 2.0
    post = stats.beta(1.0 / 3.0 + hits, 1.0 / 3.0 + trials - hits)
    return float(post.ppf(tail)), float(post.ppf(1.0 - tail))


def summarise(rows: Sequence[Dict], level: float = 0.95) -> Dict[str, float]:
    """Pooled coverage, per-cell misses against the Beta criterion, and bias."""
    reps = sum(r["reps"] for r in rows)
    out = {
        "n_cells": float(len(rows)),
        "n_fits": float(reps),
        "cov90_pooled": sum(r["hits90"] for r in rows) / reps,
        "cov50_pooled": sum(r["hits50"] for r in rows) / reps,
        "iroas_median_max_abs_dev": max(
            abs(r["median_iroas"] - TRUE_IROAS) for r in rows),
        "iroas_median_pooled_abs_dev": abs(
            float(np.median([r["median_iroas"] for r in rows])) - TRUE_IROAS),
    }
    for nominal, key in zip(LEVELS, ("cov90", "cov50")):
        off = 0
        for r in rows:
            lo, hi = beta_interval(r[f"hits{key[3:]}"], r["reps"], level)
            off += not (lo <= nominal <= hi)
        out[f"{key}_cells_off_nominal"] = float(off)
    # Squared bias as a share of MSE, the statistic section 5.2 reports. It is
    # a diagnostic and not a target: for any unbiased estimator its expectation
    # is 1/n, so it measures the replication count. See
    # results/monte_carlo_floor.txt.
    bias_sq = np.mean([(r["mean_iroas"] - TRUE_IROAS) ** 2 for r in rows])
    mse = np.mean([r["mse_iroas"] for r in rows])
    out["bias_share_of_mse_pct"] = float(100.0 * bias_sq / mse)
    return out
