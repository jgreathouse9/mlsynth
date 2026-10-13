"""Kerman, Wang and Vaver (2017) section 5.2: coverage and iROAS bias.

    python -m benchmarks.studies.tbr_geo.simulation [reps]

Runs anywhere out of the box: the design is the paper's, so nothing outside
this package is needed.

Two targets:

* Figure 7 -- the 90% and 50% posterior intervals attain nominal coverage
  across every ``(rho, c)`` cell and every pretest length.
* Section 5.2 -- the iROAS posterior median is practically unbiased at a true
  value of 2.0, with squared bias as a fraction of MSE equal to 0.04%, with
  standard deviation 0.06%, over all scenarios.

Section 5.1 and 5.2 set the design: 20 geos, four test points, no cooldown,
2000 replications per cell, and an incremental cost taken to be a known
constant. The treated share is half the geos, drawn afresh each replication,
which the paper does not state.
"""
from __future__ import annotations

import sys

import numpy as np

from .dgp import panel
from .tbr import TBR

RHOS = (0.0, 0.5, 0.8)
CS = (0.15, 0.25, 0.5)
PRES = (10, 20, 30, 40)
TRUE_IROAS = 2.0
LIFT = 0.10


def one_replication(rng, n_pre, rho, c, n_geos=20, n_test=4, sigma_log=1.0):
    """One simulated experiment: returns (in 90%, in 50%, iROAS estimate)."""
    y = panel(n_geos, n_pre, n_test, rho, c, rng, sigma_log)
    idx = rng.permutation(n_geos)
    treated, control = idx[: n_geos // 2], idx[n_geos // 2:]
    Y, X = y[:, treated].sum(1), y[:, control].sum(1)

    # Section 5.2 takes the incremental cost to be a known constant, so the
    # true response is a constant too. Deriving it from the realised treatment
    # volume instead correlates the truth with test-period noise and inflates
    # the measured bias by about a factor of two. The geo shares sum to one, so
    # the expected treatment volume over the test window is 0.5 * n_test.
    true_response = LIFT * 0.5 * n_test
    cost = true_response / TRUE_IROAS

    Y = Y.copy()
    Y[n_pre:] += true_response / n_test
    fit = TBR().fit(Y[:n_pre], X[:n_pre])
    dist = fit.cumulative(Y[n_pre:], X[n_pre:])
    delta = np.atleast_1d(dist.kwds["loc"])[-1]
    return (dist.ppf(0.05)[-1] <= true_response <= dist.ppf(0.95)[-1],
            dist.ppf(0.25)[-1] <= true_response <= dist.ppf(0.75)[-1],
            delta / cost)


def cell_rng(seed, rho, c, n_pre):
    """A generator belonging to one cell, and to nothing else.

    One generator consumed across the grid makes a cell's draws depend on which
    cells preceded it, so a cell requested alone and the same cell inside the
    full grid are different experiments. Seeding per cell makes a cell's result
    a function of ``(seed, rho, c, n_pre)`` and nothing else.
    """
    return np.random.default_rng([int(seed), int(round(rho * 1_000_000)),
                                  int(round(c * 1_000_000)), int(n_pre)])


def run(reps=2000, pres=PRES, seed=11, sigma_log=1.0):
    rows = []
    for rho in RHOS:
        for c in CS:
            for n_pre in pres:
                rng = cell_rng(seed, rho, c, n_pre)
                hits90 = hits50 = 0
                est = np.empty(reps)
                for r in range(reps):
                    a, b, ir = one_replication(rng, n_pre, rho, c,
                                               sigma_log=sigma_log)
                    hits90 += a
                    hits50 += b
                    est[r] = ir
                rows.append(dict(rho=rho, c=c, n_pre=n_pre,
                                 cov90=hits90 / reps, cov50=hits50 / reps,
                                 median=float(np.median(est)), est=est))
    return rows


def squared_bias_fraction(rows):
    """Section 5.2's statistic, per scenario, in percent."""
    out = []
    for r in rows:
        e = r["est"]
        out.append((e.mean() - TRUE_IROAS) ** 2 /
                   np.mean((e - TRUE_IROAS) ** 2))
    return np.asarray(out) * 100.0


def monte_carlo_floor(counts=(250, 500, 1000, 2000, 4000)) -> int:
    """What section 5.2's statistic measures.

    For an unbiased estimator the estimated squared bias has expectation
    sigma^2 / n, and the MSE it is divided by estimates sigma^2, so the ratio
    has expectation 1 / n whatever the estimator does. Across a sixteen-fold
    range of n the measured value tracks that floor, which places the paper's
    0.04% at 1 / 2000 and makes the statistic a report on the replication
    count, not on the bias.
    """
    print("Does section 5.2's statistic measure bias, or 1 / n?\n")
    print(f"  {'reps':>6} {'sq bias / MSE':>15} {'1 / n':>9} {'ratio':>7}")
    for n in counts:
        fr = squared_bias_fraction(run(reps=n, pres=(30,), seed=7))
        print(f"  {n:6d} {fr.mean():14.4f}% {100.0 / n:8.4f}% "
              f"{fr.mean() / (100.0 / n):7.2f}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "floor":
        return monte_carlo_floor()
    reps = int(argv[0]) if argv else 2000
    rows = run(reps)
    print(f"Kerman et al. (2017) section 5.2 -- {reps} replications per cell, "
          f"20 geos, four test points, true iROAS {TRUE_IROAS}\n")
    print(f"  {'rho':>4} {'c':>5} {'T0':>4} {'cov90':>7} {'cov50':>7} "
          f"{'iROAS med':>10}")
    for r in rows:
        print(f"  {r['rho']:4} {r['c']:5} {r['n_pre']:4} {r['cov90']:7.3f} "
              f"{r['cov50']:7.3f} {r['median']:10.4f}")
    c90 = np.array([r["cov90"] for r in rows])
    c50 = np.array([r["cov50"] for r in rows])
    print(f"\n  coverage 90%: mean {c90.mean():.4f}  min {c90.min():.3f}  "
          f"max {c90.max():.3f}   (nominal 0.90)")
    print(f"  coverage 50%: mean {c50.mean():.4f}  min {c50.min():.3f}  "
          f"max {c50.max():.3f}   (nominal 0.50)")
    fr = squared_bias_fraction(rows)
    print(f"\n  squared bias / MSE: mean {fr.mean():.4f}%  sd {fr.std():.4f}%"
          f"   (paper: 0.04%, sd 0.06%)")
    print(f"  the Monte Carlo floor for {reps} replications is "
          f"{100.0 / reps:.4f}%; see the README on what this statistic "
          f"measures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
