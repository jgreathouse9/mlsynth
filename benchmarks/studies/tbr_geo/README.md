# Time-Based Regression: a demonstrate-first replication

Validates Time-Based Regression before any estimator is written, against both
the papers behind `google/matched_markets` and the reference implementation
itself.

* Kerman, J., Wang, P. and Vaver, J. (2017). *Estimating Ad Effectiveness using
  Geo Experiments in a Time-Based Regression Framework.* Google.
* Au, T. C. (2018). *A Time-Based Regression Matched Markets Approach for
  Designing Geo Experiments.* Google LLC.

The reference is Apache 2.0 and is not vendored. `simulation.py` needs nothing
outside this package; `xval.py` skips unless `MLSYNTH_MATCHED_MARKETS` points
at a checkout.

```
python -m benchmarks.studies.tbr_geo.simulation          # Kerman section 5.2
python -m benchmarks.studies.tbr_geo.simulation floor    # what 0.04% measures
MLSYNTH_MATCHED_MARKETS=/path/to/matched_markets \
    python -m benchmarks.studies.tbr_geo.xval            # cross-validation
```

## The method

`tbr.py` is written from the paper, not from the reference. Section 3.2 fits
`y_t = alpha + beta x_t + eps_t` on the pretest, where `y` and `x` are the
treatment and control groups aggregated across geos. Section 9.1 gives the
cumulative effect and its posterior in closed form under a noninformative
prior:

```
Delta(T)    = T (ybar_T - alpha - xbar_T beta)                     eqn 4
scale(T)    = T s (v_a + 2 xbar_T v_ab + v_b xbar_T^2 + 1/T)^(1/2) eqn 6
Delta(T)    ~ t(n - 2, loc = Delta(T), scale = scale(T))
```

`v_a`, `v_b`, `v_ab` are entries of the unscaled `V = (X'X)^-1` and `s` is the
classical residual standard deviation. Section 3.4 defines
`iROAS(T) = Delta_resp(T) / Delta_cost(T)`, simulated from the two posteriors,
and collapses to a scaled `t` when the cost is a known constant.

## Cross-validation (`results/xval.txt`)

On the reference's own `salesandcost.csv`, every quantity agrees:

| quantity | port | reference | difference |
| --- | --- | --- | --- |
| alpha, beta | -421.72940805, 0.99970012 | identical | 0 |
| `s^2` | 358331.83595838 | identical | 0 |
| cumulative response `Delta(T)` | 143028.620641 | 143028.620641 | 6.4e-10 |
| cumulative cost `Delta(T)` | 50000.000000 | 50000.000000 | 7.3e-12 |
| iROAS point | 2.86057241 | 2.86057241 | 7.1e-15 |
| iROAS 90% interval | [2.56727291, 3.15387191] | identical | 2.4e-12 |

Three findings came out of getting there.

The reference panel is unbalanced. 75 of 9300 geo-days are absent, across 17
geos, and they fall on the small ones: mean sales 44.2 for the geos with gaps
against 716.2 for the complete ones, with the count of missing days rising as
size falls (geo 100 missing 16, geo 99 missing 11, geo 98 missing 7). That is
dropped zero-sales days. `dataprep` refuses the panel, correctly, because
treatment is then not sustained. Summing over present rows, which is what the
reference does, equals summing a zero-filled full grid, so filling the grid
reproduces the reference exactly and lets ingestion through.

The cost arm is rank deficient. Cost is identically zero through the pretest,
so `X'X` is singular. `numpy.linalg.inv` raises; statsmodels uses a
pseudoinverse and returns zeros, which is why the reference survives its own
example. This is section 3.4's special case, where the cost counterfactual is
zero with certainty and `Delta_cost(T)` is the total test-period spend. An
implementation needs that branch, and the reference's `_is_fixed_cost_scenario`
detects it by summing the pretest cost over every geo plus the control group's
test-period cost.

The reference suite is green on the method. 529 of 549 of its own tests pass
here. Eighteen failures are pandas 2.x drift in data plumbing and datetime
handling; the two in `test_tbr_iroas.py` are a hard-coded seven-decimal
absolute tolerance on a number of magnitude 3.4e4, failing at a relative
difference of 1.2e-11. None is a method error.

## Kerman section 5.2 (`results/simulation.txt`)

`dgp.py` is section 5.1's design, written from the paper. The shipped
`matched_markets/examples/data_simulator.py` is a different design: linear geo
sizes, power-law heteroskedasticity, and no common component, so it produces no
cross-geo correlation and cannot sweep `rho`. The paper's design reproduces its
own parameters, with realised correlation within 0.007 of `rho` at every cell
and a realised coefficient of variation of `c / 2`, which is what
`y_it = m_i(0.5 W_t + 0.5 Z_it)` implies.

Coverage reproduces, over 36 cells at 2000 replications each:

| interval | measured | range | nominal |
| --- | --- | --- | --- |
| 90% | 0.8999 | 0.878 to 0.912 | 0.90 |
| 50% | 0.4981 | 0.475 to 0.519 | 0.50 |

The paper does not state the lognormal's shape. It does not matter: over an
eightfold sweep of it, 0.25 to 2.0, coverage stays within 0.0014 of nominal and
the iROAS median within 0.003 of 2.0.

## What the 0.04% is (`results/monte_carlo_floor.txt`)

Section 5.2 reports squared bias as a fraction of MSE at 0.04%, standard
deviation 0.06%, and reads it as evidence the iROAS median is unbiased. The
statistic cannot carry that reading. For any unbiased estimator the estimated
squared bias has expectation `sigma^2 / n` and the MSE estimates `sigma^2`, so
the ratio has expectation `1 / n` however good the estimator is. Measured
across a sixteenfold range:

| replications | squared bias / MSE | 1 / n |
| --- | --- | --- |
| 250 | 0.4765% | 0.4000% |
| 500 | 0.0520% | 0.2000% |
| 1000 | 0.0795% | 0.1000% |
| 2000 | 0.0435% | 0.0500% |
| 4000 | 0.0237% | 0.0250% |

It falls as `1 / n` and does not settle on a positive value. At the paper's own
2000 replications it lands on 0.0435%, which is the published 0.04%. So the
number is reproducible and the estimator is consistent with being unbiased; the
0.04% measures the replication count, and a larger simulation would have
produced a smaller figure from the same estimator.

One correction to the harness produced this. Setting the injected effect as a
share of the realised treatment volume correlates the truth with test-period
noise and put the statistic at 0.100%, twice the floor. Section 5.2 takes the
incremental cost as a known constant, so the true response is constant too;
with that fixed the figure falls to 0.0733% over the full grid and to 0.0435%
on the single-pretest sweep.
