# `mlsynth`

![coverage](coverage.svg)

`mlsynth` is a Python framework for **synthetic control causal inference and
synthetic-control-based experimental design**. It bundles over 20 modern estimators
under a single typed `Config` / `.fit()` / typed-results interface, so swapping
between, say, Forward DiD, TASC, and SPCD is a one-line change.

Documentation: <https://mlsynth.readthedocs.io/> · Landing page:
<https://jgreathouse9.github.io/mlsynth/webdir/mlsynthlanding.html>

---

## Why `mlsynth`

The synthetic control literature has fragmented across econometrics, statistics,
marketing science, and machine learning. Reference implementations are scattered
across paper-supplemental GitHub repositories in heterogeneous languages and
with idiosyncratic APIs, so comparing several methods on a single panel
typically means translating four or five codebases into a common shape before
any analysis begins. `mlsynth` removes that translation cost.

Three things distinguish it from other SCM libraries. First, **breadth**: it
covers regularized variants of the canonical convex-hull estimator (ridge-
augmented, $L_\infty$, low-rank, and Bayesian shrinkage), methods in which the
donor pool is chosen explicitly rather than implicitly through convex weighting
(forward selection, clustering, proximal identification, matrix completion),
and synthetic-control-based experimental design. Second, **a uniform API**:
every estimator takes a Pydantic-validated config and returns a typed dataclass
of results. Third,
**experimental design as a first-class concern**: `SCDI`, `MAREX`, `LEXSCM`,
and `SPCD` jointly optimize *which* units to treat with minimum detectable
effect (MDE) analysis and, in the case of `LEXSCM`, explicit unit-level cost
and budget constraints solved by branch-and-bound. This last capability is
largely unavailable in other SCM packages.

## Install

```bash
pip install -U git+https://github.com/jgreathouse9/mlsynth.git
```

`mlsynth` supports Python 3.8 and later.

## Quickstart

```python
import numpy as np
import pandas as pd
from mlsynth import FDID

rng = np.random.default_rng(0)
N, T1, T2 = 60, 24, 12            # 60 controls, 24 pre, 12 post
T = T1 + T2

def factors(T, rng, burn=200):    # f1: AR(1); f2: ARMA(1,1); f3: MA(2)
    Tt = T + burn; v = rng.standard_normal((Tt, 3)); f = np.zeros((Tt, 3))
    for t in range(1, Tt): f[t, 0] = 0.8 * f[t-1, 0] + v[t, 0]
    for t in range(1, Tt): f[t, 1] = -0.6 * f[t-1, 1] + v[t, 1] + 0.8 * v[t-1, 1]
    f[1, 2] = v[1, 2] + 0.9 * v[0, 2]
    for t in range(2, Tt): f[t, 2] = v[t, 2] + 0.9 * v[t-1, 2] + 0.4 * v[t-2, 2]
    f[0, 2] = v[0, 2]
    return f[burn:]

sf = factors(T, rng).sum(1)                        # common factor path
y_tr = 1 + sf + rng.standard_normal(T)             # treated: loading 1, true ATT = 0
loads = np.where(np.arange(N) < N // 2, 1.0, 2.0)  # first 30 match, last 30 mismatch
Y = 1 + np.outer(sf, loads) + rng.standard_normal((T, N))

rows = [{"unit": "treated", "time": t, "gdp": y_tr[t], "treat": int(t >= T1)}
        for t in range(T)]
for j in range(N):
    rows += [{"unit": f"c{j}", "time": t, "gdp": Y[t, j], "treat": 0} for t in range(T)]
df = pd.DataFrame(rows)

res = FDID({"df": df, "outcome": "gdp", "treat": "treat",
            "unitid": "unit", "time": "time", "display_graphs": False}).fit()

sel = res.fdid.selected_names
matching = sum(int(s[1:]) < N // 2 for s in sel)
print(f"FDID: ATT={res.fdid.att:+.3f}  R2={res.fdid.r_squared:.3f}  "
      f"selected {len(sel)} donors, {matching} from the matching group")
print(f"DID : ATT={res.did.att:+.3f}  R2={res.did.r_squared:.3f}  (all {N} donors)")
```

The same five `df`/`outcome`/`unitid`/`time`/`treat` fields work for every
estimator. Swap `FDID` for `TASC`, `CLUSTERSC`, `BVSS`, or any other class in the
table below; only the class name and any estimator-specific hyperparameters
change.

## Estimators

| Estimator | Reference | Class |
|---|---|---|
| [L1PDA](https://doi.org/10.1016/j.jeconom.2016.01.011) | Li & Bell (2017), *Journal of Econometrics* 197(1):65–75 | `PDA` |
| [Forward-Selected Panel Data Approach](https://doi.org/10.1016/j.jeconom.2021.04.009) | Shi & Huang (2023), *Journal of Econometrics* 234(2):512–535 | `PDA` |
| [L2-Relaxation for Economic Prediction](https://doi.org/10.13140/RG.2.2.11670.97609) | Shi & Wang (2024) | `PDA` |
| [Synthetic Control Method (vanilla SCM)](https://doi.org/10.1198/jasa.2009.ap08746) | Abadie, Diamond, Hainmueller (2010), *JASA* 105(490):493–505 | `TSSC` |
| [Two-Step Synthetic Control](https://doi.org/10.1287/mnsc.2023.4878) | Li & Shankar (2024), *Management Science* 70(6):3734–3755 | `TSSC` |
| [Forward Difference-in-Differences](https://doi.org/10.1287/mksc.2022.0212) | Li (2024), *Marketing Science* 43(2):267–279 | `FDID` |
| [Factor Model Approach](https://doi.org/10.1177/00222437221137533) | Li & Sonnier (2023), *JMR* 60(3) | `FMA` |
| [Optimal Initial Donor Selection for SCM](https://doi.org/10.1016/j.econlet.2024.111976) | Cerulli (2024), *Economics Letters* 244:111976 | `FSCM` |
| [Principal Component Regression](https://doi.org/10.1080/01621459.2021.1928513) | Agarwal et al. (2021), *JASA* 116(536):1731–1745 | `CLUSTERSC` |
| [Robust PCA Synthetic Control](https://academicworks.cuny.edu/gc_etds/4984) | Bayani (2022), CUNY Academic Works | `CLUSTERSC` |
| [CLUSTERSC](https://doi.org/10.48550/arXiv.2503.21629) | Rho, Tang, Bergam, Cummings, Misra (2024), arXiv:2503.21629 | `CLUSTERSC` |
| [SCM with Nonlinear Outcomes](https://arxiv.org/abs/2306.01967) | Tian (2023), arXiv:2306.01967 | `NSC` |
| [Proximal SCM Framework](https://arxiv.org/abs/2108.13935) | Shi, Li, Miao, Hu, Tchetgen Tchetgen (2023), arXiv:2108.13935 | `PROXIMAL` |
| [Proximal Causal Inference for SCM (Surrogates)](https://arxiv.org/abs/2308.09527) | Liu, Tchetgen Tchetgen, Varjão (2023), arXiv:2308.09527 | `PROXIMAL` |
| [SCM with Multiple Outcomes](https://arxiv.org/abs/2304.02272) | Tian, Lee, Panchenko (2023), arXiv:2304.02272 | `SCMO` |
| [Synthetic Difference-in-Differences](https://www.aeaweb.org/articles?id=10.1257/aer.20190159) | Arkhangelsky et al. (2021), *AER* 111(12):4088–4118 | `SDID` |
| [Synthetic Historical Control](https://ssrn.com/abstract=4995085) | Chen, Yang, Yang (2024), SSRN | `SHC` |
| [Synthetic Regressing Control](https://arxiv.org/abs/2306.02584) | — | `SRC` |
| [Synthetic Interventions](https://arxiv.org/abs/2006.07691) | Agarwal, Shah, Shen (2020), arXiv:2006.07691 | `SI` |
| [Relaxed Balanced Synthetic Control](https://arxiv.org/abs/2508.01793) | Liao, Shi, Zheng (2025), arXiv:2508.01793 | `RESCM` |
| [$L_\infty$ Synthetic Control](https://arxiv.org/abs/2510.26053) | Wang, Xing, Ye (2025), arXiv:2510.26053 | `RESCM` |
| [Bayesian SC with Soft Simplex Constraint](https://arxiv.org/abs/2503.06454) | Xu & Zhou (2025), arXiv:2503.06454 | `BVSS` |
| [Time-Aware Synthetic Control](https://arxiv.org/abs/2601.03099) | Rho, Illick, Narasipura, Abadie, Hsu, Misra (2026), arXiv:2601.03099 | `TASC` |
| [Synthetic Business Cycle](https://arxiv.org/abs/2505.22388) | Shi, Xi, Xie (2025), arXiv:2505.22388 | `SBC` |
| [Synthetic Controls for Experimental Design](https://arxiv.org/abs/2108.02196) | Abadie & Zhao (2025), arXiv:2108.02196 | `SCDI` |
| [Lexicographic Synthetic Control](https://economics.mit.edu/sites/default/files/2026-02/Synthetic%20Controls%20for%20Experimental%20Design%20Feb%202026.pdf) | Vives-i-Bastida (2022) | `LEXSCM` |
| [Synthetic Principal Component Design](https://arxiv.org/abs/2211.15241) | Lu, Li, Ying, Blanchet (2022), arXiv:2211.15241 | `SPCD` |
| [Synthetic Experimental Design](https://arxiv.org/abs/2108.02196) | Abadie & Zhao (2025), arXiv:2108.02196 | `MAREX` |

Several entries share a class because the underlying estimator generalizes
several methods through a single configuration (`PDA` covers three norm
choices, `RESCM` covers relaxation- and $L_\infty$-balanced variants, and so
on).

## Inference and design tools

Beyond point estimates, `mlsynth` ships three families of inference: moving-block conformal prediction intervals for `SPCD` and
`LEXSCM`; and posterior credible intervals for the Bayesian and state-space
estimators (`BVSS`, `TASC`).

For experimental design, `SCDI`, `SPCD`,
`LEXSCM`, and `MAREX` jointly select treated units and donor weights, and
expose pre-experiment minimum-detectable-effect curves so power can be
evaluated before the experiment is run.

## Contributing

Small fixes are always welcome. For new estimators, inference tests, or
larger changes, please email Jared first. Estimators currently on the
development list include continuous-treatment synthetic controls, Bayesian
factor SCMs, random-forest-based SCMs, simplex-weight inference, and
prediction-interval refinements.

Whatever change is proposed, it must reproduce either a canonical benchmark
application from the SCM literature (Basque Country, California Proposition 99,
or West Germany reunification) or the empirical findings reported in the
originating methodological paper. Continuous integration runs the unit test
suite, a fresh-install smoke check, and coverage reporting on every pull
request.

In addition to code, you can also develop tutorials, presentations, and
educational materials using `mlsynth`, promote it on LinkedIn or in the
classroom, and help with outreach and onboarding new contributors.

## License

`mlsynth` is open source and distributed under the [MIT License](LICENSE).
