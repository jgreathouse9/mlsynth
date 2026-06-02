# `mlsynth`

![coverage](coverage.svg)

`mlsynth` is a Python framework for **synthetic control causal inference and
synthetic-control-based experimental design**. It bundles over 20 modern estimators
under a single typed `Config` / `.fit()` / typed-results interface, so swapping
between, say, Forward DiD, TASC, and SPCD is a one-line change.

[Documentation](https://mlsynth.readthedocs.io/)

---

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

`mlsynth` implements 38 estimator classes spanning the full synthetic-control landscape. Several classes expose multiple methods through one configuration (e.g. `PDA` covers three norm choices; `RESCM` covers relaxed- and $L_\infty$-balanced variants; `PROXIMAL` dispatches six proximal estimators).

| Estimator | Reference | Class |
|---|---|---|
| **— Canonical & convex-hull —** | | |
| [Synthetic Control Method (vanilla SCM)](https://doi.org/10.1198/jasa.2009.ap08746) | Abadie, Diamond & Hainmueller (2010), *JASA* 105(490):493–505 | `TSSC` |
| [Two-Step Synthetic Control](https://doi.org/10.1287/mnsc.2023.4878) | Li & Shankar (2024), *Management Science* 70(6):3734–3755 | `TSSC` |
| Modified Unbiased Synthetic Control | Bottmer, Imbens, Spiess & Warnick (2024), *JBES* 42(2):762–773 | `MUSC` |
| Matching & Synthetic Control (MASC) | Kellogg, Mogstad, Pouliot & Torgovitsky (2021), *JASA* | `MASC` |
| **— Donor selection / forward —** | | |
| [Forward Difference-in-Differences](https://doi.org/10.1287/mksc.2022.0212) | Li (2024), *Marketing Science* 43(2):267–279 | `FDID` |
| [Optimal Initial Donor Selection (FSCM)](https://doi.org/10.1016/j.econlet.2024.111976) | Cerulli (2024), *Economics Letters* 244:111976 | `FSCM` |
| Panel Data Approach (HCW) | Hsiao, Ching & Wan (2012), *J. Applied Econometrics* | `PDA` |
| [L1-PDA](https://doi.org/10.1016/j.jeconom.2016.01.011) | Li & Bell (2017), *J. Econometrics* 197(1):65–75 | `PDA` |
| [Forward-Selected Panel Data Approach](https://doi.org/10.1016/j.jeconom.2021.04.009) | Shi & Huang (2023), *J. Econometrics* 234(2):512–535 | `PDA` |
| [L2-Relaxation](https://doi.org/10.13140/RG.2.2.11670.97609) | Shi & Wang (2024) | `PDA` |
| **— High-dimensional / robust / relaxed-hull —** | | |
| [Principal Component Regression SC](https://doi.org/10.1080/01621459.2021.1928513) | Agarwal et al. (2021), *JASA* 116(536); Amjad, Shah & Shen (2018) | `CLUSTERSC` |
| [Robust PCA Synthetic Control](https://academicworks.cuny.edu/gc_etds/4984) | Bayani (2022), CUNY Academic Works | `CLUSTERSC` |
| [CLUSTERSC (donor clustering)](https://arxiv.org/abs/2503.21629) | Rho, Tang, Bergam, Cummings & Misra (2024), arXiv:2503.21629 | `CLUSTERSC` |
| Sparse Synthetic Control (L1 predictor selection) | Vives-i-Bastida (2023), *Predictor Selection for Synthetic Controls* | `SparseSC` |
| [Relaxed Balanced Synthetic Control](https://arxiv.org/abs/2508.01793) | Liao, Shi & Zheng (2025), arXiv:2508.01793 | `RESCM` |
| [$L_\infty$ Synthetic Control](https://arxiv.org/abs/2510.26053) | Wang, Xing & Ye (2025), arXiv:2510.26053 | `RESCM` |
| **— Factor / time-series / Bayesian —** | | |
| [Factor Model Approach](https://doi.org/10.1177/00222437221137533) | Li & Sonnier (2023), *JMR* 60(3):449–472 | `FMA` |
| Harmonic Synthetic Control | Liu & Xu (2026), *The Harmonic Synthetic Control Method* | `HSC` |
| [Time-Aware Synthetic Control](https://arxiv.org/abs/2601.03099) | Rho, Illick, Narasipura, Abadie, Hsu & Misra (2026), arXiv:2601.03099 | `TASC` |
| [Synthetic Business Cycle](https://arxiv.org/abs/2505.22388) | Shi, Xi & Xie (2025), arXiv:2505.22388 | `SBC` |
| [Bayesian SC with Soft Simplex Constraint](https://arxiv.org/abs/2503.06454) | Xu & Zhou (2025), arXiv:2503.06454 | `BVSS` |
| [Dynamic SC for Auto-Regressive Processes](https://doi.org/10.1093/jrsssb/qkad103) | Zheng & Chen (2024), *JRSS-B* 86(1):155–176 | `DSCAR` |
| **— SDID family / staggered adoption —** | | |
| [Synthetic Difference-in-Differences](https://doi.org/10.1257/aer.20190159) | Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021), *AER* 111(12):4088–4118 | `SDID` |
| [Sequential Synthetic Difference-in-Differences](https://arxiv.org/abs/2404.00164) | Arkhangelsky & Samkov (2025), arXiv:2404.00164 | `SequentialSDID` |
| [Partially Pooled SCM (staggered)](https://doi.org/10.1111/rssb.12448) | Ben-Michael, Feller & Rothstein (2022), *JRSS-B* 84(2):351–381 | `PPSCM` |
| **— Spillover / interference (SUTVA) —** | | |
| [SCM with Spillover Effects](https://arxiv.org/abs/1902.07343) | Cao & Dowd (2023) | `SPILLSYNTH` |
| Spatial Synthetic Difference-in-Differences | Serenini & Masek (2024), SSRN 4736857 | `SpSyDiD` |
| Imperfect Synthetic Controls | Powell (2026), *J. Applied Econometrics* 41(3):253–264 | `ISCM` |
| **— Multiple outcomes / interventions / proximal —** | | |
| [SCM with Multiple Outcomes](https://arxiv.org/abs/2304.02272) | Tian, Lee & Panchenko (2023), arXiv:2304.02272 | `SCMO` |
| [Synthetic Interventions](https://arxiv.org/abs/2006.07691) | Agarwal, Shah & Shen (2026), *Operations Research* | `SI` |
| [Proximal SCM Framework](https://arxiv.org/abs/2108.13935) | Shi, Li, Miao, Hu & Tchetgen Tchetgen (2023), arXiv:2108.13935 | `PROXIMAL` |
| [Proximal SC with Surrogates](https://arxiv.org/abs/2308.09527) | Liu, Tchetgen Tchetgen & Varjão (2023), arXiv:2308.09527 | `PROXIMAL` |
| [Single Proxy Synthetic Control](https://doi.org/10.1515/jci-2023-0079) | Park & Tchetgen Tchetgen (2025), *J. Causal Inference* | `PROXIMAL` |
| [Doubly Robust Proximal Synthetic Controls](https://doi.org/10.1093/biomtc/ujae055) | Qiu, Shi, Miao, Dobriban & Tchetgen Tchetgen (2024), *Biometrics* 80(2) | `PROXIMAL` |
| **— Matrix completion / missing data —** | | |
| [Matrix Completion with Nuclear Norm Minimization](https://arxiv.org/abs/1710.10251) | Athey, Bayati, Doudchenko, Imbens & Khosravi (2021), *JASA* 116(536) | `MCNNM` |
| [Synthetic Nearest Neighbors / Causal Matrix Completion](https://arxiv.org/abs/2109.15154) | Agarwal, Dahleh, Shah & Shen (2021), arXiv:2109.15154 | `SNN` |
| **— Distributional / nonlinear / continuous / IV / micro —** | | |
| [Distributional Synthetic Controls](https://doi.org/10.3982/ECTA18260) | Gunsilius (2023), *Econometrica* 91(3):1105–1117 | `DSC` |
| [SCM with Nonlinear Outcomes](https://arxiv.org/abs/2306.01967) | Tian (2023), arXiv:2306.01967 | `NSC` |
| Continuous-Treatment Synthetic Control | Powell (2022), *JBES* 40(3):1302–1314 | `CTSC` |
| Synthetic IV Estimation in Panels | Gulek & Vives-i-Bastida (2024), Job Market Paper | `SIV` |
| Micro-level Balancing Synthetic Control | Robbins & Davenport (2021) | `MicroSynth` |
| Synthetic Control with Disaggregated Data | Bottmer (2026), Stanford job-market paper | `MLSC` |
| [Synthetic Historical Control](https://ssrn.com/abstract=4995085) | Chen, Yang & Yang (2024), SSRN 4995085 | `SHC` |
| **— Experimental design —** | | |
| [Synthetic Controls for Experimental Design](https://arxiv.org/abs/2108.02196) | Abadie & Zhao (2026) | `MAREX` |
| Synthetic Design (optimization approach) | Doudchenko, Khosravi, Pouget-Abadie, Lahaie, Lubin, Mirrokni, Spiess & Imbens (2021) | `SYNDES` |
| Lexicographic Synthetic Control (validity → power) | Abadie & Zhao (2026); Vives-i-Bastida (2022) | `LEXSCM` |
| [Synthetic Principal Component Design](https://arxiv.org/abs/2211.15241) | Lu, Li, Ying & Blanchet (2022), arXiv:2211.15241 | `SPCD` |
| Parallel-Trends Supergeo Design | `mlsynth` (PANGEO), extending Supergeo Design — Chen, Doudchenko, Jiang, Stein & Ying (2023) | `PANGEO` |

## Contributing

Fixes are always welcome. For new estimators, inference tests, or
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
