# SSC replication data — Guanajuato police reform

Data for the empirical application (Section 4) of Cao, Lu & Wu (2026),
*Synthetic Control Inference for Staggered Adoption*, which revisits Alcocer
(2025), *Increasing Intergovernmental Coordination to Fight Crime: Evidence
from Mexico*. `N = 33` municipalities, 10 of which adopt the "Mando Único"
police reform in a staggered fashion starting 2014.

## Source & rights
Publicly available from the Harvard Dataverse
(Alcocer 2024, doi:10.7910/DVN/PBTTDM) and redistributed via the authors'
replication package (`MS3714-step2`), whose README certifies permission to
redistribute. No restricted or confidential data are included.

## Files
| File | What |
|---|---|
| `guanajuato_crime_ssc.csv` | **mlsynth-ready** (UTF-8, long). Monthly municipality panel; columns `idunico`, `time`, `Policial` (treatment), and outcomes `hom_all_rate`, `hom_ym_rate`, `theft_violent_rate`, `theft_nonviolent_rate`. |
| `guanajuato_cartel_ssc.csv` | **mlsynth-ready** (UTF-8, long). Annual panel; columns `idunico`, `Year`, `policial` (treatment), and outcomes `presence_strength`, `co_num`, `war`. |
| `psrm_crime_data_full.csv` | The authors' full cleaned crime file (all columns; latin-1). |
| `psrm_cartel_data_full.csv` | The authors' full cleaned cartel file (all columns; latin-1). |
| `results_ssc_reference.csv` | The authors' reference SSC event-time ATT estimates (for verification). |

## Sample windows used in the paper
* Homicide (`hom_all_rate`, `hom_ym_rate`): `time < 253`  → `T0 = 174`, `S = 78`.
* Theft (`theft_violent_rate`, `theft_nonviolent_rate`): `time >= 133` → `T0 = 42`, `S = 90` (point estimates only; no end-of-sample band since `T0 < S`).
* Cartel (`presence_strength`, `co_num`, `war`): full annual panel → `T0 = 15`, `S = 7`.

## Reproduce with mlsynth
```python
import pandas as pd
from mlsynth import SSC

crime = pd.read_csv("guanajuato_crime_ssc.csv")
sub = crime[crime["time"] < 253]            # homicide window
res = SSC({"df": sub, "outcome": "hom_all_rate", "treat": "Policial",
           "unitid": "idunico", "time": "time",
           "inference": True, "display_graphs": True}).fit()
print(res.event_att)        # event-time ATT path (matches results_ssc_reference.csv)
```

mlsynth's SSC reproduces the reference event-time ATTs to ~1e-4 (homicide,
theft) and ~1e-3 (cartel); the residual is the synthetic-control weight solver
(cvxpy vs. the reference's MATLAB `fmincon`).
