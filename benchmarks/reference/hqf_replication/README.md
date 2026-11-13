# HQF replication -- Wang, Li, So & Liu (2023), Table 1

Path B replication of `CLUSTERSC`'s `rpca_method="HQF"` against the paper it
cites, with a value-for-value cross-check against the authors' own MATLAB under
Octave. No library code is changed here.

* Paper: Wang, Z.-Y., Li, X. P., So, H. C. & Liu, Z. (2023), "Robust PCA via
  non-convex half-quadratic regularization", Signal Processing 204, 108816.
* Code: <https://github.com/bestzywang/Robust-PCA-via-non-convex-half-quadratic-regularization>
  @ `ca4d23a` (`RPCA_HQF.m`, `Gaussian_noise.m`, `main.m`). No licence file, so
  it is not vendored; `octave_crosscheck.m` runs it from a clone.

## Verdict

The algorithm reproduces. The library's defaults do not, and they are the
cause of HQF ranking last among RPCA-SC's denoisers in the RPCA spike
(`benchmarks/reference/rpca_l1_spike/`).

1. A line-for-line port (`port.py`) matches the authors' `RPCA_HQF.m` to a
   relative 2.3e-14 on 25 cases (5 SNRs x 5 runs), and the paper's Table 1 in
   every cell (below).
2. mlsynth's `hqf_decompose`, called with the paper's settings (true rank,
   `ip=3`, `lam=0.001`, `max_iter=100`), equals the port with mlsynth's three
   deliberate changes to 1e-12, and its RMSE is within 0.0004 of the authors'
   in every cell. The changes are harmless.
3. `hqf_decompose` as `CLUSTERSC` calls it is 15 times worse than the paper at
   every SNR. Its default rank, the smallest with 99.9% of the spectral energy
   of the corrupted matrix, picks 330-423 of a true rank 10 at m = 500 and
   661-847 of 20 at m = 1000: the outliers' energy is spread over every
   singular value, so the rule keeps nearly all of them and the "low-rank"
   part is the data. The paper takes the rank as known (main.m sets
   `rak = m/50`); the 0.999 rule is mlsynth's, attributed to Bayani (2021).

## Table 1, mean RMSE of the low-rank estimate

m = 500, 20 runs per cell (`table1_m500.parquet`):

| SNR (dB) | paper | authors' code (port) | mlsynth, paper settings | mlsynth, default |
|---|---|---|---|---|
| 3 | 0.1508 | 0.1512 | 0.1509 | 2.2280 |
| 6 | 0.1074 | 0.1072 | 0.1070 | 1.5833 |
| 9 | 0.0757 | 0.0758 | 0.0757 | 1.1140 |
| 12 | 0.0534 | 0.0540 | 0.0539 | 0.7911 |
| 15 | 0.0383 | 0.0381 | 0.0381 | 0.5556 |

m = 1000, 10 runs per cell (`table1_m1000.parquet`):

| SNR (dB) | paper | authors' code (port) | mlsynth, paper settings | mlsynth, default |
|---|---|---|---|---|
| 3 | 0.2141 | 0.2139 | 0.2135 | 3.1513 |
| 6 | 0.1514 | 0.1523 | 0.1520 | 2.2465 |
| 9 | 0.1072 | 0.1074 | 0.1073 | 1.5844 |
| 12 | 0.0757 | 0.0759 | 0.0758 | 1.1147 |
| 15 | 0.0536 | 0.0537 | 0.0536 | 0.7800 |

Every port cell is within two of the paper's standard deviations except
m = 500 at 12 dB (0.0540 against 0.0534 +- 0.0006) and m = 1000 at 6 dB
(0.1523 against 0.1514 +- 0.0012), both within the run-to-run spread of the
authors' own code under Octave (`crosscheck.parquet`: 0.0526-0.0556 at 12 dB).

## Where mlsynth departs from the authors' code

| | `RPCA_HQF.m` / `main.m` | `hqf.py` | effect on Table 1 |
|---|---|---|---|
| U/V update | `inv(V*V' - lam)`, `- lam*U_p` (MATLAB subtracts the scalar from every entry) | `+ lam I`, `+ lam U_prev`, the step that minimises eq. (14)-(15) | none (`+proper_prox`) |
| threshold path | starts at 10 x MAD, then `min(previous, ip x MAD)` | `ip x MAD` each iteration | <= 0.0004 (`+recompute`) |
| stagnation counter | never resets | resets on an improving step | none (`+reset_stall`) |
| `ip` default | 3 | 1.0 | in the default arm |
| `lam` default | 0.001 | 1/sqrt(max(m, n)) | in the default arm |
| rank | given | cumulative variance 0.999 | the 15x |

The arms in `run_tableone.py` switch these on one at a time from the same
matrix and the same random start; the library's U draw is pinned to the port's
for the comparison.

## Files

| file | contents |
|---|---|
| `port.py` | the port of `RPCA_HQF.m` and `Gaussian_noise.m` |
| `run_tableone.py`, `table1_m500.*`, `table1_m1000.*` | Table 1 and the attribution arms |
| `octave_crosscheck.m`, `crosscheck.py`, `crosscheck.parquet`, `crosscheck.log` | the authors' code under Octave 8.4 (statistics 1.6.3), replayed through the port and the library |

Results are parquet; `pd.read_parquet` reads them.
