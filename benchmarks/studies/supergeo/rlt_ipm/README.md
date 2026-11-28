# Solvers for the pair-product relaxation, and the stall RCA

Study scripts for solving the pair-product (first-level RLT) relaxation of
PANGEO's column LP, and for the root-cause analysis of why the
support-restricted interior point stalled at 24 markets. The findings are in
`../PANGEO_ROADMAP.md`, section "Measured: solving the pair relaxation, and the
restricted interior point's stall", and the sources in `../SOURCES.md`. No
library code is changed.

Every script takes the path of a checkout of this repository (`<wt>`) and builds
the same LP: `make_seasonal_sales_panel(units_per_arm=n, arms=("A",), T=60,
seed=0)`, design window the first 42 periods, Q = 2, the groups whose root
reduced cost fits in the gap to the optimum and the disjoint pairs of them whose
reduced costs fit together. All single-threaded (OMP, OpenBLAS, MKL and numba at
1 thread), on 4-core containers. The optima: 20 markets 5.0738116454, 24
4.5336033175, 30 4.3834205759, 40 4.8890564695 (`columns.solve_exact`).

| script | what it measures |
| --- | --- |
| `rlt_scale.py <wt> <n> <solver> <time_limit> [opt=val ...]` | builds the LP (vectorised), solves it with highspy (`simplex`, `ipm`, `pdlp`) or Clarabel (`clarabel`), and forms the weak-duality bound from the returned duals |
| `pge.py <wt> <out.json> 40` | the pair-gap elementary inequalities (the relaxation's projection that needs no pair variables) |
| `rlt_lag.py`, `rlt_da.py` | the Lagrangian bound by blocks: subgradient and volume steps, and a monotone dual ascent (negative results) |
| `sparse_ipm.py`, `sparse_ipm2.py` | the support-restricted interior point with pair pricing (Zanetti and Gondzio), first and second version; `sparse_ipm2.py` writes a MOSEK-style log to a file |
| `sparse_ipm_rca.py` | `sparse_ipm2.py` with the RCA's switches: `SHIFT=maxdiag\|diagrel`, `GUARD=0\|1`, `LEAVE=off`, `LEAVE_CAP=f`, `FR=<npz>`, `DUMP_AT`/`DUMP_PREFIX`; per-iteration linear-system residuals at DEBUG |
| `iter_cost.py` | one iteration's time by step at 24 markets |
| `rca24_restricted.py <wt> <dump.npz>` | the frozen support's restricted LP (HiGHS) and the rank of A_J |
| `rca24_direction.py <wt> <dump.npz> ...` | the Newton direction recomputed with six solve variants |
| `rca_rank.py <wt> <n>` | the rank of the full standard-form LP |
| `rca_presolve.py <wt> <n>` | singleton-row and forcing-constraint presolve, rank and strict feasibility after it |
| `rca_facial.py <wt> <n> <out.npz>` | primal-side facial reduction (uncertified; at 24 markets it removed feasible columns) |
| `rca_facial2.py <wt> <n> <out.npz>` | dual-side facial reduction with exposing-vector certificates (`HIGHS_LOG=<file>` for the solver's log) |
| `run_general.sh <wt> <python> <out> <n> <seed>` | step 1 after the RCA, for one arm: the HiGHS reference, the certified facial reduction, and the working configuration on the face and on the unreduced LP; one summary line per step in `<out>/summary.log` |
| `run_scaling.sh <wt> <python> <out> <g> <p> [...]` | step 2, criterion 3: for each pair of factors, the reference and the fixed configuration on the unreduced LP, one line per LP in `<out>/scaling.log` |


`GROUP_GAP_FACTOR=<g>` and `PAIR_GAP_FACTOR=<p>` (default 1) keep the groups, and the pairs, whose reduced costs fit in that multiple of the gap: weaker but valid relaxations with more pairs per row.

`PANEL_SEED=<s>` (default 0) selects the panel seed for every script that builds the LP through `rlt_scale.py`; for a seed other than 0 the optimum comes from `columns.solve_exact`.

The RCA's runs, in the order the ladder used them (24 markets, reference value
4.344031 from HiGHS simplex on the full LP):

```
# the stall, reproduced, with state dumps
DUMP_AT="40,50,53,55,58,62,65" DUMP_PREFIX=rca24 python -u sparse_ipm_rca.py <wt> 24 4.344031 rca24.log 0.5 0.1 0.001 0.25 66
python -u rca24_restricted.py <wt> rca24_53.npz
python -u rca24_direction.py <wt> rca24_40.npz rca24_53.npz rca24_58.npz
# the ablation arms (log arguments: init 0.5, enter cap 0.1 m, enter tolerance 1e-3, cheap share 0.25, 200 iterations)
SHIFT=diagrel GUARD=0                 python -u sparse_ipm_rca.py <wt> 24 4.344031 A1.log 0.5 0.1 0.001 0.25 200
SHIFT=maxdiag GUARD=1                 ... A2.log ...
SHIFT=diagrel GUARD=1                 ... A3.log ...
SHIFT=diagrel GUARD=1 LEAVE=off       ... A4.log ...
SHIFT=diagrel GUARD=1 LEAVE_CAP=0.02  ... A5.log ...
SHIFT=diagrel GUARD=0 LEAVE=off       ... A6.log ...
# facial reduction, certified, then the original and the fixed settings on the face
HIGHS_LOG=fr2_24_highs.log python -u rca_facial2.py <wt> 24 fr2_24.npz
FR=fr2_24.npz                                          python -u sparse_ipm_rca.py <wt> 24 4.344031 face_orig.log 0.5 0.1 0.001 0.25 200
FR=fr2_24.npz SHIFT=diagrel GUARD=1 LEAVE_CAP=0.02     ... face_fixed.log ...
FR=fr2_24.npz SHIFT=diagrel GUARD=1 LEAVE=off          ... face_fixed_noleave.log ...
```
