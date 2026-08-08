# DSC Path-B targets — Zhang, Zhang & Zhang (2026), Section 5.1

`benchmarks/cases/dsc_mc.py` is scored against Figures 1 and 2 of
[arXiv:2405.00953v3](https://arxiv.org/abs/2405.00953). Section 5 of that paper
prints no tables, so the target numbers are not transcribed from the text —
they are read out of the figures themselves.

This directory holds the reader, not a dump: the case carries the sixteen
recovered values inline, and `digitize_figures.py` exists so they can be
re-derived from the source instead of taken on trust. The paper's PDFs are not
vendored.

## Re-deriving the targets

```bash
curl -o 2405.00953v3.tar.gz https://arxiv.org/e-print/2405.00953v3
mkdir -p src && tar -xzf 2405.00953v3.tar.gz -C src
python benchmarks/reference/dsc_mc/digitize_figures.py src
```

The four panels are vector PDFs written by R 4.2.2, so each series survives as
an explicit `m`/`l` polyline in the page content stream. The script pulls those
polylines, then maps device coordinates onto the axes.

Calibration is against the major gridlines, not the tick labels. R anchors label
text at its left edge and centres it vertically on the break, so a label sits
about 4.3 device units below its tick and a horizontal distance that depends on
the string's width to its left. Calibrating on the text instead shifts the
recovered weight-error values up by 0.0031 uniformly — small, but a third of the
distance between the last two points of that curve.

Resolution is about 6e-5 per device unit on the ratio panels and 3e-4 on the
weight-error panels, both far below the Monte-Carlo noise in the plotted values.

## What comes out

Figure 1, the risk ratio, and Figure 2, the weight error, are what
`_PAPER_RATIO` and `_PAPER_NORM` in the case hold:

| figure | J | M=50 | M=100 | M=200 | M=400 |
|---|---|---|---|---|---|
| 1 ratio | 20 | 1.0238 | 1.0145 | 1.0075 | 1.0030 |
| 1 ratio | 50 | 1.0274 | 1.0180 | 1.0099 | 1.0043 |
| 2 error | 20 | 0.1856 | 0.1221 | 0.0668 | 0.0276 |
| 2 error | 50 | 0.2789 | 0.2123 | 0.1301 | 0.0593 |

The script also reads Figures 3 and 4, the Section 5.2 quantile-factor design,
which the benchmark does not reproduce — `docs/replications/dsc_mc.rst` says
why.

## Ruling the solver out as a cause of the M = 50 divergence

The reconstruction sits above the published risk ratio at the two smallest-draw
cells, by 0.006 at J = 20 and 0.028 at J = 50. The obvious suspect was the
solver. At J = 50, M = 50 the donor matrix is square with a condition number of
2.0e4, and `mlsynth/utils/dsc_helpers/weights.py` already records that a unique
minimum value does not imply a unique argmin -- on the Stata Journal tenure
panel, projected gradient matched an exact QP's objective to 0.00 percent while
its weights differed enough to miss the published values by 0.0047. If two exact
QP solvers picked different argmins on a near-degenerate design, the population
risk evaluated at those weights would differ, and the reported ratio with it.

They do not. `dump_cells.py` writes the exact matrices the benchmark's solver
sees at four cells spanning the conditioning range; `disco_solver.R` solves each
with `pracma::lsqlincon` under DiSCo's own argument construction, copied from
`R/DiSCo_weights_reg.R` at `ed2b3d94` (all-ones equality row, spectral-norm
rescale, `lb = 0`, `ub = 1`); `compare_solvers.py` compares:

| cell | cond(A) | max abs weight diff | relative objective gap |
|---|---|---|---|
| J=50, M=50 | 2.0e4 | 3.3e-9 | 1.2e-10 |
| J=50, M=100 | 7.7e2 | 1.7e-9 | 9.1e-11 |
| J=20, M=50 | 9.1e1 | 8.3e-10 | 1.4e-10 |
| J=20, M=400 | 1.6e1 | 4.8e-9 | 2.2e-9 |

CLARABEL and quadprog agree to 3e-9 everywhere, and the population risk at their
respective weights agrees to six decimals. The divergence from the published
figure is therefore upstream of the solver, in the design specification, which
is where `docs/replications/dsc_mc.rst` attributes it.

The same result bears on issue #304. The Dube weight disagreement between
mlsynth and the `DiSCos` R package is not the solver either: hand both the same
matrix and they return the same answer. What differs is the grid rule -- DiSCo
draws its quadrature points with `runif`, so its weights are a Monte Carlo
estimate that varies across seeds by up to 0.119 on that panel, while mlsynth
uses the deterministic closed grid that matches the Stata implementation. That
is the conclusion `benchmarks/cases/disco_tenure.py` reaches from the other
direction.

Reproducing it needs the R reference (`bash benchmarks/R/install_discos.sh`,
15-25 minutes cold):

```bash
cd benchmarks/reference/dsc_mc
python dump_cells.py
Rscript disco_solver.R J50_M50 J50_M100 J20_M50 J20_M400
python compare_solvers.py
```
