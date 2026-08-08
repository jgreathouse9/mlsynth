# DSC Path-B targets — Zhang, Zhang & Zhang (2024), Section 5.1

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
