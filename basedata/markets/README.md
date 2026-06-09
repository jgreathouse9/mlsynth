# Market geographies (`basedata/markets/`)

Bundled market-area contiguity for spillover-aware experimental design
(e.g. `LEXSCM(..., adjacency=...)`).

## `dma_adjacency.csv`
A **206 × 206 symmetric 0/1 contiguity matrix** over US Nielsen Designated Market
Areas (DMAs), indexed and columned by DMA name (e.g. `"Atlanta, GA"`). Entry 1 ⇔
the two DMAs share a border. Ready to pass straight to `LEXSCM(adjacency=...)`:

```python
import pandas as pd
adj = pd.read_csv("basedata/markets/dma_adjacency.csv", index_col=0)
```

## `dma_metadata.csv`
Per-DMA `dma_name`, `dma_code` (Nielsen number), `state`, `latitude`,
`longitude` (centroids) — handy for filtering a region or building a
distance-based graph instead.

## Provenance
Derived from the public `simzou/nielsen-dma` TopoJSON via shared-arc contiguity;
see `build_dma_adjacency.py` (rerun it to regenerate). Six DMAs that the source
topology leaves arc-isolated (Erie, Harrisonburg, Lima, Miami-Fort Lauderdale,
Salisbury, Santa Barbara) are patched to their nearest centroids so the graph is
fully connected.
