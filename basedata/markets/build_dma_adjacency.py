"""Build the Nielsen DMA contiguity matrix for ``basedata/markets/``.

Source
------
Nielsen Designated Market Area (DMA) TopoJSON from the public ``simzou/nielsen-dma``
repository::

    https://raw.githubusercontent.com/simzou/nielsen-dma/master/nielsentopo.json

Method
------
TopoJSON encodes shared boundaries as shared *arcs*, so DMA contiguity falls out
of the topology directly -- two DMAs border each other iff their geometries
reference a common arc (negative arc indices are normalized via ``~a == -a-1``,
the reverse traversal of arc ``a``). No GIS library or geometry math required.

A handful of DMAs (Erie PA, Harrisonburg VA, Lima OH, Miami-Fort Lauderdale FL,
Salisbury MD, Santa Barbara CA) come out isolated because this topology encodes a
few coincident boundaries as separate arcs rather than one shared arc. Those
degree-0 DMAs are patched to their 3 nearest neighbours by great-circle
(haversine) distance between the bundled centroids, yielding a fully-connected,
symmetric, zero-diagonal 0/1 contiguity matrix.

Outputs (regenerate by running this module)
-------------------------------------------
* ``dma_adjacency.csv`` -- 206 x 206 symmetric 0/1 matrix indexed and columned by
  DMA name; ready to pass to ``LEXSCM(adjacency=...)``.
* ``dma_metadata.csv``  -- per-DMA ``dma_name``, ``dma_code``, ``state``,
  ``latitude``, ``longitude``.
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/simzou/nielsen-dma/master/nielsentopo.json"
)
HERE = Path(__file__).resolve().parent


def _arc_indices(geometry) -> set:
    """All (normalized) arc indices a TopoJSON geometry references."""
    out: set = set()

    def walk(x):
        if isinstance(x, list):
            for y in x:
                walk(y)
        else:
            out.add(x if x >= 0 else ~x)   # ~x == -x-1 : reversed-arc normalization

    walk(geometry["arcs"])
    return out


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def build(topo: dict, patch_k: int = 3):
    geoms = topo["objects"]["nielsen_dma"]["geometries"]
    names = [g["properties"]["dma1"] for g in geoms]
    if len(set(names)) != len(names):
        raise ValueError("DMA names are not unique")
    lat = np.array([g["properties"]["latitude"] for g in geoms])
    lon = np.array([g["properties"]["longitude"] for g in geoms])
    code = [g["properties"]["dma"] for g in geoms]
    states = [(re.findall(r"\b([A-Z]{2})\b", n) or [""])[-1] for n in names]
    n = len(names)

    owner = defaultdict(list)
    for i, g in enumerate(geoms):
        for a in _arc_indices(g):
            owner[a].append(i)
    A = np.zeros((n, n), dtype=int)
    for who in owner.values():
        for x in range(len(who)):
            for y in range(x + 1, len(who)):
                A[who[x], who[y]] = A[who[y], who[x]] = 1

    # patch degree-0 DMAs (topology artifacts) via nearest centroids
    patched = []
    for i in range(n):
        if A[i].sum() == 0:
            dist = np.array([_haversine_km(lat[i], lon[i], lat[j], lon[j])
                             if j != i else np.inf for j in range(n)])
            for j in np.argsort(dist)[:patch_k]:
                A[i, j] = A[j, i] = 1
            patched.append(names[i])

    adjacency = pd.DataFrame(A, index=names, columns=names)
    metadata = pd.DataFrame({"dma_name": names, "dma_code": code, "state": states,
                             "latitude": lat, "longitude": lon})
    return adjacency, metadata, patched


def main():
    with urllib.request.urlopen(SOURCE_URL) as resp:
        topo = json.loads(resp.read().decode("utf-8"))
    adjacency, metadata, patched = build(topo)
    adjacency.to_csv(HERE / "dma_adjacency.csv")
    metadata.to_csv(HERE / "dma_metadata.csv", index=False)
    n = adjacency.shape[0]
    print(f"DMAs={n}  edges={int(adjacency.values.sum() // 2)}  "
          f"avg_deg={adjacency.values.sum(1).mean():.1f}  patched={patched}")


if __name__ == "__main__":
    main()
