"""Correlation-based market similarity for geo-experiment market selection.

These are the first, purely treatment-agnostic leaves of the market-selection
port: they take the wide ``time x unit`` outcome matrix (as produced by
:func:`mlsynth.utils.datautils.geoex_dataprep`) and answer "which units look
like which". Everything stays label-carrying — units flow through as the
``Ywide`` column :class:`pandas.Index` rather than positional bookkeeping.

``correlation_matrix`` is the single seam for the similarity metric; swapping
in a residualized / mean-based metric later leaves :func:`rank_markets_by_correlation`
and every downstream consumer untouched.
"""

import numpy as np
import pandas as pd

from mlsynth.exceptions import MlsynthDataError


def correlation_matrix(Ywide: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation between units (the columns of ``Ywide``).

    Faithful to GeoLift's ``MarketCorrelations``: raw Pearson on the level
    series, no detrending or scaling. This is deliberately the GeoLift baseline
    so the port has a direct validation target; it is also the one place to
    substitute a better-behaved similarity metric.

    Parameters
    ----------
    Ywide : pd.DataFrame
        Wide outcome matrix, shape ``(n_periods, n_units)`` — rows are time,
        columns are units.

    Returns
    -------
    pd.DataFrame
        An ``(n_units, n_units)`` correlation matrix indexed by the unit
        :class:`~pandas.Index` on both axes. Constant (zero-variance) units
        yield ``NaN`` correlations.
    """
    return Ywide.corr()


def rank_markets_by_correlation(Ywide: pd.DataFrame) -> pd.DataFrame:
    """Rank, for each unit, the other units from most to least correlated.

    The label-carrying analogue of GeoLift's ranked-neighbor matrix. Each
    anchor unit (a row) gets the *other* units ordered by descending Pearson
    correlation; the anchor itself is excluded (its self-correlation is a
    trivial ``1``). Units whose correlation is undefined — a constant series
    gives ``NaN`` — sort to the bottom rather than raising.

    Parameters
    ----------
    Ywide : pd.DataFrame
        Wide outcome matrix, shape ``(n_periods, n_units)``.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_units, n_units - 1)``. The index is the anchor unit
        :class:`~pandas.Index`; columns are integer ranks ``1 .. n_units - 1``
        (``rank``); each cell holds the unit id at that similarity rank for the
        anchor.

    Raises
    ------
    MlsynthDataError
        If fewer than two units are present (nothing to rank against).
    """
    units = Ywide.columns
    if len(units) < 2:
        raise MlsynthDataError(
            "Need at least 2 units to rank markets by correlation; "
            f"got {len(units)}."
        )

    correlations = correlation_matrix(Ywide)

    # Vectorized ranking: a single argsort over the whole correlation matrix
    # instead of one pandas sort per anchor. Set the diagonal to +inf so each
    # anchor's self-correlation sorts to the front (then dropped by index, which
    # is robust even when a constant anchor's self-correlation is NaN); map any
    # remaining NaN (constant series) to -inf so it sorts last. ``argsort(-C,
    # kind="stable")`` gives descending order with ties keeping the original
    # column order -- identical to the per-anchor ``sort_values(ascending=False,
    # kind="stable")`` it replaces.
    C = correlations.to_numpy().astype(float, copy=True)
    np.fill_diagonal(C, np.inf)
    C = np.where(np.isnan(C), -np.inf, C)
    order = np.argsort(-C, kind="stable")                  # (n_units, n_units)

    n = C.shape[0]
    keep = order != np.arange(n)[:, None]                  # drop each row's own anchor
    neighbours = order[keep].reshape(n, n - 1)
    ranked = units.to_numpy()[neighbours]                  # indices -> unit labels

    return pd.DataFrame(
        ranked,
        index=pd.Index(units, name=units.name),
        columns=pd.Index(range(1, len(units)), name="rank"),
    )
