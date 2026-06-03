"""Grossi et al. (2025) partial-interference SCM subpackage.

Direct and spillover effects of an intervention when interference is confined
to the treated unit's cluster (partial interference). The treated unit and its
cluster-mates each receive a penalized synthetic control built from the clean
(far-cluster) controls; the treated gap is the direct effect, the cluster-mate
gaps are spillover effects.

Module layout:

* :mod:`.pipeline` -- public dispatcher :func:`run_grossi`.
"""

from __future__ import annotations

from .pipeline import run_grossi

__all__ = ["run_grossi"]
