"""Frozen, NumPy-first containers for the Modified Unbiased Synthetic
Control (MUSC) estimator.

MUSC is the Bottmer, Imbens, Spiess & Warnick (2024 JBES) modification
of the Synthetic Control estimator that adds a column-sums-to-zero
restriction on the weight matrix, making the resulting average
treatment effect estimator **unbiased under random assignment** of
which unit is treated (Lemma 1 of the paper). Everything below is
pure NumPy; the only DataFrame touchpoint is
:func:`mlsynth.utils.musc_helpers.setup.prepare_musc_inputs`.

Units and time are addressed through :class:`IndexSet` (immutable
label-to-integer maps) so downstream code never reaches back into
pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

# IndexSet currently lives in ``fast_scm_helpers.structure`` on main;
# importing from there keeps MUSC in lock-step with SCMO / FDID style.
from ..fast_scm_helpers.structure import IndexSet


# Variant names exposed to the public API.
SC = "SC"            # standard SC (no column-sum constraint) — comparator
MUSC = "MUSC"        # MUSC with column-sums-to-zero — unbiased variant


@dataclass(frozen=True)
class MUSCInputs:
    """Preprocessed, NumPy-only panel for the MUSC engine.

    Parameters
    ----------
    unit_index : IndexSet
        All ``N`` units; row order of ``Y``.
    time_index : IndexSet
        All ``T`` periods; column order of ``Y``.
    treated_idx : int
        Row index (into ``unit_index``) of the treated unit.
    donor_idx : np.ndarray
        Row indices of the donor pool.
    Y : np.ndarray
        Outcome panel of shape ``(N, T)`` (rows = units).
    T0 : int
        Number of pre-treatment periods. The first treated period is
        the first column of ``Y_post``.
    metadata : dict
        Free-form provenance.
    """

    unit_index: IndexSet
    time_index: IndexSet
    treated_idx: int
    donor_idx: np.ndarray
    Y: np.ndarray
    T0: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.Y.shape[1])

    @property
    def N(self) -> int:
        return int(self.Y.shape[0])

    @property
    def n_donors(self) -> int:
        return int(self.donor_idx.shape[0])

    @property
    def donor_labels(self) -> np.ndarray:
        return self.unit_index.get_labels(self.donor_idx)

    @property
    def treated_label(self) -> Any:
        return self.unit_index.get_labels([self.treated_idx])[0]

    @property
    def Y_pre(self) -> np.ndarray:
        """Pre-treatment outcomes, shape ``(T0, N)`` (time-major)."""
        return self.Y[:, : self.T0].T

    @property
    def Y_post(self) -> np.ndarray:
        """Post-treatment outcomes, shape ``(T - T0, N)`` (time-major)."""
        return self.Y[:, self.T0 :].T

    @property
    def y_treated(self) -> np.ndarray:
        return self.Y[self.treated_idx]


@dataclass(frozen=True)
class MUSCVariantFit:
    """A single estimator variant fit (SC or MUSC).

    Parameters
    ----------
    name : str
        Variant label, ``"SC"`` or ``"MUSC"``.
    M : np.ndarray
        Full ``(N, N+1)`` weight matrix as defined in the paper: the
        first column is the per-unit intercept; the remaining ``N``
        columns are the within-row weights, with ``M[i, i+1] = 1`` and
        off-diagonals in ``[-1, 0]``.
    weights_on_treated : np.ndarray
        Donor weights from the treated unit's row, length ``n_donors``,
        in the canonical SC sign (non-negative, summing to one).
    intercept : float
        The treated unit's row-intercept ``M[treated_idx, 0]``.
    counterfactual : np.ndarray
        Length-``T`` synthetic counterfactual for the treated unit:
        ``-intercept - Σ_j M[treated, j+1] Y_{j, t}``.
    gap : np.ndarray
        Length-``T`` treated-minus-counterfactual gap.
    att : float
        Mean of ``gap`` over the post-treatment window.
    pre_rmse : float
        Root mean squared error of ``gap`` over the pre-treatment
        window. Used as a simple measure of pre-period fit quality.
    column_sum_residual : float
        ``max_j |Σ_i M[i, j+1]|`` — the binding diagnostic for the
        MUSC unbiasedness constraint; should be ~0 for ``MUSC`` and
        bounded away from 0 for ``SC``.
    donor_weights : dict
        Donor-label-keyed dict of the treated unit's row weights, in
        canonical SC sign (non-negative).
    """

    name: str
    M: np.ndarray
    weights_on_treated: np.ndarray
    intercept: float
    counterfactual: np.ndarray
    gap: np.ndarray
    att: float
    pre_rmse: float
    column_sum_residual: float
    donor_weights: Dict[Any, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MUSCInference:
    """Inference outputs for the recommended (``MUSC``) variant.

    Parameters
    ----------
    variance : float
        Proposition-1 unbiased estimate of ``Var[τ̂]`` at the first
        post-treatment period, computed under the random-unit-
        assignment design (Bottmer et al. 2024, equation 3.3).
    se : float
        Square root of ``variance`` (``nan`` if ``variance`` is
        negative due to finite-sample noise).
    ci_normal : (float, float)
        Normal-approximation ``(1 − alpha)`` CI for the ATT.
    ci_randomization : (float, float)
        Exact randomization-based ``(1 − alpha)`` CI built by
        inverting a placebo permutation test (Section 3.5 of the
        paper). ``nan`` entries when the design is too small or the
        QP failed on every placebo.
    placebo_atts : np.ndarray
        Placebo ATTs from the leave-one-out estimator applied to each
        non-treated unit; populates ``ci_randomization``.
    alpha : float
        Two-sided significance level.
    """

    variance: float
    se: float
    ci_normal: Tuple[float, float]
    ci_randomization: Tuple[float, float]
    placebo_atts: np.ndarray
    alpha: float


@dataclass(frozen=True)
class MUSCResults:
    """Top-level container returned by :meth:`mlsynth.MUSC.fit`."""

    inputs: MUSCInputs
    fits: Dict[str, MUSCVariantFit]
    inference: MUSCInference
    selected_variant: str = MUSC
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def _primary(self) -> MUSCVariantFit:
        return self.fits.get(self.selected_variant, next(iter(self.fits.values())))

    @property
    def att(self) -> float:
        return self._primary.att

    @property
    def att_ci(self) -> Tuple[float, float]:
        """Randomization-based ``(1 − alpha)`` CI for the ATT."""
        return self.inference.ci_randomization

    @property
    def counterfactual(self) -> np.ndarray:
        return self._primary.counterfactual

    @property
    def gap(self) -> np.ndarray:
        return self._primary.gap

    @property
    def donor_weights(self) -> Dict[Any, float]:
        return self._primary.donor_weights

    @property
    def pre_rmse(self) -> float:
        return self._primary.pre_rmse

    def att_by_variant(self) -> Dict[str, float]:
        return {name: fit.att for name, fit in self.fits.items()}
