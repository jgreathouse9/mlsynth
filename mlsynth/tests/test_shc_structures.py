"""Tests for mlsynth.utils.shc_helpers.structures dataclasses.

Covers the computed properties (T, n, N) on SHCInputs and the
weights_by_block property on SHCResults, plus basic construction of the
frozen containers.
"""

import numpy as np
import pytest

from mlsynth.utils.helperutils import IndexSet
from mlsynth.utils.shc_helpers.structures import (
    SHCDesign,
    SHCInference,
    SHCInputs,
    SHCResults,
)


def _make_inputs(T=10, T0=7, m=3):
    time_index = IndexSet.from_labels(np.arange(T))
    y = np.arange(T, dtype=float)
    return SHCInputs(
        time_index=time_index,
        y=y,
        T0=T0,
        m=m,
        treated_label="A",
        metadata={"foo": "bar"},
    )


def test_shcinputs_properties():
    inp = _make_inputs(T=10, T0=7, m=3)
    # T is derived from y length.
    assert inp.T == 10
    assert isinstance(inp.T, int)
    # n = T - T0
    assert inp.n == 10 - 7
    # N = T0 - n - (m - 1)
    assert inp.N == 7 - inp.n - (3 - 1)


def test_shcinputs_metadata_default():
    time_index = IndexSet.from_labels(np.arange(4))
    inp = SHCInputs(
        time_index=time_index,
        y=np.zeros(4),
        T0=3,
        m=2,
        treated_label=1,
    )
    assert inp.metadata == {}


def test_shcinputs_is_frozen():
    inp = _make_inputs()
    with pytest.raises(Exception):
        inp.T0 = 99  # frozen dataclass -> FrozenInstanceError


def _make_design():
    return SHCDesign(
        bandwidth=1.5,
        latent_pre=np.arange(7, dtype=float),
        weights=np.array([0.5, 0.5]),
        selected_blocks=[0, 1],
        block_weights={"b0": 0.5, "b1": 0.5},
        counterfactual_window=np.arange(6, dtype=float),
        use_augmented=False,
    )


def test_shcdesign_defaults():
    design = _make_design()
    # best_lambda defaults to None for plain SHC.
    assert design.best_lambda is None
    assert design.use_augmented is False


def _make_inference():
    return SHCInference(
        method="conformal_permutation",
        test_statistic=1.0,
        p_value=0.3,
        critical_values={0.05: 2.0},
        reject={0.05: False},
        num_resamples=100,
        null_distribution=np.zeros(100),
        conformal_lower=np.zeros(3),
        conformal_upper=np.ones(3),
        confidence_level=0.9,
    )


def test_shcresults_weights_by_block_property():
    inp = _make_inputs()
    design = _make_design()
    results = SHCResults(
        inputs=inp,
        design=design,
        att=0.5,
        att_percent=10.0,
        observed=np.arange(6, dtype=float),
        counterfactual=np.arange(6, dtype=float),
        gap=np.zeros(6),
        time_labels=np.arange(6),
        fit_diagnostics={"pre_rmse": 0.1},
        inference=_make_inference(),
    )
    # weights_by_block delegates to design.block_weights.
    assert results.weights_by_block == design.block_weights
    # metadata default factory.
    assert results.metadata == {}
    # inference default is exercised here as a provided value.
    assert results.inference is not None


def test_shcresults_inference_optional_default():
    inp = _make_inputs()
    design = _make_design()
    results = SHCResults(
        inputs=inp,
        design=design,
        att=0.0,
        att_percent=0.0,
        observed=np.zeros(6),
        counterfactual=np.zeros(6),
        gap=np.zeros(6),
        time_labels=np.arange(6),
        fit_diagnostics={},
    )
    assert results.inference is None
    assert results.weights_by_block == design.block_weights
