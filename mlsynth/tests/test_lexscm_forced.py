"""Forced-in / forbidden treated markets for LEXSCM.

LEXSCM already honours the cluster / adjacency / stratum-quota / size-band
constraints; the one gap versus SYNDES is the pair of hard market
lists ``to_be_treated`` (always treated) and ``not_to_be_treated`` (never
treated, stays a donor). These tests pin that behaviour test-first, at both the
search layer (``select_treated_designs`` with ``forced``, exact and heuristic
paths) and the estimator layer (``LEXSCM`` config).
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.fast_scm_helpers.lexsearch import select_treated_designs


# === search layer: forced indices, exact enumeration ===

def _gram(n, seed=0):
    """A PSD Gram matrix over n units (random but reproducible)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + np.eye(n)


def test_forced_index_in_every_enumerated_design():
    G = _gram(8)
    out = select_treated_designs(G, candidate_idx=list(range(8)), m=3, top_K=5,
                                 method="enumerate", forced=[2])
    assert out["top_designs"]
    for design in out["top_designs"]:
        assert 2 in set(np.asarray(design.indices).tolist())


def test_multiple_forced_indices_all_present():
    G = _gram(9)
    out = select_treated_designs(G, candidate_idx=list(range(9)), m=4, top_K=5,
                                 method="enumerate", forced=[1, 7])
    assert out["top_designs"]
    for design in out["top_designs"]:
        s = set(np.asarray(design.indices).tolist())
        assert {1, 7} <= s


def test_forced_equals_m_yields_exactly_that_set():
    G = _gram(8)
    out = select_treated_designs(G, candidate_idx=list(range(8)), m=3, top_K=5,
                                 method="enumerate", forced=[0, 3, 5])
    designs = out["top_designs"]
    assert len(designs) == 1                       # only one feasible tuple
    assert set(np.asarray(designs[0].indices).tolist()) == {0, 3, 5}


def test_forced_index_in_every_heuristic_design():
    """The heuristic path (large pool) must also pin the forced units."""
    G = _gram(14, seed=2)
    out = select_treated_designs(G, candidate_idx=list(range(14)), m=4, top_K=6,
                                 method="heuristic", forced=[5], n_starts=8,
                                 random_state=1)
    assert out["top_designs"]
    for design in out["top_designs"]:
        assert 5 in set(np.asarray(design.indices).tolist())


def test_forced_subset_of_candidates_required():
    G = _gram(6)
    with pytest.raises(MlsynthConfigError):
        select_treated_designs(G, candidate_idx=[0, 1, 2, 3], m=2,
                               method="enumerate", forced=[5])   # 5 not a candidate


def test_too_many_forced_for_m_raises():
    G = _gram(6)
    with pytest.raises(MlsynthConfigError, match="forced|to_be_treated"):
        select_treated_designs(G, candidate_idx=list(range(6)), m=2,
                               method="enumerate", forced=[0, 1, 2])


# === estimator layer: LEXSCM config to_be_treated / not_to_be_treated ===

def _panel(seed=4, T=24, n=8):
    """A small long panel with a candidate flag (all eligible)."""
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    rows = []
    for i in range(n):
        s = base + rng.normal(scale=1.0, size=T) + i
        for t in range(T):
            rows.append({"unit": f"u{i}", "time": t, "Y": float(s[t]),
                         "elig": True})
    return pd.DataFrame(rows)


def _fit(df, **over):
    from mlsynth.estimators.lexscm import LEXSCM
    cfg = dict(df=df, outcome="Y", unitid="unit", time="time",
               candidate_col="elig", m=3, top_K=5, display_graph=False,
               verbose=False)
    cfg.update(over)
    return LEXSCM(cfg).fit()


def _treated_sets(res):
    return [set(c.treated_weight_dict_full) for c in res.search.candidates]


def test_estimator_not_to_be_treated_never_treated():
    res = _fit(_panel(), not_to_be_treated=["u3"])
    for s in _treated_sets(res):
        assert "u3" not in s


def test_estimator_not_to_be_treated_still_a_donor():
    """A forbidden market is removed from treatment but stays an eligible donor."""
    res = _fit(_panel(), not_to_be_treated=["u3"])
    donors_seen = set()
    for c in res.search.candidates:
        donors_seen |= set(c.control_weight_dict_full)
    assert "u3" in donors_seen


def test_estimator_to_be_treated_always_treated():
    res = _fit(_panel(), to_be_treated=["u1"])
    sets = _treated_sets(res)
    assert sets
    for s in sets:
        assert "u1" in s


def test_estimator_forced_and_forbidden_together():
    res = _fit(_panel(), to_be_treated=["u1"], not_to_be_treated=["u5"])
    for s in _treated_sets(res):
        assert "u1" in s and "u5" not in s


def test_estimator_forced_unit_must_be_candidate():
    df = _panel()
    df.loc[df.unit == "u2", "elig"] = False        # u2 not a candidate
    with pytest.raises(MlsynthConfigError):
        _fit(df, to_be_treated=["u2"])


def test_estimator_overlap_forced_forbidden_raises():
    # LEXSCM surfaces config-validation failures as MlsynthDataError.
    with pytest.raises(MlsynthDataError, match="both|overlap|to_be_treated"):
        _fit(_panel(), to_be_treated=["u1"], not_to_be_treated=["u1"])
