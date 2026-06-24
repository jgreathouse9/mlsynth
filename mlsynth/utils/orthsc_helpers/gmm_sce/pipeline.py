"""GMM-SCE orchestrator: dataprep -> control/instrument split -> one-step GMM
weights -> treatment-effect path and standardized results (Fry 2024).

The control/instrument split is either supplied on the config or chosen by the
Andrews--Lu downward-testing procedure (``model_selection=True``). The reported
ATT is the post-period average of the gap between the treated unit and the
GMM-weighted controls; the over-identification J-statistic is carried as a
diagnostic.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ....config_models import (
    BaseEstimatorResults,
    InferenceResults,
    MethodDetailsResults,
)
from ....exceptions import MlsynthDataError
from ...datautils import dataprep
from ...results_helpers import build_effect_submodels, make_weights_results
from .solver import gmm_sc_weights
from .selection import select_controls


def _prep(config):
    """dataprep -> treated series, donor matrix, names, pre split, time labels."""
    prep = dataprep(
        df=config.df,
        unit_id_column_name=config.unitid,
        time_period_column_name=config.time,
        outcome_column_name=config.outcome,
        treatment_indicator_column_name=config.treat,
    )
    if "y" not in prep or "donor_matrix" not in prep:  # pragma: no cover - single-treated dataprep guard (mirrors run_orthsc)
        raise MlsynthDataError(
            "GMM-SCE requires a single treated unit (dataprep returned a "
            "multi-cohort structure).")
    y = np.asarray(prep["y"], dtype=float).ravel()
    Y0 = np.asarray(prep["donor_matrix"], dtype=float)            # (T, J_all)
    pre = int(prep["pre_periods"])
    donor_names = [str(d) for d in prep["donor_names"]]
    return {
        "y": y, "Y0": Y0, "pre": pre,
        "time_labels": np.asarray(prep["time_labels"]),
        "donor_names": donor_names,
        "treated_name": str(prep.get("treated_unit_name", "treated")),
    }


def _resolve_split(config, ins) -> tuple[List[str], List[str]]:
    """Return ``(controls, instruments)`` label lists, by selection or config."""
    donor_names = ins["donor_names"]
    col = {name: j for j, name in enumerate(donor_names)}
    Y0, pre, y = ins["Y0"], ins["pre"], ins["y"]

    def _check(labels, what):
        missing = [u for u in labels if u not in col]
        if missing:
            raise MlsynthDataError(
                f"{what} not in the donor pool: {missing}; donors are {donor_names}.")

    if config.model_selection:
        guaranteed = [str(u) for u in (config.guaranteed_instruments or [])]
        _check(guaranteed, "guaranteed_instruments")
        candidates = [d for d in donor_names if d not in set(guaranteed)]
        if not candidates:
            raise MlsynthDataError(
                "GMM-SCE model selection needs at least one candidate donor that "
                "is not a guaranteed instrument.")
        YN0 = Y0[:pre][:, [col[c] for c in candidates]]
        YN1 = (Y0[:pre][:, [col[g] for g in guaranteed]] if guaranteed else None)
        ctrl_local = select_controls(
            y[:pre], YN0, YN1,
            weight_tol=config.weight_tol, include_constant=config.include_constant)
        controls = [candidates[i] for i in ctrl_local]
        instruments = [c for c in candidates if c not in set(controls)] + guaranteed
    else:
        instruments = [str(u) for u in config.instruments]
        _check(instruments, "instrument unit(s)")
        if config.controls is not None:
            controls = [str(u) for u in config.controls]
            _check(controls, "control unit(s)")
        else:
            instr_set = set(instruments)
            controls = [d for d in donor_names if d not in instr_set]

    if not controls:  # pragma: no cover - unreachable: selection and defaults yield >=1 control
        raise MlsynthDataError("GMM-SCE needs at least one control unit.")
    if not instruments:  # pragma: no cover - unreachable: the split yields >=1 instrument
        raise MlsynthDataError("GMM-SCE needs at least one instrument unit.")
    return controls, instruments


def run_gmm_sce(config) -> BaseEstimatorResults:
    """Fit GMM-SCE from a config and assemble standardized results."""
    ins = _prep(config)
    controls, instruments = _resolve_split(config, ins)
    Y0, pre, y = ins["Y0"], ins["pre"], ins["y"]
    col = {name: j for j, name in enumerate(ins["donor_names"])}

    ctrl_cols = [col[c] for c in controls]
    instr_cols = [col[z] for z in instruments]
    res = gmm_sc_weights(
        y[:pre], Y0[:pre][:, ctrl_cols], Y0[:pre][:, instr_cols],
        include_constant=config.include_constant)
    w = res["weights"]

    counterfactual = Y0[:, ctrl_cols] @ w                         # (T,)
    effect = y - counterfactual
    att = float(np.mean(effect[pre:])) if pre < len(y) else float("nan")

    weights = make_weights_results(
        {c: float(wj) for c, wj in zip(controls, w)},
        constraint="simplex (non-negative, sum to 1)",
        extra={
            "instruments": instruments,
            "jstatistic": res["jstatistic"],
            "n_instruments": res["n_instruments"],
            "selected_by": "andrews-lu" if config.model_selection else "supplied",
        },
    )
    inference = InferenceResults(
        method="GMM-SCE over-identification J-statistic (Fry 2024); point "
               "inference deferred to the paper's Section 6",
        details={
            "jstatistic": float(res["jstatistic"]),
            "n_instruments": int(res["n_instruments"]),
            "n_controls": len(controls),
        },
    )
    submodels = build_effect_submodels(
        observed_outcome=y, counterfactual_outcome=counterfactual,
        n_pre_periods=pre, n_post_periods=int(len(y) - pre),
        time_periods=ins["time_labels"], weights=weights, inference=inference,
        effects_overrides={"att": att},
        intervention_time=(ins["time_labels"][pre]
                           if pre < len(ins["time_labels"]) else None),
    )
    return BaseEstimatorResults(
        **submodels,
        method_details=MethodDetailsResults(
            method_name="ORTHSC (gmm_sce)",
            parameters_used={
                "method": "gmm_sce",
                "model_selection": config.model_selection,
                "n_controls": len(controls),
                "n_instruments": int(res["n_instruments"]),
                "include_constant": config.include_constant,
            },
        ),
        additional_outputs={
            "treated_name": ins["treated_name"],
            "controls": controls,
            "instruments": instruments,
            "pre_periods": pre,
            "jstatistic": float(res["jstatistic"]),
        },
    )
