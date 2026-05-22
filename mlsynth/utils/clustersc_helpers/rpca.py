"""Robust PCA Synthetic Control pipeline.

Wraps the existing :func:`mlsynth.utils.estutils.RPCASYNTH` routine
(PCP -- Candes, Li, Ma and Wright 2011; HQF -- Wang, Li, So and Liu
2023) and packages its output into a clean :class:`MethodFit`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthEstimationError
from ..estutils import RPCASYNTH
from .structures import CLUSTERSCInputs, MethodFit


def run_rpca(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    inputs: CLUSTERSCInputs,
    rpca_method: str = "PCP",
) -> MethodFit:
    """Run the RPCA-SC pipeline and assemble a :class:`MethodFit`.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel (the underlying RPCASYNTH routine pivots
        it internally).
    outcome, treat, unitid, time : str
        Column names.
    inputs : CLUSTERSCInputs
        Preprocessed inputs container (provides treated outcome and
        T / T0 for downstream computation).
    rpca_method : {"PCP", "HQF"}
        Robust-PCA decomposition method.
    """
    if rpca_method not in {"PCP", "HQF"}:
        raise MlsynthEstimationError(
            f"rpca_method must be 'PCP' or 'HQF'; got {rpca_method!r}."
        )

    raw = RPCASYNTH(
        panel_dataframe=df,
        estimation_config={
            "unitid": unitid,
            "time": time,
            "outcome": outcome,
            "treat": treat,
            "ROB": rpca_method,
        },
        preprocessed_data_dict={
            "treated_unit_name": inputs.treated_unit_name,
            "pre_periods": inputs.T0,
            "post_periods": inputs.n_post,
            "y": inputs.treated_outcome,
        },
    )

    vectors = raw.get("Vectors", {})
    counterfactual = np.asarray(
        vectors.get("Counterfactual"), dtype=float
    ).flatten()
    if counterfactual.size != inputs.T:
        raise MlsynthEstimationError(
            f"RPCASYNTH counterfactual length mismatch: "
            f"{counterfactual.size} vs {inputs.T}."
        )

    gap = inputs.treated_outcome - counterfactual
    att = (
        float(np.mean(gap[inputs.T0:])) if inputs.n_post > 0
        else float("nan")
    )
    pre_rmse = float(np.sqrt(np.mean(gap[:inputs.T0] ** 2)))

    donor_weights_dict = {
        str(k): float(v) for k, v in dict(raw.get("weights", {})).items()
    }
    selected = list(donor_weights_dict.keys()) or list(inputs.donor_names)

    return MethodFit(
        name="rpca",
        counterfactual=counterfactual,
        gap=gap,
        att=att,
        pre_rmse=pre_rmse,
        donor_weights=donor_weights_dict,
        selected_donors=np.asarray(selected),
        metadata={"rpca_method": rpca_method},
    )
