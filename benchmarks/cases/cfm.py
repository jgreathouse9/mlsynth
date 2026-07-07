"""Path A benchmark: Bai & Wang (2026) Causal Factor Model, both empirical
applications.

Reproduces the numbers the paper reports for its two empirical analyses on
the authors' data (``basedata/smoking_data.csv`` and
``basedata/german_reunification.csv``): the Ahn-Horenstein ER/GR factor
counts, the Chow structural-break F-statistic at the intervention date, and
the intercept-shift t-statistics. The paper releases no code, so the target
is the reported quantities; they are sensitive to the extracted factor, so
matching them exercises the factor extraction and the treated regressions,
not merely an endpoint.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlsynth.utils.cfm_helpers.factors import extract_cfm_factors
from mlsynth.utils.cfm_helpers.inference import cfm_inference
from mlsynth.utils.cfm_helpers.pipeline import chow_break_statistic
from mlsynth.utils.cfm_helpers.setup import prepare_cfm_inputs

_BASE = Path(__file__).resolve().parents[2] / "basedata"


def _california() -> dict:
    df = pd.read_csv(_BASE / "smoking_data.csv")
    df["treat"] = ((df.state == "California") & (df.year >= 1989)).astype(int)
    inputs = prepare_cfm_inputs(df, "cigsale", "treat", "state", "year")
    n_er, _, _, _ = extract_cfm_factors(inputs.control_outcomes, selection="er")
    n_gr, _, _, _ = extract_cfm_factors(inputs.control_outcomes, selection="gr")
    _, F1, _, _ = extract_cfm_factors(inputs.control_outcomes, n_factors=1)
    _, F2, _, _ = extract_cfm_factors(inputs.control_outcomes, n_factors=2)
    chow = chow_break_statistic(inputs.treated_outcome, F1, inputs.T0)
    k1 = cfm_inference(inputs.treated_outcome, F1, inputs.control_outcomes,
                       inputs.T0, factor_variance=False)["kappa_t"]
    k2 = cfm_inference(inputs.treated_outcome, F2, inputs.control_outcomes,
                       inputs.T0, factor_variance=False)["kappa_t"]
    return {"ca_er": float(n_er), "ca_gr": float(n_gr), "ca_chow": float(chow),
            "ca_kappa_t_1f": float(k1), "ca_kappa_t_2f": float(k2)}


def _germany() -> dict:
    df = pd.read_csv(_BASE / "german_reunification.csv")
    # paper convention: 1990 marked; treated periods 1991-2003 (T0 = 1990)
    df["treat"] = ((df.country == "West Germany") & (df.year >= 1991)).astype(int)
    inputs = prepare_cfm_inputs(df, "gdp", "treat", "country", "year")
    n_er, _, _, _ = extract_cfm_factors(inputs.control_outcomes, selection="er")
    n_gr, _, _, _ = extract_cfm_factors(inputs.control_outcomes, selection="gr")
    _, F1, _, _ = extract_cfm_factors(inputs.control_outcomes, n_factors=1)
    chow = chow_break_statistic(inputs.treated_outcome, F1, inputs.T0)
    k1 = cfm_inference(inputs.treated_outcome, F1, inputs.control_outcomes,
                       inputs.T0, factor_variance=False)["kappa_t"]
    return {"de_er": float(n_er), "de_gr": float(n_gr), "de_chow": float(chow),
            "de_kappa_t_1f": float(k1)}


def run() -> dict:
    out = {}
    out.update(_california())
    out.update(_germany())
    return out


EXPECTED = {
    # California Prop 99 (Bai & Wang Sec 7.1)
    "ca_er": (1.0, 0.0),
    "ca_gr": (1.0, 0.0),
    "ca_chow": (16.84, 0.1),
    "ca_kappa_t_1f": (1.38, 0.05),
    "ca_kappa_t_2f": (0.10, 0.05),
    # German reunification (Bai & Wang Sec 7.2)
    "de_er": (1.0, 0.0),
    "de_gr": (1.0, 0.0),
    "de_chow": (634.5, 1.0),
    "de_kappa_t_1f": (11.81, 0.3),   # mlsynth 11.77 under the block-HC1 form
}


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(run(), indent=2))
