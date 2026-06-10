"""Data preparation for the PROXIMAL estimator.

Pivots a long panel into the typed :class:`PROXIMALInputs` container:
the treated outcome, the donor outcome matrix ``W`` and donor proxy
matrix ``Z0``, and -- when surrogate units are configured -- the cleaned
surrogate outcome matrix ``X`` and surrogate proxy matrix ``Z1``.

Surrogate outcomes are residualized against the donor proxies/outcomes on
the pre-period via :func:`mlsynth.utils.datautils.clean_surrogates2`,
matching the construction in Liu, Tchetgen Tchetgen and Varjao (2023).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import clean_surrogates2, dataprep, proxy_dataprep
from .structures import PROXIMALInputs


def _hac_bandwidth(num_post_periods: int) -> int:
    """Rule-of-thumb Bartlett bandwidth ``floor(4 (T_post / 100)^(2/9))``."""
    return int(np.floor(4 * (num_post_periods / 100) ** (2 / 9)))


def prepare_proximal_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    treat: str,
    donors: List[Union[str, int]],
    surrogates: List[Union[str, int]],
    vars: Dict[str, List[str]],
    methods: Sequence[str] = ("PI",),
    spsc_detrend: bool = True,
    spsc_lambda: Optional[float] = None,
    spsc_spline_df: int = 5,
    spsc_basis_degree: int = 1,
    spsc_conformal: bool = False,
    spsc_conformal_periods: Optional[Sequence[int]] = None,
) -> PROXIMALInputs:
    """Pivot a long panel into the typed inputs the PROXIMAL pipeline expects.

    Only the matrices the requested ``methods`` need are built: donor
    proxies (``Z0``) are built when a method consuming them (PI/PIS/PIPost)
    is requested, and surrogate matrices when PIS/PIPost is requested. SPSC
    needs only the donor outcomes and the treated series.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel data.
    outcome, unitid, time, treat : str
        Column names identifying the outcome, units, time periods, and the
        treatment indicator.
    donors : list
        Donor unit identifiers used to build ``W``.
    surrogates : list
        Surrogate unit identifiers used to build ``X``/``Z1``.
    vars : dict
        Proxy-variable map (``"donorproxies"`` / ``"surrogatevars"``).
    methods : sequence of str
        Estimators to prepare for.
    spsc_detrend, spsc_lambda, spsc_spline_df, spsc_conformal, spsc_conformal_periods
        SPSC options, forwarded onto :class:`PROXIMALInputs`.

    Returns
    -------
    PROXIMALInputs
        Prepared outcome/proxy matrices and label metadata.

    Raises
    ------
    MlsynthDataError
        If no configured donors are present in the panel.
    """

    needs_donor_proxies = any(m in methods for m in ("PI", "PIS", "PIPost", "DR", "PIPW"))
    needs_surrogates = any(m in methods for m in ("PIS", "PIPost"))

    prepared = dataprep(df, unitid, time, outcome, treat)

    valid_donors = [d for d in donors if d in prepared["Ywide"].columns]
    if not valid_donors:
        raise MlsynthDataError("None of the configured donor units are present in the panel.")

    donor_outcomes = prepared["Ywide"][valid_donors].to_numpy()

    donor_proxies: Optional[np.ndarray] = None
    if needs_donor_proxies:
        donor_proxy_pivot = df.pivot(index=time, columns=unitid, values=vars["donorproxies"][0])
        donor_proxies = donor_proxy_pivot[valid_donors].to_numpy()

    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    num_post = int(prepared["post_periods"])

    surrogate_outcomes: Optional[np.ndarray] = None
    surrogate_proxies: Optional[np.ndarray] = None
    if needs_surrogates:
        X_raw, Z1 = proxy_dataprep(
            df=df,
            surrogate_units=surrogates,
            proxy_variable_column_names_map=vars,
            unit_id_column_name=unitid,
            time_period_column_name=time,
            num_total_periods=T,
        )
        # Residualize surrogate outcomes against donor proxies/outcomes on the pre-period.
        surrogate_outcomes = clean_surrogates2(X_raw, donor_proxies, donor_outcomes, T0)
        surrogate_proxies = Z1

    return PROXIMALInputs(
        y=np.asarray(prepared["y"], dtype=float).ravel(),
        donor_outcomes=donor_outcomes,
        donor_proxies=donor_proxies,
        surrogate_outcomes=surrogate_outcomes,
        surrogate_proxies=surrogate_proxies,
        T=T,
        T0=T0,
        bandwidth=_hac_bandwidth(num_post),
        time_labels=np.asarray(prepared["Ywide"].index.to_numpy()),
        treated_unit_name=prepared["treated_unit_name"],
        donor_names=list(valid_donors),
        methods=tuple(methods),
        spsc_detrend=spsc_detrend,
        spsc_lambda=spsc_lambda,
        spsc_spline_df=spsc_spline_df,
        spsc_basis_degree=spsc_basis_degree,
        spsc_conformal=spsc_conformal,
        spsc_conformal_periods=spsc_conformal_periods,
    )
