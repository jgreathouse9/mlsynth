"""Data preparation for SDID.

Calls :func:`mlsynth.utils.datautils.dataprep` and packages its return
shape (single-treated or cohorts) into a uniform ``cohorts_dict`` that
the math helpers consume. This replaces the inline ``if "cohorts" not in
prep`` restructuring block that used to live in ``SDID.fit()``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from .structures import SDIDInputs


def apply_ddd_transform(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    subgroup: str,
    target_subgroup: Any,
):
    """Zhuang (2024) triple-difference-to-DID transform for SC-DDD mode.

    Demeans the outcome by the non-target subgroup within each
    treatment-group-by-time cell and returns a reduced ``(unit, time)`` panel
    over the target subgroup, ready for the ordinary SDID pipeline.

    For each row the transformed outcome is

    .. math::

        W_{it} = Y_{it} - \\bar Y_{\\text{non-target},\\, g(i),\\, t},

    where :math:`g(i)` is the unit's treatment-group indicator (1 if the unit is
    ever treated, 0 otherwise) and :math:`\\bar Y_{\\text{non-target}, g, t}` is
    the mean outcome over the non-target subgroup rows in that
    group-by-time cell (Zhuang 2024, eq. 10-12). A difference-in-differences on
    :math:`W` over the target subgroup identifies the triple-difference effect,
    so SDID applied to :math:`W` yields the synthetic triple difference.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel with a ``subgroup`` column (unit x subgroup x time).
    outcome, treat, unitid, time : str
        Column names. ``outcome`` must be numeric.
    subgroup : str
        Column naming the within-unit subgroup dimension.
    target_subgroup : Any
        Value of ``subgroup`` identifying the policy-exposed subgroup.

    Returns
    -------
    (pd.DataFrame, str)
        The reduced ``(unitid, time, treat, <outcome>__ddd)`` panel over the
        target subgroup, and the transformed-outcome column name.

    Raises
    ------
    MlsynthDataError
        If the outcome is non-numeric, a treatment-group-by-time cell has no
        non-target rows to demean by, or the target subgroup is not unique per
        ``(unit, time)``.
    """
    d = df.copy()
    y = pd.to_numeric(d[outcome], errors="coerce")
    if y.isna().any():
        raise MlsynthDataError(
            f"SC-DDD: outcome '{outcome}' has non-numeric or missing values; "
            "the demeaning transform needs a numeric outcome.")
    d[outcome] = y.to_numpy()

    # Treatment-group indicator g(i): 1 for units ever treated, else 0.
    d["_ddd_grp"] = (d.groupby(unitid, observed=True)[treat].transform("max") > 0).astype(int)

    non_target = d[d[subgroup] != target_subgroup]
    nt_mean = (non_target.groupby(["_ddd_grp", time], observed=True)[outcome].mean()
               .rename("_ddd_nt").reset_index())
    d = d.merge(nt_mean, on=["_ddd_grp", time], how="left")

    target = d[d[subgroup] == target_subgroup].copy()
    if target.duplicated([unitid, time]).any():
        raise MlsynthDataError(
            f"SC-DDD: target subgroup {target_subgroup!r} is not unique per "
            f"({unitid}, {time}); each unit-time must have a single target row.")
    if target["_ddd_nt"].isna().any():
        raise MlsynthDataError(
            "SC-DDD: some treatment-group-by-time cells have no non-target rows "
            "to demean by; the transform is undefined there.")

    new_outcome = f"{outcome}__ddd"
    target[new_outcome] = target[outcome] - target["_ddd_nt"]
    reduced = (target[[unitid, time, treat, new_outcome]]
               .sort_values([unitid, time]).reset_index(drop=True))
    return reduced, new_outcome


def _pre_period_covariate_means(df, unitid, treat, cols, unit_order):
    """Per-unit covariate summary for 'match': the mean over untreated rows.

    Shape ``(K, n_units)``, columns in ``unit_order``. The paper's empirical
    work uses either the mean of the covariates or their last pre-treatment
    value, and reports both; the mean is the default here and is what its
    Tables 1-3 label "mean of covariates". Untreated rows rather than all rows,
    so a covariate that itself responds to treatment cannot feed the
    post-treatment period back into the matching.
    """
    untreated = df[df[treat] == 0]
    means = untreated.groupby(unitid, observed=True)[list(cols)].mean()
    absent = [u for u in unit_order if u not in means.index]
    if absent:
        raise MlsynthDataError(
            f"unit(s) {absent[:5]} have no untreated rows, so their covariate "
            "means for covariates={'match': ...} are undefined.")
    out = means.reindex(list(unit_order)).to_numpy(dtype=float).T
    if not np.all(np.isfinite(out)):
        raise MlsynthDataError(
            "covariate means for covariates={'match': ...} contain NaN/inf; "
            "check the covariate columns for missing values.")
    return np.atleast_2d(out)


def _apply_optimized_covariates(payload, df, unitid, time, cols, donor_names,
                                treated_names):
    """Fit and remove one cohort's 'optimized' covariate effect, in place.

    Per cohort rather than panel-wide: the donor pool, the horizon and the
    ridge all vary by adoption date in a staggered design, so a single
    coefficient would not be the quantity any cohort's objective defines. This
    is the same reason ``match`` attaches its covariate summaries per cohort.
    """
    from .covariates import optimized_covariate_beta

    wide = df.pivot_table(index=time, columns=unitid, values=list(cols))
    order = sorted(wide.index)
    donor_cov = np.stack([
        wide[c].reindex(index=order, columns=list(donor_names)).to_numpy(float)
        for c in cols])
    treated_cov = np.stack([
        wide[c].reindex(index=order, columns=list(treated_names))
        .to_numpy(float).mean(axis=1) for c in cols])

    beta = optimized_covariate_beta(
        donor_outcomes=np.asarray(payload["donor_matrix"], dtype=float),
        treated_outcome=np.asarray(payload["y"], dtype=float).mean(axis=1),
        donor_covariates=donor_cov,
        treated_covariates=treated_cov,
        pre_periods=int(payload["pre_periods"]),
        n_treated=len(treated_names),
    )
    payload["donor_matrix"] = (np.asarray(payload["donor_matrix"], dtype=float)
                               - np.tensordot(beta, donor_cov, axes=(0, 0)))
    # ``y`` keeps its (T, n_treated) shape, so the coefficients come off each
    # treated unit's own covariate path rather than off their mean.
    treated_paths = np.stack([
        wide[c].reindex(index=order, columns=list(treated_names)).to_numpy(float)
        for c in cols])
    payload["y"] = (np.asarray(payload["y"], dtype=float)
                    - np.tensordot(beta, treated_paths, axes=(0, 0)))
    payload["optimized_beta"] = beta
    return payload


def prepare_sdid_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    match_covariates=None,
    match_pre_periods=None,
    optimized_covariates=None,
    zeta=None,
) -> SDIDInputs:
    """Prepare panel data for the SDID pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form balanced panel.
    outcome, treat, unitid, time : str
        Column names identifying the outcome, treatment indicator, units,
        and time periods.
    zeta : float, optional
        Unit-weight ridge penalty to use in place of the data-driven one. When
        given it is written onto every cohort payload, so each cohort uses the
        same penalty rather than one scaled to its own donors and horizon --
        which is the point of overriding it.

    Returns
    -------
    SDIDInputs
        Pre-processed cohorts payload and metadata.
    """

    keys = balance(df, unitid, time)
    prep: Dict[str, Any] = dataprep(df, unitid, time, outcome, treat, keys=keys)

    if "cohorts" in prep:
        # ``dataprep`` keys cohorts by the actual time *label* (e.g. 2010),
        # but the cohort estimator's ``ell = arange(T) - (a - 1)`` math expects
        # ``a`` to be the cohort's time *index* (1-based). Build a label -> index
        # map from the panel's time axis so the event-time labels come out
        # centered on each cohort's first treated period.
        time_labels_arr = np.asarray(prep["time_labels"])
        label_to_index = {
            label: position + 1
            for position, label in enumerate(time_labels_arr)
        }
        cohorts_dict = {
            int(label_to_index[k]): _coerce_cohort_payload(v)
            for k, v in prep["cohorts"].items()
        }
        if match_covariates:
            # Per cohort, because the donor pool and the treated set both vary
            # by cohort in a staggered design. The treated summary is the mean
            # over that cohort's treated units, matching how the cohort's
            # treated outcome path is itself a mean over them.
            for key, payload in cohorts_dict.items():
                payload["donor_covariates"] = _pre_period_covariate_means(
                    df, unitid, treat, match_covariates,
                    list(payload["donor_names"]))
                payload["treated_covariates"] = _pre_period_covariate_means(
                    df, unitid, treat, match_covariates,
                    list(payload["treated_indices"])).mean(axis=1)
                payload["match_pre_periods"] = match_pre_periods
        if optimized_covariates:
            for key, payload in cohorts_dict.items():
                _apply_optimized_covariates(
                    payload, df, unitid, time, list(optimized_covariates),
                    list(payload["donor_names"]),
                    list(payload["treated_indices"]))
        # Earliest cohort drives the pre/post counts surfaced on inputs.
        earliest = min(cohorts_dict.keys())
        n_pre = int(cohorts_dict[earliest]["pre_periods"])
        n_post = int(cohorts_dict[earliest]["post_periods"])
        # Treated unit name: in the cohort path, dataprep does not return a
        # single 'treated_unit_name'; surface the first treated label of the
        # earliest cohort instead (used for plotting only).
        first_treated = cohorts_dict[earliest]["treated_indices"]
        treated_unit_name = first_treated[0] if first_treated else None
        # Donor pool labels come from any cohort (they're shared across the
        # cohorts return shape).
        first_label = sorted(prep["cohorts"].keys())[0]
        donor_names = list(prep["cohorts"][first_label]["donor_names"])
    else:
        pre = prep.get("pre_periods")
        post = prep.get("post_periods")
        total = prep.get("total_periods")
        if pre is None or post is None:
            raise MlsynthDataError(
                "dataprep output missing pre_periods/post_periods for the "
                "single-treated-unit case."
            )
        if total is None:
            warnings.warn(
                "'total_periods' missing from dataprep single-unit output; "
                "computing as pre_periods + post_periods.",
                UserWarning,
            )
            total = pre + post

        # ``cohort_key`` is the 1-based index of the *first treated period* so
        # that ``ell = arange(T) - (cohort_key - 1)`` puts ell = 0 on the first
        # treated period (e.g. 1989 for Prop 99). Setting it to ``pre`` instead
        # would shift the post-treatment mask one period early and include the
        # last pre-treatment period in the post-treatment ATT — see
        # Arkhangelsky et al. (2021), Table 1 for the canonical -15.6 value.
        cohort_key = int(pre) + 1
        cohorts_dict = {
            cohort_key: {
                "y": prep["y"].reshape(-1, 1),
                "donor_matrix": prep["donor_matrix"],
                "treated_indices": [prep["treated_unit_name"]],
                "pre_periods": int(pre),
                "post_periods": int(post),
                "total_periods": int(total),
            }
        }
        if match_covariates:
            donor_cov = _pre_period_covariate_means(
                df, unitid, treat, match_covariates, list(prep["donor_names"]))
            treated_cov = _pre_period_covariate_means(
                df, unitid, treat, match_covariates,
                [prep["treated_unit_name"]]).ravel()
            cohorts_dict[cohort_key]["donor_covariates"] = donor_cov
            cohorts_dict[cohort_key]["treated_covariates"] = treated_cov
            cohorts_dict[cohort_key]["match_pre_periods"] = match_pre_periods
        if optimized_covariates:
            _apply_optimized_covariates(
                cohorts_dict[cohort_key], df, unitid, time,
                list(optimized_covariates), list(prep["donor_names"]),
                [prep["treated_unit_name"]])
        n_pre = int(pre)
        n_post = int(post)
        treated_unit_name = prep["treated_unit_name"]
        donor_names = list(prep["donor_names"])

    if zeta is not None:
        for payload in cohorts_dict.values():
            payload["zeta_override"] = float(zeta)

    Ywide = prep["Ywide"]
    time_labels = np.asarray(prep["time_labels"])

    return SDIDInputs(
        cohorts_dict=cohorts_dict,
        treated_unit_name=treated_unit_name,
        donor_names=donor_names,
        time_labels=time_labels,
        n_pre=n_pre,
        n_post=n_post,
        Ywide=Ywide,
        outcome=outcome,
    )


def _coerce_cohort_payload(raw_cohort: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt ``dataprep`` cohort entries to the schema the math expects.

    ``dataprep`` returns the cohort treated-outcome matrix as ``y`` but
    keys the treated-unit list as ``treated_units``. The math helpers
    expect ``treated_indices``. Both naming conventions co-exist in the
    library; this shim bridges them without touching either.
    """

    payload = dict(raw_cohort)
    if "treated_indices" not in payload:
        payload["treated_indices"] = list(raw_cohort.get("treated_units", []))
    # The disaggregate-cohort y is already (T, n_treated_in_cohort).
    return payload
