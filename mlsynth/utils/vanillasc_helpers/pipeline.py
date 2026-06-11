"""Orchestration for the VanillaSC estimator.

dataprep -> (optional) covariate matrices -> bilevel engine -> ATT, fit
diagnostics, in-space placebo inference -> standardized results.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...config_models import (
    BaseEstimatorResults,
    InferenceResults,
    MethodDetailsResults,
)
from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from ..helperutils import IndexSet
from ..results_helpers import build_effect_submodels, make_weights_results
from ..bilevel import BilevelSCM

_EPS = 1e-12


def _covariate_means(
    df: pd.DataFrame,
    units: List[Any],
    covariates: List[str],
    windows: Dict[Any, Tuple[Any, Any]],
    pre_labels: List[Any],
    unitid: str,
    time: str,
) -> np.ndarray:
    """Per-unit covariate means over each covariate's window -> ``(K, N)``.

    With a single shared window (the default, and augsynth's covariate spec) the
    aggregation is a joint ``na.omit``: a pre-period contributes to the means
    only if *every* covariate is observed there, matching augsynth's
    ``X <- na.omit(...)`` (which drops a row carrying any missing covariate
    before averaging). Per-covariate windows fall back to independent means.
    """
    for cov in covariates:
        if cov not in df.columns:
            raise MlsynthDataError(f"Covariate {cov!r} not in DataFrame.")

    def _years(cov):
        win = windows.get(cov)
        if win is None:
            return list(pre_labels)
        lo, hi = win
        return [t for t in pre_labels if lo <= t <= hi] or list(pre_labels)

    year_sets = [_years(c) for c in covariates]

    if all(ys == year_sets[0] for ys in year_sets):
        # joint na.omit over the shared window (augsynth convention)
        years = year_sets[0]
        sub = df[df[time].isin(years)]
        stack = np.stack(
            [sub.pivot_table(index=unitid, columns=time, values=cov)
                .reindex(index=units, columns=years).to_numpy()
             for cov in covariates],
            axis=2,
        )                                            # (N, T_win, K)
        row_ok = ~np.isnan(stack).any(axis=2)        # period kept iff no NA
        means = np.full((len(units), len(covariates)), np.nan)
        for u in range(len(units)):
            if row_ok[u].any():
                means[u] = stack[u][row_ok[u]].mean(0)
        X = means.T                                  # (K, N)
    else:
        rows = []
        for cov, years in zip(covariates, year_sets):
            g = df[df[time].isin(years)].groupby(unitid)[cov].mean()
            rows.append([float(g.get(u, np.nan)) for u in units])
        X = np.asarray(rows, dtype=float)

    if not np.all(np.isfinite(X)):
        raise MlsynthDataError("Covariate means contain NaN (check windows/coverage).")
    return X


def _scale_unit_variance(X: np.ndarray) -> np.ndarray:
    """Scale each predictor row to unit variance across units (Synth convention)."""
    sd = X.std(axis=1, ddof=1, keepdims=True)
    sd[sd < _EPS] = 1.0
    return X / sd


def _covariate_balance(
    pred_names: List[str], Xall: np.ndarray, W: np.ndarray
) -> Dict[str, Any]:
    """Abadie Table-1 style balance: treated vs synthetic vs donor average.

    ``Xall`` is the raw (unscaled) ``(P, N)`` predictor-means matrix with the
    treated unit in column 0 and donors (in ``W`` order) in columns ``1:``.
    The synthetic value of each predictor is the donor-weighted mean; the donor
    average is the unweighted mean across donors. All in the predictors' own
    units, matching what practitioners report.
    """
    treated = Xall[:, 0]
    donors = Xall[:, 1:]
    w = np.asarray(W, dtype=float)
    s = w.sum()
    if abs(s) > _EPS:
        w = w / s
    synthetic = donors @ w
    donor_avg = donors.mean(axis=1)

    def mape(ref: np.ndarray) -> float:
        denom = np.where(np.abs(treated) > _EPS, treated, np.nan)
        return float(np.nanmean(np.abs((ref - treated) / denom)) * 100.0)

    return {
        "predictors": list(pred_names),
        "treated": treated.tolist(),
        "synthetic": synthetic.tolist(),
        "donor_average": donor_avg.tolist(),
        "mean_abs_pct_gap": {
            "synthetic": mape(synthetic),
            "donor_average": mape(donor_avg),
        },
    }


def _rmspe_ratio(y: np.ndarray, cf: np.ndarray, pre: int) -> Tuple[float, float, float]:
    """(pre_rmspe, post_rmspe, ratio) for an outcome/counterfactual pair."""
    gap = y - cf
    pre_r = float(np.sqrt(np.mean(gap[:pre] ** 2)))
    post_r = float(np.sqrt(np.mean(gap[pre:] ** 2))) if gap[pre:].size else float("nan")
    ratio = post_r / pre_r if pre_r > _EPS else float("inf")
    return pre_r, post_r, ratio


def run_vanillasc(config) -> BaseEstimatorResults:
    """Fit VanillaSC and assemble :class:`BaseEstimatorResults`."""
    covariates = list(config.covariates or [])
    windows = dict(config.covariate_windows or {})

    prep = dataprep(
        df=config.df,
        unit_id_column_name=config.unitid,
        time_period_column_name=config.time,
        outcome_column_name=config.outcome,
        treatment_indicator_column_name=config.treat,
    )
    if "y" not in prep or "donor_matrix" not in prep:
        raise MlsynthDataError(
            "VanillaSC requires a single treated unit (dataprep returned a "
            "multi-cohort structure)."
        )
    y = np.asarray(prep["y"], dtype=float).ravel()
    Y0 = np.asarray(prep["donor_matrix"], dtype=float)
    pre = int(prep["pre_periods"])
    time_labels = np.asarray(prep["time_labels"])
    J = Y0.shape[1]

    # All unit bookkeeping goes through IndexSets, which preserve the original
    # label dtype (so groupby lookups match the DataFrame) and centralise the
    # label <-> position mapping. ``donors`` indexes the columns of Y0;
    # ``units`` is the treated unit followed by the donors (covariate order).
    donors = IndexSet.from_labels(prep["donor_names"])
    treated_label = prep.get("treated_unit_name", "treated")
    units = IndexSet.from_labels([treated_label, *donors.labels])
    treated_name = str(treated_label)
    donor_names = [str(lbl) for lbl in donors.labels]   # string labels for reporting

    # Build predictor matrices (treated + donors, donor order matches Y0 columns).
    X1 = X0 = None
    Xall = None
    pred_names: List[str] = []
    if covariates:
        pre_labels = list(time_labels[:pre])
        Xall = _covariate_means(
            config.df, list(units.labels), covariates, windows, pre_labels,
            config.unitid, config.time,
        )
        Xs = _scale_unit_variance(Xall)
        X1, X0 = Xs[:, 0], Xs[:, 1:]
        pred_names = list(covariates)

    engine = BilevelSCM(
        config.backend,
        canonical_v=config.canonical_v,
        seed=config.seed,
        augment=config.augment,
        ridge_lambda=config.ridge_lambda,
        residualize=config.residualize,
        maxiter=config.mscmt_maxiter,
        popsize=config.mscmt_popsize,
        prune_shady=config.mscmt_prune_shady,
        cv=config.penalized_cv,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = engine.fit(
            y[:pre], Y0[:pre],
            X1=X1, X0=X0, donor_names=donor_names, predictor_names=pred_names,
        )

    counterfactual = res.counterfactual(Y0)
    gap = y - counterfactual
    pre_r, post_r, ratio_tr = _rmspe_ratio(y, counterfactual, pre)

    mode = config.inference
    mode = "placebo" if mode is True else ("none" if not mode else str(mode).lower())
    inference = None

    # SCPI prediction intervals (Cattaneo, Feng & Titiunik 2021).
    if mode == "scpi" and gap[pre:].size:
        from .scpi import scpi_intervals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = scpi_intervals(
                y, Y0, pre, res.W, sims=config.scpi_sims,
                u_alpha=config.alpha, e_alpha=config.alpha,
                e_method=config.scpi_e_method, seed=config.seed,
            )
        post_labels = list(time_labels[pre:])
        inference = InferenceResults(
            ci_lower=float(sc.metadata["att_lower"]),
            ci_upper=float(sc.metadata["att_upper"]),
            confidence_level=1.0 - 2.0 * config.alpha,
            method="scpi prediction intervals (Cattaneo-Feng-Titiunik 2021)",
            details={
                "periods": post_labels,
                "tau": sc.tau, "pi_lower": sc.lower, "pi_upper": sc.upper,
                "counterfactual_lower": sc.cf_lower,
                "counterfactual_upper": sc.cf_upper,
                "att": sc.metadata["att"],
                "in_sample_lower": sc.M1_lower, "in_sample_upper": sc.M1_upper,
                "out_of_sample_lower": sc.M2_lower, "out_of_sample_upper": sc.M2_upper,
                "sims": sc.metadata["sims"], "e_method": sc.metadata["e_method"],
            },
        )

    # Conformal test-inversion prediction intervals (Chernozhukov, Wuthrich &
    # Zhu 2021; augsynth's default ASCM inference). Reuses the fitted ridge
    # penalty across refits, matching augsynth.
    if mode == "conformal" and gap[pre:].size:
        from ..bilevel import conformal_intervals
        Z0 = X0.T if X0 is not None else None
        z1 = X1 if X1 is not None else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ci = conformal_intervals(
                y, Y0, pre, lambda_=res.lambda_, Z0=Z0, z1=z1,
                alpha=config.alpha, ns=config.scpi_sims, seed=config.seed,
                ridge_kwargs={"residualize": config.residualize},
            )
        # per-period counterfactual bands: gap tau in [lower, upper] => cf in
        # [y - upper, y - lower]; pre-period left as NaN (no interval).
        cf_lower = np.full_like(y, np.nan, dtype=float)
        cf_upper = np.full_like(y, np.nan, dtype=float)
        cf_lower[pre:] = y[pre:] - ci.upper
        cf_upper[pre:] = y[pre:] - ci.lower
        inference = InferenceResults(
            ci_lower=float(np.nanmean(ci.lower)) if ci.lower.size else None,
            ci_upper=float(np.nanmean(ci.upper)) if ci.upper.size else None,
            p_value=float(ci.joint_p_value),
            confidence_level=1.0 - config.alpha,
            method="conformal prediction intervals (Chernozhukov-Wuthrich-Zhu 2021)",
            details={
                "periods": list(time_labels[pre:]),
                "tau": ci.att, "pi_lower": ci.lower, "pi_upper": ci.upper,
                "counterfactual_lower": cf_lower,
                "counterfactual_upper": cf_upper,
                "period_p_value": ci.p_value,
                "joint_p_value": ci.joint_p_value,
                "lambda": res.lambda_,
            },
        )

    # Leave-Two-Out refined placebo test (Lei & Sudijono 2025).
    if mode == "lto" and J >= 3 and gap[pre:].size:
        from .lto import lto_placebo_test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lto = lto_placebo_test(
                engine, y, Y0, pre, X1=X1, X0=X0, alpha=config.alpha,
                max_pairs=config.lto_max_pairs, seed=config.seed,
            )
        inference = InferenceResults(
            p_value=lto["p_value"],
            method="leave-two-out refined placebo (Lei-Sudijono 2025)",
            confidence_level=1.0 - config.alpha,
            details={
                "treated_rmspe_ratio": ratio_tr,
                "p_powered": lto["p_powered"],
                "powered_offset_c": lto["c"],
                "type_i_bound": lto["type_i_bound"],
                "reject_at_alpha": lto["reject"],
                "n_pairs": lto["n_pairs"],
                "treated_losses": lto["treated_losses"],
                "n_units": lto["N"],
                "alpha": lto["alpha"],
                "subsampled": lto["subsampled"],
            },
        )

    # In-space placebo inference (Abadie): reassign treatment to each donor.
    if mode == "placebo" and J >= 2 and gap[pre:].size:
        ratios = []
        for j in range(J):
            others = [k for k in range(J) if k != j]
            yj = Y0[:, j]
            Y0j = Y0[:, others]
            X1j = X0j = None
            if covariates:
                X1j = Xs[:, 1 + j]
                X0j = Xs[:, [1 + k for k in others]]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rj = engine.fit(yj[:pre], Y0j[:pre], X1=X1j, X0=X0j)
                cfj = rj.counterfactual(Y0j)
                _, _, ratio_j = _rmspe_ratio(yj, cfj, pre)
                if np.isfinite(ratio_j):
                    ratios.append(ratio_j)
            except Exception:  # pragma: no cover - defensive placebo-refit guard
                continue
        all_ratios = np.array(ratios + [ratio_tr], dtype=float)
        p_value = float(np.mean(all_ratios >= ratio_tr))
        inference = InferenceResults(
            p_value=p_value,
            method="in-space placebo (RMSPE ratio)",
            confidence_level=1.0 - config.alpha,
            details={
                "treated_rmspe_ratio": ratio_tr,
                "n_placebos": len(ratios),
                "rank": int(np.sum(all_ratios >= ratio_tr)),
            },
        )

    weights = make_weights_results(
        res.donor_weights,
        constraint=("simplex (non-negative, sum to 1)"),
        extra={
            "backend": res.backend,
            "predictor_weights": (
                {n: float(v) for n, v in zip(res.predictor_names, res.V)}
                if res.V is not None else None
            ),
            "v_agreement": res.v_agreement,
        },
    )
    # Canonical effect / fit / time-series sub-models (results_helpers is the
    # single source of truth for the series-derived quantities).
    submodels = build_effect_submodels(
        observed_outcome=y, counterfactual_outcome=counterfactual,
        n_pre_periods=pre, n_post_periods=int(len(y) - pre),
        time_periods=time_labels, weights=weights, inference=inference,
        additional_effects={"rmspe_ratio": ratio_tr},
    )
    results = BaseEstimatorResults(
        **submodels,
        method_details=MethodDetailsResults(
            method_name=f"VanillaSC[{res.backend}]",
            parameters_used={
                "backend": res.backend,
                "augment": config.augment,
                "covariates": covariates,
                "canonical_v": config.canonical_v,
                "v_agreement": res.v_agreement,
            },
        ),
        additional_outputs={
            "donor_names": donor_names,
            "treated_name": treated_name,
            "pre_periods": pre,
            "solver_diagnostics": res.diagnostics,
            "covariate_balance": (
                _covariate_balance(pred_names, Xall, res.W)
                if covariates and Xall is not None else None
            ),
        },
    )

    if config.display_graphs or config.save:
        _plot_vanillasc(config, y, counterfactual, time_labels, pre,
                        treated_name, res.backend, inference)
    return results


def _full_band(arr, T: int, pre: int) -> np.ndarray:
    """Align a (possibly post-only) band array to the full T-length axis."""
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == T:
        return a
    full = np.full(T, np.nan)
    full[pre:pre + a.size] = a
    return full


def _plot_vanillasc(config, y, counterfactual, time_labels, pre,
                    treated_name, backend, inference) -> None:
    """Render the observed-vs-synthetic plot through the shared Plotter,
    shading the prediction-interval band when conformal/SCPI inference ran."""
    import matplotlib.pyplot as plt

    from ..plotting import Plotter, mlsynth_style

    T = len(time_labels)
    interval = None
    if inference is not None and getattr(inference, "details", None):
        lo = inference.details.get("counterfactual_lower")
        hi = inference.details.get("counterfactual_upper")
        if lo is not None and hi is not None:
            interval = (_full_band(lo, T, pre), _full_band(hi, T, pre))

    intervention = time_labels[pre] if 0 <= pre < T else None
    with mlsynth_style():
        plotter = Plotter.from_config(getattr(config, "plot", None))
        ax = plotter.observed_vs_counterfactual(
            times=time_labels, observed=y, counterfactuals=[counterfactual],
            labels=[f"Synthetic {treated_name}"], treated_label=treated_name,
            intervention=intervention, interval=interval,
            outcome=config.outcome, time=config.time,
            title=f"VanillaSC[{backend}]: {treated_name}",
        )
        fig = ax.figure
        if config.save:
            fname = (config.save if isinstance(config.save, str)
                     else f"VanillaSC_{treated_name}.png")
            fig.savefig(fname, bbox_inches="tight")
        if config.display_graphs:
            plt.show()
        plt.close(fig)
