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
from ...exceptions import (MlsynthConfigError, MlsynthDataError,
                           MlsynthEstimationError)
from ..datautils import dataprep
from ..helperutils import IndexSet
from ..results_helpers import build_effect_submodels, make_weights_results
from ..bilevel import BilevelSCM, bias_corrected_gaps

_EPS = 1e-12


class _OracleFit:
    """Drop-in for the bilevel fit when donor weights are user-specified.

    Exposes the same surface ``run_vanillasc`` reads off a fitted engine
    (``W``, ``donor_weights``, ``backend``, ``V``, ``v_agreement``,
    ``predictor_names``, ``counterfactual``) but skips the optimization.
    """

    def __init__(self, w: np.ndarray, donor_names: List[str]):
        self.W = np.asarray(w, dtype=float)
        self.donor_weights = {n: float(v) for n, v in zip(donor_names, self.W)}
        self.backend = "oracle"
        self.V = None
        self.v_agreement = None
        self.predictor_names: List[str] = []
        self.diagnostics: Dict[str, Any] = {"backend": "oracle",
                                            "note": "user-specified weights"}

    def counterfactual(self, Y0: np.ndarray) -> np.ndarray:
        return np.asarray(Y0, dtype=float) @ self.W


class _ConstrainedFit:
    """Drop-in for the bilevel fit when a ``w_constr`` family is requested.

    Solves scpi's constrained weight program (Cattaneo, Feng, Palomba and
    Titiunik) on the pre-treatment outcomes instead of the bilevel simplex QP,
    and exposes the same surface ``run_vanillasc`` reads off a fitted engine.
    ``wc`` carries the normalised constraint, including the budget ``Q`` that
    ridge and L1-L2 estimate from the data.
    """

    def __init__(self, y_pre: np.ndarray, Y0_pre: np.ndarray,
                 donor_names: List[str], w_constr: Any):
        from .staggered_engine import b_est

        J = Y0_pre.shape[1]
        x, wc = b_est(np.asarray(y_pre, dtype=float).reshape(-1, 1),
                      np.asarray(Y0_pre, dtype=float), J, 0, w_constr)
        self.W = np.asarray(x, dtype=float).ravel()[:J]
        self.wc = wc
        self.donor_weights = {n: float(v) for n, v in zip(donor_names, self.W)}
        self.backend = f"w_constr:{wc['name']}"
        self.V = None
        self.v_agreement = None
        self.predictor_names: List[str] = []
        self.diagnostics: Dict[str, Any] = {
            "backend": self.backend,
            "w_constr": wc["name"],
            "w_constr_Q": wc.get("Q"),
            "w_constr_Q2": wc.get("Q2"),
            "w_constr_lambda": wc.get("lambda"),
        }

    def counterfactual(self, Y0: np.ndarray) -> np.ndarray:
        return np.asarray(Y0, dtype=float) @ self.W


def _align_oracle(oracle_weights: Dict[Any, float],
                  donor_names: List[str]) -> np.ndarray:
    """Align a ``{donor_id: weight}`` map to the Y0 column order (missing -> 0)."""
    d = {str(k): float(v) for k, v in oracle_weights.items()}
    unknown = sorted(set(d) - set(donor_names))
    if unknown:
        raise MlsynthDataError(
            f"oracle_weights references non-donor unit(s) {unknown}; valid "
            f"donors are {donor_names}."
        )
    return np.array([d.get(n, 0.0) for n in donor_names], dtype=float)


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

    Missing values are omitted per covariate, not per period: each covariate is
    averaged over the periods in which *it* is reported, whatever its neighbours
    do. This is augsynth's rule -- ``extract_covariates`` (``R/format.R``) passes
    ``na.action = NULL`` to ``model.frame``, so no rows are dropped, and then
    aggregates each column with its default ``cov_agg``,
    ``function(x) mean(x, na.rm = TRUE)``.

    The alternative -- dropping a period from every covariate when any one of them
    is missing there -- is a tempting reading and a costly one. Mixing an annual
    series with quarterly ones is ordinary in panel data, and listwise deletion
    then discards three quarters in four from the covariates that were fully
    observed. On augsynth's own Kansas study that moved the covariate ASCM's ATT
    from -0.0609 to -0.0663.

    A unit with a covariate missing in *every* period still yields NaN here, which
    the finiteness check below turns into a reported :class:`MlsynthDataError`
    rather than a silently degenerate fit.
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
        # shared window: one mean per covariate, NAs omitted column-wise
        # (augsynth's ``mean(x, na.rm = TRUE)`` per covariate)
        years = year_sets[0]
        sub = df[df[time].isin(years)]
        stack = np.stack(
            [sub.pivot_table(index=unitid, columns=time, values=cov)
                .reindex(index=units, columns=years).to_numpy()
             for cov in covariates],
            axis=2,
        )                                            # (N, T_win, K)
        with warnings.catch_warnings():
            # an all-missing (unit, covariate) slice is a legitimate input error,
            # reported by the finiteness check below -- not a warning to surface
            warnings.filterwarnings("ignore", "Mean of empty slice",
                                    RuntimeWarning)
            X = np.nanmean(stack, axis=1).T           # (K, N)
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


def _v_concentration(V) -> Tuple[Optional[int], Optional[float]]:
    """How concentrated are the predictor weights, and is ``V`` pinned down?

    ``V`` is generically non-identified -- several predictor-weight vectors can
    attain the same upper-level loss -- and backends differ in which member of
    that set they return: a corner search lands on a sparse one, a global search
    on a dense one, a closed-form rule on whatever the data imply. The returned
    weights alone do not make that visible, so two numbers are reported.

    ``n_weighted`` counts the predictors above a fixed tolerance, which is easy
    to read but depends on the tolerance. ``effective`` is the participation
    ratio :math:`1 / \sum_h v_h^2`: no tolerance, equal to ``K`` for uniform
    weights and ``1`` for a corner, moving continuously in between. Prefer it.
    """
    if V is None:
        return None, None
    v = np.asarray(V, dtype=float).ravel()
    if v.size == 0:
        return None, None
    ss = float(np.sum(v ** 2))
    effective = float(1.0 / ss) if ss > 1e-12 else None
    return int(np.sum(v > 1e-6)), effective


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
    if "cohorts" in prep:
        # Staggered adoption (several treated units, possibly at different times):
        # fit one synthetic control per treated unit on the never-treated donors.
        from .staggered import run_vanillasc_staggered
        return run_vanillasc_staggered(config, prep)
    if getattr(config, "staggered_spec", None) is not None:
        # Only the staggered engine reads ``staggered_spec``; the single-treated
        # path below never looks at it. Accepting it here would discard every
        # field on the spec -- including ``w_constr`` -- while returning a plain
        # outcome-only fit that looks like it honoured the constraint.
        raise MlsynthConfigError(
            "staggered_spec was given, but this panel has 1 treated unit and "
            "the staggered engine needs at least 2. On a single-treated panel "
            "the spec would be silently ignored, so it is rejected instead. "
            "For multi-feature matching with one treated unit use `covariates` "
            "(with `covariate_windows` and a predictor-weight `backend`); to "
            "use the staggered spec, mark the other treated units in "
            f"'{config.treat}'."
        )
    if "y" not in prep or "donor_matrix" not in prep:
        raise MlsynthDataError(
            "VanillaSC could not prepare the data (dataprep returned neither a "
            "single-treated nor a multi-cohort structure)."
        )
    y = np.asarray(prep["y"], dtype=float).ravel()
    Y0 = np.asarray(prep["donor_matrix"], dtype=float)
    pre = int(prep["pre_periods"])
    time_labels = np.asarray(prep["time_labels"])
    J = Y0.shape[1]

    # Outcome-fit window (MSCMT's ``times.dep``): restrict the dependent SSR to an
    # inclusive sub-range of the pre-treatment period. Default (None) fits the
    # full pre-period. Predictor matching is unaffected (it uses covariate_windows).
    fit_pos = np.arange(pre)
    if config.fit_window is not None:
        start, end = config.fit_window
        pre_labels_arr = time_labels[:pre]
        lo = pre_labels_arr.min() if start is None else start
        hi = pre_labels_arr.max() if end is None else end
        keep = (pre_labels_arr >= lo) & (pre_labels_arr <= hi)
        fit_pos = np.flatnonzero(keep)
        if fit_pos.size == 0:
            raise MlsynthDataError(
                f"fit_window {config.fit_window} selects no pre-treatment periods "
                f"(pre-period spans {pre_labels_arr.min()}-{pre_labels_arr.max()})."
            )

    # All unit bookkeeping goes through IndexSets, which preserve the original
    # label dtype (so groupby lookups match the DataFrame) and centralise the
    # label <-> position mapping. ``donors`` indexes the columns of Y0;
    # ``units`` is the treated unit followed by the donors (covariate order).
    donors = IndexSet.from_labels(prep["donor_names"])
    treated_label = prep.get("treated_unit_name", "treated")
    units = IndexSet.from_labels([treated_label, *donors.labels])
    treated_name = str(treated_label)
    donor_names = [str(lbl) for lbl in donors.labels]   # string labels for reporting

    # Oracle path: user-specified weights, skip the optimization entirely.
    oracle_w = None
    X1 = X0 = Xs = None
    pred_names: List[str] = []
    if config.oracle_weights is not None:
        infmode = config.inference
        compatible = (infmode is False) or (
            isinstance(infmode, str) and infmode.lower() == "ttest")
        if not compatible:
            raise MlsynthConfigError(
                "oracle_weights is supported only with inference=False or "
                "inference='ttest' (the other inference modes re-estimate the "
                "weights, which contradicts supplying them)."
            )
        oracle_w = _align_oracle(config.oracle_weights, donor_names)
        engine = None
        res = _OracleFit(oracle_w, donor_names)
    elif getattr(config, "w_constr", None) is not None:
        # Named weight-constraint family: solve scpi's constrained program on
        # the pre-treatment outcomes. The config forbids combining this with
        # covariates / a predictor-weight backend, so there is no ambiguity
        # about which estimator ran.
        engine = None
        res = _ConstrainedFit(y[fit_pos], Y0[fit_pos], donor_names,
                              config.w_constr)
    else:
        # Build predictor matrices (treated + donors, donor order matches Y0 cols).
        if covariates:
            pre_labels = list(time_labels[:pre])
            Xall = _covariate_means(
                config.df, list(units.labels), covariates, windows, pre_labels,
                config.unitid, config.time,
            )
            Xs = _scale_unit_variance(Xall)
            X1, X0 = Xs[:, 0], Xs[:, 1:]
            pred_names = list(covariates)

        # A fixed penalized penalty (numeric ``lam``) is passed straight to the
        # penalized solver, which then skips cross-validation; ``None`` keeps
        # the CV path selected by ``penalized_cv``.
        penalized_lam_kwargs = (
            {"lam": float(config.penalized_lambda)}
            if config.penalized_lambda is not None else {}
        )
        engine = BilevelSCM(
            config.backend,
            canonical_v=config.canonical_v,
            seed=config.seed,
            augment=config.augment,
            ridge_lambda=config.ridge_lambda,
            residualize=config.residualize,
            maxiter=config.mscmt_maxiter,
            popsize=config.mscmt_popsize,
            tol=config.mscmt_tol,
            prune_shady=config.mscmt_prune_shady,
            cv=config.penalized_cv,
            **penalized_lam_kwargs,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = engine.fit(
                y[fit_pos], Y0[fit_pos],
                X1=X1, X0=X0, donor_names=donor_names, predictor_names=pred_names,
            )

    counterfactual = res.counterfactual(Y0)
    if config.bias_correct:
        # Abadie-L'Hour: remove the part of the gap attributable to residual
        # predictor imbalance. X1/X0 are the *standardised* predictors the
        # weights were solved in, which is the space the correction requires.
        gap = bias_corrected_gaps(res.W, X1, X0, y, Y0,
                                  ridge=config.bias_correct_ridge)
        counterfactual = y - gap
    else:
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
                e_method=config.scpi_e_method,
                cointegrated=config.scpi_cointegrated, seed=config.seed,
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
                "pi_lower_simultaneous": sc.lower_simul,
                "pi_upper_simultaneous": sc.upper_simul,
                "counterfactual_lower_simultaneous": sc.cf_lower_simul,
                "counterfactual_upper_simultaneous": sc.cf_upper_simul,
                "w_constr": sc.metadata["w_constr"], "df": sc.metadata["df"],
                "sims": sc.metadata["sims"], "e_method": sc.metadata["e_method"],
            },
        )

    # Conformal test-inversion prediction intervals (Chernozhukov, Wuthrich &
    # Zhu 2021; augsynth's default ASCM inference). Reuses the fitted ridge
    # penalty across refits, matching augsynth.
    #
    # The refit rule follows the estimator, and must: the refit is the only
    # place the test touches the fit, and an augmented refit -- unconstrained by
    # construction -- can re-level a large post-period effect away, spreading it
    # over the pre-period residuals that form the reference distribution. Both
    # halves of the test then scale together and the p-value stalls. Against
    # ``scinference`` on the Swedish carbon tax panel that ceiling is visible
    # directly: an augmented refit gives 0.348 for an injected effect of 5 or of
    # 100, where the simplex refit gives the authors' 1 / T = 0.0217 for both.
    if mode == "conformal" and gap[pre:].size:
        from ..bilevel import conformal_intervals
        refit = "ridge" if config.augment == "ridge" else "sc"
        if refit == "sc" and covariates:
            raise MlsynthConfigError(
                "inference='conformal' with covariates needs augment='ridge': "
                "the null refit for a plain simplex SCM matches on outcomes "
                "alone (as scinference's estimation_method='sc' does), so the "
                f"covariates {list(covariates)} would not enter it. Set "
                "augment='ridge' to match on them, or drop them."
            )
        Z0 = X0.T if X0 is not None and refit == "ridge" else None
        z1 = X1 if X1 is not None and refit == "ridge" else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ci = conformal_intervals(
                y, Y0, pre, lambda_=res.lambda_, Z0=Z0, z1=z1,
                alpha=config.alpha,
                ns=int(config.conformal_n_perm or config.scpi_sims),
                seed=config.seed, refit=refit,
                conformal_type=config.conformal_type,
                grid=config.conformal_grid,
                finite_sample=config.conformal_finite_sample,
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
                "conformal_type": config.conformal_type,
                "refit": refit,
                "n_perm": int(config.conformal_n_perm or config.scpi_sims),
                "finite_sample": config.conformal_finite_sample,
                "grid": config.conformal_grid,
                "lambda": res.lambda_,
            },
        )

    # jackknife+ over pre-treatment periods (augsynth ``inf_type="jackknife+"``).
    # Only defined for the ridge-augmented fit: the procedure refits the ASCM
    # once per held-out pre-period, so without an augmentation layer there is
    # nothing being de-biased and the "refit" is just the base SCM again.
    if mode == "jackknife_plus" and gap[pre:].size:
        if config.augment != "ridge":
            raise MlsynthEstimationError(
                "VanillaSC inference='jackknife_plus' is augsynth's inference "
                "for ridge ASCM and needs augment='ridge'; got "
                f"augment={config.augment!r}."
            )
        from ..bilevel.jackknife_plus import jackknife_plus
        Z0 = X0.T if X0 is not None else None
        z1 = X1 if X1 is not None else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            jk = jackknife_plus(
                y[:pre], Y0[:pre], y[pre:], Y0[pre:].T,
                alpha=config.alpha, conservative=config.jackknife_conservative,
                Z0=Z0, z1=z1, residualize=config.residualize,
                lambda_=res.lambda_,
            )
        # jackknife_plus returns one extra entry for the post-period average;
        # the per-period band is everything before it.
        per_period_lower, per_period_upper = jk.lower[:-1], jk.upper[:-1]
        cf_lower = np.full_like(y, np.nan, dtype=float)
        cf_upper = np.full_like(y, np.nan, dtype=float)
        cf_lower[pre:] = jk.counterfactual_lower[:-1]
        cf_upper[pre:] = jk.counterfactual_upper[:-1]
        inference = InferenceResults(
            # The scalar slots carry the bound on the post-period AVERAGE --
            # augsynth's own ``average_att`` row -- not the mean of the
            # per-period bounds, which is a different and less useful quantity.
            ci_lower=float(jk.lower[-1]),
            ci_upper=float(jk.upper[-1]),
            p_value=None,
            confidence_level=1.0 - config.alpha,
            method=("jackknife+ over pre-treatment periods "
                    "(augsynth inf_type=\"jackknife+\")"),
            details={
                "periods": list(time_labels[pre:]),
                "pi_lower": per_period_lower,
                "pi_upper": per_period_upper,
                "counterfactual_lower": cf_lower,
                "counterfactual_upper": cf_upper,
                "average_att_lower": float(jk.lower[-1]),
                "average_att_upper": float(jk.upper[-1]),
                "held_out_errors": jk.held_out_errors,
                "conservative": jk.conservative,
                "lambda": jk.lambda_,
            },
        )

    # Split-conformal prediction intervals (Chernozhukov, Wuthrich & Zhu 2021):
    # the constant-width band ``counterfactual +/- q``, with ``q`` the
    # (1-alpha) order statistic of the absolute pre-period gaps. This is the
    # construction in R Synth's ``synth_inference(method="conformal")``
    # (Hainmueller's j-hai/Synth), distinct from the test-inversion "conformal"
    # band above, which widens over the post-period.
    #
    # Unlike the post-only prediction bands of the other modes, this constant
    # band spans the FULL trajectory (pre and post), as R Synth draws it: the
    # pre-period portion visualizes the conformal calibration -- by construction
    # about (1-alpha) of the pre-period points lie inside ``synthetic +/- q``.
    if mode == "conformal_split" and gap[pre:].size:
        from mlsynth.utils.inferutils import split_conformal_quantile
        q = split_conformal_quantile(gap[:pre], alpha=config.alpha)
        if np.isfinite(q):
            cf_lower, cf_upper = counterfactual - q, counterfactual + q
            pi_lower, pi_upper = gap - q, gap + q
        else:
            warnings.warn(
                "split-conformal band is uninformative (q=inf): need at least "
                f"ceil(1/alpha)-1 = {int(np.ceil(1.0 / config.alpha)) - 1} "
                f"pre-periods for finite-sample coverage at alpha={config.alpha}.",
                UserWarning, stacklevel=2,
            )
            cf_lower = cf_upper = np.full_like(y, np.nan, dtype=float)
            pi_lower = pi_upper = np.full_like(gap, np.nan, dtype=float)
        inference = InferenceResults(
            confidence_level=1.0 - config.alpha,
            method="split-conformal prediction intervals (Chernozhukov-Wuthrich-Zhu 2021)",
            details={
                "periods": list(time_labels),
                "tau": gap,
                "pi_lower": pi_lower,
                "pi_upper": pi_upper,
                "counterfactual_lower": cf_lower,
                "counterfactual_upper": cf_upper,
                "conformal_q": q,
            },
        )

    # Error-in-variables normal/t prediction intervals (Hirshberg 2021).
    if mode == "eiv" and gap[pre:].size:
        from .eiv import eiv_intervals
        ev = eiv_intervals(y, Y0, pre, res.W, alpha=config.alpha)
        cf_lower = np.full_like(y, np.nan, dtype=float)
        cf_upper = np.full_like(y, np.nan, dtype=float)
        cf_lower[pre:] = ev.cf_lower
        cf_upper[pre:] = ev.cf_upper
        inference = InferenceResults(
            ci_lower=float(ev.att_lower),
            ci_upper=float(ev.att_upper),
            confidence_level=1.0 - config.alpha,
            method="error-in-variables prediction intervals (Hirshberg 2021)",
            details={
                "periods": list(time_labels[pre:]),
                "tau": ev.tau, "pi_lower": ev.lower, "pi_upper": ev.upper,
                "counterfactual_lower": cf_lower,
                "counterfactual_upper": cf_upper,
                "att": ev.att, "att_lower": ev.att_lower, "att_upper": ev.att_upper,
                "sigma_tau": ev.metadata["sigma_tau"],
                "p_eff": ev.metadata["p_eff"],
                "theta_l2": ev.metadata["theta_l2"],
                "dof": ev.metadata["dof"],
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

    # Debiased SC t-test for the ATT (Chernozhukov, Wuthrich & Zhu 2025).
    # The cross-fit refits the configured backend on each block-complement of
    # the pre-period; inferutils owns the blocking, rescale, and t_{K-1} CI.
    # Refitting on a subset of the periods is how two modes recalibrate: the
    # debiased t-test drops folds, the cumulative conformal band rolls an origin.
    # One closure serves both so they cannot drift apart.
    if oracle_w is not None:
        # Oracle case: known weights, no per-fold refit (skip the solve).
        def _refit_weight_fn(keep_idx):
            return oracle_w
    else:
        def _refit_weight_fn(keep_idx):
            keep_idx = np.asarray(keep_idx)
            yk, Y0k = y[keep_idx], Y0[keep_idx]
            X1k = X0k = None
            if covariates:
                kept_labels = list(time_labels[keep_idx])
                Xk = _scale_unit_variance(_covariate_means(
                    config.df, list(units.labels), covariates, windows,
                    kept_labels, config.unitid, config.time))
                X1k, X0k = Xk[:, 0], Xk[:, 1:]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rk = engine.fit(yk, Y0k, X1=X1k, X0=X0k,
                                donor_names=donor_names, predictor_names=pred_names)
            return np.asarray(rk.W, dtype=float).ravel()

    if mode == "ttest" and gap[pre:].size:
        from scipy.stats import t as _tdist

        from mlsynth.utils.inferutils import debiased_sc_ttest, select_K

        T1_post = int(len(y) - pre)
        if config.ttest_K == "auto":
            K_used, k_info = select_K(pre, T1_post, gap[:pre], alpha=config.alpha)
        else:
            K_used, k_info = int(config.ttest_K), None
        tt = debiased_sc_ttest(
            y, Y0, T0=pre, T1=T1_post, K=K_used,
            alpha=config.alpha, weight_fn=_refit_weight_fn,
        )
        p_val = float(2.0 * _tdist.sf(abs(tt["tstat"]), tt["dof"]))
        inference = InferenceResults(
            p_value=p_val,
            ci_lower=tt["ci_lower"], ci_upper=tt["ci_upper"],
            confidence_level=1.0 - config.alpha,
            method="debiased SC t-test (Chernozhukov-Wuthrich-Zhu 2025)",
            details={
                "att_debiased": tt["att"],
                "att_naive": float(np.mean(gap[pre:])),
                "se": tt["se"], "tstat": tt["tstat"], "dof": tt["dof"],
                "K": tt["K"], "r": tt["r"], "tau_k": tt["tau_k"].tolist(),
                "alpha": tt["alpha"],
                "K_auto": config.ttest_K == "auto",
                "rho_hat": (k_info["rho_hat"] if k_info else None),
            },
        )

    # Cumulative-effect conformal band: the interval for the SUM of the effect
    # over the horizon, calibrated on out-of-sample windows of the same length so
    # the way period-to-period errors accumulate is measured, not assumed.
    if mode == "conformal_cumulative" and gap[pre:].size:
        from mlsynth.utils.conformal import cumulative_conformal_from_refit

        T1_post = int(len(y) - pre)
        horizon = int(config.conformal_horizon or T1_post)
        band = cumulative_conformal_from_refit(
            y, Y0, pre_periods=int(pre), horizon=horizon,
            weight_fn=_refit_weight_fn, alpha=config.alpha,
        )
        if not np.isfinite(band.half_width):
            warnings.warn(
                "cumulative conformal band is uninformative (half-width=inf): "
                f"{band.n_scores} non-overlapping calibration window(s) of length "
                f"{horizon} fit in the pre-period, but finite-sample coverage at "
                f"alpha={config.alpha} needs at least "
                f"{int(np.ceil(1.0 / config.alpha)) - 1}. Shorten "
                "conformal_horizon or extend the pre-period.",
                UserWarning, stacklevel=2,
            )
        # Dividing the band by the horizon gives the per-period mean over the same
        # window; that is the ATT only when the horizon spans the whole post-period,
        # so a partial window reports the cumulative figure alone.
        spans_post = horizon == T1_post
        inference = InferenceResults(
            ci_lower=(band.lower / horizon) if spans_post else None,
            ci_upper=(band.upper / horizon) if spans_post else None,
            confidence_level=1.0 - config.alpha,
            method="split-conformal cumulative-effect band (rolling origin)",
            details={
                "cumulative_effect": band.point,
                "cumulative_lower": band.lower,
                "cumulative_upper": band.upper,
                "conformal_q": band.half_width,
                "n_calibration_windows": band.n_scores,
                "horizon": horizon,
                "spans_post_period": spans_post,
                "alpha": band.alpha,
            },
        )

    # In-space placebo inference (Abadie): reassign treatment to each donor.
    if mode == "placebo" and J >= 2 and gap[pre:].size:
        ratios = []
        # The outcome-only refits are one leave-one-out family over the donor
        # matrix, so they are solved together. Restricted to the case where a
        # refit is exactly that one simplex QP: no covariates to weight, no
        # augmentation layer over the base, and a backend that reduces to it.
        # ``solve_simplex_loo_exact`` certifies each member and re-solves the
        # rest with ``simplex_qp``, the same solver the engine calls below, so
        # the ranks this p-value is built from are unchanged.
        loo_W = None
        if (engine is not None and not covariates and engine.augment != "ridge"
                and str(config.backend) in ("auto", "outcome-only")):
            from ..bilevel.minnorm import solve_simplex_loo_exact
            from ..bilevel.ridge_augment import simplex_qp
            try:
                loo_W = solve_simplex_loo_exact(Y0[:pre], fallback=simplex_qp)
            except Exception:  # pragma: no cover - fall back to the loop
                loo_W = None
        if loo_W is not None:
            for j in range(J):
                others = [k for k in range(J) if k != j]
                cfj = Y0[:, others] @ loo_W[j, others]
                _, _, ratio_j = _rmspe_ratio(Y0[:, j], cfj, pre)
                if np.isfinite(ratio_j):
                    ratios.append(ratio_j)
        for j in ([] if loo_W is not None else range(J)):
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

    # Never leave a requested-but-uncomputable inference as a silent ``None``: a
    # valid mode whose preconditions were not met (too few donors, no
    # post-periods) returns an explanatory ``InferenceResults`` plus a warning,
    # so the caller is never surprised by a missing band or p-value. ``"none"``
    # (``inference=False``) is an explicit opt-out and stays ``None``.
    if inference is None and mode != "none":
        if not gap[pre:].size:
            reason = "there are no post-treatment periods"
        elif mode == "placebo":
            reason = f"in-space placebo needs >=2 donors (this panel has {J})"
        elif mode == "lto":
            reason = f"leave-two-out needs >=3 donors (this panel has {J})"
        else:  # pragma: no cover - band/ttest modes only skip on empty post window
            reason = "its preconditions were not met on this panel"
        warnings.warn(
            f"VanillaSC inference={mode!r} was requested but not computed: "
            f"{reason}. No band or p-value is available.",
            UserWarning, stacklevel=2,
        )
        inference = InferenceResults(
            confidence_level=1.0 - config.alpha,
            method=f"{mode} (requested but not computed: {reason})",
            details={"requested": mode, "computed": False, "reason": reason},
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
            "n_predictors_weighted": _v_concentration(res.V)[0],
            "v_effective_count": _v_concentration(res.V)[1],
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
                "bias_correct": config.bias_correct,
                "bias_correct_ridge": (
                    float(config.bias_correct_ridge)
                    if config.bias_correct else None
                ),
                "canonical_v": config.canonical_v,
                "v_agreement": res.v_agreement,
                # How concentrated the returned V is (participation ratio).
                # Near 1 means the fit rests on a single predictor, which under
                # V's non-identification is a property of the backend's search
                # as much as of the data -- see _v_concentration.
                "v_effective_count": _v_concentration(res.V)[1],
                "penalized_lambda": (
                    float(res.diagnostics["lambda"])
                    if res.backend == "penalized"
                    and res.diagnostics.get("lambda") is not None
                    else None
                ),
                # Which weight-constraint family ran, and the budget it used
                # (estimated from the data for ridge / L1-L2). None on the
                # default path, so the reported spec is never ambiguous.
                "w_constr": res.diagnostics.get("w_constr"),
                "w_constr_Q": res.diagnostics.get("w_constr_Q"),
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


def _variant_label(config) -> str:
    """Human-readable name of the SCM variant run, for plot titles."""
    if getattr(config, "augment", None) != "ridge":
        return "Synthetic Control"
    if config.covariates:
        return ("Ridge ASCM (residualized covariates)" if config.residualize
                else "Ridge ASCM (covariates)")
    return "Ridge ASCM"


def _plot_vanillasc(config, y, counterfactual, time_labels, pre,
                    treated_name, backend, inference) -> None:
    """Render the observed-vs-synthetic plot through the shared Plotter,
    shading the prediction-interval band when conformal/SCPI inference ran."""
    import matplotlib.pyplot as plt

    from ..plotting import Plotter, mlsynth_style

    T = len(time_labels)
    # Pointwise and (SCPI-only) simultaneous prediction-interval bands.
    pointwise = simultaneous = None
    if inference is not None and getattr(inference, "details", None):
        det = inference.details
        lo, hi = det.get("counterfactual_lower"), det.get("counterfactual_upper")
        if lo is not None and hi is not None:
            pointwise = (_full_band(lo, T, pre), _full_band(hi, T, pre))
        slo = det.get("counterfactual_lower_simultaneous")
        shi = det.get("counterfactual_upper_simultaneous")
        if slo is not None and shi is not None:
            simultaneous = (_full_band(slo, T, pre), _full_band(shi, T, pre))

    # Resolve which band(s) to shade (the simultaneous band is SCPI-only; other
    # inference modes fall back to the pointwise band).
    from ..plotting import select_pi_bands
    interval, interval2, interval_label = select_pi_bands(
        pointwise, simultaneous, getattr(config, "plot_bands", "pointwise"))

    intervention = time_labels[pre] if 0 <= pre < T else None
    with mlsynth_style():
        plotter = Plotter.from_config(getattr(config, "plot", None))
        ax = plotter.observed_vs_counterfactual(
            times=time_labels, observed=y, counterfactuals=[counterfactual],
            labels=[f"Synthetic {treated_name}"], treated_label=treated_name,
            intervention=intervention, interval=interval,
            interval_label=interval_label, interval2=interval2,
            outcome=config.outcome, time=config.time,
            title=f"{_variant_label(config)}: {treated_name}",
        )
        fig = ax.figure
        if config.save:
            fname = (config.save if isinstance(config.save, str)
                     else f"VanillaSC_{treated_name}.png")
            fig.savefig(fname, bbox_inches="tight")
        if config.display_graphs:
            plt.show()
        plt.close(fig)
