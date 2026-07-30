"""Cross-validation: DTWSC against the authors' R package on the Basque panel.

Reference: ``conflictlab/dsc`` (MIT) pinned at
``b1cd241518329ac2bc8cfe21a871a798ac14d74f``; see
``benchmarks/reference/dtwsc_basque/README.md`` for the generator and
``benchmarks/R/install_dtwsc.sh`` for the dependency chain.

The reference dump is not committed (it needs a live R + ``Synth`` toolchain),
so this case reads it when present and reports ``nan`` for the reference-backed
quantities when it is absent -- the pure-Python invariants below still run.

What this case pins, and why each row is where it is:

* The warp, against the R dump -- ``cutoff`` and the first-phase speeds,
  through mlsynth's own preprocessing. This is the tight structural
  comparison.
* The warp again, this time fed R's own filtered panel (``gold_preproc.csv``)
  instead of mlsynth's. Every stage of the warping engine then agrees with R
  to the last bit: first-phase speeds 16/16 bit-identical, second-phase speeds
  to one ULP, warped donor series to 1e-14. That row is the real statement
  that the TFDTW port is exact, and it isolates the one remaining difference
  to preprocessing alone.
* The synthetic-control half, against the reference's own numbers, using
  ``sc_backend="mscmt"`` with the paper's 14 predictors and
  ``rescale_units=True``. Both arms land within 0.004 on pre-RMSE and 0.03 on
  the ATT.

  The rescaling is not optional for that comparison, and an earlier version of
  this file got it wrong in a way worth recording. ``dsc(rescale = TRUE)`` is
  the reference's default: it maps every unit onto a common pre-treatment
  range before fitting, so R reports in rescaled units, not in gdpcap. mlsynth
  without ``rescale_units`` reports an unwarped ATT of -0.6026 in gdpcap and R
  reports -0.6027 rescaled, which reads as a four-decimal match and is not one
  -- they are different quantities. In common units R's is -0.6405. The rows
  below compare like with like.

The one documented gap is the Savitzky-Golay edge treatment: the reference
pads each series with an ``auto.arima`` forecast before filtering where
mlsynth edge-pads. The filter agrees to 1e-16 in the interior and disagrees
only over the two points at each end -- but those points are large relative to
the local scale of a second derivative, so through mlsynth's own preprocessing
5 of 16 donors pick up a different alignment step.

Closing that gap would take a bit-exact ``auto.arima``, not an approximate
one. ``second_dtw`` discards outliers with a type-7 fence that, on the
near-constant speed columns this panel produces, evaluates to exactly the
value it is testing; which side of that comparison a speed falls on decides
whether it is averaged in at all. An approximate ARIMA would land on an
arbitrary side of the fence, so it would not reproduce R any more reliably
than edge-padding does. Hence the gap is recorded rather than papered over.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_REF = Path(__file__).resolve().parents[1] / "reference" / "dtwsc_basque"
_DATA = Path(__file__).resolve().parents[2] / "basedata" / "basque_data.csv"

TREATED = "Basque Country (Pais Vasco)"
#: The reference package's own README example uses 1970, not Abadie and
#: Gardeazabal's 1975 terrorism onset -- and its ``t.treat`` INCLUDES the
#: treatment year in the alignment window (``1:t.treat``), where mlsynth's
#: ``pre_periods`` stops the period before. Marking treatment from 1971 gives
#: both sides the same 16-period window (1955--1970) to learn the warp on,
#: which is what this case pins. The end-to-end ATT comparison is in
#: ``docs/replications/dtwsc.rst``, not here.
TREAT_YEAR = 1971


#: The paper's 14 special predictors, as inclusive averaging windows.
_COVARIATE_WINDOWS = {
    "invest_ratio": (1964, 1969), "popdens": (1969, 1969),
    "sec.agriculture": (1961, 1969), "sec.energy": (1961, 1969),
    "sec.industry": (1961, 1969), "sec.construction": (1961, 1969),
    "sec.services.venta": (1961, 1969), "sec.services.nonventa": (1961, 1969),
    "school.illit": (1964, 1969), "school.prim": (1964, 1969),
    "school.med": (1964, 1969), "school.high": (1964, 1969),
    "school.post.high": (1964, 1969),
}
_COVARIATES = list(_COVARIATE_WINDOWS)


def _panel(with_covariates: bool = False) -> pd.DataFrame:
    df = pd.read_csv(_DATA)
    df = df[df.regionname != "Spain (Espana)"].dropna(subset=["gdpcap"]).copy()
    df["treated"] = ((df.regionname == TREATED) & (df.year >= TREAT_YEAR)).astype(int)
    if with_covariates:
        df["invest_ratio"] = df["invest"] / df["gdpcap"]
        present = [c for c in _COVARIATES if c in df.columns]
        df[present] = df.groupby("regionname")[present].ffill().bfill()
    return df


def _reference_fit():
    """R's own fitted paths, or ``None``.

    ``value`` is the treated unit's outcome and ``synth_dsc`` / ``synth_sc``
    the two counterfactuals, all in the units R fits in -- that is, AFTER
    ``dsc(rescale = TRUE)`` has mapped each unit onto the common pre-treatment
    range.
    """
    path = _REF / "gold_fit.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _pre_rmse_and_att(observed, synthetic, n_pre):
    observed = np.asarray(observed, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    return (float(np.sqrt(np.nanmean(
                (observed[:n_pre] - synthetic[:n_pre]) ** 2))),
            float(np.nanmean(observed[n_pre:] - synthetic[n_pre:])))


def _post_path_gap(mlsynth_path, reference_path, n_pre):
    """Worst POINTWISE post-treatment gap between the two counterfactuals.

    The ATT rows above average the same difference, and averaging hides it:
    the two paths cross rather than running parallel, so sign-cancelling
    pointwise error of ~0.19 collapses to an ATT gap of ~0.03. A regression
    that tilted the counterfactual while leaving its mean intact would clear
    the ATT row untouched, so the quantity that actually moves is pinned here.

    Three donors' warped series end a period short of 1997, which makes the
    reference's own counterfactual ``NA`` there; the comparison is taken over
    the periods both sides define.
    """
    mlsynth_path = np.asarray(mlsynth_path, dtype=float)
    reference_path = np.asarray(reference_path, dtype=float)
    n = min(len(mlsynth_path), len(reference_path))
    gap = np.abs(mlsynth_path[:n] - reference_path[:n])[n_pre:]
    return float(np.nanmax(gap)) if gap.size and not np.isnan(gap).all() else float("nan")


def _load_reference():
    """``{unit: {quantity: array}}`` from the R dump, or ``None``."""
    path = _REF / "gold_tfdtw.csv"
    if not path.exists():
        return None
    out = {}
    for line in path.read_text().splitlines():
        unit, quantity, values = line.split(";")
        out.setdefault(unit, {})[quantity] = np.array(
            [np.nan if v == "NA" else float(v) for v in values.split(",")]
        )
    return out


def _load_reference_filtered():
    """``{unit: array}`` -- R's own Savitzky-Golay panel, or ``None``.

    Injecting this in place of mlsynth's filter removes the single known
    difference between the two implementations, so what remains measures the
    warping engine and nothing else.
    """
    path = _REF / "gold_preproc.csv"
    if not path.exists():
        return None
    ref = pd.read_csv(path)
    out = {}
    for unit, sub in ref.groupby("unit", sort=False):
        sub = sub.sort_values("time")
        # ``value_raw`` is the unfiltered series (affinely rescaled by the
        # reference); ``value`` is that series after R's filter. Keeping both
        # lets the substitution be looked up by the series going in.
        out[unit] = (sub["value_raw"].to_numpy(float),
                     sub["value"].to_numpy(float))
    return out


def _warp_on_reference_filter(cfg, filtered):
    """Re-run the warp with R's filtered panel substituted in.

    Keyed on the t-normalized series so the lookup is invariant to the affine
    rescaling the reference applies before dumping.
    """
    import mlsynth.utils.dtwsc_helpers.pipeline as pipeline
    from mlsynth.utils.dtwsc_helpers.warping import (
        savgol_second_derivative as _real, t_normalize)
    from mlsynth import DTWSC

    def key(v):
        return tuple(np.round(t_normalize(np.asarray(v, dtype=float)), 6))

    table = {key(raw): value for raw, value in filtered.values()}

    def substitute(Y, filter_width=5, n_poly=3):
        Y = np.asarray(Y, dtype=float)
        flat = Y.ndim == 1
        M = Y[:, None] if flat else Y
        out = np.empty_like(M)
        for c in range(M.shape[1]):
            k = key(M[:, c])
            out[:, c] = (table[k] if k in table
                         else _real(M[:, c], filter_width, n_poly))
        return out[:, 0] if flat else out

    original = pipeline.savgol_second_derivative
    pipeline.savgol_second_derivative = substitute
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return DTWSC(cfg).fit()
    finally:
        pipeline.savgol_second_derivative = original


def run() -> dict:
    from mlsynth import DTWSC

    cfg = {"df": _panel(), "outcome": "gdpcap", "treat": "treated",
           "unitid": "regionname", "time": "year", "display_graphs": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warped = DTWSC(cfg).fit()
        plain = DTWSC({**cfg, "warp": False}).fit()

    out = {
        # The paper's claim: aligning speeds tightens the pre-treatment fit.
        "dtwsc_pre_rmse_improves": float(warped.pre_rmse < plain.pre_rmse),
        "dtwsc_pre_rmse_ratio": float(warped.pre_rmse / plain.pre_rmse),
        "dtwsc_n_donors": float(len(warped.donor_weights)),
        "dtwsc_weights_sum": float(sum(warped.donor_weights.values())),
        "dtwsc_att_sign": float(np.sign(warped.att)),
    }

    ref = _load_reference()
    if ref is None:
        for key in ("dtwsc_cutoff_matches_r", "dtwsc_donors_speeds_exact_vs_r",
                    "dtwsc_speed_max_abs_diff_vs_r",
                    "dtwsc_speeds_exact_on_r_filter",
                    "dtwsc_post_speed_max_diff_on_r_filter",
                    "dtwsc_warped_max_diff_on_r_filter"):
            out[key] = float("nan")
        return out

    n_cut, n_exact, worst = 0, 0, 0.0
    for unit, cutoff in warped.cutoffs.items():
        if unit not in ref:
            continue
        n_cut += int(cutoff == int(ref[unit]["cutoff"][0]))
        gap = float(np.nanmax(np.abs(
            warped.pre_period_speeds[unit] - ref[unit]["weight.a"])))
        n_exact += int(gap < 1e-12)
        worst = max(worst, gap)
    out["dtwsc_cutoff_matches_r"] = float(n_cut)
    out["dtwsc_donors_speeds_exact_vs_r"] = float(n_exact)
    out["dtwsc_speed_max_abs_diff_vs_r"] = worst

    filtered = _load_reference_filtered()
    if filtered is None:
        out["dtwsc_speeds_exact_on_r_filter"] = float("nan")
        out["dtwsc_post_speed_max_diff_on_r_filter"] = float("nan")
        out["dtwsc_warped_max_diff_on_r_filter"] = float("nan")
        return out

    same = _warp_on_reference_filter(cfg, filtered)
    n_bit, worst_post, worst_warp = 0, 0.0, 0.0
    donors = list(same.inputs.donor_names)
    matrix = np.asarray(same.warped_donor_matrix, dtype=float)
    for j, unit in enumerate(donors):
        if unit not in ref:
            continue
        # Bit-identical, not merely close: see the module docstring on why the
        # last bit is load-bearing here.
        n_bit += int(np.array_equal(
            np.asarray(same.pre_period_speeds[unit], dtype=float),
            ref[unit]["weight.a"]))
        worst_post = max(worst_post, float(np.nanmax(np.abs(
            np.asarray(same.post_period_speeds[unit], dtype=float)
            - ref[unit]["avg.weight"]))))
        # The reference dumps the warp of an affinely rescaled series, so the
        # comparison is made after removing that rescaling.
        mine, theirs = matrix[:, j], ref[unit]["warped"]
        n = min(mine.size, theirs.size)
        mine, theirs = mine[:n], theirs[:n]
        ok = np.isfinite(mine) & np.isfinite(theirs)
        slope, intercept = np.polyfit(mine[ok], theirs[ok], 1)
        worst_warp = max(worst_warp, float(np.nanmax(
            np.abs(slope * mine + intercept - theirs))))
    out["dtwsc_speeds_exact_on_r_filter"] = float(n_bit)
    out["dtwsc_post_speed_max_diff_on_r_filter"] = worst_post
    out["dtwsc_warped_max_diff_on_r_filter"] = worst_warp

    # The synthetic-control half, compared IN R'S OWN UNITS. This row exists
    # because comparing across scales once produced a spurious four-decimal
    # agreement -- see the module docstring.
    fit = _reference_fit()
    if fit is None:
        for key in ("dtwsc_sc_pre_rmse_gap_vs_r", "dtwsc_sc_att_gap_vs_r",
                    "dtwsc_dsc_pre_rmse_gap_vs_r", "dtwsc_dsc_att_gap_vs_r"):
            out[key] = float("nan")
        return out

    panel = _panel(with_covariates=True)
    present = [c for c in _COVARIATES if c in panel.columns]
    scaled = {**cfg, "df": panel, "rescale_units": True,
              "sc_backend": "mscmt", "covariates": present,
              "covariate_windows": {k: v for k, v in _COVARIATE_WINDOWS.items()
                                    if k in present},
              "fit_window": (1960, 1969)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dsc_arm = DTWSC(dict(scaled)).fit()
        sc_arm = DTWSC({**scaled, "warp": False}).fit()

    n_pre_fit = int((fit["time"] <= 1970).sum())
    for tag, arm, column in (("sc", sc_arm, "synth_sc"),
                             ("dsc", dsc_arm, "synth_dsc")):
        r_rmse, r_att = _pre_rmse_and_att(fit["value"], fit[column], n_pre_fit)
        out[f"dtwsc_{tag}_pre_rmse_gap_vs_r"] = abs(arm.pre_rmse - r_rmse)
        out[f"dtwsc_{tag}_att_gap_vs_r"] = abs(arm.att - r_att)
        out[f"dtwsc_{tag}_post_path_gap_vs_r"] = _post_path_gap(
            arm.counterfactual, fit[column], n_pre_fit)
    return out


# Deterministic: the warp is a dynamic program with no sampling anywhere, and
# the simplex solve is convex. Re-running gives identical numbers.
#
# Tolerances, and what each row is really pinning:
#  * ``pre_rmse_improves`` / ``pre_rmse_ratio`` -- the paper's own claim, on the
#    authors' own panel, through mlsynth's public API. The ratio is pinned at
#    0.70 +/- 0.08: it depends on the SC half, which is outcome-only here
#    against the reference's 14-predictor Abadie specification, so it is a
#    same-direction check rather than a cell match.
#  * ``cutoff_matches_r`` -- 16/16 exact. The cutoff is an integer read off the
#    first-phase warp, so this is the tight structural check and gets zero
#    tolerance.
#  * ``donors_speeds_exact_vs_r`` -- 11 of 16 donors' first-phase speeds agree
#    with R to 1e-12 through mlsynth's own preprocessing. The other five differ
#    because the reference pads each series with an ``auto.arima`` forecast
#    before the Savitzky-Golay filter where mlsynth edge-pads, which moves the
#    two points at each end and can flip an alignment step. Feeding mlsynth R's
#    own filtered panel instead makes it 16/16 to one ULP -- see
#    ``docs/replications/dtwsc.rst``. Pinned exactly so a regression in the
#    warp shows up rather than hiding inside a loose bound.
#  * ``speed_max_abs_diff_vs_r`` -- the size of that edge-buffer disagreement,
#    2/3 of one period's speed on the worst donor. Pinned loosely (0.1) because
#    it is a known, documented difference, not a target to drive to zero.
#  * ``speeds_exact_on_r_filter`` -- 16/16 donors bit-identical once R's own
#    filtered panel is used, so the first-phase port has no residual at all.
#    Zero tolerance, and equality is exact rather than tolerance-based.
#  * ``post_speed_max_diff_on_r_filter`` / ``warped_max_diff_on_r_filter`` --
#    the second phase and the resulting warped series on the same footing:
#    one ULP and 1e-14. Pinned tightly, at roughly an order of magnitude of
#    headroom over the observed values, because anything larger means a real
#    regression in the warping engine rather than accumulated rounding.
EXPECTED = {
    "dtwsc_pre_rmse_improves": (1.0, 0.0),
    "dtwsc_pre_rmse_ratio": (0.70, 0.08),
    "dtwsc_n_donors": (16.0, 0.0),
    "dtwsc_weights_sum": (1.0, 1e-6),
    "dtwsc_att_sign": (-1.0, 0.0),
    "dtwsc_cutoff_matches_r": (16.0, 0.0),
    "dtwsc_donors_speeds_exact_vs_r": (11.0, 0.0),
    "dtwsc_speed_max_abs_diff_vs_r": (0.667, 0.1),
    "dtwsc_speeds_exact_on_r_filter": (16.0, 0.0),
    "dtwsc_post_speed_max_diff_on_r_filter": (0.0, 1e-15),
    "dtwsc_warped_max_diff_on_r_filter": (0.0, 1e-13),
    "dtwsc_sc_pre_rmse_gap_vs_r": (0.0041, 0.004),
    "dtwsc_sc_att_gap_vs_r": (0.0205, 0.015),
    "dtwsc_dsc_pre_rmse_gap_vs_r": (0.0042, 0.004),
    "dtwsc_dsc_att_gap_vs_r": (0.0269, 0.015),
    # Pointwise, not averaged -- see ``_post_path_gap``. An order of magnitude
    # larger than the ATT rows, and the honest measure of how far the two
    # counterfactuals actually run apart.
    "dtwsc_sc_post_path_gap_vs_r": (0.1610, 0.02),
    "dtwsc_dsc_post_path_gap_vs_r": (0.1897, 0.02),
}
