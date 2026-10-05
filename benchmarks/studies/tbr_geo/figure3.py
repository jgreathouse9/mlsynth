"""Kerman, Wang and Vaver (2017) Figure 3, on a simulated panel.

    python -m benchmarks.studies.tbr_geo.figure3

The paper's Figure 3 is TBR's own output on the authors' revenue experiment,
which is not public. The design is section 5.1's, so the figure can be redrawn
on a panel this repository generates. Three panels, as the caption specifies:

    (a) the observed treatment series y_t and its counterfactual y*_t
    (b) the pointwise effects phi_t, with their 90% posterior intervals
    (c) the cumulative effect Delta(t), with its 90% posterior intervals

The pretest half of panel (b) is the model's residuals, which the caption calls
a visual diagnostic, so it is drawn over the whole panel and not only the test
window.

The pointwise interval
----------------------

``mlsynth``'s result carries the cumulative posterior and the fitted relation,
and no pointwise posterior. Equation 6 supplies it anyway: at a horizon of one
period the cumulative effect *is* the pointwise effect, and its scale reduces
to ``s sqrt(v_a + 2 x_t v_ab + v_b x_t^2 + 1)``. So the band comes from
``cumulative_posterior`` on a one-period window, which is the library's own
code and not a second implementation of the paper's algebra. A test pins the
identity against ``y_t - (alpha + beta x_t)``.

Exposing the pointwise posterior on the result would make the estimator's own
plotter able to draw this panel; it currently draws the difference as a bare
line. That is an estimator change and belongs on its own branch.

The vertical scale
------------------

``dgp.panel`` normalises the geo sizes to sum to one, so the series are shares.
``VOLUME`` rescales them to read as revenue. It multiplies every panel by a
constant and changes nothing about the fit or the intervals.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

from mlsynth import TBR
from mlsynth.config_models import TBRConfig
from mlsynth.utils.plotting import mlsynth_style
from mlsynth.utils.tbr_helpers.posterior import cumulative_posterior, fit_pretest

from .dgp import panel

#: Series colors. The house blue at a step that clears the lightness band, and
#: the house accent. Validated as a categorical pair: worst adjacent CVD
#: separation 26.8 (protan), 42.1 normal vision, both well above the floor.
OBSERVED = "#2340C8"
COUNTERFACTUAL = "#E04E39"
INK = "#1F2328"
MUTED = "#6B7280"
BAND_ALPHA = 0.18

N_GEOS, N_PRE, N_INTERVENTION, N_COOLDOWN = 20, 40, 8, 4
RHO, C, SEED = 0.5, 0.25, 4
#: Share-to-revenue rescaling; cosmetic, see the module docstring.
VOLUME = 250_000.0
#: Incremental revenue per treated geo-period, as a share of treatment volume.
LIFT = 0.08
LEVEL = 0.90

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def simulate():
    """A panel with an effect, the named split, and the period boundaries."""
    rng = np.random.default_rng(SEED)
    n_test = N_INTERVENTION + N_COOLDOWN
    y = panel(N_GEOS, N_PRE, n_test, RHO, C, rng) * VOLUME
    treated = np.sort(rng.permutation(N_GEOS)[: N_GEOS // 2])

    # The effect runs through the intervention and the cooldown, as the paper's
    # example does. It is a constant per treated geo-period.
    treated_volume = y[N_PRE:, treated].sum(axis=1).mean()
    per_cell = LIFT * treated_volume / len(treated)
    y[N_PRE:, treated] += per_cell

    rows = []
    for j in range(N_GEOS):
        for t in range(N_PRE + n_test):
            rows.append({
                "geo": f"g{j:02d}", "t": t, "revenue": float(y[t, j]),
                "post": int(t >= N_PRE),
                "cooldown": int(t >= N_PRE + N_INTERVENTION),
                "is_treat": int(j in set(treated.tolist())),
                "is_ctrl": int(j not in set(treated.tolist())),
            })
    return pd.DataFrame(rows), treated


def pointwise(df, treated):
    """``(phi, lower, upper)`` per period, from equation 6 at one horizon."""
    wide = df.pivot_table(index="t", columns="geo", values="revenue").sort_index()
    cols = list(wide.columns)
    tre = [cols.index(f"g{j:02d}") for j in treated]
    ctl = [j for j in range(len(cols)) if j not in tre]
    Y = wide.to_numpy()[:, tre].sum(axis=1)
    X = wide.to_numpy()[:, ctl].sum(axis=1)

    fit = fit_pretest(Y[:N_PRE], X[:N_PRE])
    from scipy import stats
    tail = (1.0 - LEVEL) / 2.0
    phi, lo, hi = [], [], []
    for t in range(len(Y)):
        loc, scale = cumulative_posterior(fit, Y[t:t + 1], X[t:t + 1])
        a, b = stats.t.ppf([tail, 1.0 - tail], fit.df,
                           loc=loc[-1], scale=scale[-1])
        phi.append(float(loc[-1])); lo.append(float(a)); hi.append(float(b))
    return np.array(phi), np.array(lo), np.array(hi)


def figure(df, treated, report):
    import matplotlib.pyplot as plt

    ts = report.time_series
    observed = np.asarray(ts.observed_outcome, dtype=float).ravel()
    counterfactual = np.asarray(ts.counterfactual_outcome, dtype=float).ravel()
    periods = np.arange(len(observed))
    phi, phi_lo, phi_hi = pointwise(df, treated)
    cum = report.cumulative
    post = np.asarray(cum.periods, dtype=float)

    with mlsynth_style():
        fig, axes = plt.subplots(
            3, 1, figsize=(9.5, 9.6), sharex=True,
            gridspec_kw={"height_ratios": [1.35, 1.0, 1.0], "hspace": 0.18})

        # The three periods, shaded rather than ruled: three regions read more
        # directly than two lines, and the shading stays under the marks.
        for ax in axes:
            ax.axvspan(N_PRE - 0.5, N_PRE + N_INTERVENTION - 0.5,
                       color=MUTED, alpha=0.07, lw=0)
            ax.axvspan(N_PRE + N_INTERVENTION - 0.5, periods[-1] + 0.5,
                       color=MUTED, alpha=0.14, lw=0)

        # (a) the two series
        axes[0].plot(periods, observed, lw=2.0, color=OBSERVED,
                     label="observed treatment group")
        axes[0].plot(periods, counterfactual, lw=2.0, ls="--",
                     color=COUNTERFACTUAL, label="counterfactual")
        axes[0].set_ylabel("revenue")
        # Direct labels as well as a legend, so identity is never colour alone.
        # The legend sits above the panels as one row: inside the axes it
        # covered the period labels and the panel tag.
        for series, colour, name in ((observed, OBSERVED, "observed"),
                                     (counterfactual, COUNTERFACTUAL, "counterfactual")):
            axes[0].annotate(name, xy=(periods[-1], series[-1]),
                             xytext=(8, 0), textcoords="offset points",
                             color=colour, fontsize=11, va="center",
                             annotation_clip=False)

        # (b) the pointwise effects
        axes[1].axhline(0.0, lw=1.0, color=INK, alpha=0.5)
        axes[1].fill_between(periods, phi_lo, phi_hi, color=OBSERVED,
                             alpha=BAND_ALPHA, lw=0)
        axes[1].plot(periods, phi, lw=2.0, color=OBSERVED)
        axes[1].set_ylabel("pointwise effect")

        # (c) the cumulative effect
        axes[2].axhline(0.0, lw=1.0, color=INK, alpha=0.5)
        axes[2].fill_between(post, np.asarray(cum.lower, dtype=float),
                             np.asarray(cum.upper, dtype=float),
                             color=OBSERVED, alpha=BAND_ALPHA, lw=0)
        axes[2].plot(post, np.asarray(cum.estimate, dtype=float), lw=2.0,
                     color=OBSERVED)
        axes[2].set_ylabel("cumulative effect")
        axes[2].set_xlabel("period")

        for ax, tag in zip(axes, ("(a)", "(b)", "(c)")):
            ax.annotate(tag, xy=(0.0, 1.0), xycoords="axes fraction",
                        xytext=(2, -4), textcoords="offset points",
                        color=MUTED, fontsize=11, va="top")
            ax.margins(x=0.01)

        # The three periods, named in a strip above the top panel so the names
        # never sit on the data. The cooldown is short, so its label is nudged
        # off centre to clear the intervention's.
        for x, name, dx in ((N_PRE / 2, "pretest", 0),
                            (N_PRE + N_INTERVENTION / 2, "intervention", -6),
                            (N_PRE + N_INTERVENTION + N_COOLDOWN / 2,
                             "cooldown", 14)):
            axes[0].annotate(name, xy=(x, 1.0),
                             xycoords=("data", "axes fraction"),
                             xytext=(dx, 7), textcoords="offset points",
                             ha="center", va="bottom", color=MUTED,
                             fontsize=10, annotation_clip=False)

        handles = [
            plt.Line2D([], [], color=OBSERVED, lw=2.0,
                       label="observed treatment group"),
            plt.Line2D([], [], color=COUNTERFACTUAL, lw=2.0, ls="--",
                       label="counterfactual"),
            plt.Line2D([], [], color=OBSERVED, lw=8, alpha=BAND_ALPHA,
                       label=f"{LEVEL:.0%} posterior interval"),
        ]
        fig.legend(handles=handles, loc="upper left",
                   bbox_to_anchor=(0.09, 0.945), ncol=3, frameon=False,
                   fontsize=11)  # its own band, above the period names

        fig.suptitle(
            "TBR causal effect analysis on a simulated panel "
            f"(Kerman et al. 2017, Figure 3)\n{N_GEOS} geos, "
            f"rho={RHO}, c={C}, {LEVEL:.0%} posterior intervals",
            x=0.09, y=1.035, ha="left", fontsize=12, fontweight="bold")
        # Three bands above the panels, none of them overlapping: the title,
        # the legend, then the period names just clear of the axes.
        fig.subplots_adjust(top=0.86, right=0.86)
    return fig, (periods, observed, counterfactual, phi, phi_lo, phi_hi)


def main(argv=None) -> int:
    df, treated = simulate()
    report = TBR(TBRConfig(
        df=df, outcome="revenue", unitid="geo", time="t",
        treatment_col="is_treat", control_col="is_ctrl", post_col="post",
        cooldown_col="cooldown", level=LEVEL)).fit().report

    fig, series = figure(df, treated, report)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "figure3_simulated.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"wrote {out}")

    # The numbers behind the figure, so the panels are inspectable and not only
    # viewable. The repository's other studies record results the same way.
    periods, observed, counterfactual, phi, lo, hi = series
    cum = report.cumulative
    cum_at = dict(zip(np.asarray(cum.periods, dtype=int),
                      zip(cum.estimate, cum.lower, cum.upper)))
    lines = [
        "Kerman et al. (2017) Figure 3 on a simulated panel",
        f"  {N_GEOS} geos, pretest {N_PRE}, intervention {N_INTERVENTION}, "
        f"cooldown {N_COOLDOWN}, rho {RHO}, c {C}, seed {SEED}",
        f"  fit: alpha {report.tbr_fit.alpha:.4f}  beta {report.tbr_fit.beta:.6f}"
        f"  s^2 {report.tbr_fit.sigma_sq:.4f}  df {report.tbr_fit.df}",
        f"  cumulative effect at the final horizon: {cum.estimate[-1]:.2f}"
        f"  [{cum.lower[-1]:.2f}, {cum.upper[-1]:.2f}]  ({LEVEL:.0%})",
        "",
        f"  {'t':>4} {'observed':>12} {'counterfact':>12} {'phi':>11} "
        f"{'phi lower':>11} {'phi upper':>11} {'cumulative':>12}",
    ]
    for i, t in enumerate(periods):
        c = cum_at.get(int(t))
        lines.append(f"  {int(t):>4} {observed[i]:>12.1f} "
                     f"{counterfactual[i]:>12.1f} {phi[i]:>11.1f} "
                     f"{lo[i]:>11.1f} {hi[i]:>11.1f} "
                     + (f"{c[0]:>12.1f}" if c else f"{'':>12}"))
    (RESULTS / "figure3_simulated.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {RESULTS / 'figure3_simulated.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
