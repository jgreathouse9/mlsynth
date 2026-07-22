"""scpi-``scplotMulti``-style plots for staggered VanillaSC.

Mirrors the conventions of the estimator's cousin, ``nppackages/scpi``
(``scplotMulti``), for the multi-treated (staggered) case -- rendered through
mlsynth's matplotlib :class:`~mlsynth.utils.plotting.Plotter` rather than
plotnine. Four views, covering scpi's ``ptype`` x ``effect`` grid:

* ``series`` facets  -- treated (observed) vs synthetic control, one panel per
  treated unit with its adoption line (``ptype="series"``, ``effect="unit-time"``);
* ``treatment`` facets -- the per-period gap with a zero line, one panel per unit
  (``ptype="treatment"``, ``effect="unit-time"``);
* a per-unit ATT dot plot -- one point per treated unit, sized by the number of
  post-periods averaged, ordered like scpi's ``coord_flip`` (``effect="unit"``);
* the event-time aggregate effect -- mean effect by time-to-treatment with a zero
  line (``effect="time"``).

When SCPI inference is on (``inference="scpi"``) the per-unit counterfactual and
gap bands, the per-unit ATT credible intervals, and the event-study band are all
shaded; otherwise the point views render on their own.

Figures are saved (``config.save``) and/or shown (``config.display_graphs``) with
the single-treated path's semantics, and returned to the caller.
"""
from __future__ import annotations

import math
from typing import Any, List, Tuple

import numpy as np

from ..plotting import Plotter, mlsynth_style


def _save_prefix(save: Any) -> str:
    """Treat ``config.save`` as a filename prefix (the staggered case emits
    several figures); strip a trailing ``.png`` if the user supplied one."""
    if isinstance(save, str) and save:
        return save[:-4] if save.lower().endswith(".png") else save
    return "VanillaSC_staggered"


def _post_band_full(lower, upper, T: int):
    """Align a post-treatment-only SCPI band ``(lower, upper)`` to the full length
    ``T`` time axis, padding the pre-period with ``NaN`` (unshaded), tail-aligned
    to the post-treatment periods. Returns None if either side is absent."""
    if lower is None or upper is None:
        return None
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if lo.size == T:
        return (lo, hi)
    full_lo = np.full(T, np.nan)
    full_hi = np.full(T, np.nan)
    k = min(lo.size, T)
    full_lo[T - k:] = lo[-k:]
    full_hi[T - k:] = hi[-k:]
    return (full_lo, full_hi)


def _facet_grid(n: int):
    """A (nrows x ncols) subplot grid sized like scpi's ``facet_wrap`` (ncol=3),
    returning the figure and the flattened axes."""
    import matplotlib.pyplot as plt

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.0, nrows * 3.4),
                             squeeze=False)
    return fig, axes.ravel()


def plot_staggered(config, unit_fits, event_study, event_study_intervals=None) -> List[Any]:
    """Render the scpi-``scplotMulti``-style figures for a staggered fit.

    Returns the list of matplotlib figures (also saved / shown per ``config``).
    """
    import matplotlib.pyplot as plt

    tagged: List[Tuple[Any, str]] = []
    with mlsynth_style():
        plotter = Plotter.from_config(getattr(config, "plot", None))
        n = len(unit_fits)

        # 1) Per-unit series facets (ptype="series", effect="unit-time").
        if unit_fits:
            fig, flat = _facet_grid(n)
            for ax, (name, uf) in zip(flat, unit_fits.items()):
                plotter.observed_vs_counterfactual(
                    times=uf.time_labels, observed=uf.observed,
                    counterfactuals=[uf.counterfactual],
                    labels=["Synthetic Control"], treated_label=name,
                    intervention=uf.adoption_time,
                    interval=_post_band_full(uf.cf_lower, uf.cf_upper, len(uf.observed)),
                    outcome=config.outcome, time=config.time,
                    title=str(name), ax=ax)
            for ax in flat[n:]:
                ax.set_visible(False)
            fig.suptitle("VanillaSC (staggered): treated vs. synthetic control")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            tagged.append((fig, "series"))

        # 2) Per-unit treatment (gap) facets (ptype="treatment", effect="unit-time").
        if unit_fits:
            fig, flat = _facet_grid(n)
            for ax, (name, uf) in zip(flat, unit_fits.items()):
                plotter.gap(
                    times=uf.time_labels, gap=uf.gap,
                    intervention=uf.adoption_time,
                    interval=_post_band_full(uf.tau_lower, uf.tau_upper, len(uf.gap)),
                    outcome=f"Effect on {config.outcome}", time=config.time,
                    title=str(name), ax=ax)
            for ax in flat[n:]:
                ax.set_visible(False)
            fig.suptitle("VanillaSC (staggered): per-period treatment effect")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            tagged.append((fig, "treatment"))

        # 3) Per-unit ATT dot plot (effect="unit"): one point per unit, sized by
        #    the number of post-periods averaged, with the credible interval when
        #    SCPI inference ran. scpi flips the axes (coord_flip) -> horizontal.
        if unit_fits:
            names = list(unit_fits)
            atts = np.asarray([unit_fits[k].att for k in names], float)
            posts = np.asarray([max(unit_fits[k].post_periods, 1) for k in names], float)
            ypos = np.arange(len(names))
            fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(names) + 1.6))
            has_ci = any(np.isfinite(unit_fits[k].att_ci_lower) for k in names)
            if has_ci:
                lo = np.clip(atts - np.asarray([unit_fits[k].att_ci_lower for k in names], float), 0, None)
                hi = np.clip(np.asarray([unit_fits[k].att_ci_upper for k in names], float) - atts, 0, None)
                ax.errorbar(atts, ypos, xerr=np.vstack([lo, hi]), fmt="o",
                            color=plotter.counterfactual_colors[0], capsize=3,
                            markersize=5, label="ATT (95% PI)")
            else:
                ax.scatter(atts, ypos, s=20 + 6 * posts,
                           color=plotter.counterfactual_colors[0], label="ATT")
            ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
            ax.set_yticks(ypos)
            ax.set_yticklabels(names)
            ax.set_xlabel(f"ATT on {config.outcome}")
            ax.set_ylabel(config.unitid)
            ax.set_title("VanillaSC (staggered): treatment effect by unit")
            ax.legend()
            fig.tight_layout()
            tagged.append((fig, "att_by_unit"))

        # 4) Event-time aggregate effect (effect="time").
        if event_study:
            ells = sorted(event_study)
            effects = np.asarray([event_study[e] for e in ells], float)
            lower = upper = None
            if event_study_intervals:
                lo, hi = [], []
                for e in ells:
                    ci = (event_study_intervals.get(e) or {}).get("effect_ci")
                    lo.append(ci[0] if ci else np.nan)
                    hi.append(ci[1] if ci else np.nan)
                if not np.all(np.isnan(lo)):
                    lower, upper = np.asarray(lo, float), np.asarray(hi, float)
            ax = plotter.event_study(
                event_times=ells, effects=effects,
                ci_lower=lower, ci_upper=upper,
                outcome=f"Effect on {config.outcome}", time="Event time",
                title="VanillaSC (staggered): event-study ATT")
            tagged.append((ax.figure, "event_study"))

        figs = [fig for fig, _ in tagged]

        # Save / display with the single-treated path's semantics.
        did_output = False
        if getattr(config, "save", None):
            prefix = _save_prefix(config.save)
            for fig, tag in tagged:
                fig.savefig(f"{prefix}_{tag}.png", bbox_inches="tight")
            did_output = True
        if getattr(config, "display_graphs", False):
            plt.show()
            did_output = True
        if did_output:
            for fig in figs:
                plt.close(fig)
    return figs
