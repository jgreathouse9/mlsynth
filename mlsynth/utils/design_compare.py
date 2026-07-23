"""Cross-estimator design comparison on a common fit-vs-power plane.

The design estimators (SYNDES, LEXSCM, MAREX) differ in their optimisers but
share a grammar: each emits a design that reduces to a unit-level contrast
``c = w_treated - w_control`` (both sides summing to one) over the same panel.
From ``c`` two comparable numbers follow -- the pre-period fit RMSE
``sqrt(mean((Y_pre @ c)**2))`` and, by injecting a known effect at a fixed
horizon, a minimum detectable effect.

Scoring every design from any method through one shared simulation harness --
same horizon, same effect grid, same moving-block permutation null, same
baseline -- puts them on identical axes, so the per-method Pareto frontiers can
be overlaid and compared directly. This is the point of the module: the frontier
gap then reflects the *designs*, not two different power methodologies.

The MDE simulation rests on one fact true for any normalised design: adding an
effect ``tau`` to the treated units' outcomes shifts the contrast mean
``Y_t @ c`` by exactly ``tau`` (because the treated weights sum to one). So the
alternative distribution of the length-``h`` block mean is the null distribution
shifted by ``tau``, and the power at ``tau`` is the share of the (empirical,
moving-block) null whose shifted magnitude clears the two-sided critical value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DesignSpec:
    """A design from any estimator, in the common cross-method currency.

    Parameters
    ----------
    method : str
        Originating method, e.g. ``"SYNDES"`` or ``"LEXSCM"``.
    label : str
        Human-readable design label (shown on the plot).
    contrast : dict
        ``{unit_label: weight}`` for ``w_treated - w_control``; missing units are
        zero. Treated weights are positive and sum to one, control weights
        negative and sum to minus one.
    treated : list
        Treated-unit labels (used for the baseline and the label).
    oos_rmse : float, optional
        Out-of-sample (holdout) contrast RMSE of the design, when it was
        selected by SYNDES's holdout selector. ``None`` for in-sample SYNDES
        designs and for methods without a holdout notion.
    """

    method: str
    label: str
    contrast: Dict[Any, float]
    treated: List[Any]
    oos_rmse: Optional[float] = None


def simulated_mde(
    pre_contrast: np.ndarray,
    *,
    horizon: int,
    effects_abs: Sequence[float],
    alpha: float = 0.10,
    power_target: float = 0.80,
) -> float:
    """Minimum detectable effect from a contrast's moving-block permutation null.

    Parameters
    ----------
    pre_contrast : np.ndarray
        The pre-period contrast series ``g_t = Y_t @ c``, shape ``(T0,)``.
    horizon : int
        Post-period horizon ``h`` the test averages over.
    effects_abs : sequence of float
        Ascending grid of candidate effect sizes, in outcome units.
    alpha, power_target : float
        Two-sided significance level and target power.

    Returns
    -------
    float
        Smallest grid effect reaching ``power_target``, or ``inf`` if none does.
    """
    g = np.asarray(pre_contrast, dtype=float)
    g = g - g.mean()                      # center under the sharp null
    T = g.size
    if T < horizon + 1:
        return float("inf")
    # Empirical null of the length-h block mean: overlapping pre-period blocks.
    blocks = np.array([g[i:i + horizon].mean()
                       for i in range(T - horizon + 1)], dtype=float)
    crit = float(np.quantile(np.abs(blocks), 1.0 - alpha))
    for tau in np.sort(np.asarray(effects_abs, dtype=float)):
        # Under an additive effect tau the statistic shifts by tau; power is the
        # share of the shifted null whose magnitude clears the critical value.
        power = float(np.mean(np.abs(blocks + tau) > crit))
        if power >= power_target:
            return float(tau)
    return float("inf")


def _pareto_mask(fit: np.ndarray, mde: np.ndarray) -> np.ndarray:
    """Boolean non-dominated mask on (fit downwards, mde downwards)."""
    fit = np.asarray(fit, dtype=float)
    mde = np.asarray(mde, dtype=float)
    n = fit.size
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (fit[j] <= fit[i] and mde[j] <= mde[i]
                    and (fit[j] < fit[i] or mde[j] < mde[i])):
                keep[i] = False
                break
    return keep


def compare_pareto(
    specs: Sequence[DesignSpec],
    Ywide: pd.DataFrame,
    n_pre: int,
    *,
    horizon: int = 5,
    effects_pct: Optional[Sequence[float]] = None,
    alpha: float = 0.10,
    power_target: float = 0.80,
) -> pd.DataFrame:
    """Score designs from any method(s) on the shared fit-vs-power plane.

    Parameters
    ----------
    specs : sequence of DesignSpec
        Designs to compare (typically from :func:`from_syndes` and
        :func:`from_lexscm`, run on the same panel).
    Ywide : pd.DataFrame
        Outcome panel, rows = time, columns = unit labels (the canonical order
        every contrast is aligned to).
    n_pre : int
        Number of leading rows that are pre-treatment.
    horizon : int, default 5
        Post-period horizon for the MDE simulation.
    effects_pct : sequence of float, optional
        Ascending grid of effect sizes as a percent of the treated baseline.
        Defaults to ``0.25 .. 30`` in 0.25 steps -- the common grid every method
        is simulated over.
    alpha, power_target : float
        Two-sided significance level and target power.

    Returns
    -------
    pd.DataFrame
        One row per design: ``method``, ``label``, ``treated``, ``fit_rmse``,
        ``mde_pct`` (at ``horizon``), and a per-method ``pareto`` flag.
    """
    if effects_pct is None:
        effects_pct = np.arange(0.25, 30.25, 0.25)
    cols = list(Ywide.columns)
    pre = np.asarray(Ywide.values[:n_pre], dtype=float)
    effects_pct = np.asarray(effects_pct, dtype=float)

    rows: List[Dict[str, Any]] = []
    for s in specs:
        c = np.array([s.contrast.get(u, 0.0) for u in cols], dtype=float)
        g = pre @ c
        fit_rmse = float(np.sqrt(np.mean(g ** 2)))
        treated_pos = [cols.index(u) for u in s.treated if u in cols]
        base = float(np.mean(pre[:, treated_pos])) if treated_pos else float("nan")
        if not np.isfinite(base) or abs(base) < 1e-12:
            mde_pct = float("inf")
        else:
            eff_abs = (effects_pct / 100.0) * abs(base)
            mde_abs = simulated_mde(g, horizon=horizon, effects_abs=eff_abs,
                                    alpha=alpha, power_target=power_target)
            mde_pct = (100.0 * mde_abs / abs(base)
                       if np.isfinite(mde_abs) else float("inf"))
        rows.append({
            "method": s.method, "label": s.label, "treated": list(s.treated),
            "fit_rmse": fit_rmse, "mde_pct": mde_pct,
            "oos_rmse": (float(s.oos_rmse) if s.oos_rmse is not None
                         else float("nan")),
        })

    df = pd.DataFrame(rows)
    df["pareto"] = False
    for _, idx in df.groupby("method").groups.items():
        sub = df.loc[idx]
        mask = _pareto_mask(sub["fit_rmse"].to_numpy(),
                            np.where(np.isfinite(sub["mde_pct"]),
                                     sub["mde_pct"], np.inf))
        df.loc[idx, "pareto"] = mask
    return df


def from_syndes(res) -> List[DesignSpec]:
    """Build DesignSpecs from a fitted SYNDES result (its solution pool).

    Falls back to the single returned design when no pool exists (``top_K == 1``).
    """
    labels = list(np.asarray(res.inputs.unit_index.labels))

    def _spec(design, markets, tag, oos=None):
        c = np.asarray(design.contrast_weights, dtype=float).reshape(-1)
        contrast = {labels[j]: float(c[j]) for j in range(len(labels))}
        return DesignSpec("SYNDES", tag, contrast, list(markets), oos_rmse=oos)

    pool = getattr(res, "pool", None)
    if pool:
        return [
            _spec(e["design"], e["markets"],
                  f"S{i + 1}:{'+'.join(map(str, sorted(e['markets'])))}",
                  e.get("oos_rmse"))
            for i, e in enumerate(pool)
        ]
    d = res.design
    return [_spec(d, list(np.asarray(d.selected_unit_labels)), "S1")]


def from_lexscm(res) -> List[DesignSpec]:
    """Build DesignSpecs from a fitted LEXSCM result (its candidate search).

    The cross-method contrast is ``w_treated - w_control``; the treated side
    sums to ``+1`` and the control side to ``-1`` -- the same convention as the
    other adapters.

    Weights are taken from the candidate's full-precision dicts
    (``treated_weight_dict_full`` / ``control_weight_dict_full``), so both sides
    sum to one exactly and the net contrast is exactly zero -- the rounded
    ``*_weight_dict`` views are for display only.
    """
    specs: List[DesignSpec] = []
    for cand in res.search.candidates:
        treated_w = {str(u): float(w) for u, w in cand.treated_weight_dict_full.items()}
        control_w = {str(u): float(v) for u, v in cand.control_weight_dict_full.items()}
        treated = sorted(treated_w)
        contrast: Dict[Any, float] = dict(treated_w)
        for u, v in control_w.items():
            contrast[u] = contrast.get(u, 0.0) - v
        specs.append(DesignSpec(
            "LEXSCM", f"L:{'+'.join(map(str, treated))}", contrast, treated,
        ))
    return specs


def from_marex(res) -> List[DesignSpec]:
    """Build DesignSpecs from a fitted MAREX result.

    Uses the solution-pool menu when present (``top_K > 1``), else the single
    aggregate design. Each design's contrast is ``w_treated - v_control`` over
    units (treated side sums to +1, control to -1), matching the convention of
    the other adapters.
    """
    def _spec(labels, w, v, treated):
        w = np.asarray(w, dtype=float)
        v = np.asarray(v, dtype=float)
        contrast = {str(labels[i]): float(w[i] - v[i]) for i in range(len(labels))
                    if abs(float(w[i] - v[i])) > 1e-12}
        treated = sorted(str(t) for t in treated)
        return DesignSpec("MAREX", f"M:{'+'.join(treated)}", contrast, treated)

    pool = getattr(res, "pool", None)
    if pool:
        out = []
        for e in pool:
            raw = e["design"]
            # raw["df"] is the units x time panel; its index carries the unit
            # labels the aggregate weight vectors are aligned to.
            labels = list(raw["df"].index)
            out.append(_spec(labels, raw["w_agg"], raw["v_agg"], e["markets"]))
        return out

    # Single-design fallback (top_K == 1): globres.Y_full is a bare ndarray with
    # no labels, so assemble the contrast from the per-cluster unit-weight maps,
    # which carry the unit labels. Size-weight clusters so the treated/control
    # sides each sum to +1 / -1 across the whole design (matching *_weights_agg).
    clusters = list(res.clusters.values())
    sizes = np.array([float(c.cardinality) for c in clusters], dtype=float)
    total = float(sizes.sum()) or 1.0
    multi = len(clusters) > 1
    contrast: Dict[str, float] = {}
    treated: List[str] = []
    for c, sz in zip(clusters, sizes):
        scale = (sz / total) if multi else 1.0
        for u, w in c.unit_weight_map.get("Treated", {}).items():
            contrast[str(u)] = contrast.get(str(u), 0.0) + scale * float(w)
            treated.append(str(u))
        for u, v in c.unit_weight_map.get("Control", {}).items():
            contrast[str(u)] = contrast.get(str(u), 0.0) - scale * float(v)
    treated = sorted(set(treated))
    return [DesignSpec("MAREX", f"M:{'+'.join(treated)}", contrast, treated)]


_SYNDES_DEFAULTS: Dict[str, Any] = dict(
    mode="two_way_global", gap_limit=0.05, time_limit=20.0, run_inference=False,
)
_LEXSCM_DEFAULTS: Dict[str, Any] = dict(display_graph=False, verbose=False)

# The geographic-design constraint vocabulary, identical across the method
# configs (SYNDES / LEXSCM / MAREX). ``compare_methods`` routes a single
# ``constraints`` dict of these to every requested method, so a constraint binds
# the comparison uniformly instead of having to be hand-coded per method.
_GEO_CONSTRAINT_KEYS = frozenset({
    "to_be_treated", "not_to_be_treated",
    "cluster_col", "adjacency", "spillover_threshold",
    "stratum_col", "min_per_stratum", "max_per_stratum",
    "size_col", "min_size", "max_size",
})
_MAREX_DEFAULTS: Dict[str, Any] = dict(design="standard", display_graph=False,
                                       verbose=False)


@dataclass(frozen=True)
class DesignComparison:
    """Result of :func:`compare_methods`.

    Parameters
    ----------
    table : pd.DataFrame
        One row per design across all requested methods, with ``method``,
        ``label``, ``treated``, ``fit_rmse``, ``mde_pct`` (at ``horizon``), and a
        per-method ``pareto`` flag -- the dataframe form of the comparison.
    syndes, lexscm, marex : optional
        The underlying fitted results (``None`` for any method not requested),
        kept for inspection.
    specs : list of DesignSpec
        The common-currency designs that were scored.
    horizon : int
        The post-period horizon the MDE was simulated at.
    """

    table: pd.DataFrame
    syndes: Any = None
    lexscm: Any = None
    marex: Any = None
    specs: List[DesignSpec] = field(default_factory=list)
    horizon: int = 5

    def plot(self, ax=None):
        """Overlay the per-method Pareto frontiers (the plot form)."""
        return plot_compare_pareto(self.table, ax=ax)


def compare_methods(
    df: pd.DataFrame,
    *,
    outcome: str,
    unitid: str,
    time: str,
    treated_size: int,
    horizon: int = 5,
    post_col: Optional[str] = None,
    n_post: Optional[int] = None,
    top_K: int = 6,
    alpha: float = 0.10,
    power_target: float = 0.80,
    effects_pct: Optional[Sequence[float]] = None,
    methods: Sequence[str] = ("SYNDES", "LEXSCM"),
    constraints: Optional[Dict[str, Any]] = None,
    syndes_holdout_frac: Optional[float] = 0.3,
    syndes_options: Optional[Dict[str, Any]] = None,
    lexscm_options: Optional[Dict[str, Any]] = None,
    marex_options: Optional[Dict[str, Any]] = None,
) -> DesignComparison:
    """Fit SYNDES, LEXSCM, and/or MAREX on one panel and compare them.

    A one-call wrapper around the adapters + :func:`compare_pareto`: it fits each
    requested method on the same data with the same treated-set size and post
    window, then scores every design through the one shared MDE harness so the
    frontiers are comparable.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel.
    outcome, unitid, time : str
        Column names.
    treated_size : int
        Number of treated units (e.g. SYNDES ``K``, LEXSCM ``m``, MAREX ``m_eq``).
    horizon : int, default 5
        Post-period horizon for the simulated MDE.
    post_col : str, optional
        0/1 post-treatment column. Provide this or ``n_post``.
    n_post : int, optional
        If ``post_col`` is not given, mark the last ``n_post`` periods as post.
    top_K : int, default 6
        SYNDES solution-pool size (number of candidate SYNDES designs).
    alpha, power_target : float
        Significance level and target power for the MDE.
    effects_pct : sequence of float, optional
        Common effect grid (percent of baseline) for the MDE simulation.
    methods : sequence of str, default ``("SYNDES", "LEXSCM")``
        Which methods to fit and compare. Any of ``"SYNDES"``, ``"LEXSCM"``,
        ``"MAREX"``.
    constraints : dict, optional
        Geographic-design constraints applied uniformly to *every* requested
        method, so a constraint binds the whole comparison (all methods honour
        the same vocabulary). Any of ``to_be_treated`` /
        ``not_to_be_treated`` (forced-in / forbidden markets), ``cluster_col``
        (no two treated markets from one cluster), ``adjacency`` +
        ``spillover_threshold`` (no two bordering markets treated), ``stratum_col``
        + ``min_per_stratum`` / ``max_per_stratum`` (coverage quota), and
        ``size_col`` + ``min_size`` / ``max_size`` (treated-unit size band). A
        key set in a per-method ``*_options`` dict overrides the routed value for
        that method.
    syndes_holdout_frac : float or None, default 0.3
        Holdout fraction for SYNDES design selection: SYNDES learns its pool on
        the leading ``1 - syndes_holdout_frac`` of the pre-period and is ranked
        by out-of-sample contrast error on the tail (column ``oos_rmse``; SYNDES
        rows are ordered ascending by it). ``None`` reverts SYNDES to in-sample
        selection (``oos_rmse`` is ``NaN``). Ignored when ``top_K < 2`` (a pool
        is required to validate). An explicit ``holdout_frac`` in
        ``syndes_options`` overrides this.
    syndes_options, lexscm_options, marex_options : dict, optional
        Per-method overrides merged over the defaults (e.g.
        ``{"time_limit": 20.0}`` to give SYNDES a larger budget, or
        ``{"top_K": 8}`` to widen LEXSCM's search). For LEXSCM, ``treated_size``
        maps to ``m`` and every unit is marked eligible unless an explicit
        ``candidate_col`` is supplied in ``lexscm_options``. For MAREX,
        ``treated_size`` maps to the per-cluster ``m_eq`` (the exact number of
        treated units) and ``top_K`` drives its solution-pool menu, unless the
        caller supplies their own cardinality keys (``m_eq`` / ``m_min`` /
        ``m_max``) in ``marex_options``.

    Returns
    -------
    DesignComparison
        ``.table`` (dataframe form) and ``.plot()`` (plot form).
    """
    methods = tuple(methods)
    unknown = set(methods) - {"SYNDES", "LEXSCM", "MAREX"}
    if unknown:
        raise ValueError(f"Unknown method(s): {sorted(unknown)}; "
                         "expected any of 'SYNDES', 'LEXSCM', 'MAREX'.")
    if not methods:
        raise ValueError("methods must name at least one estimator.")

    # Geographic constraints are routed uniformly to every requested method (the
    # config field names are identical across the methods), so a constraint binds
    # the whole comparison rather than one method. Per-method options still override.
    constraints = dict(constraints or {})
    unknown_c = set(constraints) - _GEO_CONSTRAINT_KEYS
    if unknown_c:
        raise ValueError(
            f"Unknown constraint(s): {sorted(unknown_c)}; expected any of "
            f"{sorted(_GEO_CONSTRAINT_KEYS)}."
        )

    times = sorted(df[time].unique())
    df = df.copy()
    if post_col is None:
        if n_post is None:
            raise ValueError("Provide either post_col or n_post.")
        post_col = "__compare_post__"
        df[post_col] = df[time].isin(set(times[-int(n_post):])).astype(int)
        n_post_resolved = int(n_post)
    else:
        if post_col not in df.columns:
            raise ValueError(f"post_col {post_col!r} is not a column of df.")
        post_by_time = (df[[time, post_col]].drop_duplicates(subset=[time])
                        .set_index(time)[post_col])
        n_post_resolved = int(post_by_time.astype(bool).sum())
    n_pre = len(times) - n_post_resolved

    Ywide = df.pivot(index=time, columns=unitid, values=outcome).sort_index()

    common = dict(df=df, outcome=outcome, unitid=unitid, time=time,
                  post_col=post_col)
    syn = lex = mar = None
    specs: List[DesignSpec] = []

    if "SYNDES" in methods:
        from ..estimators.syndes import SYNDES
        syn_over = syndes_options or {}
        # Holdout selection needs a candidate pool (top_K >= 2); fall back to
        # in-sample otherwise so a top_K=1 caller still works. If the caller
        # picks a selection rule (or holdout_frac) explicitly, defer to it and
        # do not inject the default holdout (they are mutually exclusive).
        explicit = "selection" in syn_over or "holdout_frac" in syn_over
        sk = {**_SYNDES_DEFAULTS, **common, "K": treated_size, "top_K": top_K,
              "alpha": alpha}
        if not explicit:
            if syndes_holdout_frac is not None and top_K >= 2:
                sk["holdout_frac"] = syndes_holdout_frac
            else:
                # SYNDES now holdout-validates pools by default, so to honour
                # syndes_holdout_frac=None (disable) we ask for in-sample.
                sk["selection"] = "in_sample"
        sk.update(constraints)
        sk.update(syn_over)
        syn = SYNDES(sk).fit()
        specs += from_syndes(syn)

    if "LEXSCM" in methods:
        from ..estimators.lexscm import LEXSCM
        lex_over = lexscm_options or {}
        # LEXSCM needs a per-unit eligibility column; compare_methods treats
        # every unit as a candidate, so inject an all-eligible column unless the
        # caller names their own. treated_size maps to LEXSCM's `m`.
        cand_col = lex_over.get("candidate_col")
        if cand_col is None:
            cand_col = "__compare_candidate__"
            df[cand_col] = True
        lk = {**_LEXSCM_DEFAULTS, **common, "candidate_col": cand_col,
              "m": treated_size, "alpha": alpha, **constraints, **lex_over}
        lex = LEXSCM(lk).fit()
        specs += from_lexscm(lex)

    if "MAREX" in methods:
        from ..estimators.scexp import MAREX
        mar_over = marex_options or {}
        # treated_size -> MAREX's per-cluster exact count m_eq, unless the caller
        # specifies their own cardinality (m_eq / m_min / m_max). top_K drives
        # the solution-pool menu the adapter reduces to DesignSpecs.
        mk = {**_MAREX_DEFAULTS, **common, "top_K": top_K, "alpha": alpha,
              "power_target": power_target}
        if not ({"m_eq", "m_min", "m_max"} & set(mar_over)):
            mk["m_eq"] = treated_size
        mk.update(constraints)
        mk.update(mar_over)
        mar = MAREX(mk).fit()
        specs += from_marex(mar)

    table = compare_pareto(specs, Ywide, n_pre, horizon=horizon,
                           effects_pct=effects_pct, alpha=alpha,
                           power_target=power_target)

    # Rank the SYNDES block by out-of-sample (holdout) error when it is defined,
    # keeping every other method's rows in place. Stable sort -> deterministic.
    syn_mask = table["method"] == "SYNDES"
    if syn_mask.any() and table.loc[syn_mask, "oos_rmse"].notna().any():
        syn_sorted = table[syn_mask].sort_values("oos_rmse", kind="stable")
        table = pd.concat([syn_sorted, table[~syn_mask]], ignore_index=True)

    return DesignComparison(table=table, syndes=syn, lexscm=lex,
                            marex=mar, specs=specs, horizon=horizon)


def plot_compare_pareto(frame: pd.DataFrame, ax=None):
    """Overlay each method's fit-vs-power Pareto frontier on shared axes.

    Renders in the in-house mlsynth style. Returns the axis drawn on.
    """
    import matplotlib.pyplot as plt

    from .plotting import mlsynth_style

    _METHOD_COLORS = {"SYNDES": "blue", "LEXSCM": "red", "MAREX": "green"}
    _palette = ("purple", "orange", "brown", "olive")

    def _draw(ax):
        for k, (method, sub) in enumerate(frame.groupby("method")):
            color = _METHOD_COLORS.get(method, _palette[k % len(_palette)])
            fit = sub["fit_rmse"].to_numpy()
            mde = np.where(np.isfinite(sub["mde_pct"]), sub["mde_pct"], np.nan)
            par = sub["pareto"].to_numpy()
            ax.scatter(fit[~par], mde[~par], color=color, alpha=0.35, s=45,
                       zorder=2)
            order = np.argsort(fit[par])
            ax.plot(fit[par][order], mde[par][order], "-o", color=color,
                    label=f"{method} frontier", zorder=3)
        ax.set_xlabel("pre-period RMSE (treated vs weighted control)")
        ax.set_ylabel("simulated MDE %  (lower is better)")
        ax.set_title("Design methods: fit vs power at a common horizon")
        ax.legend()
        return ax

    if ax is not None:
        return _draw(ax)
    with mlsynth_style():
        _, ax = plt.subplots(figsize=(8, 5))
        _draw(ax)
        ax.figure.tight_layout()
        plt.show()
    return ax
