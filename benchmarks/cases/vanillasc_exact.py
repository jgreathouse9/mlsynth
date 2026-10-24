"""Abadie and Shang (2026), exact synthetic control: VanillaSC against the authors' ``run_all.py``.

Path A, cross-validated against the authors' code at the commit they published. The
reference bundle ``benchmarks/reference/vanillasc_exact/`` is a live run of
``code/scripts/run_all.py`` of lychemon/synth-AS-replication at 791a3865 (v1.0.0),
with lychemon/synthx_package at a10a28c5 beside it, which writes every data-based
table of the paper: its own ``verify_results.py`` reports every paper number
reproduced, and the bundle records the 479 numbers in its ``output/``. Here
``VanillaSC(backend="exact").fit()`` is run on the same two panels, the authors' own
CSVs (``basedata/abadie_shang_smoking.csv`` and ``abadie_shang_germany.csv``, byte for
byte), with the authors' predictor specification, and compared with that run number by
number:

- Tables 1 and 5: the diagonal-class donor weights and validation losses, and the
  leakage-free California sample that drops the years 1975, 1980 and 1988, whose
  outcomes are predictors, from validation (``fit_exclude``);
- Tables D.1 and D.2: the positive definite class (``exact_v_class="pd"``), the number
  of visible facets, and the facet holding the optimum;
- the mean post-treatment gaps, 1989-2000 and 1991-2003, which are VanillaSC's ATT;
- Tables 2, 4 and 6: the diagonal and positive definite predictor weights that select
  the diagonal optimum, their analytic centres and the identified ranges
  (``exact_ranges=True``), at the four decimals ``run_all.py`` writes and at full
  precision against the authors' functions called with ``decimals=None``;
- Table 3: the faces of the donor hull by dimension (``exact_face_counts=True``).

The predictor specification follows the authors' ``data_preparation.py``: California's
log income, youth share and retail price are averaged over 1980-1988, beer over
1984-1988, and cigarette sales enter at 1988, 1980 and 1975; West Germany's GDP, trade,
inflation and industry share are averaged over 1971-1980, schooling over 1970 and 1975,
and the 1970s investment rate is read at 1980, with validation over 1981-1990. Both
standardise each predictor by its standard deviation across all units (ddof=1), as
VanillaSC does.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "basedata")
CA_DATA = os.path.abspath(os.path.join(_BASE, "abadie_shang_smoking.csv"))
GE_DATA = os.path.abspath(os.path.join(_BASE, "abadie_shang_germany.csv"))

# covariate column -> (averaging window, the authors' predictor name), in the authors' order
CA_PREDICTORS = {
    "lnincome": ((1980, 1988), "Log per-capita state income"),
    "age15to24": ((1980, 1988), "Share of population aged 15-24"),
    "retprice": ((1980, 1988), "Retail cigarette price"),
    "beer": ((1984, 1988), "Per-capita beer consumption"),
    "cigsale_1988": ((1988, 1988), "Cigarette sales per capita (1988)"),
    "cigsale_1980": ((1980, 1980), "Cigarette sales per capita (1980)"),
    "cigsale_1975": ((1975, 1975), "Cigarette sales per capita (1975)"),
}
GE_PREDICTORS = {
    "gdp": ((1971, 1980), "Per-capita GDP (1971-80)"),
    "trade": ((1971, 1980), "Trade openness (1971-80)"),
    "infrate": ((1971, 1980), "Inflation rate (1971-80)"),
    "industry": ((1971, 1980), "Industry share (1971-80)"),
    "schooling": ((1970, 1975), "Schooling (1970, 1975)"),
    "invest70": ((1980, 1980), "Investment rate (1970s)"),
}
LEAKAGE_YEARS = [1975, 1980, 1988]


def california_panel() -> pd.DataFrame:
    """The authors' Proposition 99 panel, California treated from 1989, with the three
    cigarette-sales lags as covariate columns (copies of the outcome, each read in its year)."""
    d = pd.read_csv(CA_DATA)
    d["year"] = d["year"].astype(int)
    d["treated"] = ((d["state"] == "California") & (d["year"] >= 1989)).astype(int)
    for year in (1988, 1980, 1975):
        d[f"cigsale_{year}"] = d["cigsale"]
    return d


def germany_panel() -> pd.DataFrame:
    """The authors' reunification panel, West Germany treated from 1991."""
    d = pd.read_csv(GE_DATA)
    d["treated"] = ((d["country"] == "West Germany") & (d["year"] >= 1991)).astype(int)
    return d


def config(case: str, **overrides) -> dict:
    """The VanillaSC configuration of ``case`` ("california", "california_leakage_free", "germany")."""
    if case == "germany":
        spec, cfg = GE_PREDICTORS, {"df": germany_panel(), "outcome": "gdp", "unitid": "country",
                                    "fit_window": (1981, 1990)}
    else:
        spec, cfg = CA_PREDICTORS, {"df": california_panel(), "outcome": "cigsale", "unitid": "state"}
        if case == "california_leakage_free":
            cfg["fit_exclude"] = LEAKAGE_YEARS
    cfg.update(treat="treated", time="year", backend="exact", covariates=list(spec),
               covariate_windows={c: w for c, (w, _) in spec.items()}, inference=False,
               display_graphs=False)
    cfg.update(overrides)
    return cfg


def fit(case: str, **overrides):
    from mlsynth import VanillaSC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(config(case, **overrides)).fit()


def _ref():
    return load_reference("vanillasc_exact")["values"]


def _ref_weights(ref, prefix):
    return {k[len(prefix) + 1:-1]: v for k, v in ref.items() if k.startswith(prefix + "[")}


def _named(res, case):
    """The exact details' vectors keyed by the authors' predictor names."""
    spec = GE_PREDICTORS if case == "germany" else CA_PREDICTORS
    names = [spec[c][1] for c in res.exact.predictor_names]
    return lambda v: dict(zip(names, np.asarray(v, float)))


def _maxdev(got: dict, want: dict) -> float:
    return float(max(abs(got.get(k, 0.0) - v) for k, v in want.items()))


def _tag(case):
    return "california" if case == "california" else case


def _predictors(case):
    """X1, X0 standardised as the authors and VanillaSC standardise them, donors in panel order."""
    spec = GE_PREDICTORS if case == "germany" else CA_PREDICTORS
    cfg = config(case)
    d, unit = cfg["df"], cfg["unitid"]
    treated = d.loc[d["treated"] == 1, unit].iloc[0]
    rows = [d[d["year"].between(*w)].groupby(unit)[c].mean() for c, (w, _) in spec.items()]
    X = pd.concat(rows, axis=1).T
    X = X.div(X.std(axis=1, ddof=1), axis=0)
    donors = [u for u in X.columns if u != treated]
    return X[treated].to_numpy(float), X[donors].to_numpy(float), donors


def _pd_objectives(X1, X0, w, V):
    """log det V and log det V + sum_j log s_j(V), s_j = -(X_j - X_j0)'V r over the inactive donors."""
    S = np.flatnonzero(w > 1e-7)
    r = X1 - X0 @ w
    s = np.array([-(X0[:, j] - X0[:, S[0]]) @ V @ r for j in np.setdiff1d(np.arange(len(w)), S)])
    logdet = float(np.linalg.slogdet(V)[1])
    return logdet, logdet + float(np.log(s).sum())


def _diag_objectives(Sm, strict, d, support):
    """sum log d over the support, and the same plus the log of the strict slacks (Sm d)_i."""
    logdet = float(np.log(d[list(support)]).sum())
    return logdet, logdet + float(np.log((Sm @ d)[strict]).sum())


def _matrix(ref, key, K):
    V = np.zeros((K, K))
    for k, l in zip(*np.triu_indices(K)):
        V[k, l] = V[l, k] = ref[f"{key}[{k},{l}]"]
    return V


def _fits():
    """Every fit the comparison needs, keyed by (case, class)."""
    return {("california", "diag"): fit("california", exact_ranges=True, exact_face_counts=True),
            ("california", "pd"): fit("california", exact_v_class="pd"),
            ("california_leakage_free", "diag"): fit("california_leakage_free", exact_ranges=True),
            ("germany", "diag"): fit("germany", exact_ranges=True, exact_face_counts=True),
            ("germany", "pd"): fit("germany", exact_v_class="pd")}


def measure(fits=None) -> dict:
    """mlsynth against run_all.py: the largest gap per table, and the counts that must agree."""
    ref, fits = _ref(), fits or _fits()
    out = {}
    for (case, cls), res in fits.items():
        tag = _tag(case)
        w = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
        out[f"{tag}.{cls}.w_maxdev"] = _maxdev(w, _ref_weights(ref, f"{tag}.{cls}.w"))
        loss = float(res.exact.validation_loss)
        out[f"{tag}.{cls}.loss_reldev"] = abs(loss - ref[f"{tag}.{cls}.loss"]) / ref[f"{tag}.{cls}.loss"]
        if f"{tag}.{cls}.mean_gap" in ref:
            out[f"{tag}.{cls}.mean_gap_dev"] = abs(float(res.effects.att) - ref[f"{tag}.{cls}.mean_gap"])
        named = _named(res, case)
        if cls == "pd":
            out[f"{tag}.pd.n_visible_facets_dev"] = abs(res.exact.n_visible_facets
                                                        - ref[f"{tag}.pd.n_visible_facets"])
            donors = list(res.additional_outputs["donor_names"])
            facet = tuple(sorted(donors.index(d) for d in _ref_weights(ref, f"{tag}.pd.facet")))
            out[f"{tag}.pd.facet_held"] = float(facet in res.exact.facets)
            continue
        dg, pd_ = res.exact.diagonal, res.exact.pd
        for label, vec in (("logdet", dg.logdet), ("analytic_center", dg.center)):
            out[f"{tag}.diag_v.{label}_maxdev"] = _maxdev(named(vec), _ref_weights(ref, f"{tag}.diag_v.{label}"))
            out[f"{tag}.diag_v.{label}.full_maxdev"] = _maxdev(
                named(vec), _ref_weights(ref, f"{tag}.diag_v.{label}.full"))
        support = {n for n, v in named(np.isin(np.arange(len(dg.logdet)), dg.support)).items() if v}
        out[f"{tag}.diag_v.support_same"] = float(support == set(_ref_weights(ref, f"{tag}.diag_v.support")))
        out[f"{tag}.diag_v.n_implicit_dev"] = abs(dg.n_implicit - ref[f"{tag}.diag_v.n_implicit_equalities"])
        for label, vec in (("logdet", pd_.logdet_shares), ("center", pd_.center_shares),
                           ("range_lo", pd_.range_lo), ("range_hi", pd_.range_hi)):
            out[f"{tag}.pd_v.{label}_maxdev"] = _maxdev(named(vec), _ref_weights(ref, f"{tag}.pd_v.{label}"))
            out[f"{tag}.pd_v.{label}.full_maxdev"] = _maxdev(
                named(vec), _ref_weights(ref, f"{tag}.pd_v.{label}.full"))
        out[f"{tag}.pd_v.logdet_value_dev"] = abs(float(np.linalg.slogdet(pd_.logdet)[1])
                                                  - ref[f"{tag}.pd_v.logdet_value"])
        out[f"{tag}.pd_v.logdet_min_eig_reldev"] = abs(pd_.logdet_min_eig - ref[f"{tag}.pd_v.logdet_min_eig"]) \
            / ref[f"{tag}.pd_v.logdet_min_eig"]
        # The authors solve the centres and the diagonal log-det maximiser at Clarabel's default
        # tolerances; on the same programs, our points must score at least as well as theirs.
        X1, X0, donors = _predictors(case)
        w_vec = np.array([w.get(str(u), 0.0) for u in donors])
        K = X0.shape[0]
        ours_l, ours_c = _pd_objectives(X1, X0, w_vec, pd_.logdet), _pd_objectives(X1, X0, w_vec, pd_.center)
        theirs_l = _pd_objectives(X1, X0, w_vec, _matrix(ref, f"{tag}.pd_v.logdet_matrix", K))
        theirs_c = _pd_objectives(X1, X0, w_vec, _matrix(ref, f"{tag}.pd_v.center_matrix", K))
        out[f"{tag}.pd_v.logdet_objective_shortfall"] = max(0.0, theirs_l[0] - ours_l[0])
        out[f"{tag}.pd_v.center_objective_shortfall"] = max(0.0, theirs_c[1] - ours_c[1])
        spec = GE_PREDICTORS if case == "germany" else CA_PREDICTORS
        order = [spec[c][1] for c in res.exact.predictor_names]
        theirs_dl = np.array([ref[f"{tag}.diag_v.logdet.full[{n}]"] for n in order])
        theirs_dc = np.array([ref[f"{tag}.diag_v.analytic_center.full[{n}]"] for n in order])
        from mlsynth.utils.bilevel.exact import rationalize
        _, Sm = rationalize._diag_system(X1, X0, w_vec)
        s_ours = Sm @ dg.center                       # the strict rows: positive at our centre
        strict = s_ours > 1e-9 * max(1.0, float(np.abs(s_ours).max(initial=0.0)))
        out[f"{tag}.diag_v.logdet_objective_shortfall"] = max(
            0.0, _diag_objectives(Sm, strict, theirs_dl, dg.support)[0]
            - _diag_objectives(Sm, strict, dg.logdet, dg.support)[0])
        out[f"{tag}.diag_v.analytic_center_objective_shortfall"] = max(
            0.0, _diag_objectives(Sm, strict, theirs_dc, dg.support)[1]
            - _diag_objectives(Sm, strict, dg.center, dg.support)[1])
        if res.exact.face_counts is not None:
            fc, K = res.exact.face_counts, len(dg.logdet)
            out[f"{tag}.faces_mismatch"] = float(sum(
                getattr(fc, kind)[d] != ref[f"{tag}.faces.{kind}[{d}]"]
                for kind in ("total", "visible", "attainable") for d in range(K)))
    return out


def run() -> dict:
    return measure()


def comparison() -> dict:
    """mlsynth next to run_all.py, quantity by quantity: the donor weights, losses and mean gaps
    of both classes and both applications, the leakage-free weights, and the predictor weights."""
    ref, fits = _ref(), _fits()
    rows = []
    for (case, cls), res in fits.items():
        tag = _tag(case)
        w = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
        for donor, v in _ref_weights(ref, f"{tag}.{cls}.w").items():
            if v > 0 or w.get(donor, 0.0) > 0:
                rows.append({"quantity": f"{tag} {cls} weight[{donor}]", "mlsynth": w.get(donor, 0.0),
                             "reference": v})
        rows.append({"quantity": f"{tag} {cls} validation loss", "mlsynth": float(res.exact.validation_loss),
                     "reference": ref[f"{tag}.{cls}.loss"]})
        if f"{tag}.{cls}.mean_gap" in ref:
            rows.append({"quantity": f"{tag} {cls} mean post-treatment gap", "mlsynth": float(res.effects.att),
                         "reference": ref[f"{tag}.{cls}.mean_gap"]})
        if cls == "diag":
            named = _named(res, case)
            for label, vec in (("diagonal log-det", res.exact.diagonal.logdet),
                               ("diagonal centre", res.exact.diagonal.center),
                               ("PD log-det", res.exact.pd.logdet_shares),
                               ("PD centre", res.exact.pd.center_shares)):
                key = {"diagonal log-det": "diag_v.logdet", "diagonal centre": "diag_v.analytic_center",
                       "PD log-det": "pd_v.logdet", "PD centre": "pd_v.center"}[label]
                for name, v in named(vec).items():
                    rows.append({"quantity": f"{tag} {label} V[{name}]", "mlsynth": v,
                                 "reference": ref[f"{tag}.{key}.full[{name}]"]})
    cfg = {k: v for k, v in config("california").items() if k != "df"}
    return {"rows": rows,
            "mlsynth_call": {"estimator": "VanillaSC", "config": cfg},
            "reference": {"impl": "lychemon/synth-AS-replication run_all.py (live run, captured)",
                          "version": "791a3865 (v1.0.0), synthx_package a10a28c5"}}


EXPECTED = {
    # donor weights at the full precision run_all.py writes, and the losses
    **{f"{t}.{c}.w_maxdev": (0.0, 1e-7) for t, c in [("california", "diag"), ("california", "pd"),
                                                       ("california_leakage_free", "diag"),
                                                       ("germany", "diag"), ("germany", "pd")]},
    **{f"{t}.{c}.loss_reldev": (0.0, 1e-9) for t, c in [("california", "diag"), ("california", "pd"),
                                                         ("california_leakage_free", "diag"),
                                                         ("germany", "diag"), ("germany", "pd")]},
    # the paper's mean gaps (-19.0, -19.8 packs; -1391, -1582 USD) are VanillaSC's ATT
    **{f"{t}.{c}.mean_gap_dev": (0.0, 1e-6) for t, c in [("california", "diag"), ("california", "pd"),
                                                          ("germany", "diag"), ("germany", "pd")]},
    "california.pd.n_visible_facets_dev": (0.0, 0.5),
    "germany.pd.n_visible_facets_dev": (0.0, 0.5),
    "california.pd.facet_held": (1.0, 0.5),
    "germany.pd.facet_held": (1.0, 0.5),
    # Predictor weights the authors solve at tolerances of 1e-12 (the PD log-det maximiser and its
    # ranges): their four decimals to rounding, and their full-precision values to 1e-6.
    **{f"{t}.{k}_maxdev": (0.0, 5e-5 + 1e-9) for t in ("california", "california_leakage_free", "germany")
       for k in ("pd_v.logdet", "pd_v.range_lo", "pd_v.range_hi")},
    **{f"{t}.{k}.full_maxdev": (0.0, 1e-6) for t in ("california", "california_leakage_free", "germany")
       for k in ("pd_v.logdet", "pd_v.range_lo", "pd_v.range_hi")},
    # Those they solve at Clarabel's default tolerances (the centres, the diagonal log-det maximiser):
    # their full-precision values carry up to 2.3e-5 of solver error, so the four decimals agree to
    # rounding plus 3e-5 (one West German centre share rounds to 0.1764 here and 0.1763 there) ...
    **{f"{t}.{k}_maxdev": (0.0, 5e-5 + 3e-5) for t in ("california", "california_leakage_free", "germany")
       for k in ("diag_v.logdet", "diag_v.analytic_center", "pd_v.center")},
    **{f"{t}.{k}.full_maxdev": (0.0, 3e-5) for t in ("california", "california_leakage_free", "germany")
       for k in ("diag_v.logdet", "diag_v.analytic_center", "pd_v.center")},
    # ... and on the authors' own programs our points score at least as well as theirs.
    **{f"{t}.{k}_objective_shortfall": (0.0, 1e-9) for t in ("california", "california_leakage_free", "germany")
       for k in ("diag_v.logdet", "diag_v.analytic_center", "pd_v.logdet", "pd_v.center")},
    **{f"{t}.diag_v.support_same": (1.0, 0.5) for t in ("california", "california_leakage_free", "germany")},
    **{f"{t}.diag_v.n_implicit_dev": (0.0, 0.5) for t in ("california", "california_leakage_free", "germany")},
    **{f"{t}.pd_v.logdet_value_dev": (0.0, 1e-6) for t in ("california", "california_leakage_free", "germany")},
    **{f"{t}.pd_v.logdet_min_eig_reldev": (0.0, 1e-4)
       for t in ("california", "california_leakage_free", "germany")},
    # Table 3
    "california.faces_mismatch": (0.0, 0.5),
    "germany.faces_mismatch": (0.0, 0.5),
}
