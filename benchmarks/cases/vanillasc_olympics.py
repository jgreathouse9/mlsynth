"""Tokyo 2020 Olympics and COVID-19 incidence -- Yoneoka et al. (2022), VanillaSC.

Yoneoka, D., Eguchi, A., Fukumoto, K., Kawashima, T., Tanoue, Y., Tabuchi, T.,
Miyata, H., Ghaznavi, C., Shibuya, K., & Nomura, S. (2022), *Effect of the Tokyo
2020 Summer Olympic Games on COVID-19 incidence in Japan: a synthetic control
approach*, BMJ Open 12:e061444,
`10.1136/bmjopen-2022-061444 <https://doi.org/10.1136/bmjopen-2022-061444>`_.
Replication package: https://github.com/kingqwert/R/tree/master/Synthetic_Olympic

The paper asks whether hosting the Games raised COVID-19 case counts in Japan,
and answers it with an ordinary Abadie-Diamond-Hainmueller synthetic control:
Japan against 42 donor countries, daily confirmed cases per million (7-day
moving average), 34 predictors, treated from the opening ceremony on 23 July
2021. It reports about 53,900 excess infections over the Olympic period.

Two reference bases, because they answer different questions
------------------------------------------------------------

The authors committed their own outputs -- donor weights, the balance table,
the placebo distribution and the Fisher p-values, at all three intervention
timings. That artifact is the Path A gold: it is what produced the printed
table, so comparing to it asks whether this case has identified the paper's
estimand.

A live tidysynth 0.2.0 run of the authors' own script is the second base, and
it asks whether the specification is reproducible today. It is not, in the part
that gets quoted:

    timing   max |W_authors - W_live|   max |synthetic path difference|   p
    -7 days            0.178                        5.54                 same
     main              0.183                        2.79                 same
    +7 days            0.038                        0.43                 same

Germany's weight moves from 0.547 to 0.730 in the main specification, Italy's
from 0.182 to 0.078, South Korea's from 0.018 to zero. The observed series is
identical to 0.0, so this is the optimizer, not the data. What does not move is
the answer: the Fisher exact p-value is the same in all three timings, and the
closing-day counterfactual shifts from 50.98 to 51.81 against an observed
109.21.

That is the ordinary non-identification of the synthetic-control weight vector,
and it is why this case gates the p-value and the estimand and records the
weights.

What the paper's numbers are, and where they come from
------------------------------------------------------

The estimand had to be recovered from the script. The paper's cumulative
figures are sums over ``date2`` 53 to 69 inclusive -- 23 July to 8 August, the
opening day through the closing day, seventeen days -- converted from per-
million to counts at a population of 126.05 million. At that window the authors'
committed output gives 143,072 observed against 89,210 counterfactual, +60.4%,
which is the paper's sentence to the digit.

``paper_cum_observed_cases`` below recomputes the observed side from the
vendored panel, so it pins the estimand and the data together: a window off by
one period, or a panel that drifted, misses 143,072.

One arithmetic inconsistency in the paper. It reports the closing-day observed
as 109.2 per million and the counterfactual as 51.0, both of which reproduce
(109.210 and 50.975), and then calls the first 115.7% higher than the second.
109.2 / 51.0 is 114.2%. The cumulative pair is exact, so this is the text and
not the estimator; the case pins 114.2% against the reproduced values.

Where mlsynth lands
-------------------

VanillaSC returns the same Fisher p-value -- 0.0233, Japan ranked first of 43 --
at a pre-treatment RMSPE of 0.139 against tidysynth's 0.838. That is a six-fold
better fit to the same pre-period, on the criterion the reference is itself
minimising, and the better fit gives a larger effect: a closing-day
counterfactual of 46.0, so +137% where the paper says +115.7%.

The improvement is recorded, not gated as a win. A bug that lowered the
objective for the wrong reason would also pass such a gate. What is gated is
that mlsynth does not do *worse* than the reference on the reference's own
criterion, and that the inference agrees exactly.

Why the 34 predictors barely bind
----------------------------------

The V search puts 93.7% of its mass on the five lagged-outcome predictors. The
authors' own balance table shows what that leaves: park mobility is -1.3 for
Japan against 98.2 for its synthetic control, vaccination 18.7 against 35.0,
testing 0.46 against 3.06. mlsynth's ``outcome-only`` backend, which ignores all
34 predictors, reproduces the qualitative answer (+79% against +90% excess).
``v_mass_on_lags`` pins this, because a specification that reads as covariate-
rich and behaves as outcome-only is the thing a reader of the paper would want
to know.

The sensitivity analysis is weaker than the paper suggests
-----------------------------------------------------------

The paper shifts the intervention by +/- 7 days and reports the results
"consistent with our main findings". In sign and significance they are, in both
implementations. In magnitude they are not: at -7 days the excess is 93.8%
(mlsynth) against 94.5% (authors), but at +7 days it is 31.1% against 69.9%.
That window already contains the surge, so the counterfactual chases it, and
how much it catches depends on how tightly the pre-period is fitted. Recorded,
not gated.

Running the R directly
----------------------

The committed gold under ``benchmarks/reference/vanillasc_olympics/`` is what
the case reads by default, so it runs in CI without R. To re-run it live::

    bash benchmarks/R/install_tidysynth_olympics.sh
    Rscript benchmarks/reference/vanillasc_olympics/reference.R

``authors/`` alongside it holds the authors' own committed CSVs, copied
unchanged from their repository at commit ``bde42e2``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parents[1]
_ROOT = _HERE.parent
_REF = _HERE / "reference" / "vanillasc_olympics"
_AUTHORS = _REF / "authors"

#: tidysynth's ``i_time`` is the last PRE-treatment period (its own
#: ``grab_significance`` splits on ``time_unit <= trt_time``), so the treatment
#: indicator is ``date2 > TIMING``.
MAIN = 53
TIMINGS = (46, 53, 60)
_AUTHOR_TAG = {46: "minus7days_treattime46", 53: "treattime53",
               60: "plus7days_treattime60"}
_LAGS = {46: (25, 32, 39, 46), 53: (25, 32, 39, 46, 53),
         60: (25, 32, 39, 46, 53, 60)}

#: The paper's cumulative window: 23 July through 8 August inclusive.
WINDOW = (53, 69)
#: The population OWID used for Japan's per-million series, recovered from the
#: paper's own 143,072 / 89,210 pair.
POP = 126.05e6

_PERIOD = ["new_tests_smoothed_per_thousand", "people_fully_vaccinated_per_hundred",
           "stringency_index", "temparature", "rainfall", "retail", "grocery",
           "transit", "work", "park", "resident"]
_SPOT = ["median_age", "Population_14", "Population_65", "Population_density",
         "democracy", "Mortality_rate_infant", "gdp_per_capita",
         "human_development_index", "diabetes_prevalence",
         "Current health expenditure", "life_expectancy", "Surface area",
         "Unemployment", "PM2.5", "International migrant stock",
         "houseshold_size", "hospital_beds_per_thousand", "Asia"]

EXPECTED = {
    "n_units": (43.0, 0.5),
    "n_periods": (50.0, 0.5),
    "n_donors": (42.0, 0.5),
    # the vendored panel is the authors' input, unchanged
    "panel_matches_authors": (0.0, 1e-12),
    # Path A: the paper's printed quantities, from the authors' own artifact
    "paper_cum_observed_cases": (143072.0, 1.0),
    "paper_cum_counterfactual_cases": (89210.0, 1.0),
    "paper_excess_pct": (60.4, 0.05),
    "paper_closing_day_observed": (109.2, 0.05),
    "paper_closing_day_counterfactual": (51.0, 0.05),
    "paper_closing_day_pct": (114.2, 0.1),
    "authors_pvalue_main": (0.023255813953488372, 1e-12),
    # cross-validation: mlsynth against a live tidysynth run of the same script
    "mlsynth_pvalue_main": (0.023255813953488372, 1e-12),
    "mlsynth_rank_main": (1.0, 0.5),
    "mlsynth_beats_reference_pre_fit": (1.0, 0.5),
    "mlsynth_excess_positive_all_timings": (1.0, 0.5),
    # recorded: the weight vector is not identified, the inference is
    "authors_vs_live_weight_maxdiff": (0.183, 0.02),
    "authors_vs_live_pvalues_identical": (1.0, 0.5),
    "v_mass_on_lags": (0.937, 0.01),
}


def _panel() -> pd.DataFrame:
    return pd.read_parquet(_ROOT / "basedata" / "yoneoka_olympics_covid.parquet")


def _design(timing: int):
    """The authors' 34-predictor spec, as a (df, covariates, windows) triple.

    tidysynth's ``generate_predictor(time_window = t, x = <outcome>)`` is a spot
    value of the outcome; mlsynth windows a named column, so each lag gets its
    own copy of the outcome column matched over ``(t, t)``.
    """
    d = _panel().copy()
    for t in _LAGS[timing]:
        d[f"cases_t{t}"] = d["new_cases_smoothed_per_million"]
    d["treated"] = ((d.location == "Japan") & (d.date2 > timing)).astype(int)
    covs = _PERIOD + _SPOT + [f"cases_t{t}" for t in _LAGS[timing]]
    wins = {c: (25, timing) for c in _PERIOD}
    wins.update({c: (25, 25) for c in _SPOT})
    wins.update({f"cases_t{t}": (t, t) for t in _LAGS[timing]})
    return d, covs, wins


def _fit(timing: int, *, backend: str = "mscmt", covariates: bool = True,
         inference=False):
    from mlsynth import VanillaSC

    d, covs, wins = _design(timing)
    cfg = {"df": d, "outcome": "new_cases_smoothed_per_million",
           "treat": "treated", "unitid": "location", "time": "date2",
           "backend": backend, "seed": 0, "inference": inference,
           "display_graphs": False}
    if covariates:
        cfg.update({"covariates": covs, "covariate_windows": wins})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(cfg).fit(), d


def _paths(res, d: pd.DataFrame) -> pd.DataFrame:
    times = sorted(d.date2.unique())
    obs = (d[d.location == "Japan"].sort_values("date2")
           .new_cases_smoothed_per_million.values)
    cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float).ravel()
    return pd.DataFrame({"t": times, "obs": obs, "cf": cf})


def _authors(kind: str, timing: int) -> pd.DataFrame:
    return pd.read_csv(_AUTHORS / f"res_{kind}_{_AUTHOR_TAG[timing]}.csv")


def _rmspe(obs, cf) -> float:
    return float(np.sqrt(np.mean((np.asarray(obs) - np.asarray(cf)) ** 2)))


def run() -> dict:
    d = _panel()
    out: dict[str, float] = {
        "n_units": float(d.location.nunique()),
        "n_periods": float(d.date2.nunique()),
        "n_donors": float(d.location.nunique() - 1),
    }

    # ---- the vendored panel is the authors' input ----
    gold = _authors("trend_diff_data", MAIN)
    jp = (d[d.location == "Japan"].sort_values("date2")
          .set_index("date2").new_cases_smoothed_per_million)
    out["panel_matches_authors"] = float(
        np.abs(jp.loc[gold.time_unit].values - gold.real_y.values).max())

    # ---- Path A: the paper's printed quantities ----
    lo, hi = WINDOW
    win = gold[gold.time_unit.between(lo, hi)]
    obs_win = float(jp.loc[lo:hi].sum())          # from OUR panel
    out["paper_cum_observed_cases"] = obs_win * POP / 1e6
    out["paper_cum_counterfactual_cases"] = float(win.synth_y.sum()) * POP / 1e6
    out["paper_excess_pct"] = (obs_win / float(win.synth_y.sum()) - 1) * 100
    close = gold[gold.time_unit == hi]
    out["paper_closing_day_observed"] = float(close.real_y.iloc[0])
    out["paper_closing_day_counterfactual"] = float(close.synth_y.iloc[0])
    out["paper_closing_day_pct"] = float(
        close.real_y.iloc[0] / close.synth_y.iloc[0] - 1) * 100
    ap = _authors("fishers_exact_pvalue", MAIN)
    out["authors_pvalue_main"] = float(
        ap[ap.unit_name == "Japan"].fishers_exact_pvalue.iloc[0])

    # ---- the weight vector is not identified; the inference is ----
    wdiffs, same_p = [], True
    for T in TIMINGS:
        aw = _authors("weight_data", T)
        aw = aw[aw.type == "Control Unit Weights (W)"].set_index("unit").weight
        lw = pd.read_csv(_REF / f"gold_weights_{T}.csv").set_index("unit").weight
        j = pd.concat([aw.rename("a"), lw.rename("l")], axis=1).fillna(0.0)
        wdiffs.append(float(j.a.sub(j.l).abs().max()))
        a_p = _authors("fishers_exact_pvalue", T)
        l_p = pd.read_csv(_REF / f"gold_signif_{T}.csv")
        same_p &= np.isclose(
            float(a_p[a_p.unit_name == "Japan"].fishers_exact_pvalue.iloc[0]),
            float(l_p[l_p.unit_name == "Japan"].fishers_exact_pvalue.iloc[0]))
    out["authors_vs_live_weight_maxdiff"] = float(max(wdiffs))
    out["authors_vs_live_pvalues_identical"] = float(same_p)

    # The predictor weights concentrate on the lagged outcomes, which is why the
    # covariate-rich spec behaves like an outcome-only one.
    v = pd.read_csv(_REF / f"gold_predw_{MAIN}.csv")
    lag_mass = v[v.variable.str.contains("lagged cases")].weight.sum()
    out["v_mass_on_lags"] = float(lag_mass / v.weight.sum())

    # ---- mlsynth: same inference, better fit ----
    res, dd = _fit(MAIN, inference="placebo")
    p = _paths(res, dd)
    out["mlsynth_pvalue_main"] = float(res.inference.p_value)
    out["mlsynth_rank_main"] = float(res.inference.details["rank"])
    ml_pre = _rmspe(p[p.t <= MAIN].obs, p[p.t <= MAIN].cf)
    ref_pre = _rmspe(gold[gold.time_unit <= MAIN].real_y,
                     gold[gold.time_unit <= MAIN].synth_y)
    out["mlsynth_beats_reference_pre_fit"] = float(ml_pre <= ref_pre)

    positive = True
    for T in TIMINGS:
        r, dt = _fit(T)
        q = _paths(r, dt)
        post = q[q.t > T]
        positive &= bool(post.obs.sum() > post.cf.sum())
    out["mlsynth_excess_positive_all_timings"] = float(positive)

    return out
