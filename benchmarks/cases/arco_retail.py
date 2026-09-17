"""Path A -- Masini & Medeiros (2021) Table 5 Panel (a), the retail price experiment.

The empirical application of *"Counterfactual Analysis With Artificial Controls:
Inference, High Dimensions, and Nonstationarity"*, JASA 116(536), 1773-1788,
Section 6, on the authors' own data (the panel in ``codes/`` of their
replication package, shipped here as ``basedata/masini_retail_sales.parquet``).

A Brazilian retail chain raised the price of one product in 107 municipalities
on 2016-10-18 and held it there for 14 days, leaving 126 municipalities at the
old price. The question is what the increase did to quantities sold. The series
trends and the trend is heterogeneous across municipalities -- the authors show
an augmented Dickey-Fuller p-value histogram spread across the whole unit
interval (their Figure 1c) -- which is the case the method is built for and the
case that rules out both difference-in-differences and their own earlier ArCo.

The counterfactual explains the treated group's total daily quantity with the
126 untouched municipalities plus seven day-of-week indicators, so 133
regressors against 120 pre-intervention days: more regressors than observations,
which is why the first stage is penalized. Their reported cells are

===================================== ==========
:math:`\\Delta`                        -1,147
:math:`\\Delta` / number of shops      -4.33
p-value, :math:`\\phi(x) = mean(x^2)`  0
p-value, :math:`\\phi(x) = mean(|x|)`  0
regressors                            133
selected regressors                   26
===================================== ==========

What this case establishes
--------------------------
The port reproduces every cell. :math:`\\Delta` comes back at -1147.29 against a
table rounded to -1,147, the per-shop effect at -4.329 against -4.33, the
selected-regressor count at 26 exactly, and both resampling p-values at 0. A
port that agrees to five digits on a penalized selection over 133 regressors
agrees on the selection itself, since one regressor in or out moves the third
digit.

The size of the effect is the applied point: 1,147 units a day fewer across 265
shops, and the authors extrapolate the daily figure over the chain to more than
4,000 units.

The weights do not bind
-----------------------
Their Table 1 assigns each regressor a penalty weight from its trend type, and
on this panel all four schemes -- the I(0) weight of 1, the driftless-``I(1)``
``sqrt(T0)``, the trend weight ``|X_{i,T0}|``, and the per-column augmented
Dickey-Fuller pretest of their Section 4 -- return the same :math:`\\Delta` to
machine precision. The weights enter as a diagonal rescaling of the design and
standardizing the columns is another one, so standardization absorbs whichever
weights were asked for. MATLAB's ``lasso`` standardizes by default and the
authors did not turn it off.

The case pins this both ways. ``weights_inert_standardized`` is the spread in
:math:`\\Delta` across the four schemes with standardization on, which is zero;
``delta_unstandardized_level`` is what the trend weights give with it off, which
is -1475.13 against -1612.26 for unit weights, a 9% move on the headline number
and a selected set of 68 regressors against 17. So the weight table is reachable
and does change the answer; it is standardization that decides whether it is
consulted, and the published numbers are the ``w = 1`` column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.masini_common import (
    TREAT_DATE,
    load_retail,
    partial_resampling,
    trend_weights,
    weekday_dummies,
    wlasso,
)

#: The authors' own count for this specification (126 controls + 7 dummies).
N_REGRESSORS = 133


def _design() -> tuple:
    """Ingest through ``dataprep`` and assemble the paper's regressor matrix."""
    from mlsynth.utils.datautils import dataprep

    prepped = dataprep(
        load_retail(),
        unit_id_column_name="municipality",
        time_period_column_name="date",
        outcome_column_name="quantity",
        treatment_indicator_column_name="treat",
    )
    y = np.asarray(prepped["y"], float).ravel()
    donors = np.asarray(prepped["donor_matrix"], float)
    T0 = int(prepped["pre_periods"])
    dates = pd.DatetimeIndex(prepped["time_labels"])

    assert T0 == 120, T0
    assert int(prepped["post_periods"]) == 14, prepped["post_periods"]
    assert dates[T0] == TREAT_DATE, dates[T0]
    assert donors.shape[1] == 126, donors.shape

    X = np.column_stack([donors, weekday_dummies(dates)])
    shops = int(load_retail().query("municipality == 0")["shops"].iloc[0])
    return y, X, T0, shops


def _fit_delta(y, X, T0, *, mode="unit", standardize=True):
    w = trend_weights(X[:T0], mode)
    b0, beta, k, _ = wlasso(y[:T0], X[:T0], w, standardize=standardize)
    counterfactual = b0 + X @ beta
    return (float(np.mean(y[T0:] - counterfactual[T0:])), k,
            y[:T0] - counterfactual[:T0], y[T0:] - counterfactual[T0:])


def run() -> dict:
    y, X, T0, shops = _design()

    delta, k, residuals, gaps = _fit_delta(y, X, T0)
    p_square, _, _, _ = partial_resampling(lambda x: np.mean(x ** 2), residuals, gaps)
    p_abs, _, _, _ = partial_resampling(lambda x: np.mean(np.abs(x)), residuals, gaps)

    # Every Table 1 weight scheme, standardized: the spread across them.
    deltas = [_fit_delta(y, X, T0, mode=m)[0]
              for m in ("unit", "sqrt", "level", "auto")]
    spread = float(np.ptp(deltas))

    # The same two extremes with standardization off, where the weights bite.
    delta_raw_unit, k_raw_unit, _, _ = _fit_delta(
        y, X, T0, mode="unit", standardize=False)
    delta_raw_level, k_raw_level, _, _ = _fit_delta(
        y, X, T0, mode="level", standardize=False)

    return {
        "delta": delta,
        "delta_per_shop": delta / shops,
        "n_selected": float(k),
        "n_regressors": float(X.shape[1]),
        "p_square": p_square,
        "p_absolute": p_abs,
        "weights_inert_standardized": spread,
        "delta_unstandardized_unit": delta_raw_unit,
        "delta_unstandardized_level": delta_raw_level,
        "n_selected_unstandardized_level": float(k_raw_level),
        # ordering indicators (1.0 == holds)
        "unstandardized_weights_bite": float(
            abs(delta_raw_level - delta_raw_unit) > 50.0
            and k_raw_level > k_raw_unit),
    }


# The fit is deterministic -- no resampling of the design, no seed -- so these are
# reproducible to the solver's tolerance. The tolerances on the reported cells
# are set by the paper's own rounding: the table prints Delta to the unit and the
# per-shop effect to two decimals, so 0.6 and 0.01 are the widths within which a
# port cannot be distinguished from the published number. The counts are exact.
# The unstandardized cells are this port's measurements and not the paper's, and
# carry a coordinate-descent tolerance.
EXPECTED = {
    "delta": (-1147.29, 0.6),                    # paper -1,147
    "delta_per_shop": (-4.329, 0.01),            # paper -4.33
    "n_selected": (26.0, 0.0),                   # paper 26
    "n_regressors": (133.0, 0.0),                # paper 133
    "p_square": (0.0, 0.0),                      # paper 0
    "p_absolute": (0.0, 0.0),                    # paper 0
    "weights_inert_standardized": (0.0, 1e-8),   # all four schemes coincide
    "delta_unstandardized_unit": (-1612.26, 5.0),
    "delta_unstandardized_level": (-1475.13, 5.0),
    "n_selected_unstandardized_level": (68.0, 2.0),
    "unstandardized_weights_bite": (1.0, 0.0),
}
