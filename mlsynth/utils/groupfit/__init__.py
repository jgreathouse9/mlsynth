"""The two-group aggregate regression, behind one entry point.

Three estimators here fit the same equation on a pair of group aggregates. FDID's
ADID arm regresses the treated series on the control *mean* (Li and Van den
Bulte's eqn 2.4), TBR and through it TBRMM regress it on the control *sum*
(Kerman, Wang and Vaver's eqn 1), and PANGEO regresses it on the control
aggregate with an optional trend. The two notations are one regression: the
aggregations differ by the group size, so the intercept, the fitted values and
the residual variance are the same numbers and only the slope's units change.

Two of the three also form the prediction variance that sets their interval's
width, independently, in different arithmetic. One definition is enough, and this
is the part where a second one is dangerous: a wrong point estimate usually looks
wrong, and a wrong interval width does not.

    >>> from mlsynth.utils.groupfit import aggregate_group, fit_two_group
    >>> x = aggregate_group(panel, control_units, how="mean")   # doctest: +SKIP
    >>> y = aggregate_group(panel, treated_units, how="sum")    # doctest: +SKIP
    >>> fit = fit_two_group(y[:n_pre], x[:n_pre])               # doctest: +SKIP
    >>> prediction_variance(fit.sums, x[n_pre:].mean())         # doctest: +SKIP

What this package does not decide is what to do when the regressor is constant,
because the three estimators disagree on purpose: one holds the slope at one, one
takes the pseudoinverse because a constant regressor is its zero-cost case, one
refuses the design. Every function here requires an identified design and
:func:`is_identified` is the gate, so that choice stays at the call site.
"""
from .aggregate import AGGREGATIONS, aggregate_group
from .fit import MIN_WINDOW, fit_on_sums, fit_two_group
from .structures import GroupSums, TwoGroupFit
from .sums import group_sums, is_identified
from .variance import prediction_variance, unscaled_cov

__all__ = [
    "AGGREGATIONS",
    "GroupSums",
    "MIN_WINDOW",
    "TwoGroupFit",
    "aggregate_group",
    "fit_on_sums",
    "fit_two_group",
    "group_sums",
    "is_identified",
    "prediction_variance",
    "unscaled_cov",
]
