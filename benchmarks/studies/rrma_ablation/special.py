"""The distribution functions the RRMA study needs, in NumPy and ``math``.

Three calls replace ``scipy.stats``:

* ``norm_ppf``, the normal quantile, is Wichura's AS241 (``PPND16``, Applied
  Statistics 37, 1988), the algorithm R's ``qnorm`` uses; ``norm_sf`` and
  ``norm_cdf`` come from ``math.erfc``, so a tail probability is never formed
  as ``1 - (something near 1)``: the subtraction would be exact (Sterbenz), but
  a stored probability in ``[1/2, 1)`` is already off by up to ``2^-54``
  absolute (Russinoff 2022, Lemma 6.45), which a tail below 1e-16 cannot carry.
* ``chi2_cdf``, ``chi2_sf`` and ``chi2_ppf`` rest on the regularized incomplete
  gamma function, by its series when ``x < a + 1`` and by Lentz's continued
  fraction otherwise, each returning the tail it computes accurately with the
  prefactor taken in logs. The quantile solves for the smaller tail, ``P = q``
  when ``q <= 1/2`` and ``Q = 1 - q`` otherwise (``1 - q`` is exact for
  ``1/2 <= q <= 2``, Russinoff 2022, Corollary 4.16), by Newton steps on the
  log of that tail from a Wilson-Hilferty start. A Newton step is taken only
  when it lands inside the bracket and is shorter than half the step before
  last, the halving test of Brent's zeroin, so step lengths contract by a
  fixed factor every two iterations (the condition Nocedal and Wright 2006
  put on a safeguarded contraction, p. 37); otherwise bisection, or doubling
  while no upper end is known. It stops when the step and the log residual
  are both at the level of rounding, or the bracket closes to ``4 eps``
  relative (p. 62). The log residual is ``log1p((f - q) / q)`` near the root,
  since ``log f - log q`` cancels to its last bit there (p. 615), and the step
  uses ``P / pdf``, which can only underflow, not ``pdf / P``, which overflows
  when the root is subnormal.
* ``kendall_pvalue`` is ``cor.test(x, y, method = "kendall")``: the exact null
  distribution below 50 observations without ties, the normal approximation
  with R's tie-corrected variance otherwise. The exact branch counts
  permutations by number of inversions in Python integers and sums the tail it
  needs directly. R forms that tail as ``1 - pkendall(q - 1, n)``, which loses
  every digit near ``1e-16`` (it returns ``-4.4e-16`` for a Basque donor); the
  sum here is a rational number until the final division, so the p-value is
  correctly rounded, within half a unit in the last place. Ties are found by
  ``x_i - x_j == 0``, which holds exactly when ``x_i == x_j`` under IEEE
  gradual underflow (a nonzero difference of doubles is at least the smallest
  subnormal); on hardware that flushes subnormals to zero it does not.

Every function validates its arguments and raises ``ValueError`` outside its
domain; the iterations raise ``RuntimeError`` if they fail to converge, which
the tests make sure does not happen on the domain the study uses.
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import numpy as np

_EPS = np.finfo(float).eps
_TINY = 1e-300

# ---------------------------------------------------------------- the normal


def norm_cdf(z: float) -> float:
    """``P(Z <= z)``, from ``erfc`` so the lower tail keeps its digits."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def norm_sf(z: float) -> float:
    """``P(Z > z)``, from ``erfc`` so the upper tail keeps its digits."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """The normal quantile by Wichura's AS241 (R's ``qnorm``), for ``0 < p < 1``."""
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"norm_ppf needs 0 < p < 1, got {p!r}")
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r
                    + 67265.770927008700853) * r + 45921.953931549871457) * r
                  + 13731.693765509461125) * r + 1971.5909503065514427) * r
                + 133.14166789178437745) * r + 3.387132872796366608)
        den = (((((((r * 5226.495278852545925 + 28729.085735721942674) * r
                    + 39307.89580009271061) * r + 21213.794301586595867) * r
                  + 5394.1960214247511077) * r + 687.1870074920579083) * r
                + 42.313330701600911252) * r + 1.0)
        return q * num / den
    r = p if q < 0 else 1.0 - p                     # exact for 1/2 <= p <= 2 (Sterbenz)
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        num = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r
                    + 0.24178072517745061177) * r + 1.27045825245236838258) * r
                  + 3.64784832476320460504) * r + 5.7694972214606914055) * r
                + 4.6303378461565452959) * r + 1.42343711074968357734)
        den = (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r
                    + 0.0151986665636164571966) * r + 0.14810397642748007459) * r
                  + 0.68976733498510000455) * r + 1.6763848301838038494) * r
                + 2.05319162663775882187) * r + 1.0)
    else:
        r -= 5.0
        num = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r
                    + 0.0012426609473880784386) * r + 0.026532189526576123093) * r
                  + 0.29656057182850489123) * r + 1.7848265399172913358) * r
                + 5.4637849111641143699) * r + 6.6579046435011037772)
        den = (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r
                    + 1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r
                  + 0.0148753612908506148525) * r + 0.13692988092273580531) * r
                + 0.59983220655588793769) * r + 1.0)
    val = num / den
    return -val if q < 0 else val


# ----------------------------------------------------------- the chi-square


def _direct(a: float, x: float) -> bool:
    """Whether ``x^a e^-x / Gamma(a)`` can be formed without overflow, or ``e^-x`` underflowing.

    ``x^a`` may be subnormal: it is then as precise as the result can be.
    """
    return a < 171.0 and x < 708.0 and a * math.log(x) < 709.0


def _prefactor(a: float, x: float) -> float:
    """``x^a e^-x / Gamma(a)``, the factor both expansions share.

    Formed directly unless a piece overflows: through logs, an absolute error
    of ``eps |a log x|`` in the exponent becomes that relative error in the
    tail, 120 eps at ``x = 1e-52`` and 700 at the edge of the normal range
    (Nocedal and Wright 2006, p. 616).
    """
    if _direct(a, x):
        return x ** a * math.exp(-x) / math.gamma(a)
    return math.exp(a * math.log(x) - x - math.lgamma(a))


def _gamma_series(a: float, x: float) -> float:
    """``P(a, x)`` by its power series, for ``x < a + 1``."""
    term = total = 1.0 / a
    ap = a
    for _ in range(100_000):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) <= abs(total) * _EPS:
            return total * _prefactor(a, x)
    raise RuntimeError(f"incomplete gamma series did not converge (a={a}, x={x})")  # pragma: no cover


def _gamma_fraction(a: float, x: float) -> float:
    """``Q(a, x)`` by its continued fraction (modified Lentz), for ``x >= a + 1``."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, 100_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = _TINY if abs(d) < _TINY else d
        c = b + an / c
        c = _TINY if abs(c) < _TINY else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if not math.isfinite(h):  # pragma: no cover - needs d at the _TINY guard, then an overflow
            raise RuntimeError(f"incomplete gamma fraction overflowed (a={a}, x={x})")
        if abs(delta - 1.0) <= _EPS:
            return h * _prefactor(a, x)
    raise RuntimeError(f"incomplete gamma fraction did not converge (a={a}, x={x})")  # pragma: no cover


def _gamma_pq(a: float, x: float) -> tuple:
    """``(P, Q)``: the series below ``x = a + 1``, the continued fraction above.

    The other tail is the complement. Its relative error is amplified by
    ``v / (1 - v)``, 6.4 at ``a = 1, x = 2``; taking each expansion outside its
    region instead was measured and is worse: against ``erfc``, the continued
    fraction for ``Q(1/2, z)`` errs by 5e-15 to 1e-13 at ``z`` from 0.05 to 0.8,
    where ``1 - series`` errs by at most 7e-16.
    """
    if x < a + 1.0:
        p = _gamma_series(a, x)
        return p, 1.0 - p
    q = _gamma_fraction(a, x)
    return 1.0 - q, q


def gamma_p(a: float, x: float) -> float:
    """The regularized lower incomplete gamma ``P(a, x)``."""
    if a <= 0 or x < 0:
        raise ValueError(f"gamma_p needs a > 0 and x >= 0, got a={a}, x={x}")
    if x == 0:
        return 0.0
    return _gamma_pq(a, x)[0]


def gamma_q(a: float, x: float) -> float:
    """The regularized upper incomplete gamma ``Q(a, x) = 1 - P(a, x)``, computed as itself."""
    if a <= 0 or x < 0:
        raise ValueError(f"gamma_q needs a > 0 and x >= 0, got a={a}, x={x}")
    if x == 0:
        return 1.0
    return _gamma_pq(a, x)[1]


def chi2_cdf(x: float, df: float) -> float:
    """``P(chi2_df <= x)``."""
    if df <= 0:
        raise ValueError(f"chi2_cdf needs df > 0, got {df}")
    return 0.0 if x <= 0 else gamma_p(df / 2.0, x / 2.0)


def chi2_sf(x: float, df: float) -> float:
    """``P(chi2_df > x)``, computed as an upper tail."""
    if df <= 0:
        raise ValueError(f"chi2_sf needs df > 0, got {df}")
    return 1.0 if x <= 0 else gamma_q(df / 2.0, x / 2.0)


def _chi2_logpdf(x: float, df: float) -> float:
    a = df / 2.0
    return (a - 1.0) * math.log(x) - x / 2.0 - a * math.log(2.0) - math.lgamma(a)


def _tail_rel_err(a: float, x: float) -> float:
    """A bound on the relative error of the computed tail at ``(a, x)``, in units of the iteration.

    64 eps for the expansions, plus ``eps |a log x - x - lgamma(a)|`` when the
    prefactor has to go through logs.
    """
    err = 64.0 * _EPS
    if not _direct(a, x):
        err += 64.0 * _EPS * (abs(a * math.log(x)) + x + abs(math.lgamma(a)))
    return err


def chi2_ppf(q: float, df: float) -> float:
    """The chi-square quantile: ``x`` with ``P(chi2_df <= x) = q``, for ``0 < q < 1``."""
    q, df = float(q), float(df)
    if not (0.0 < q < 1.0) or not df > 0:
        raise ValueError(f"chi2_ppf needs 0 < q < 1 and df > 0, got q={q!r}, df={df!r}")
    upper = q > 0.5
    target = 1.0 - q if upper else q                # exact when q > 1/2
    tail = chi2_sf if upper else chi2_cdf
    log_target = math.log(target)
    # Wilson-Hilferty start. P(a, z) <= z^a / Gamma(a + 1) for every z, so the
    # root of that leading term is a lower bound on the quantile: the bracket's
    # lower end, and the start when Wilson-Hilferty falls below it. Its exponent
    # carries an absolute error of a few eps (|log q| + |lgamma(a + 1)|) / a,
    # which exp makes relative (702 eps at q = 1.7e-305); the bound is lowered
    # by eight times that so it stays below the root.
    a = df / 2.0
    k = 2.0 / (9.0 * df)
    x = df * (1.0 - k + norm_ppf(q) * math.sqrt(k)) ** 3
    expo = (math.log(q) + math.lgamma(a + 1.0)) / a
    x0 = 2.0 * math.exp(expo - 8.0 * _EPS * (2.0 + (abs(math.log(q)) + abs(math.lgamma(a + 1.0))) / a))
    if not x > x0:
        x = x0
    lo, hi = x0, math.inf
    steps = [math.inf, math.inf]                    # |step| one and two iterations back
    for _ in range(200):
        f = tail(x, df)
        if f == target:
            return x
        # P rises with x and Q falls: keep the root bracketed.
        if (f < target) != upper:
            lo = x
        else:
            hi = x
        if math.isfinite(hi) and hi - lo <= 4.0 * _EPS * hi:
            return 0.5 * (lo + hi)
        x_new = math.nan
        if f > 0:
            rel = (f - target) / target
            dlog = math.log1p(rel) if abs(rel) <= 0.5 else math.log(f) - log_target
            # Newton on log(tail): step = dlog * tail / pdf, signed by the tail's direction
            step = dlog * math.exp(math.log(f) - _chi2_logpdf(x, df)) * (-1.0 if upper else 1.0)
            if math.isfinite(step):
                if abs(step) <= 4.0 * _EPS * x and abs(dlog) <= _tail_rel_err(a, x):
                    return x - step
                x_new = x - step
        if not (lo < x_new < hi) or abs(x_new - x) >= 0.5 * steps[0]:
            x_new = 0.5 * (lo + hi) if math.isfinite(hi) else 2.0 * x
        if x_new == x:                              # no representable point between: x is the root
            return x
        steps = [steps[1], abs(x_new - x)]
        x = x_new
    raise RuntimeError(f"chi2_ppf did not converge (q={q}, df={df})")  # pragma: no cover


# ---------------------------------------------------------------- Kendall


@lru_cache(maxsize=None)
def _inversion_counts(n: int) -> tuple:
    """Permutations of ``n`` items by number of inversions, as exact integers."""
    counts = [1]
    for m in range(2, n + 1):
        width = len(counts) + m - 1
        new = [0] * width
        window = 0
        for k in range(width):
            if k < len(counts):
                window += counts[k]
            if k - m >= 0:
                window -= counts[k - m]
            new[k] = window
        counts = new
    return tuple(counts)


def kendall_upper_tail(q: int, n: int) -> Fraction:
    """``P(T >= q)`` for ``T`` the number of concordant pairs under independence, exactly."""
    counts = _inversion_counts(n)
    q = max(int(q), 0)
    return Fraction(sum(counts[q:]), math.factorial(n))


def _tie_sizes(v: np.ndarray) -> np.ndarray:
    _, counts = np.unique(v, return_counts=True)
    return counts[counts > 1].astype(float)


def kendall_pvalue(x, y) -> float:
    """Two-sided p-value of ``cor.test(x, y, method = "kendall")``."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n != len(y) or n < 2 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("kendall_pvalue needs two finite vectors of equal length >= 2")
    i, j = np.triu_indices(n, 1)
    sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
    S = int(np.sum(sx * sy))
    T0 = n * (n - 1) // 2
    ties = bool(np.any(sx == 0) or np.any(sy == 0))
    if n < 50 and not ties:
        conc = (S + T0) // 2
        tail = kendall_upper_tail(conc, n) if 2 * conc > T0 else kendall_upper_tail(T0 - conc, n)
        return float(min(Fraction(1), 2 * tail))
    tx, ty = _tie_sizes(x), _tie_sizes(y)
    v0 = n * (n - 1) * (2 * n + 5)
    vt = np.sum(tx * (tx - 1) * (2 * tx + 5))
    vu = np.sum(ty * (ty - 1) * (2 * ty + 5))
    v1 = np.sum(tx * (tx - 1)) * np.sum(ty * (ty - 1))
    v2 = np.sum(tx * (tx - 1) * (tx - 2)) * np.sum(ty * (ty - 1) * (ty - 2))
    var = (v0 - vt - vu) / 18.0 + v1 / (2.0 * n * (n - 1)) + v2 / (9.0 * n * (n - 1) * (n - 2))
    if not var > 0:
        raise ValueError("kendall_pvalue is undefined when a vector is constant")
    return math.erfc(abs(S) / math.sqrt(var) / math.sqrt(2.0))
