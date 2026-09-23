r"""Shi & Huang (2023) Table-1 Monte Carlo DGP for the forward-selected PDA.

Reusable simulator behind the ``pda_table1`` Path-B benchmark (forward selection
vs LASSO). Re-implements the four-factor data-generating process of Shi & Huang
(2023), *"Forward-selected panel data approach for program evaluation,"* Journal
of Econometrics 234(2), 512-535, Table 1.

Provenance
----------
The parameterization is traced to the authors' released code, ``zhentaoshi/fsPDA``
at ``5f4542c``: ``simulation/nonsparse/FS.simulation.dense.R`` (the functions
``FS.simulation.dense``, ``DGP.Lambda`` and ``DGP.Delta``) and the
``loading.RData`` that ``Run.simulation.dense.R`` loads before calling them.
Where the paper's prose and that code differ, the code is what Table 1 was run
on, and the code is what this module follows; the two places they differ are
recorded below.

The DGP
-------
One treated unit and ``N`` controls are observed over ``T = T1 + T2`` periods
(the paper sets ``T1 = T2``). Each outcome is a noisy combination of four common
factors plus idiosyncratic noise,

.. math::

   y_{it} = \lambda_i' f_t + \varepsilon_{it},
   \qquad \varepsilon_{it} \sim N(0, 0.5^2),

so the noise has standard deviation ``0.5`` (``sigma.eta = 0.5``), not variance
0.5.

The paper studies two factor structures. Under the **i.i.d.** structure
(``dynamic_factors=False``) factor :math:`\ell` is drawn ``N(0, \ell^2)`` for
:math:`\ell = 1, \dots, 4` -- their ``rnorm(Tn, sd = k * sigma.u)``, so the
fourth factor carries sixteen times the variance of the first. The paper's
Section 4.1 prints this as ``N(0, I^2)``, which is the index :math:`\ell` set in
an upright font. Under the **dynamic** structure the factors carry distinct
serial dependence with unit innovations: ``f1`` i.i.d. ``N(0,1)``; ``f2`` AR(1)
0.9; ``f3`` MA(2) (0.8, 0.4); ``f4`` ARMA(1,1) (0.5, 0.5), giving unconditional
variances ``(1, 5.26, 1.8, 2.33)``.

The factor loadings :math:`\lambda_i \in \mathbb{R}^4` separate the donor pool:
the treated unit and the **first four** controls draw their loadings from
``U(1, 2)`` (the *relevant* donors), while the remaining ``N - 4`` controls draw
from ``U(-h, h)`` (the *irrelevant* donors). The paper's Section 4.1 gives
``h = 0.1``; the committed ``loading.RData`` behind Table 1 has ``h = 0.5``, and
that is the default here (``irrelevant_loading``). The wider block is the
paper's own point -- the irrelevant donors carry real signal, so the regression
is dense and no method can select its way to a sparse truth. Forward selection
recovers a handful of donors; LASSO takes in more.

Their driver loads one fixed ``loading.RData`` and reuses it across every
replication. Pass that matrix as ``loadings=`` to reproduce their cells; leaving
it ``None`` redraws the loadings with each replication, which makes each draw an
independent sample from the DGP.

The post-period treatment effect follows one of the paper's seven shock
processes :math:`\Delta_t` (``shock=`` one of ``"D1"``..``"D7"``), added to the
treated unit over the ``T2`` post periods:

* ``D1`` -- :math:`\Delta_t = 0` (null; the clean size cell);
* ``D2`` -- :math:`\Delta_t \sim N(0, 1)` (null in mean);
* ``D3`` -- :math:`\Delta_t = 0.3\,\Delta_{t-1} + w_t,\; w_t \sim N(0,1)` (null);
* ``D4`` -- :math:`\Delta_t \sim N(0.5, 1)` (power);
* ``D5`` -- :math:`\Delta_t \sim N(1, 1)` (power);
* ``D6`` -- :math:`\Delta_t = 0.35 + 0.3\,\Delta_{t-1} + w_t` (power);
* ``D7`` -- :math:`\Delta_t = 0.7 + 0.3\,\Delta_{t-1} + w_t` (power).

The null is true under ``D1``-``D3`` and false under ``D4``-``D7``. All six
non-degenerate processes are built from one innovation sequence
:math:`w_t` (their ``DGP.Delta`` draws it once), so ``D4`` sits exactly ``0.5``
above ``D2`` period by period and ``D6`` exactly ``0.5`` above ``D3`` -- the
autoregressive columns are burned in, so they start at their stationary mean
:math:`\mu / (1 - \rho)` and not at zero. :func:`simulate_pda_shocks` returns all
seven columns of one draw, which is what a size-and-power table needs; ``shock=``
on :func:`simulate_pda_panel` takes one column of such a draw.

Determinism
-----------
Loadings, factor innovations, and idiosyncratic noise are drawn from the *same*
generator, so a fixed ``seed`` reproduces a draw bit-for-bit. With ``loadings``
left ``None`` they are redrawn alongside the shocks, so each replication is an
independent draw from the DGP; supplying them holds the design matrix fixed
across replications, which is what their driver does and what reproducing Table
1 requires.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

__all__ = ["PDASimSample", "simulate_pda_panel", "simulate_pda_shocks"]

_N_RELEVANT = 4                  # relevant controls (plus the treated unit)
_N_FACTORS = 4
_NOISE_SD = 0.5                  # sigma.eta: idiosyncratic N(0, 0.5^2)
_AR1 = 0.9                        # f2: AR(1)
_MA2 = (0.8, 0.4)                # f3: MA(2)
_ARMA = (0.5, 0.5)               # f4: ARMA(1,1) (ar, ma)
_NULL_SHOCKS = frozenset({"D1", "D2", "D3"})
_SHOCK_COLUMN = {f"D{j + 1}": j for j in range(7)}
#: ``DGP.Delta``'s ``mu`` and ``rho``, in the order it builds its columns
#: (which is D3, D4, D5, D6, D7 -- D2 is the innovation sequence itself).
_SHOCK_MU = (0.0, 0.5, 1.0, 0.35, 0.7)
_SHOCK_RHO = (0.3, 0.0, 0.0, 0.3, 0.3)
_SHOCK_BURN = 1000               # their DGP.Delta burn-in
_FACTOR_BURN = 1000              # their arima.sim n.start


@dataclass(frozen=True)
class PDASimSample:
    """One draw from the Shi & Huang (2023) factor DGP.

    Attributes
    ----------
    df : pd.DataFrame
        Long panel (columns ``unit`` / ``time`` / ``y`` / ``treat``) ready for
        :class:`mlsynth.PDA`. Treated unit ``"treated"``; relevant donors are
        ``c000``-``c003``.
    Y_treated, Y_controls : np.ndarray
        Treated path ``(T,)`` (incl. effect) and control matrix ``(N, T)``.
    relevant_donors : list of str
        Labels of the 4 relevant controls.
    loadings : np.ndarray
        The factor loadings used, shape ``(N + 1, 4)``; row 0 is the treated
        unit and rows ``1..N`` the controls in label order.
    factors : np.ndarray
        The latent factor paths, shape ``(T, 4)``. Recovering them from the
        panel by least squares projects the noise as well, so the residual of
        that fit is short of the true noise by the rank of the loading matrix;
        carrying the paths makes the DGP's second moments checkable exactly.
    T1, T2 : int
        Pre- / post-treatment period counts.
    shock : str
        The ``D1``..``D7`` shock label (or ``"custom"`` for a numeric effect).
    is_null : bool
        Whether the data-generating ATE is zero (size cell).
    """

    df: pd.DataFrame
    Y_treated: np.ndarray
    Y_controls: np.ndarray
    relevant_donors: List[str]
    loadings: np.ndarray
    factors: np.ndarray
    T1: int
    T2: int
    shock: str
    is_null: bool


def _factors(T: int, rng: np.random.Generator, dynamic: bool,
             burn: int = _FACTOR_BURN) -> np.ndarray:
    """Four common factor paths, shape ``(T, 4)``.

    ``dynamic=False`` draws factor ``l`` from ``N(0, l^2)`` for ``l = 1..4``,
    their ``rnorm(Tn, sd = k * sigma.u)``; ``dynamic=True`` returns the
    (i.i.d., AR(1), MA(2), ARMA(1,1)) structure with unit innovations. A burn-in
    washes out the zero initial condition for the serially-dependent factors.
    """
    if not dynamic:
        return rng.standard_normal((T, _N_FACTORS)) * np.arange(
            1.0, _N_FACTORS + 1.0)

    L = T + burn
    v = rng.standard_normal((L, _N_FACTORS))   # innovations
    f = np.zeros((L, _N_FACTORS))
    ar_arma, ma_arma = _ARMA
    m1, m2 = _MA2
    for t in range(L):
        f[t, 0] = v[t, 0]                                          # f1 i.i.d.
        f[t, 1] = (_AR1 * f[t - 1, 1] if t >= 1 else 0.0) + v[t, 1]  # f2 AR(1)
        lag1 = v[t - 1, 2] if t >= 1 else 0.0
        lag2 = v[t - 2, 2] if t >= 2 else 0.0
        f[t, 2] = v[t, 2] + m1 * lag1 + m2 * lag2                 # f3 MA(2)
        prev = f[t - 1, 3] if t >= 1 else 0.0
        vlag = v[t - 1, 3] if t >= 1 else 0.0
        f[t, 3] = ar_arma * prev + v[t, 3] + ma_arma * vlag       # f4 ARMA(1,1)
    return f[burn:]


def simulate_pda_shocks(
    T2: int,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    burn: int = _SHOCK_BURN,
) -> np.ndarray:
    """All seven treatment-shock processes of one draw, shape ``(T2, 7)``.

    Column ``j`` is ``D{j+1}``: ``D1`` is identically zero, ``D2`` is the
    innovation sequence :math:`w_t \\sim N(0,1)` itself, and ``D3``-``D7`` are
    ``mu + rho * lag + w`` for the paper's five ``(mu, rho)`` pairs. Their
    ``DGP.Delta`` draws ``w`` once and builds every column from it, so the seven
    columns of a draw are the seven treatment effects the *same* replication
    would have produced -- which is what lets one fit answer the size and the
    power cells of Table 1 together.

    The autoregressive columns run through ``burn`` extra periods that are then
    discarded, so they enter the returned window at their stationary mean
    ``mu / (1 - rho)``. Without that the first periods sit below it and the
    finite-sample mean of the effect is not the ``mu / (1 - rho)`` the power
    cells are read against.

    Raises
    ------
    ValueError
        If ``T2`` is not positive.
    """
    if T2 <= 0:
        raise ValueError(f"T2 must be positive; got {T2}.")
    if rng is None:
        rng = np.random.default_rng(seed)
    w = rng.standard_normal(T2 + burn)
    out = np.zeros((T2, 7))
    out[:, 1] = w[burn:]
    for k, (mu, rho) in enumerate(zip(_SHOCK_MU, _SHOCK_RHO), start=2):
        if rho == 0.0:
            out[:, k] = mu + w[burn:]
            continue
        d, path = 0.0, np.empty(T2 + burn)
        for t in range(T2 + burn):
            d = mu + rho * d + w[t]
            path[t] = d
        out[:, k] = path[burn:]
    return out


def _shock(name: str, T2: int, rng: np.random.Generator) -> np.ndarray:
    """The post-period treatment shock :math:`\\Delta_t`, length ``T2``."""
    if name not in _SHOCK_COLUMN:
        raise ValueError(f"unknown shock {name!r}; expected D1..D7.")
    return simulate_pda_shocks(T2, rng=rng)[:, _SHOCK_COLUMN[name]]


def simulate_pda_panel(
    N: int = 100,
    T1: int = 100,
    T2: Optional[int] = None,
    shock: str = "D1",
    dynamic_factors: bool = False,
    effect: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    loadings: Optional[np.ndarray] = None,
    irrelevant_loading: float = 0.5,
) -> PDASimSample:
    """Draw one sample from the Shi & Huang (2023) Table-1 factor DGP.

    Parameters
    ----------
    N : int, default 100
        Number of control units (the paper uses ``N = 100``).
    T1 : int, default 100
        Pre-treatment period count.
    T2 : int, optional
        Post-treatment period count. Defaults to ``T1`` (paper sets ``T1=T2``).
    shock : str, default "D1"
        Treatment-shock process, one of ``"D1"``..``"D7"`` (see module docstring).
        Ignored when ``effect`` is given.
    dynamic_factors : bool, default False
        ``False`` -> i.i.d. factors (benchmark default); ``True`` -> the paper's
        AR/MA/ARMA factor structure.
    effect : float, optional
        Constant post-period shift; overrides ``shock`` when set (a null cell at
        ``0.0``, a power cell when positive).
    rng : np.random.Generator, optional
        RNG; takes precedence over ``seed``.
    seed : int, optional
        Convenience seed used to build a default generator when ``rng`` is None.
    loadings : np.ndarray, optional
        Fixed ``(N + 1, 4)`` loading matrix, row 0 the treated unit. Their
        driver loads one ``loading.RData`` and holds it across replications;
        pass it here to reproduce their cells. ``None`` redraws per call.
    irrelevant_loading : float, default 0.5
        Half-width of the ``U(-h, h)`` range the ``N - 4`` irrelevant controls
        draw from. The default is the committed ``loading.RData``'s; the paper's
        Section 4.1 says ``0.1``. Ignored when ``loadings`` is given.

    Raises
    ------
    ValueError
        If ``N``, ``T1`` or ``T2`` is out of range, if ``shock`` is not one of
        ``D1``..``D7``, or if ``loadings`` is not ``(N + 1, 4)``.
    """
    if N <= _N_RELEVANT:
        raise ValueError(f"N must exceed {_N_RELEVANT} relevant controls; got {N}.")
    if T1 <= 0:
        raise ValueError(f"T1 must be positive; got {T1}.")
    if rng is None:
        rng = np.random.default_rng(seed)
    T2 = T1 if T2 is None else T2
    if T2 <= 0:
        raise ValueError(f"T2 must be positive; got {T2}.")
    T = T1 + T2

    f = _factors(T, rng, dynamic=dynamic_factors)            # (T, 4)

    if loadings is None:
        # DGP.Lambda: K + 1 major rows (the treated unit and the relevant
        # controls) stacked on N - K minor rows.
        lam = np.empty((N + 1, _N_FACTORS))
        lam[:_N_RELEVANT + 1] = rng.uniform(
            1.0, 2.0, size=(_N_RELEVANT + 1, _N_FACTORS))
        lam[_N_RELEVANT + 1:] = rng.uniform(
            -irrelevant_loading, irrelevant_loading,
            size=(N - _N_RELEVANT, _N_FACTORS))
    else:
        lam = np.asarray(loadings, dtype=float)
        if lam.shape != (N + 1, _N_FACTORS):
            raise ValueError(
                f"loadings must have shape {(N + 1, _N_FACTORS)} (row 0 the "
                f"treated unit); got {lam.shape}.")
    lam_treated, lam_ctrl = lam[0], lam[1:]

    y_tr = f @ lam_treated + _NOISE_SD * rng.standard_normal(T)
    if effect is not None:
        label, is_null = "custom", (effect == 0.0)
        y_tr[T1:] += float(effect)
    else:
        label, is_null = shock, shock in _NULL_SHOCKS
        y_tr[T1:] += _shock(shock, T2, rng)

    y_ctrl = lam_ctrl @ f.T + _NOISE_SD * rng.standard_normal((N, T))

    rows = [{"unit": "treated", "time": t, "y": float(y_tr[t]),
             "treat": int(t >= T1)} for t in range(T)]
    for i in range(N):
        rows.extend({"unit": f"c{i:03d}", "time": t,
                     "y": float(y_ctrl[i, t]), "treat": 0} for t in range(T))
    df = pd.DataFrame(rows)
    relevant = [f"c{i:03d}" for i in range(_N_RELEVANT)]
    return PDASimSample(df=df, Y_treated=y_tr, Y_controls=y_ctrl,
                        relevant_donors=relevant, loadings=lam, factors=f,
                        T1=T1, T2=T2,
                        shock=label, is_null=is_null)


# --------------------------------------------------------------------------- #
# Li & Bell (2017) DGP3 -- the LASSO-PDA out-of-sample-prediction simulation
# (Table 2: a dense three-factor model with N > T1).
# --------------------------------------------------------------------------- #
_LIBELL_BURN = 100


def _libell_factors(T: int, rng: np.random.Generator) -> np.ndarray:
    """Li & Bell (2017) Eq. 5.1 three factors, shape ``(T, 3)``.

    ``f1`` AR(1) 0.8; ``f2`` ARMA(1,1) (-0.68, 0.8); ``f3`` MA(2) (0.9, 0.4);
    innovations i.i.d. ``N(0, 1)``. A burn-in removes the zero initial state.
    """
    L = T + _LIBELL_BURN
    v = rng.standard_normal((L, 3))
    f = np.zeros((L, 3))
    for t in range(L):
        f[t, 0] = (0.8 * f[t - 1, 0] if t >= 1 else 0.0) + v[t, 0]
        f[t, 1] = ((-0.68 * f[t - 1, 1] if t >= 1 else 0.0)
                   + v[t, 1] + (0.8 * v[t - 1, 1] if t >= 1 else 0.0))
        f[t, 2] = (v[t, 2] + (0.9 * v[t - 1, 2] if t >= 1 else 0.0)
                   + (0.4 * v[t - 2, 2] if t >= 2 else 0.0))
    return f[_LIBELL_BURN:]


def simulate_libell_panel(
    N: int = 31, T1: int = 25, T2: int = 10, sigma2: float = 1.0,
    rng: Optional[np.random.Generator] = None, seed: Optional[int] = None,
) -> pd.DataFrame:
    """Draw one untreated panel from Li & Bell (2017) DGP3 (Eq. 5.2).

    ``y_it^0 = a_i + b_i' f_t + u_it`` with ``a_i = 1``, loadings
    ``b_ji ~ N(1, 1)`` (a *dense* factor model), idiosyncratic
    ``u_it ~ N(0, sigma2)``, and the Eq. 5.1 factors. Unit ``0`` is the
    "treated" unit (no effect is injected -- the case measures out-of-sample
    prediction of its untreated path), treated over the ``T2`` post-periods.
    Returns a long ``unit``/``time``/``y``/``treat`` frame for :class:`mlsynth.PDA`.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    T = T1 + T2
    f = _libell_factors(T, rng)                       # (T, 3)
    a = np.ones(N)
    B = rng.normal(1.0, 1.0, size=(N, 3))             # b_ji ~ N(1, 1)
    u = rng.normal(0.0, np.sqrt(sigma2), size=(N, T))
    Y0 = a[:, None] + B @ f.T + u                      # (N, T)

    rows = [{"unit": "treated" if i == 0 else f"c{i:03d}", "time": t,
             "y": float(Y0[i, t]), "treat": int(i == 0 and t >= T1)}
            for i in range(N) for t in range(T)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Shi & Wang (2024) Section 5.1-5.2 -- the L2-relaxation PDA size/power
# simulation (Table 2: a dense strong-factor model with N = 100).
# --------------------------------------------------------------------------- #
# Innovation SDs that normalise each factor to unit unconditional variance.
_SW_FACTOR_SD = (1.0, 0.19 ** 0.5, (5.0 / 9.0) ** 0.5, (3.0 / 7.0) ** 0.5)
_SW_NOISE_SD = 0.5 ** 0.5          # idiosyncratic N(0, 0.5)
_SW_NULL_SHOCKS = frozenset({"D1", "D2", "D3"})


def _shiwang_factors(T: int, rng: np.random.Generator, burn: int = 200) -> np.ndarray:
    """Shi & Wang (2024) Sec. 5.1 four factors, shape ``(T, 4)``.

    ``f1`` i.i.d.; ``f2`` AR(1) 0.9; ``f3`` MA(2) (0.8, 0.4); ``f4`` ARMA(1,1)
    (0.5, 0.5); innovation variances set so each factor has unit unconditional
    variance. A burn-in removes the zero initial state.
    """
    L = T + burn
    v = rng.standard_normal((L, 4)) * np.asarray(_SW_FACTOR_SD)
    f = np.zeros((L, 4))
    for t in range(L):
        f[t, 0] = v[t, 0]
        f[t, 1] = (0.9 * f[t - 1, 1] if t >= 1 else 0.0) + v[t, 1]
        f[t, 2] = (v[t, 2] + (0.8 * v[t - 1, 2] if t >= 1 else 0.0)
                   + (0.4 * v[t - 2, 2] if t >= 2 else 0.0))
        f[t, 3] = ((0.5 * f[t - 1, 3] if t >= 1 else 0.0) + v[t, 3]
                   + (0.5 * v[t - 1, 3] if t >= 1 else 0.0))
    return f[burn:]


def _shiwang_loadings(n: int, rng: np.random.Generator) -> np.ndarray:
    """Strong loadings ~ Uniform([-0.5,-0.3] U [0.3,0.5]), shape ``(n, 4)``."""
    return rng.choice([-1.0, 1.0], size=(n, 4)) * rng.uniform(0.3, 0.5, size=(n, 4))


def simulate_shiwang_panel(
    N: int = 100, T1: int = 50, T2: Optional[int] = None, shock: str = "D1",
    rng: Optional[np.random.Generator] = None, seed: Optional[int] = None,
):
    """Draw one sample from Shi & Wang (2024) Table-2 (single-treated, strong factors).

    ``y_it = lambda_i' f_t + u_it`` with strong loadings, ``u_it ~ N(0, 0.5)``,
    and unit-variance factors; the treated unit (index 0) gets the post-period
    shock ``Delta_t`` (``D1`` = 0 for size; ``D4`` adds a constant 0.3 for power).
    Returns ``(y_treated, Y_controls, T1)`` -- arrays, since the L2 size study
    drives the estimator at the array level. The null holds for ``D1``-``D3``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    T2 = T1 if T2 is None else T2
    T = T1 + T2
    f = _shiwang_factors(T, rng)                       # (T, 4)
    y = f @ _shiwang_loadings(1, rng)[0] + _SW_NOISE_SD * rng.standard_normal(T)
    Yc = _shiwang_loadings(N, rng) @ f.T + _SW_NOISE_SD * rng.standard_normal((N, T))
    if shock in ("D4", "D5", "D6", "D7", "D8", "D9"):
        const = {"D4": 0.3, "D5": 0.3, "D6": 0.3, "D7": 0.5, "D8": 0.5, "D9": 0.5}[shock]
        y[T1:] += const
    elif shock not in ("D1", "D2", "D3"):
        raise ValueError(f"unknown shock {shock!r}; expected D1..D9.")
    return y, Yc, T1
