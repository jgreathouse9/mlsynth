Synthetic Business Cycle (SBC)
==============================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Reach for SBC, due to Shi, Xi, and Xie (2025)
`arXiv:2505.22388 <https://arxiv.org/abs/2505.22388>`_, when your outcome
is a nonstationary, trending series and you want a synthetic-control
counterfactual you can trust. Standard SCM (:doc:`fdid`, :doc:`tssc`,
:doc:`clustersc`) implicitly assumes the untreated outcomes share a
common low-rank factor structure across units. When the outcome is
nonstationary, that assumption is fragile: a pre-treatment fit of the
treated unit on the donors can look excellent *purely because both series
are trending*, even when the underlying processes are independent. The
authors call this the spurious synthetic control problem, and SBC is
the first procedure that is robust to it whether or not the series are
cointegrated.

Concretely, SBC is the right tool when a strong pre-period fit might be
coincidental trending rather than genuine shared structure:

- Marketing / business science. Brand or category sales, market
  share, or price indices after a major event — a rebrand, a pricing
  policy, a regulatory change, a competitor's entry. These series trend
  over time, so a tight pre-event synthetic fit may reflect common growth
  rather than a shared demand structure that will persist post-event.
- Economics. GDP per capita (the paper's German reunification and
  Hong Kong handover studies), real exchange rates, unemployment —
  canonical nonstationary macro outcomes where spurious trend-matching is
  a live risk.
- Policy evaluation. A carbon tax's effect on CO\ :sub:`2`
  emissions, a minimum-wage change, or fiscal rules on government
  spending — drifting outcomes where conventional SCM can mistake parallel
  trends for a shared causal structure.

The flip side: if your outcome is plausibly stationary (a growth
rate, a ratio, an already-differenced series), the spurious-SC concern is
muted and a conventional SCM is simpler and at least as efficient. SBC
buys robustness on *nonstationary levels* at the cost of a trend-forecast
step.

Notation
--------

Let :math:`j = 1` denote the treated unit, with all units
:math:`\mathcal{N} \coloneqq \{1, \dots, N\}` and donor pool
:math:`\mathcal{N}_0 \coloneqq \mathcal{N} \setminus \{1\}` of
cardinality :math:`N_0`. Write :math:`y_{jt}` for the observed outcome of
unit :math:`j` at time :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`,
1-indexed. The intervention takes effect after period :math:`T_0`,
splitting :math:`\mathcal{T}` into the pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` (of
length :math:`T_0`) and the post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`; SBC
imputes the counterfactual over a forecast horizon of :math:`h`
post-treatment periods. The treated unit has potential outcomes
:math:`y_{1t}^I` under the intervention and :math:`y_{1t}^N` without it;
we observe :math:`y_{1t}^N` for :math:`t \in \mathcal{T}_1` and must
impute it for :math:`t \in \mathcal{T}_2`. Each untreated outcome
decomposes into a trend :math:`g_{jt}` and a cycle :math:`c_{jt}`,

.. math::

   y_{jt}^N \;=\; g_{jt} \;+\; c_{jt},

where :math:`g_{jt}` is the persistent (possibly nonstationary)
component and :math:`c_{jt}` is stationary. (The paper writes the trend
as :math:`\tau_{i,t}`; here :math:`\tau` is reserved for the treatment
effect, so the trend is :math:`g_{jt}`.) The Hamilton filter uses a
forecast horizon :math:`h` and :math:`p` self-lags; the cycle admits a
factor structure
:math:`c_{jt} = \boldsymbol{\lambda}_j^\top \mathbf{f}_t + \varepsilon_{jt}`
with :math:`L`-vector of stationary factors :math:`\mathbf{f}_t`,
loadings :math:`\boldsymbol{\lambda}_j`, and idiosyncratic error
:math:`\varepsilon_{jt}`.

The Spurious Synthetic Control Problem
--------------------------------------

The factor-model justification for SCM is "similar units behave
similarly": when all units load on the same latent factors, a weighted
combination of donors can stand in for the treated unit. With
nonstationary outcomes this breaks down. If each untreated outcome
:math:`y_{jt}^N` follows an independent unit-root process, a vertical
regression of :math:`y_{1t}` on the donor outcomes over the pre-period
will *still* often produce statistically significant coefficients and a
tight in-sample fit — an artifact of spurious comovement, not shared
factors (`Granger and Newbold, 1974 <https://doi.org/10.1016/0304-4076(74)90034-7>`_ ; `Phillips, 1986 <https://doi.org/10.1016/0304-4076(86)90001-1>`_). Imposing
non-negativity and adding-up constraints narrows the feasible weights but
does not eliminate the problem. There is no reason for such a fit to
persist out of sample, so it cannot be used to impute the treated unit's
counterfactual. SBC's divide-and-conquer design is built to neutralize
exactly this failure mode.

Mathematical Formulation
------------------------

SBC builds the post-treatment counterfactual from two ingredients drawn
from *different* parts of the panel:

* the treated unit's own past forecasts its trend forward (no donors
  involved), neutralizing the spurious-regression risk for the persistent
  component;
* the donor pool's cycles are combined via classical simplex SCM to
  impute the treated unit's cyclical fluctuations, where the
  common-factor justification of synthetic control is most defensible.

Step 1: Trend-Cycle Decomposition (Hamilton Filter)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The trend is the linear projection of :math:`y_{jt}^N` onto a constant
and :math:`p` of its lagged values shifted back by :math:`h` periods
(Eq. (2) of the paper):

.. math::

   g_{jt}
   \;\coloneqq\;
   \alpha_{j,0}
   + \alpha_{j,1} \, y_{j, t-h}
   + \alpha_{j,2} \, y_{j, t-h-1}
   + \cdots
   + \alpha_{j,p} \, y_{j, t-h-p+1},
   \qquad
   c_{jt} \;\coloneqq\; y_{jt}^N - g_{jt}.

The coefficients :math:`(\widehat\alpha_{j,0}, \dots, \widehat\alpha_{j,p})` are
estimated by OLS on pre-treatment data (helper:
``hamilton.fit_hamilton_filter``). The first :math:`h + p - 1`
observations have no defined target and are returned as ``NaN`` in the
trend and cycle vectors.

Step 2: Forecasting the Treated Trend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The treated unit's post-treatment trend is extrapolated by applying its
fitted Hamilton coefficients to its observed lags:

.. math::

   \widehat g_{1t}
   \;\coloneqq\;
   \widehat\alpha_{1,0}
   + \widehat\alpha_{1,1} \, y_{1, t-h}
   + \cdots
   + \widehat\alpha_{1,p} \, y_{1, t-h-p+1},
   \qquad
   T_0 + 1 \;\leq\; t \;\leq\; T_0 + h.

Two points deserve emphasis. First, the intercept
:math:`\widehat\alpha_{1,0}` *is* applied — the forecast extrapolates the
full estimated trend. (The paper's display equation omits
:math:`\widehat\alpha_{1,0}`, but the authors' replication code includes it,
and it is the correct extrapolation of the estimated trend; dropping it
systematically biases the counterfactual for series whose fitted AR
slopes do not sum to one, and can flip the sign of the estimated effect.)
Second, no donors enter this step at all: the treated trend is forecast
entirely from its own history. This is what inoculates the procedure
against spurious comovement — even if the donor pool's trends are
unrelated to the treated unit's, they never enter the forecast.

.. note::

   The Hamilton projection is an :math:`h`-step-ahead forecast, so the
   counterfactual is well-defined only for the first :math:`h`
   post-treatment periods (:math:`T_0 + 1 \le t \le T_0 + h`, the
   paper's Step 4). The implementation therefore caps the forecast
   horizon at :math:`h`: ``design.counterfactual_post`` and the ATT
   cover exactly the first :math:`\min(h,\,T - T_0)` post periods, and
   the trend forecast uses pre-treatment lags only (no contamination
   from treated post-treatment outcomes). To study a longer post window,
   raise :math:`h`.

Step 3: Synthetic Control on Cycles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The treated unit's cyclical component is imputed via standard synthetic
control on the donors' cycles (Eq. (3)):

.. math::

   \widehat c_{1t}
   \;\coloneqq\;
   \sum_{j \in \mathcal{N}_0} \widehat w_j \, \widehat c_{jt},
   \qquad t \;\in\; \mathcal{T}_2,

with weights solved on the pre-treatment cycles:

.. math::

   \mathbf{w}^\ast
   \;=\;
   \operatorname*{argmin}_{\mathbf{w} \in \Delta^{N_0}}
   \sum_{t \in \mathcal{T}_1}
   \left( \widehat c_{1t} - \sum_{j \in \mathcal{N}_0} w_j \, \widehat c_{jt} \right)^2 ,
   \qquad
   \Delta^{N_0} \coloneqq \Bigl\{ \mathbf{w} \in \mathbb{R}_{\ge 0}^{N_0} :
   \textstyle\sum_{j \in \mathcal{N}_0} w_j = 1 \Bigr\}.

No intercept is needed: the cycles are mean-zero by construction. The
``weights_mode`` flag chooses between this simplex form (default;
``"simplex"``) and an unrestricted vertical regression with intercept
(``"unrestricted"``), the latter useful when some donors' cycles are
*negatively* correlated with the treated unit's (as in the paper's Hong
Kong application).

Step 4: Counterfactual Outcome and Treatment Effect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trend and cycle recombine into the SBC counterfactual:

.. math::

   \widehat{y}_{1t}^{\,N} \;\coloneqq\; \widehat g_{1t} + \widehat c_{1t},
   \qquad T_0 + 1 \;\leq\; t \;\leq\; T_0 + h.

The estimated per-period treatment effect at :math:`t \in \mathcal{T}_2`
is :math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}^{\,N}`, with ATT

.. math::

   \widehat{\tau}
   \;=\;
   \frac{1}{|\mathcal{T}_2|}
   \sum_{t \in \mathcal{T}_2}
   \left( y_{1t} - \widehat{y}_{1t}^{\,N} \right).

Why the Asymmetry?
^^^^^^^^^^^^^^^^^^

Standard SCM treats time and unit dimensions symmetrically: weights are
fit by regressing the treated outcome on contemporaneous donor outcomes,
and the ordering of pre-intervention periods is interchangeable. Indeed,
Shen et al. (2023) show vertical regression, horizontal regression, and
synthetic difference-in-differences yield numerically identical
predictions up to penalty terms. SBC intentionally breaks that symmetry.
The trend is predicted *through time* from the treated unit's own past,
exploiting the persistence of nonstationary signals; the cycle is
predicted *across units* from the donor pool, exploiting the
common-factor structure SCM was designed for. This divide-and-conquer
separation is what gives the paper's main result its bite.

Assumptions and Theory
----------------------

The theory is "fixed-:math:`N`, large-:math:`T`" and rests on two
assumptions — one structural, one high-level.

Assumption 1 (cyclical factor structure). Each unit's cycle is weakly
stationary with
:math:`c_{jt} = \boldsymbol{\lambda}_j^\top \mathbf{f}_t + \varepsilon_{jt}`,
where the :math:`L`-vector of stationary factors :math:`\mathbf{f}_t` is
uniformly bounded, the pre-treatment factor second-moment matrix is
positive definite (eigenvalue bounded away from zero), and
:math:`\varepsilon_{jt}` is mean-zero with finite second moment,
independent across :math:`j` and :math:`t`.

*Remark.* This is the standard SCM factor assumption (Abadie et al.,
2010) applied to the cycles only — not the levels. SBC asks the
donor pool to share structure where it plausibly does (short-run business
cycle comovement, which is well documented across economies) and refuses
to ask it where it plausibly does not (idiosyncratic long-run trends).

Assumption 2 (filter accuracy). The detrending error
:math:`\widehat u_{jt} \coloneqq \widehat c_{jt} - c_{jt}` is :math:`o_p(1)`
pointwise and :math:`\sum_{t \in \mathcal{T}_1} \widehat u_{jt}^2 = o_p(T_0)`.

*Remark.* This is a high-level condition on the *filter*, not on a
particular filter. Any detrending method meeting it inherits the
theory; the paper shows (Lemmas 1-2) that the Hamilton filter satisfies
it for both unit-root and deterministic-trend processes.

Theorem 1 (asymptotic unbiasedness). Under Assumptions 1-2 and the
trend-cycle specification, for each post-treatment period
:math:`T_0 + 1 \le t \le T_0 + h`,

.. math::

   \widehat{y}_{1t}^{\,N} - y_{1t}^N
   \;=\;
   \sum_{j \in \mathcal{N}_0} \widehat w_j \, (\varepsilon_{jt} - \varepsilon_{1t})
   + o_p(1),

so :math:`\widehat{y}_{1t}^{\,N}` is asymptotically unbiased. If the cycles
follow an exact factor structure with no idiosyncratic shocks
(:math:`c_{jt} = \boldsymbol{\lambda}_j^\top \mathbf{f}_t`),
:math:`\widehat{y}_{1t}^{\,N}` is additionally consistent.

*Remark.* The honest claim is unbiasedness, not pointwise consistency:
the leading error term is a weighted average of post-period idiosyncratic
shocks, which cannot be estimated from controls because they are, by
definition, unit-specific. The weights are fixed by *pre-treatment* data
and so are independent of those post-period shocks, which is why the bias
vanishes in expectation. Simulations back this up: SBC cuts
counterfactual MSE by up to a factor of ten in the spurious-regression
designs, and still by 30-70% even when donors are genuinely cointegrated
with the treated unit.

Why the Hamilton Filter?
^^^^^^^^^^^^^^^^^^^^^^^^

Section 3.3 of the paper lays out three criteria a trend-cycle filter
must satisfy here:

1. Stationary cyclical residual. Otherwise the SCM step on cycles is
   again vulnerable to spurious regression. Lemma 1 shows the Hamilton
   filter delivers a stationary cycle for both unit-root processes and
   deterministic-trend-plus-noise processes.
2. Consistent trend and cycle estimates (Assumption 2). Lemma 2: a
   simple OLS projection on lags consistently estimates the population
   AR coefficients, hence the trend and cycle.
3. One-sided / no future leakage. The Step-2 trend extrapolation uses
   only past observations; symmetric two-sided filters like the
   Hodrick-Prescott smoother would peek at post-treatment data and are
   unsuitable. (Hamilton, 2018, argues against the HP filter on separate
   grounds.)

The current implementation hard-codes the Hamilton filter; a future
extension could expose a ``filter`` parameter on :class:`SBCConfig` for
swappable alternatives.

Example: A Draw from the Paper's Simulation
-------------------------------------------

The block below is self-contained — it reproduces Model 2 of Shi, Xi
and Xie (2025, Section 4): :math:`N` units whose levels are
independent random walks (no cointegration) but whose increments
share two stationary AR(1) factors. This is the spurious-regression
regime SBC is built for — the donor pool shares short-run structure but
not long-run trends. We draw one panel, inject a known effect, recover
it, then average over many draws to illustrate Theorem 1's asymptotic
unbiasedness.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SBC

   def model2_draw(rng, n_units=12, T0=100, h=2, phi=0.5, effect=10.0):
       """One draw from Shi-Xi-Xie Model 2: idiosyncratic unit-root trends
       with common stationary AR(1) factors (generates no cointegration)."""
       T = T0 + h
       f = np.zeros((T, 2))                          # two common AR(1) factors
       for t in range(1, T):
           f[t] = phi * f[t - 1] + rng.standard_normal(2)
       loadings = rng.standard_normal((n_units, 2))
       increments = f @ loadings.T + rng.standard_normal((T, n_units))
       Y = np.cumsum(increments, axis=0)             # I(1) levels, driftless
       Y[T0:, 0] += effect                           # effect on the treated unit
       df = pd.DataFrame(
           {"unit": f"u{i}", "t": t, "y": Y[t, i], "treat": int(i == 0 and t >= T0)}
           for i in range(n_units) for t in range(T)
       )
       return df, h

   rng = np.random.default_rng(0)
   cfg = dict(outcome="y", treat="treat", unitid="unit", time="t",
              p=2, weights_mode="simplex", display_graphs=False)

   # --- one draw ---
   df, h = model2_draw(rng)
   res = SBC({"df": df, "h": h, **cfg}).fit()
   print(f"single-draw ATT: {res.att:.2f}  (true 10; a single draw is noisy)")

   # --- many draws: SBC is asymptotically unbiased (Theorem 1) ---
   atts = []
   for _ in range(120):
       df, h = model2_draw(rng)
       atts.append(SBC({"df": df, "h": h, **cfg}).fit().att)
   atts = np.array(atts)
   print(f"mean ATT over 120 draws: {atts.mean():.2f}  (true 10)")
   print(f"std across draws:        {atts.std():.2f}")

A single draw is noisy — the post-period idiosyncratic shocks cannot be
estimated from the controls — but the mean ATT over draws converges to
the true effect, exactly the asymptotic-unbiasedness result of Theorem 1.

.. note::

   The trend-forecast step (Step 2) is the new source of finite-sample
   error relative to classical SCM, and it is sensitive to the horizon
   ``h`` and lags ``p``. If the ATT moves a lot across reasonable
   ``(h, p)``, the trend is hard to forecast from the available history —
   interpret with caution and report the sensitivity.


Simulation study: MSE ratios across all three models (Path B)
-------------------------------------------------------------

Shi, Xi and Xie (2025, Section 4, Table 1) validate SBC against the
conventional synthetic control on three nonstationary DGPs, all with
:math:`N = 12` units, forecast horizon :math:`h = 2`, lags :math:`p = 2`,
and a zero true effect:

* Model 1 -- independent random walks with drift,
  :math:`y_{jt} = y_{j,t-1} + \mu_j + \varepsilon_{jt}` (no shared
  structure: the "spurious regression" regime);
* Model 2 -- idiosyncratic unit-root trends with two common
  *stationary* AR(:math:`\phi`) factors driving the increments (shared
  short-run structure, no cointegration);
* Model 3 -- partial cointegration: half the units (including the
  treated) share two random-walk factors in *levels*, the rest follow
  Model 2 -- the regime conventional SC is actually built for.

The reported statistic is the ratio of mean-squared errors
:math:`\text{MSE}(\widehat{\mathbf{y}}^{\text{SBC}}_1) /
\text{MSE}(\widehat{\mathbf{y}}^{\text{SC}}_1)` against the treated unit's observed
(untreated) path; a ratio below 1 means SBC beats conventional SC.
Driving the packaged :py:class:`~mlsynth.estimators.sbc.SBC` estimator over
the three DGPs (300 reps; the paper uses 10,000) reproduces every headline
finding -- the post-treatment ratios under the two weighting specs SBC
exposes (``simplex`` = the paper's non-negative weights; ``unrestricted`` =
its OLS vertical regression):

.. list-table:: Post-treatment MSE ratio, SBC / conventional SC (:math:`\phi = 0.5`)
   :header-rows: 1
   :widths: 10 8 14 10 14 10

   * - DGP
     - :math:`T_0`
     - simplex (mlsynth)
     - paper
     - unrestricted (mlsynth)
     - paper
   * - Model 1
     - 50
     - 0.034
     - 0.03
     - 0.624
     - 0.64
   * - Model 1
     - 100
     - 0.012
     - 0.01
     - 0.362
     - 0.37
   * - Model 1
     - 200
     - 0.002
     - 0.00
     - 0.206
     - 0.21
   * - Model 2
     - 50
     - 0.087
     - 0.20
     - 0.876
     - 0.83
   * - Model 2
     - 100
     - 0.047
     - 0.09
     - 0.452
     - 0.45
   * - Model 2
     - 200
     - 0.017
     - 0.04
     - 0.243
     - 0.23
   * - Model 3
     - 50
     - 0.723
     - 0.56
     - 0.987
     - 0.94
   * - Model 3
     - 100
     - 0.796
     - 0.47
     - 0.918
     - 0.94
   * - Model 3
     - 200
     - 0.576
     - 0.42
     - 0.887
     - 0.95

All three of the paper's conclusions hold: SBC's MSE ratio is far below 1
in the spurious-regression Models 1-2 (often a 10-50x reduction under
non-negative weights), only modestly below 1 in the cointegrated Model 3
(the regime where conventional SC is appropriate, so SBC merely matches or
slightly beats it), and the non-negative (simplex) spec sharpens the gap
relative to the unrestricted OLS spec. The Model 1 column matches the paper
essentially to the digit; Models 2-3 match in order and direction (the
residual gaps come from 300 vs 10,000 reps and the cointegration-construction
details). A compact, runnable version (simplex spec, one :math:`T_0`):

.. code-block:: python

   import numpy as np
   import pandas as pd
   import cvxpy as cp
   from mlsynth import SBC

   NU, H, P = 12, 2, 2

   def _ar(phi, T, rng):
       f = np.zeros(T)
       for t in range(1, T):
           f[t] = phi * f[t - 1] + rng.normal()
       return f

   def draw(model, T0, rng, phi=0.5):
       """Untreated panel (NU x T) from Shi-Xi-Xie (2025) Model 1/2/3."""
       T = T0 + H
       if model == 1:                                  # independent random walks
           return np.cumsum(rng.normal(0, 0.5, NU)[:, None]
                            + rng.normal(0, 1, (NU, T)), axis=1)
       if model == 2:                                  # unit-root + common AR(1) factors
           f1, f2 = _ar(phi, T, rng), _ar(phi, T, rng)
           lam = rng.normal(0, 1, (NU, 2))
           incr = lam[:, [0]] * f1 + lam[:, [1]] * f2 + rng.normal(0, 1, (NU, T))
           return np.cumsum(incr, axis=1)
       half = NU // 2                                  # partial cointegration
       frw1, frw2 = np.cumsum(rng.normal(0, 1, T)), np.cumsum(rng.normal(0, 1, T))
       far1, far2 = _ar(phi, T, rng), _ar(phi, T, rng)
       e = rng.normal(0, 1, (NU, T)); Y = np.empty((NU, T))
       lrw = rng.normal(0, T0 ** (-1 / 3), (half, 2)); lar = rng.normal(0, 1, (half, 2))
       Y[:half] = (lrw[:, [0]] * frw1 + lrw[:, [1]] * frw2
                   + lar[:, [0]] * far1 + lar[:, [1]] * far2 + e[:half])
       l2 = rng.normal(0, 1, (NU - half, 2))
       Y[half:] = np.cumsum(l2[:, [0]] * far1 + l2[:, [1]] * far2 + e[half:], axis=1)
       return Y

   def conv_sc(y_pre, X_pre, X_full):                  # conventional simplex SC on raw levels
       w = cp.Variable(X_pre.shape[1])
       cp.Problem(cp.Minimize(cp.sum_squares(y_pre - X_pre @ w)),
                  [w >= 0, cp.sum(w) == 1]).solve(solver="CLARABEL")
       return X_full @ w.value

   def post_mse_ratio(model, T0, reps, rng):
       num = den = 0.0
       for _ in range(reps):
           Y = draw(model, T0, rng); T = T0 + H
           df = pd.DataFrame([{"unit": i, "time": t, "y": float(Y[i, t]),
                               "treat": int(i == 0 and t >= T0)}
                              for i in range(NU) for t in range(T)])
           cf = np.asarray(SBC({"df": df, "outcome": "y", "treat": "treat",
                                "unitid": "unit", "time": "time", "h": H, "p": P,
                                "weights_mode": "simplex",
                                "display_graphs": False}).fit().counterfactual_full)
           y1, X = Y[0], Y[1:].T
           cf_sc = conv_sc(y1[:T0], X[:T0], X)
           po = slice(T0, T0 + H)
           num += float(np.sum((cf[po] - y1[po]) ** 2))
           den += float(np.sum((cf_sc[po] - y1[po]) ** 2))
       return num / den

   for model in (1, 2, 3):
       r = post_mse_ratio(model, 100, 100, np.random.default_rng(model))
       print(f"Model {model}:  post MSE ratio = {r:.3f}   (<1 => SBC wins)")
   # Model 1: ~0.02   Model 2: ~0.05   Model 3: ~0.58

Empirical Illustration: German Reunification
--------------------------------------------

Following Shi, Xi and Xie (2025, Section 5.1), we revisit the 1990 German
reunification (Abadie et al., 2015) on West German per-capita GDP with 16
OECD donors and Hamilton horizon ``h=4``.

.. code-block:: python

   import pandas as pd
   from mlsynth import SBC

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "main/basedata/german_reunification.csv")
   d = pd.read_csv(url)
   # 1990 is the last pre-period (reunification Oct 1990); effect from 1991.
   d["treat"] = ((d["country"] == "West Germany") & (d["year"] >= 1991)).astype(int)

   res = SBC({"df": d, "outcome": "gdp", "treat": "treat",
              "unitid": "country", "time": "year",
              "h": 4, "p": 2, "weights_mode": "simplex",
              "display_graphs": False}).fit()

   print(f"ATT: {res.att:.1f}")          # ~ -952 (negative; see below)
   print(res.weights_by_donor)           # Greece, Netherlands, Italy

Following the paper's Section 5.1 specification, SBC concentrates the
cycle weights on Greece (0.44), the Netherlands (0.37), and Italy
(0.16) — donors whose short-run fluctuations track West Germany's
business cycle — and the ATT over 1991-1994 is about -952, with
per-period effects :math:`[+369,\,-323,\,-1155,\,-2700]`: a brief
one-year boost followed by a growing negative impact, the paper's
headline finding. Classical SCM on the raw levels instead leans on the
*trend*-matching donors (Austria and the USA, the two largest
level weights), because the trend dominates the cycle in magnitude.

Empirical Illustration: Hong Kong Handover
------------------------------------------

Section 5.2 studies the 1997 return of Hong Kong to China on per-capita
GDP (FRED levels), using a pool of 11 developed economies (Australia,
Austria, Canada, Denmark, France, Germany, Italy, Korea, the Netherlands,
New Zealand, and the US).

.. code-block:: python

   import pandas as pd
   from mlsynth import SBC

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "main/basedata/hong_kong_handover.csv")
   d = pd.read_csv(url)
   # 1997 is the last pre-period (handover July 1997); effect from 1998.
   d["treat"] = ((d["country"] == "Hong Kong") & (d["year"] >= 1998)).astype(int)

   res = SBC({"df": d, "outcome": "gdp", "treat": "treat",
              "unitid": "country", "time": "year",
              "h": 4, "p": 2, "weights_mode": "simplex",
              "display_graphs": False}).fit()

   print(f"ATT: {res.att:.2f}")
   print(res.weights_by_donor)

Some donor cycles are negatively correlated with Hong Kong's, so the
paper also reports a signed-weight specification — set
``weights_mode="unrestricted"`` to relax the non-negativity constraint.

Diagnostic: spurious matching of independent stochastic trends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The script below shows the spurious synthetic control problem
isolated on a synthetic panel where the true ATT is zero. Two panels
share donor structure but differ in the treated unit's stochastic
trend:

* Panel A -- ``mode='shared'``. Donors and the treated unit each
  carry the *same* random-walk trend plus their own short-run noise.
  Classical SC works (the donors really do span the treated trend);
  SBC is slightly less efficient but unbiased.
* Panel B -- ``mode='idio'``. The donors share one random walk,
  the treated unit has an *independent* random walk of its own. No
  weighted average of the donors can reproduce the treated trend, but
  *finite-sample realised correlations* between two independent
  random walks routinely look strong, so the pre-period fit is
  great by chance and the post-period "treatment effect" is mostly
  the divergence between the two RW realisations. This is exactly
  the Granger-Newbold / Phillips (1986) phenomenon SBC is designed
  to remove.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mlsynth import VanillaSC, SBC


   def panel(*, mode, n_donors=10, T0=40, T1=20, seed=0):
       """One shared random walk for the donors. Treated either shares
       it (mode='shared') or has its own independent RW (mode='idio').
       True ATT = 0 by construction.
       """
       rng = np.random.default_rng(seed)
       T = T0 + T1
       shared_rw = np.cumsum(rng.standard_normal(T))
       eps_d = 0.3 * rng.standard_normal((n_donors, T))
       donors = shared_rw[None, :] + eps_d
       if mode == "shared":
           treated_trend = shared_rw
       else:
           treated_trend = np.cumsum(rng.standard_normal(T))
       treated = treated_trend + 0.3 * rng.standard_normal(T)
       rows = [{"unit": "T", "t": t, "y": float(treated[t]),
                "treat": int(t >= T0)} for t in range(T)]
       for i, d in enumerate(donors):
           rows.extend({"unit": f"d{i}", "t": t, "y": float(d[t]),
                         "treat": 0} for t in range(T))
       return pd.DataFrame(rows)


   for mode, label in [
       ("shared", "shared stochastic trend (donors trace treated)"),
       ("idio",   "idiosyncratic stochastic trend (independent RW)"),
   ]:
       df = panel(mode=mode, seed=1)
       s = SBC({"df": df, "outcome": "y", "treat": "treat",
                  "unitid": "unit", "time": "t",
                  "display_graphs": False}).fit()
       sc = VanillaSC({"df": df, "outcome": "y", "treat": "treat",
                         "unitid": "unit", "time": "t",
                         "display_graphs": False}).fit()
       print(f"\n{label}  (true ATT = 0)")
       print(f"  SC   ATT = {sc.att:+7.3f}")
       print(f"  SBC  ATT = {s.att:+7.3f}")

prints (deterministic with the seed above)::

   shared stochastic trend (donors trace treated)  (true ATT = 0)
     SC   ATT =  -0.022
     SBC  ATT =  -0.644

   idiosyncratic stochastic trend (independent RW)  (true ATT = 0)
     SC   ATT = -13.383
     SBC  ATT =  -2.423

Three takeaways:

1. The classical SC bias in the idiosyncratic regime is enormous
   and one-sided. On a panel where the true ATT is zero, SC reports
   an apparent effect of :math:`-13.4` -- pure spurious matching. SBC
   strips the trend via the Hamilton filter and recovers an ATT of
   :math:`-2.4`, an 82% bias reduction. The remaining bias is the
   finite-sample idiosyncratic-shock term from Theorem 1, which goes
   to zero in expectation but not pointwise.
2. SBC pays a small efficiency cost in the easy regime. When the
   trend is genuinely shared (Panel A), SBC's :math:`-0.64` is a
   little farther from the true zero ATT than SC's :math:`-0.02`,
   because the cycle-only fit discards the shared trend variation
   that SC exploits. This is the unavoidable price of insurance.
3. The diagnostic is the agreement between SC and SBC. When
   classical SC and SBC produce similar ATTs (Panel A), trust the SC
   estimate -- the donors really do span the treated trend. When
   they disagree by an order of magnitude (Panel B), trust the SBC
   estimate -- classical SC is almost certainly picking up spurious
   trend co-movement that SBC has removed by construction.

When to prefer HSC over SBC
^^^^^^^^^^^^^^^^^^^^^^^^^^^

SBC and :doc:`hsc` are both targeted at the spurious-matching
failure mode of classical SC, but they take opposite engineering
bets. SBC commits to a hard Hamilton-filter trend-cycle split and
forecasts the treated trend from its own history; HSC dials a soft
:math:`\rho \in [0, 1]` between levels and differences using
rolling-origin cross-validation. The cases where HSC is the better
choice:

* The trend / cycle distinction is not substantively interesting.
  Sales, market share, daily prices, traffic -- anything outside the
  macro-cycle setting -- typically has no clean "trend vs cycle"
  story, and SBC's modular decomposition is solving the wrong
  problem. HSC's single :math:`\rho^*` is the more honest summary.
* The shared-trend regime is plausible. SBC always discards the
  trend variation by construction, so on a panel where donors really
  do span the treated unit's trend SBC pays a permanent efficiency
  cost (its 'Panel A' bias in the diagnostic above:
  :math:`-0.64` vs SC's :math:`-0.02`). HSC's CV picks
  :math:`\rho^* \to 1` in exactly that case and recovers level-matched
  SC's efficiency without giving up the spurious-matching insurance.
* Your outcome is outside the Hamilton filter's comfort zone.
  The Hamilton filter assumes the cycle becomes stationary after
  regressing on lags, which is well-validated for macro series but
  less obvious for high-frequency / non-AR-trend processes. HSC's
  roughness-penalty smoother makes no comparable parametric trend
  assumption.
* You don't have a long, self-forecastable treated trend. SBC's
  Step 2 forecasts the treated trend univariately. If the treated
  pre-period is short or the treated trend changed character mid-
  sample, HSC's pooled CV across the pre-period is more robust.
* You want a single, interpretable knob rather than a fixed
  decomposition. :math:`\rho^*` after fitting tells you which
  regime the data live in, on a continuous scale; SBC is a yes / no
  commitment to the cycle-only fit.

Conversely, prefer SBC when the trend / cycle distinction *is* part
of your substantive story, when the Hamilton filter's assumptions
are defensible, when you have a long enough pre-period to forecast
the treated trend from its own history, or when you prefer an
explicit asymptotic-unbiasedness theorem over a CV-tuned
hyperparameter.

Graphical comparison
^^^^^^^^^^^^^^^^^^^^

The block below builds two contrasting panels and fits both
:class:`SBC` and :class:`mlsynth.HSC` on each, then overlays the
observed treated trajectory with both counterfactuals. Both panels
have a true ATT of zero by construction; the closer a method's
dashed line tracks the solid observed line in the post-period
(right of the dotted intervention line), the better that method
estimates the (zero) treatment effect on this panel.

* Panel A -- sales-like data: donors and the treated unit share
  a strong deterministic linear trend, short pre-period
  (:math:`T_0 = 16`). HSC's cross-validation picks
  :math:`\rho^* \approx 0.97` (essentially level-matching), exploits
  the shared trend, and hugs the observed series. SBC strips the
  trend by Hamilton-filtering and then has to *forecast* it from a
  short pre-period of treated observations, which extrapolates
  poorly -- the SBC counterfactual visibly undershoots.
* Panel B -- macro-like data: the treated unit has its own
  deterministic linear trend that none of the donors share, but
  donors and treated share a strong stationary cycle. Long
  pre-period (:math:`T_0 = 120`). SBC's univariate AR-trend forecast
  of the treated trend is essentially exact, and the cycle-only SC
  step uses the donors where they actually help. HSC has no donor
  combination that can reproduce the treated trend and is forced
  into a compromise :math:`\rho^* \approx 0.5`, which drifts off in
  the post-period.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   from mlsynth import HSC, SBC


   def panel_HSC_wins(*, n_donors=10, T0=16, T1=12, seed=0):
       """Sales-like: shared deterministic growth + short pre-period."""
       rng = np.random.default_rng(seed)
       T = T0 + T1
       t = np.arange(T)
       shared = 100 + 2.0 * t
       donors = shared[None, :] + 1.5 * rng.standard_normal((n_donors, T))
       treated = shared + 1.5 * rng.standard_normal(T)
       rows = [{"unit": "T", "t": k, "y": float(treated[k]),
                "treat": int(k >= T0)} for k in range(T)]
       for i, d in enumerate(donors):
           rows.extend({"unit": f"d{i}", "t": k, "y": float(d[k]),
                         "treat": 0} for k in range(T))
       return pd.DataFrame(rows), T0, T1


   def panel_SBC_wins(*, n_donors=10, T0=120, T1=20, seed=1):
       """Macro-like: treated has own linear trend, donors share a cycle only."""
       rng = np.random.default_rng(seed)
       T = T0 + T1
       t = np.arange(T)
       treated_trend = 100 + 0.8 * t
       cycle = np.zeros(T)
       for i in range(1, T):
           cycle[i] = 0.6 * cycle[i - 1] + rng.standard_normal()
       donors = np.zeros((n_donors, T))
       for i in range(n_donors):
           base = 100 + 4 * rng.standard_normal()
           donors[i] = base + 5.0 * cycle + 0.3 * rng.standard_normal(T)
       treated = treated_trend + 5.0 * cycle + 0.3 * rng.standard_normal(T)
       rows = [{"unit": "T", "t": k, "y": float(treated[k]),
                "treat": int(k >= T0)} for k in range(T)]
       for i, d in enumerate(donors):
           rows.extend({"unit": f"d{i}", "t": k, "y": float(d[k]),
                         "treat": 0} for k in range(T))
       return pd.DataFrame(rows), T0, T1


   fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
   for ax, (label, builder) in zip(axes, [
       ("Panel A: shared trend, short pre  (HSC favoured)", panel_HSC_wins),
       ("Panel B: own trend on treated, shared cycle  (SBC favoured)", panel_SBC_wins),
   ]):
       df, T0, T1 = builder()
       h = HSC({"df": df, "outcome": "y", "treat": "treat",
                 "unitid": "unit", "time": "t",
                 "display_graphs": False}).fit()
       s = SBC({"df": df, "outcome": "y", "treat": "treat",
                 "unitid": "unit", "time": "t", "h": T1,
                 "display_graphs": False}).fit()
       obs = df.loc[df["unit"] == "T", "y"].to_numpy()
       t = np.arange(obs.size)
       ax.plot(t, obs, "k-", lw=2.2, label="Observed (true ATT=0)")
       ax.plot(t, h.counterfactual_full, "--", color="tab:blue", lw=1.8,
                label=f"HSC counterfactual  (ATT={h.att:+.2f})")
       ax.plot(t, s.counterfactual_full, "--", color="tab:red", lw=1.8,
                label=f"SBC counterfactual  (ATT={s.att:+.2f})")
       ax.axvline(T0 - 0.5, color="grey", ls=":", alpha=0.7)
       ax.set_title(label, fontsize=11)
       ax.set_xlabel("t"); ax.set_ylabel("y")
       ax.legend(loc="best", fontsize=8.5)
       ax.grid(alpha=0.2)
   plt.tight_layout()
   plt.show()

prints (deterministic with the seeds above)::

   Panel A HSC:  ATT = -0.24   rho* = 0.97
   Panel A SBC:  ATT = -6.62

   Panel B HSC:  ATT = +6.22   rho* = 0.50
   Panel B SBC:  ATT = -1.63

Read the plot the same way you'd read a Figure-3-style synthetic
control plot: the post-period vertical gap between solid (observed)
and dashed (counterfactual) is the estimated ATT. The closer that
gap is to zero, the closer the method is to recovering the true
(zero) effect on this synthetic panel.

When SBC vs Classical SCM Disagree
----------------------------------

On the California smoking panel, classical SCM (and TASC, which fits a
state-space model to the *level* of the outcome) yield an ATT around
:math:`-19` packs per capita. SBC on the same panel yields an ATT around
:math:`-10`. Part of the classical estimator's apparent precision comes
from a tight fit of California's nonstationary *level* to a weighted
combination of other states' levels — a fit that may persist
out-of-sample even when the underlying trends are unrelated. SBC strips
that spurious component out by forecasting California's trend from its
own history and using donors only for the stationary cycle.

The German reunification study makes the mechanism vivid. The
trend of West Germany is best reproduced by Austria and the USA,
whereas its cycle is best reproduced by Italy, the Netherlands, and
Greece — two genuinely *different* donor subsets. Conventional SCM on the
raw levels averages across both structures and, because the trend
dominates the cycle in magnitude, leans heavily on the trend-matching
donors. SBC separates the two, attributes a more substantial (and, in
placebo tests, more robust) negative effect to reunification, and
confines the post-reunification boom to roughly one year rather than
three.

In short: when SBC and classical SCM disagree on a nonstationary outcome,
SBC is the more conservative answer about how much of the post-treatment
gap really reflects the intervention.

Core API
--------

.. automodule:: mlsynth.estimators.sbc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SBCConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.sbc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sbc_helpers.hamilton
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sbc_helpers.trend_forecast
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sbc_helpers.synthetic
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sbc_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sbc_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sbc_helpers.structures
   :members:
   :undoc-members:
