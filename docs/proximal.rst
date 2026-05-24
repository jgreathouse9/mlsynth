Proximal Inference Synthetic Control (PROXIMAL)
===============================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The synthetic control (SC) method of Abadie and co-authors [ABADIE2010]_
is justified by a latent-factor model: each unit's outcome is driven by a
common, time-varying confounder :math:`\boldsymbol{\lambda}_t` (the
"interactive fixed effect") loaded differently across units. Classical SC
regresses the treated unit's pre-treatment outcomes on the donors' and
takes the fitted weights as the synthetic control. Abadie shows this is
(approximately) unbiased only **as the number of pre-treatment periods
grows without bound**, and even then only when a good pre-treatment fit is
attainable.

That leaves two regimes where classical SC is unreliable, and where
PROXIMAL is the right tool:

1. **Short pre-period / poor pre-fit.** With few pre-treatment periods, or
   when no convex combination of donors closely tracks the treated unit,
   the bias bound does not bite and the OLS/WLS weights are
   *inconsistent* -- the donor outcomes are error-laden proxies of
   :math:`\boldsymbol{\lambda}_t`, so the regressor is correlated with the
   residual (a textbook errors-in-variables problem). The bias does **not**
   vanish as the pre-period grows.

2. **Long or structurally-broken post-period.** When the post-period is
   long or contains trend breaks, extrapolating a pre-period fit forward is
   fragile. If you also observe **surrogates** -- post-treatment series
   predictive of the treatment effect -- PROXIMAL can borrow that
   post-period information to sharpen the estimate, which classical SC
   simply discards.

The fix, due to Shi, Li, Miao, Hu and Tchetgen Tchetgen [ProxSCM]_, is to
stop using *every* control as a regressor. Instead, **split the controls**:
some become **donors** that build the synthetic control, and the rest
become **proxies** (negative controls) that are associated with the units
only through the latent factor :math:`\boldsymbol{\lambda}_t`. The proxies
serve as instruments that purge the measurement error, yielding consistent
weights and valid inference via the generalized method of moments (GMM).
Liu, Tchetgen Tchetgen and Varjão [LiuTchetgenVar]_ extend this to
**surrogates**, time-varying correlates of the causal effect observed
post-treatment.

.. note::

   PROXIMAL is the panel/synthetic-control instance of **proximal causal
   inference**, whose theoretical roots are in the negative-control
   literature -- in particular the double-negative-control identification
   and multiply-robust estimation theory of Shi, Miao, Nelson and Tchetgen
   Tchetgen [ShiNegControl]_. There, a *negative-control exposure* and a
   *negative-control outcome* that share a confounding mechanism with an
   unmeasured confounder identify a causal effect despite that confounding.
   PROXIMAL repurposes the same logic: donor outcomes are negative-control
   *outcomes* and the excluded controls are negative-control *exposures*
   for the latent factor :math:`\boldsymbol{\lambda}_t`.

The Three Methods
-----------------

``PROXIMAL`` runs up to three estimators on the same panel, all reported
side by side so they can be compared:

.. list-table::
   :header-rows: 1
   :widths: 14 40 28

   * - Method
     - What it uses
     - Paper
   * - **PI**
     - Donors + donor proxies; pre-period moments only.
     - Shi et al. [ProxSCM]_
   * - **PIS**
     - Adds surrogates + surrogate proxies; pre *and* post data.
     - Liu et al. [LiuTchetgenVar]_
   * - **PIPost**
     - Surrogates, **post-treatment data only**.
     - Liu et al. [LiuTchetgenVar]_

**PI** always runs. **PIS** and **PIPost** run only when surrogate units
are configured. A striking result of [LiuTchetgenVar]_ is that PIPost can
identify the effect from post-treatment data *alone*, because the treated
outcome decomposes into a latent-factor component (matched by donors) and a
surrogate-driven effect component.

Notation
--------

We index units by :math:`j`, with :math:`j = 0` the sole **treated** unit
and :math:`\mathcal{N} = \{1, \ldots, N\}` the **control** units. A subset
:math:`\mathcal{D} \subseteq \mathcal{N}` is the **donor pool** used to
build the synthetic control; the remaining controls are repurposed as
**proxies**. Time runs over :math:`t \in \{1, \ldots, T\}`, split by the
intervention into a pre-treatment window
:math:`\mathcal{T}_1 = \{1, \ldots, T_0\}` and a post-treatment window
:math:`\mathcal{T}_2 = \{T_0 + 1, \ldots, T\}`; the post-period has
:math:`T - T_0` periods (Shi et al.'s :math:`T_1`). Potential outcomes are
:math:`y^0_{jt}` and :math:`y^1_{jt}`, and we observe

.. math::

   y_{0t} =
   \begin{cases}
       y^0_{0t}, & t \in \mathcal{T}_1, \\
       y^1_{0t}, & t \in \mathcal{T}_2.
   \end{cases}

Stacking the donor pool, let :math:`\mathbf{W}_t \in \mathbb{R}^{|\mathcal{D}|}`
be the donor outcomes at time :math:`t`, with weight vector
:math:`\boldsymbol{\alpha}`. Let :math:`\mathbf{Z}_{0t}` be the **donor
proxies**, :math:`\mathbf{X}_t \in \mathbb{R}^{H}` the **surrogate
outcomes** with coefficients :math:`\boldsymbol{\gamma}`, and
:math:`\mathbf{Z}_{1t}` the **surrogate proxies**. The estimand is the
average treatment effect on the treated,

.. math::

   \tau = \frac{1}{T - T_0} \sum_{t \in \mathcal{T}_2}
       \bigl(y^1_{0t} - y^0_{0t}\bigr).

.. admonition:: Notation bridge

   The source papers write the treated outcome :math:`Y_t`, donors
   :math:`W_t`, donor proxies :math:`Z_{0,t}`, surrogates :math:`X_t`,
   surrogate proxies :math:`Z_{1,t}`, the donor latent factor
   :math:`\lambda_t`, and the effect's latent factor :math:`\rho_t`. We
   keep :math:`\mathbf{W}, \mathbf{Z}_0, \mathbf{X}, \mathbf{Z}_1,
   \boldsymbol{\lambda}, \boldsymbol{\rho}` and write the treated unit as
   :math:`j = 0`.

Why Standard SC Fails Here
--------------------------

Assume the interactive fixed-effects model

.. math::

   y^0_{jt} = \boldsymbol{\mu}_j^\top \boldsymbol{\lambda}_t + \varepsilon_{jt},

where :math:`\boldsymbol{\lambda}_t` is an unobserved common factor and
:math:`\boldsymbol{\mu}_j` a unit-specific loading. A synthetic control
exists if the treated loading is a weighted average of the donor loadings,
:math:`\boldsymbol{\mu}_0 = \sum_{j \in \mathcal{D}} \alpha_j
\boldsymbol{\mu}_j`. Then in the pre-period

.. math::

   y_{0t} = \sum_{j \in \mathcal{D}} \alpha_j y_{jt}
       + \Bigl(\varepsilon_{0t} - \sum_{j \in \mathcal{D}} \alpha_j \varepsilon_{jt}\Bigr).

The donor outcomes :math:`y_{jt}` are **noisy proxies** of
:math:`\boldsymbol{\lambda}_t`: they carry the idiosyncratic errors
:math:`\varepsilon_{jt}`, which also appear in the residual. Regressing
:math:`y_{0t}` on them is therefore an errors-in-variables regression, and
the OLS/WLS weights are inconsistent **even as** :math:`T_0 \to \infty`
(Ferman and Pinto). PROXIMAL breaks this correlation with an instrument.

Mathematical Formulation
------------------------

Proximal Inference (PI)
~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we observe proxies :math:`\mathbf{Z}_{0t}` -- e.g. the outcomes of
controls *excluded* from the donor pool, or contemporaneous covariates --
that are associated with the units only through :math:`\boldsymbol{\lambda}_t`
in the pre-period. Then the pre-period residual
:math:`y_{0t} - \mathbf{W}_t^\top \boldsymbol{\alpha}` is **orthogonal** to
the proxies, giving the moment condition

.. math::

   \mathbb{E}\!\left[\mathbf{Z}_{0t}\bigl(y_{0t} - \mathbf{W}_t^\top
   \boldsymbol{\alpha}\bigr)\right] = 0, \qquad t \in \mathcal{T}_1.

Unlike the OLS normal equation
:math:`\mathbb{E}[\mathbf{W}_t(y_{0t} - \mathbf{W}_t^\top
\boldsymbol{\alpha})] = 0`, this estimating function is **mean-zero at the
truth** because :math:`\mathbf{Z}_{0t}` is uncorrelated with the
measurement error. Solving it by GMM yields a consistent
:math:`\hat{\boldsymbol{\alpha}}`, and the ATT is the mean post-period gap

.. math::

   \hat{\tau} = \frac{1}{T - T_0} \sum_{t \in \mathcal{T}_2}
       \bigl(y_{0t} - \mathbf{W}_t^\top \hat{\boldsymbol{\alpha}}\bigr).

Adding Surrogates (PIS)
~~~~~~~~~~~~~~~~~~~~~~~~~

Surrogates :math:`\mathbf{X}_t` are post-treatment series driven by the
same latent factors :math:`\boldsymbol{\rho}_t` as the treatment effect:

.. math::

   y^1_{0t} - y^0_{0t} = \boldsymbol{\rho}_t^\top \boldsymbol{\theta} + \delta_t,
   \qquad
   \mathbf{X}_t = \boldsymbol{\Phi}^\top \boldsymbol{\rho}_t + \boldsymbol{\epsilon}_{X,t}.

With surrogate proxies :math:`\mathbf{Z}_{1t}` instrumenting
:math:`\mathbf{X}_t`, the effect coefficient
:math:`\boldsymbol{\gamma}` (with :math:`\boldsymbol{\Phi}
\boldsymbol{\gamma} = \boldsymbol{\theta}`) is identified by a second,
post-period moment. The stacked conditions are

.. math::

   \mathbb{E}\!\left[\mathbf{Z}_{0t}\bigl(y_{0t} - \mathbf{W}_t^\top
   \boldsymbol{\alpha}\bigr)\right] = 0,\ t \in \mathcal{T}_1,
   \qquad
   \mathbb{E}\!\left[\mathbf{Z}_{1t}\bigl(y_{0t} - \mathbf{W}_t^\top
   \boldsymbol{\alpha} - \mathbf{X}_t^\top \boldsymbol{\gamma}\bigr)\right] = 0,\
   t \in \mathcal{T}_2,

and the ATT is :math:`\hat{\tau} = (T - T_0)^{-1} \sum_{t \in \mathcal{T}_2}
\mathbf{X}_t^\top \hat{\boldsymbol{\gamma}}`.

Post-Treatment-Only (PIPost)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the post-period outcome carries both a latent-factor component
(matched by donors) and a surrogate-driven effect component, both
:math:`\boldsymbol{\alpha}` and :math:`\boldsymbol{\gamma}` can be
estimated from a **single post-period IV fit**, using
:math:`(\mathbf{Z}_{0t}, \mathbf{Z}_{1t})` to instrument
:math:`(\mathbf{W}_t, \mathbf{X}_t)`:

.. math::

   \mathbb{E}\!\left[
   \begin{pmatrix} \mathbf{Z}_{0t} \\ \mathbf{Z}_{1t} \end{pmatrix}
   \bigl(y_{0t} - \mathbf{W}_t^\top \boldsymbol{\alpha}
   - \mathbf{X}_t^\top \boldsymbol{\gamma}\bigr)\right] = 0,
   \qquad t \in \mathcal{T}_2.

This is the most economical method -- it needs no pre-period -- but also
the least efficient, since it discards pre-treatment information.

Inference: GMM Sandwich with HAC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each method stacks its moment conditions into :math:`U_t(\theta)` for
parameters :math:`\theta = (\boldsymbol{\alpha}, \boldsymbol{\gamma},
\tau)` and solves the GMM problem
:math:`\hat{\theta} = \arg\min_\theta\, \bar{U}(\theta)^\top \Omega^{-1}
\bar{U}(\theta)`. Standard errors come from the sandwich variance

.. math::

   \mathrm{Cov} = \mathbf{G}^{-1} \boldsymbol{\Omega}
       \bigl(\mathbf{G}^{-1}\bigr)^\top,
   \qquad
   \mathrm{SE}(\hat{\tau}) = \sqrt{\frac{\mathrm{Cov}[-1,-1]}{T}},

where :math:`\mathbf{G}` is the Jacobian of the moment conditions and
:math:`\boldsymbol{\Omega}` is the **heteroskedasticity- and
autocorrelation-consistent (HAC)** long-run variance of the moments,

.. math::

   \boldsymbol{\Omega} = \frac{1}{T} \sum_{\ell=-J}^{J} k(\ell, J)
       \sum_{t} \mathbf{g}_t \mathbf{g}_{t+\ell}^\top,

with :math:`k(\cdot)` the Bartlett kernel and bandwidth
:math:`J = \bigl\lfloor 4 (\,(T - T_0)/100\,)^{2/9} \bigr\rfloor`. (For
PIPost the normalization uses the post-period count :math:`T - T_0` in
place of :math:`T`.) The HAC middle is what makes the intervals valid under
serially correlated errors.

Assumptions
-----------

**Assumption 1 (interactive fixed effects).** The untreated outcome obeys
:math:`y^0_{jt} = \boldsymbol{\mu}_j^\top \boldsymbol{\lambda}_t +
\varepsilon_{jt}` with :math:`\mathbb{E}[\varepsilon_{jt} \mid
\boldsymbol{\lambda}_t] = 0`, and there is no interference (the treated
unit's status does not affect controls).

*Remark.* The latent factor :math:`\boldsymbol{\lambda}_t` is the
unmeasured confounder: it both drives the outcome and is associated with
treatment timing. This is the standard SC data-generating model; PROXIMAL
does not need it to be stationary, so trending or non-stationary factors
are allowed.

**Assumption 2 (existence of a synthetic control).** There exist weights
:math:`\boldsymbol{\alpha}` with :math:`\boldsymbol{\mu}_0 = \sum_{j \in
\mathcal{D}} \alpha_j \boldsymbol{\mu}_j` (and, for surrogates,
:math:`\boldsymbol{\gamma}` with :math:`\boldsymbol{\Phi}
\boldsymbol{\gamma} = \boldsymbol{\theta}`).

*Remark.* A necessary condition is that the donor pool be at least as large
as the number of latent factors (:math:`|\mathcal{D}| \ge \dim
\boldsymbol{\lambda}_t`), and likewise that there be at least as many
surrogates as effect factors. Weights need **not** be non-negative or sum
to one -- the simplex is optional, used only for interpretability or to
avoid extrapolation.

**Assumption 3 (valid proxies).** The proxies satisfy
:math:`\mathbf{Z}_{0t} \perp\!\!\!\perp \{y_{0t}, \mathbf{W}_t\} \mid
\boldsymbol{\lambda}_t` for :math:`t \in \mathcal{T}_1` (and analogously
for :math:`\mathbf{Z}_{1t}` in the post-period).

*Remark.* Proxies must touch the units **only through the latent factor** --
they carry information about :math:`\boldsymbol{\lambda}_t` but have no
direct causal link to the treated outcome. Outcomes of controls excluded
from the donor pool (e.g. units dropped for similar interventions or
spillover risk) and treatment-free contemporaneous covariates are natural
candidates. Proxy choice is a *pre-specified, domain-knowledge* decision,
not a data-driven search.

**Assumption 4 (relevance / completeness).** The cross-moment
:math:`\mathbb{E}[\mathbf{Z}_{0t} \mathbf{W}_t^\top]` has full column rank
(and a completeness condition holds for nonparametric identification).

*Remark.* This is the instrument-relevance condition: the proxies must be
**strongly associated** with the latent factor, so that variation in
:math:`\mathbf{W}_t` is recoverable from variation in
:math:`\mathbf{Z}_{0t}`. It fails precisely when the proxies are unrelated
to :math:`\boldsymbol{\lambda}_t`, in which case they cannot purge the
measurement error.

**Assumption 5 (stationary, weakly dependent errors).** The error processes
are stationary and weakly dependent.

*Remark.* This is weaker than i.i.d. errors: it permits serial correlation,
which is why inference uses the HAC variance rather than a white-noise
formula. The *latent factors themselves* may still be non-stationary.

.. admonition:: Contaminated surrogates

   In practice "pure" surrogates are rare. Often a surrogate is an
   alternative outcome of the treated unit, or the outcome of another
   affected unit, and so is **contaminated** by the donor latent factor
   :math:`\boldsymbol{\lambda}_t` as well as the effect factor
   :math:`\boldsymbol{\rho}_t` (Appendix A.3 of [LiuTchetgenVar]_).
   ``mlsynth`` handles this by residualizing the surrogate outcomes against
   the donor proxies and donor outcomes on the pre-period (a
   confounding-bridge projection) before the surrogate stage, so the
   surrogates used downstream carry the effect signal net of
   :math:`\boldsymbol{\lambda}_t`.

Example
-------

The block below is self-contained: simulate one panel from the surrogate
data-generating process of [LiuTchetgenVar]_ -- two trending donor factors
:math:`\boldsymbol{\lambda}_t`, one effect factor :math:`\boldsymbol{\rho}_t`
with mean one (so the true ATT is :math:`\approx 1`), and **contaminated**
surrogates that load on both -- then fit ``PROXIMAL`` and read off the ATT
and standard error for all three methods.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import PROXIMAL

   rng = np.random.default_rng(4)
   F, T0, T, H = 2, 100, 200, 2            # donor factors, pre, total, surrogates
   post = np.arange(T) >= T0
   noise = 0.3

   lam = np.log(np.arange(1, T + 1))[:, None] + rng.normal(size=(T, F))  # trending factors
   rho = 1.0 + rng.normal(size=T)                                        # effect factor, mean 1
   Theta = np.array([[0.6, 0.4], [0.4, 0.6]])                            # surrogate contamination

   Y = lam.sum(1) + rng.normal(scale=noise, size=T)
   Y[post] += rho[post]                                                  # apply the effect
   true_att = rho[post].mean()

   W  = lam + rng.normal(scale=noise, size=(T, F))     # donor outcomes
   Z0 = lam + rng.normal(scale=noise, size=(T, F))     # donor proxies
   X  = lam @ Theta + np.outer(rho * post, np.ones(H)) + rng.normal(scale=noise, size=(T, H))
   Z1 = np.outer(rho, np.ones(H)) + lam @ Theta + rng.normal(scale=noise, size=(T, H))

   # Long panel: each donor unit carries (outcome=W, donorproxy=Z0); each surrogate
   # unit carries (donorproxy column = surrogate outcome X, surrogatevar = Z1).
   rows = []
   for t in range(T):
       rows.append({"unit": "treated", "time": t, "y": Y[t], "dp": 0.0, "sv": 0.0,
                    "treat": int(post[t])})
       for j in range(F):
           rows.append({"unit": f"donor{j}", "time": t, "y": W[t, j], "dp": Z0[t, j],
                        "sv": 0.0, "treat": 0})
       for k in range(H):
           rows.append({"unit": f"surr{k}", "time": t, "y": 0.0, "dp": X[t, k],
                        "sv": Z1[t, k], "treat": 0})
   df = pd.DataFrame(rows)

   res = PROXIMAL({
       "df": df, "outcome": "y", "treat": "treat", "unitid": "unit", "time": "time",
       "donors": [f"donor{j}" for j in range(F)],
       "surrogates": [f"surr{k}" for k in range(H)],
       "vars": {"donorproxies": ["dp"], "surrogatevars": ["sv"]},
       "display_graphs": False,
   }).fit()

   print(f"true ATT = {true_att:.3f}")
   for name, fit in res.methods.items():
       print(f"{name:6s} ATT = {fit.att:+.3f}  SE = {fit.att_se:.3f}")

A representative run prints (true ATT ≈ 1.05)::

   PI     ATT = +1.001  SE = 0.138
   PIS    ATT = +1.018  SE = 0.129
   PIPost ATT = +1.080  SE = 0.120

``res`` is a
:class:`~mlsynth.utils.proximal_helpers.structures.PROXIMALResults`:
``res.pi`` / ``res.pis`` / ``res.pipost`` hold the per-method
:class:`~mlsynth.utils.proximal_helpers.structures.ProximalMethodFit`
objects, ``res.methods`` maps the names that ran, and convenience accessors
(``res.att``, ``res.att_se``, ``res.donor_weights``,
``res.att_by_method()``) forward to the headline PI fit.

Empirical Illustration: Panic of 1907
--------------------------------------

[LiuTchetgenVar]_ apply the surrogate method to the Panic of 1907, using
data from [fohlin2021]_. The crisis brought down the Knickerbocker Trust, a
major New York bank. We have log stock prices for 59 trusts, with
Knickerbocker as the treated unit. Two other trusts also suffered bank
runs and seven were tied to major firms; dropping one trust missing a
period leaves 49 potential controls. The logged **bid price** of the 49
controls serves as the donor proxy for Knickerbocker's log price -- a
sensible proxy, since the bid reflects macro forces driving the overall
price.

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from mlsynth import PROXIMAL
    import matplotlib
    import os
    from theme import jared_theme

    matplotlib.rcParams.update(jared_theme)

    file_path = os.path.join(os.path.dirname(__file__), '..', 'basedata', 'trust.dta')
    df = pd.read_stata(file_path)
    df = df[df["ID"] != 1]  # Drop the unbalanced unit

    surrogates = df[df['introuble'] == 1]['ID'].unique().tolist()  # affected trusts
    donors = df[df['type'] == "normal"]['ID'].unique().tolist()    # pure controls

    vars = ["bid_itp", "ask_itp"]
    df[vars] = df[vars].apply(np.log)  # log, per the paper
    df['Panic'] = np.where((df['time'] > 229) & (df['ID'] == 34), 1, 0)

    treat, outcome, unitid, time = "Panic", "prc_log", "ID", "date"
    var_dict = {"donorproxies": ["bid_itp"], "surrogatevars": ["ask_itp"]}

    # Donors-only proximal inference (PI)
    res_pi = PROXIMAL({
        "df": df, "treat": treat, "time": time, "outcome": outcome, "unitid": unitid,
        "treated_color": "black", "counterfactual_color": ["blue"],
        "display_graphs": True, "vars": var_dict, "donors": donors,
    }).fit()

    # Adding surrogates (PI, PIS, PIPost)
    res_surr = PROXIMAL({
        "df": df, "treat": treat, "time": time, "outcome": outcome, "unitid": unitid,
        "treated_color": "black", "counterfactual_color": ["blue", "red", "lime"],
        "display_graphs": True, "vars": var_dict, "donors": donors,
        "surrogates": surrogates,  # the affected trusts, repurposed as surrogates
    }).fit()

    print(res_surr.att_by_method())

.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/examples/PROXIMAL/PanicProx.png
   :alt: Proximal Synthetic Control Estimation
   :align: center
   :width: 600px

Using the bid price as a proxy, the synthetic control fits the
pre-intervention series well. The affected trusts -- which would be
*discarded* in a classical SC analysis because they violate the
no-interference assumption -- are instead repurposed as surrogates: they do
not enter the donor pool, but their post-intervention movements help pin
down the latent effect factors. The asking price of those trusts is their
surrogate proxy.

.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/examples/PROXIMAL/PanicSurrogates.png
   :alt: Surrogate Synthetic Control Estimation
   :align: center
   :width: 600px

Even using only post-intervention data (PIPost), the estimate largely
agrees with the donors-only proximal inference.

Empirical Illustration: 1990 German Reunification
-------------------------------------------------

The headline application of [ProxSCM]_ is the classic German reunification
study: West Germany is the treated unit and 16 OECD countries are controls,
on annual per-capita GDP, 1960--2003. Six countries that receive non-zero
weight in a constrained SC (Austria, Italy, Japan, the Netherlands,
Switzerland, the USA) form the **donor pool**; the other ten are used as
**proxies**. The paper's average post-reunification effect is about
**-1709 USD** for PI (and -1719 USD, 90% CI -3669 to -610, for the
simplex-constrained variant), with a GMM 90% confidence interval of about
-2806 to -616 USD for PI -- a negative impact on West Germany's GDP. In the
1975 placebo (falsification) test, the proximal methods produce a much
smaller pre-treatment prediction error than unconstrained OLS (mean
absolute error ≈ 1115 for PI vs. ≈ 3316 for OLS).

Replication Status
------------------

.. note::

   **Reference-code validation (Path A).** ``mlsynth``'s PI, PIS and
   PIPost were checked **value-for-value** against the authors' reference
   implementation (``freshtaste/proximal``) on identical data-generating
   draws. Both the **ATT and the GMM/HAC standard error** match to machine
   precision for all three methods. A coverage Monte Carlo confirms the
   inference is correct: nominal-95% Wald intervals attain ≈ 93.8%
   coverage (PI), identical to the reference -- restored from a 63.8%
   undercoverage caused by an earlier Jacobian-scaling bug in the GMM
   sandwich.

   **Simulation (Path B).** The robustness claim of [LiuTchetgenVar]_ Sec.
   4.1 reproduces: under a trending latent factor
   (:math:`\boldsymbol{\lambda}_t \sim N(\log t, 1)`), classical SC and
   SC-with-surrogates lose all coverage (→ 0%) while PI/PIS/PIPost remain
   near nominal with low MSE; PIS attains the lowest MSE in most cells. See
   *Example* for a one-draw illustration.

   Per the project's replication contract
   (``agents/agents_estimators.md``), PROXIMAL is considered validated on
   the strength of the machine-precision agreement with the reference code
   plus the reproduced simulation behavior.

Core API
--------

.. automodule:: mlsynth.estimators.proximal
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.PROXIMALConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``PROXIMAL.fit()`` returns a
:class:`~mlsynth.utils.proximal_helpers.structures.PROXIMALResults`, whose
``pi`` / ``pis`` / ``pipost`` fields each hold a
:class:`~mlsynth.utils.proximal_helpers.structures.ProximalMethodFit`
(counterfactual, gap, ATT, GMM/HAC standard error, pre/post RMSE, donor
weights) for the methods that ran. The prepared panel is exposed as a
:class:`~mlsynth.utils.proximal_helpers.structures.PROXIMALInputs`.

.. automodule:: mlsynth.utils.proximal_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- pivots the long panel, builds the donor/surrogate
outcome and proxy matrices, residualizes contaminated surrogates, and packs
everything into the typed
:class:`~mlsynth.utils.proximal_helpers.structures.PROXIMALInputs`.

.. automodule:: mlsynth.utils.proximal_helpers.setup
   :members:
   :undoc-members:

The Bartlett kernel and HAC long-run variance used in the GMM sandwich.

.. automodule:: mlsynth.utils.proximal_helpers.inference
   :members:
   :undoc-members:

The three two-stage estimators (PI, PIS, PIPost), each closing with the
GMM/HAC standard error for the ATT.

.. automodule:: mlsynth.utils.proximal_helpers.estimation
   :members:
   :undoc-members:

Drives the methods on a prepared panel and assembles the per-method fits.

.. automodule:: mlsynth.utils.proximal_helpers.orchestration
   :members:
   :undoc-members:

The trajectories-and-gap overlay plot across methods.

.. automodule:: mlsynth.utils.proximal_helpers.plotter
   :members:
   :undoc-members:
