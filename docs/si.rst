Synthetic Interventions (SI)
============================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Classical synthetic control answers a single counterfactual question: *what
would the treated unit have done under the status quo?* Synthetic Interventions
(SI), due to Agarwal, Shah and Shen (2026) [SI]_, generalises this to **many
interventions at once**: *what would a unit have done under each of several
alternative treatments it did not receive?*

The motivating example is the canonical Proposition 99 tobacco study. In 1989
California enacted a large anti-tobacco program. Over the following decade other
states instead adopted **anti-tobacco programs** (Arizona, Massachusetts,
Oregon, Florida) or **raised cigarette taxes** (Alaska, Hawaii, Maryland,
Michigan, New Jersey, New York, Washington). SI lets you ask not only "what
would California have done under the status quo?" but also "what would
California's cigarette sales have been had it instead *raised taxes* or *run a
program*?" — by borrowing the post-treatment trajectories of the states that
actually did those things.

Reach for SI when:

* **You have multiple, distinct interventions** across units (policies, product
  launches, treatment arms) and want to compare a focal unit's counterfactual
  across them, not just against control.
* **A low-rank factor structure is plausible.** SI rests on a latent-factor
  (interactive fixed-effects) model in which each unit's latent loadings are
  *shared across time and across interventions* — the structural bridge that
  lets weights learned on pre-period control data transfer to post-period
  outcomes under a different intervention.
* **You want valid inference.** The bias-corrected SI-PCR estimator
  (the default here) is **asymptotically normal**, yielding closed-form
  confidence intervals — a feature absent from most SC point estimators.

The flip side: SI assumes **no interference and no dynamic effects**
(Assumption 1), and its factor model assumes each donor pool is observed under a
*single* intervention throughout the post-period. Staggered adoption is not
modelled (the paper only approximates it with a common post-window).

Notation
--------

Index time by :math:`t \in [T]`, units by :math:`i \in [N]`, and interventions
by :math:`d \in \{0, 1, \dots, D\}` (with :math:`d = 0` the status quo). Let
:math:`Y_{ti}(d)` be the potential outcome of unit :math:`i` at time :math:`t`
under intervention :math:`d`. The first :math:`T_0` periods are pre-treatment
(all units under control); the remaining :math:`T_1 = T - T_0` are
post-treatment. :math:`I(d)` is the set of :math:`N_d` units assigned to
intervention :math:`d` after :math:`T_0` (the **donor pool** for :math:`d`).

For a focal target unit :math:`i`, write :math:`Y_{\text{pre}, i} = [Y_{ti0} :
t \le T_0] \in \mathbb{R}^{T_0}` for its pre-period (control) outcomes and
:math:`Y_{\text{pre}, I(d)} \in \mathbb{R}^{T_0 \times N_d}` for the donor pool's
pre-period (control) outcomes. The estimand is

.. math::

   \theta_i(d) \;=\; \tfrac{1}{T_1}\textstyle\sum_{t > T_0} Y_{ti}(d),

the focal unit's average post-period outcome **had it received intervention**
:math:`d`.

Assumptions
-----------

SI inherits the SC identification conditions and adds one structural assumption
that does the real work — invariance of unit factors *across interventions*.
Each is stated with a plain-language remark.

**Assumption 1 (SUTVA / observation pattern).** Pre-treatment, every unit is
observed under control (:math:`Y_{ti0} = Y_{ti}(0)` for :math:`t \le T_0`);
post-treatment, each unit is observed under its assigned intervention
(:math:`Y_{tid} = Y_{ti}(d)` for :math:`t > T_0`, :math:`i \in I(d)`).

*Remark.* This rules out spillovers between units and, with the static factor
model below, dynamic (carry-over) treatment effects. Estimating a
counterfactual is then exactly a **tensor-completion** problem: imputing the
unobserved :math:`(t, i, d)` cells of the potential-outcome tensor.

**Assumption 2 (tensor factor model — the SI assumption).** Each potential
outcome factorises as

.. math::

   Y_{ti}(d) \;=\; \langle u_t(d),\, v_i \rangle + \varepsilon_{ti}(d),
   \qquad u_t(d), v_i \in \mathbb{R}^r,

where the **unit factors** :math:`v_i` are invariant across *both* time and
interventions, and only the time-intervention factors :math:`u_t(d)` depend on
:math:`d`.

*Remark.* This is the crux. In single-intervention SC the unit factors are
invariant across *time*, which lets pre-period weights predict post-period
control outcomes. SI strengthens this to invariance across *interventions*:
weights learned from pre-period **control** data can be applied to post-period
outcomes under a **different** intervention :math:`d`. Conceptually, each unit
has stable intrinsic traits (:math:`v_i`) that any intervention acts on through
:math:`u_t(d)`.

**Assumption 3 (low rank).** The signal :math:`\mathbb{E}[Y_{\text{pre}, I(d)}
\mid \mathcal{E}]` is low rank (rank :math:`r_{\text{pre}} \le r`).

*Remark.* This is what makes the spectral (PCR) denoising step meaningful: the
large singular values of the noisy donor pre-matrix capture the signal, the
small ones capture noise.

**Assumption 4 (span / linear span condition).** The focal unit's factor lies in
the span of the donor pool's factors, so a weight vector :math:`w^{(i,d)}` exists
with :math:`v_i = \sum_{j \in I(d)} w^{(i,d)}_j v_j`.

*Remark.* The multi-intervention analogue of "the treated unit lies in the
convex hull of the donors." A strong pre-period fit (small
:math:`\|Y_{\text{pre},i} - Y_{\text{pre},I(d)} w\|`) is the data-driven sanity
check; a poor fit warns that the span condition or low-rank structure fails.

**Assumption 5 (homoskedastic noise).** The idiosyncratic noise is mean-zero
with common variance :math:`\sigma^2`.

*Remark.* Used only for the variance estimate :math:`\hat\sigma^2` (eq. 14)
behind the confidence interval; the point estimator does not need it.

**Assumptions 6-8 (regularity for normality).** Bounded factors / sub-Gaussian
noise / incoherence-type conditions, plus the rate constraints :math:`N_d < T_0`
and :math:`T_1 = \tilde o(\min\{r_{\text{pre}}^{-3} N_d,\, r_{\text{pre}}^{-1}
\sqrt{T_0}\})`.

*Remark.* These are what Theorem 2 needs for asymptotic normality. The practical
content: the **post-window** :math:`T_1` must be *small* relative to the
pre-window :math:`T_0`, and the target must have a non-vanishing pre-period
signal. The Monte Carlo below shows the CI's coverage degrade exactly when
:math:`T_1` is pushed too large.

Mathematical Formulation
------------------------

The SI Estimator (Proposition 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under Assumptions 1-4 there is a weight vector :math:`w^{(i,d)}` such that the
estimand is recovered from donor-pool outcomes under :math:`d`, and the weights
are identified from **pre-period control** data alone. The SI estimator is two
steps:

.. math::

   \hat w^{(i,d)} \in \arg\min_{w \in \mathcal{W}}
   \; \| Y_{\text{pre}, i} - Y_{\text{pre}, I(d)}\, w \|_2^2,
   \qquad
   \hat\theta_i(d) = \tfrac{1}{T_1}\sum_{t > T_0}\sum_{j \in I(d)} \hat w^{(i,d)}_j\, Y_{tjd}.

The choice of constraint set :math:`\mathcal{W}` recovers the usual SC variants
(simplex, ridge, lasso, OLS). ``mlsynth`` implements the **PCR** variant.

SI-PCR (eq. 10)
^^^^^^^^^^^^^^^

Let the SVD of the donor pre-matrix be :math:`Y_{\text{pre}, I(d)} = \sum_\ell
\hat s_\ell \hat u_\ell \hat v_\ell^\top`. Keeping the top :math:`k` components,

.. math::

   \hat w^{(i,d)} \;=\; \Big( \textstyle\sum_{\ell=1}^{k} (1/\hat s_\ell)\,
   \hat v_\ell \hat u_\ell^\top \Big) Y_{\text{pre}, i}.

SI-PCR projects the donor pre-matrix onto its top-:math:`k` principal subspace
(denoising it under Assumption 3) and regresses the target onto the result. The
rank :math:`k` is chosen by the **Gavish-Donoho** optimal hard threshold. The
default ``rank_method="donoho"`` reproduces the authors' exact rule (the
:math:`\omega(\beta)` approximation evaluated at :math:`\beta = T_0/N_d`);
``"usvt"`` is the same threshold at the canonical ``min/max`` aspect ratio,
``"cumvar"`` keeps a spectral-energy fraction, and ``"fixed"`` takes an explicit
:math:`k`. SI-PCR reuses the HSVT primitives shared with :doc:`clustersc`.

Bias-Corrected SI-PCR and Inference (Section 4.3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plain SI-PCR is consistent (Corollary 1) but converges too slowly for
normality, because spreading weight across many near-collinear donors *dilutes*
the weight norm and deflates the variance term. The **bias-corrected** estimator
fixes this by restricting to a **rank-complete donor subset** :math:`\Omega
\subset I(d)` with :math:`|\Omega| = k` columns of full rank, and fitting by
pseudo-inverse:

.. math::

   \hat w^{(i,d,\Omega)} = (Y^k_{\text{pre}, \Omega})^{\dagger}\, Y_{\text{pre}, i},

where :math:`Y^k` is the rank-:math:`k` approximation. This is a *second* layer
of (structured) sparsity: contributions outside :math:`\Omega` are zeroed by
explicit model selection, concentrating weight along independent directions and
stabilising the weight norm. ``mlsynth`` selects :math:`\Omega` by
column-pivoted QR on the denoised donor matrix.

The estimator is then asymptotically normal (Theorem 2):
:math:`\sqrt{T_1}\,(\hat\theta_i^\Omega(d) - \theta_i(d)) / (\sigma
\|w\|_2) \to \mathcal{N}(0, 1)`, giving the closed-form confidence interval

.. math::

   \text{CI}(\alpha) = \hat\theta_i^\Omega(d) \;\pm\;
   z_{\alpha/2}\, \hat\sigma\, \frac{\| \hat w^{(i,d,\Omega)} \|_2}{\sqrt{T_1}},
   \qquad
   \hat\sigma^2 = \frac{\| (I - \hat U_k \hat U_k^\top)\, Y_{\text{pre}, i}\|_2^2}{T_0 - k},

where :math:`\hat U_k` are the left singular vectors of the rank-:math:`k`
donor approximation (eq. 14) and the noise estimate is the residual of
regressing the target's pre-period onto the donor subspace.

``mlsynth`` exposes three noise-variance estimators via ``variance``: the
main-text ``"units"`` estimator (eq. 14 above), a ``"time_iv"`` estimator from
the donor post-period residual, and the degrees-of-freedom-weighted ``"double"``
combination (the default, matching the authors' code). The interval can be the
eq.-13 confidence interval (``interval="confidence"``) or the wider prediction
interval (``interval="prediction"``, half-width
:math:`z_{\alpha/2}\hat\sigma\sqrt{1 + \|\hat w\|_2^2}/\sqrt{T_1}`) the case
study uses for coverage validation.

Monte Carlo: Coverage of the Confidence Interval
------------------------------------------------

The block below draws low-rank panels (a focal unit plus a donor pool sharing
:math:`r` latent factors), fits the bias-corrected estimator, and checks whether
the CI covers the *true* (noiseless) counterfactual mean
:math:`\theta_i(d)`. The focal unit receives no genuine effect, so the
counterfactual is its own noiseless post-period mean.

.. code-block:: python

   import numpy as np
   from mlsynth.utils.si_helpers.estimation import bias_corrected_fit

   rng = np.random.default_rng(0)
   N, T0, T1, r, sigma = 10, 80, 4, 3, 1.0     # paper regime: T1 small vs T0

   cov = 0
   reps = 600
   for _ in range(reps):
       T = T0 + T1
       F = rng.normal(0, 1, (T, r))            # bounded (stationary) factors
       lam = rng.normal(0, 1, (N, r))
       L = lam @ F.T
       Y = L + sigma * rng.standard_normal((N, T))
       donor_pre, target_pre = Y[1:, :T0].T, Y[0, :T0]
       omega, w, sig = bias_corrected_fit(donor_pre, target_pre, rank=r)
       theta_hat = (Y[1:, T0:].T[:, omega] @ w).mean()
       theta_true = L[0, T0:].mean()
       half = 1.96 * sig * np.linalg.norm(w) / np.sqrt(T1)
       cov += theta_hat - half <= theta_true <= theta_hat + half
   print(f"95% CI coverage: {cov / reps:.3f}")   # ~0.933

Under the paper's regime (:math:`T_0 = 80`, :math:`T_1 = 4`, bounded factors) the
empirical coverage is **0.933**, close to the nominal 0.95 — the CI is valid.

.. note::

   Coverage degrades sharply when Theorem 2's conditions are violated. Repeating
   the experiment with **random-walk (nonstationary) factors** and a large
   post-window (:math:`T_1 = 10` vs :math:`T_0 = 30`) drops coverage to
   :math:`\approx 0.52`: weight-estimation error multiplied by an unbounded,
   growing post-period signal swamps the variance the CI accounts for. The
   practical lesson mirrors the paper — keep :math:`T_1` **short** relative to
   :math:`T_0`, which is also why the empirical study below fixes a short
   post-window.

Empirical Application: Proposition 99 (California)
--------------------------------------------------

We reproduce the paper's case study (Section 6) on the 50-state cigarette-sales
panel (1970-2015): California (focal) under three interventions — **control**
(38 status-quo states), a **cigarette-tax increase** (Alaska, Hawaii, Maryland,
Michigan, New Jersey, New York, Washington), and an **anti-tobacco program**
(Arizona, Massachusetts, Oregon, Florida). Following the paper, weights are fit
on 1970-1988 (:math:`T_0 = 19`) and the counterfactual is reported over the
common 1999-2002 window (:math:`T_1 = 4`).

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SI

   raw = pd.read_csv(
       "https://raw.githubusercontent.com/jehangiramjad/tslib/"
       "refs/heads/master/tests/testdata/prop99.csv"
   )
   raw = raw[(raw.Year >= 1970) & (raw.Year <= 2015)]
   raw = raw[raw.SubMeasureDesc == "Cigarette Consumption (Pack Sales Per Capita)"]
   d = raw[["LocationDesc", "Year", "Data_Value"]].rename(
       columns={"LocationDesc": "state", "Year": "year", "Data_Value": "cigsale"})
   d = d[d.state != "District of Columbia"]
   d = d[(d.year <= 1988) | ((d.year >= 1999) & (d.year <= 2002))]   # fit + report

   tax = ["Alaska", "Hawaii", "Maryland", "Michigan", "New Jersey", "New York", "Washington"]
   program = ["Arizona", "Massachusetts", "Oregon", "Florida"]  # California is a program state
   treated = set(tax) | set(program) | {"California"}
   d["control"] = (~d.state.isin(treated)).astype(int)
   d["taxes"]   = d.state.isin(tax).astype(int)
   d["program"] = d.state.isin([s for s in program if s != "California"] + ["California"]).astype(int)
   d["Prop99"]  = ((d.state == "California") & (d.year >= 1999)).astype(int)

   res = SI({
       "df": d, "outcome": "cigsale", "unitid": "state", "time": "year",
       "treat": "Prop99", "inters": ["control", "taxes", "program"],
       "interval": "prediction", "display_graphs": True,
   }).fit()

   for iv, arm in res.arms.items():
       lo, hi = arm.cf_mean_ci
       print(f"{iv:>8}: k={arm.selected_rank}  cf={arm.cf_mean:.1f}  95% PI=({lo:.1f}, {hi:.1f})")

The Gavish-Donoho threshold selects **k = 5** for the control donor pool and
**k = 1** for both the tax and program pools — *exactly* the ranks the paper
reports (Section 6.2.1) — and the bias-corrected estimator's prediction interval
matches the published numbers:

.. list-table:: California's counterfactual per-capita sales, 1999-2002 (95% prediction interval)
   :header-rows: 1
   :widths: 22 8 18 22

   * - Intervention
     - k
     - Counterfactual
     - 95% PI
   * - Status quo (control)
     - 5
     - **75.8**
     - (70.9, 80.6)
   * - Tax increase
     - 1
     - **57.5**
     - (48.0, 67.1)
   * - Anti-tobacco program
     - 1
     - **59.1**
     - (49.3, 68.9)

The reading mirrors the paper: California's *observed* 1999-2002 sales (~50
packs) sit below all three counterfactuals, and the **control** counterfactual
(75.8) is far higher than the tax (57.5) or program (59.1) ones — i.e. relative
to having done nothing, Prop 99 cut sales sharply, while relative to a tax or a
program the additional effect is modest. The tax and program counterfactuals
overlap heavily, consistent with the paper's conclusion that the two policy
levers deliver similar trajectories.

Replication against the authors' code (Path A)
----------------------------------------------

Per the project's replication contract, SI is checked **against the authors'
own published code** (``opre.2025.1590.cd``), not merely against the paper's
prose. Running the authors' functions and ``mlsynth``'s ``si_helpers`` on
*identical* inputs gives machine-precision agreement on every primitive, both
simulation studies, and the case-study tables:

.. list-table:: SI Path-A replication: ``mlsynth`` vs. the authors' code
   :header-rows: 1
   :widths: 40 30 30

   * - Quantity
     - Comparison
     - Result
   * - SI-PCR weights (eq. 10)
     - 300 random panels, max\|diff\|
     - ``2.2e-16``
   * - Bias-corrected weights (eq. 12)
     - 300 random panels, max\|diff\|
     - ``0``
   * - Variance estimators (units/time-iv/double)
     - 300 random panels, max\|diff\|
     - ``< 1e-15``
   * - Donoho rank selection
     - 300 random panels
     - ``0`` mismatches
   * - Consistency sim (Sec 5.1), :math:`|\hat\theta-\theta|`
     - :math:`T_0 \in \{40,100,400\}`
     - identical to 4 d.p.
   * - Inference sim (Sec 5.2), 95% coverage
     - :math:`T_0 \in \{80,200,600\}`
     - identical (0.922 / 0.891 / 0.947)
   * - Case study, validation coverage
     - control / taxes / program
     - identical (0.684 / 0.857 / 0.600)
   * - Case study, California counterfactual + PI
     - control / taxes / program
     - identical (max\|diff\| ``0``)

The bridge is design rather than luck: ``mlsynth`` reuses the same HSVT
truncation, the authors' exact Gavish-Donoho rank rule
(``rank_method="donoho"``, :math:`\beta = T_0/N_d`), QR-pivot subset selection,
pseudo-inverse fit, and degrees-of-freedom-weighted variance
(``variance="double"``). The one-time side-by-side harness above ran against the
authors' replication package (``opre.2025.1590.cd``).

The **durable replication does not depend on the authors' code**. Both paths are
reproduced from public data and mlsynth's own DGPs, and locked in as a test
(:mod:`mlsynth.tests.test_si_replication`):

* **Path A (empirical)** loads the vendored public pack-sales panel
  (``basedata/prop99_packsales.csv``) and pins the case-study numbers above —
  the :math:`k = 5/1/1` rank selection, California's counterfactuals
  (75.8 / 57.5 / 59.1), and the validation coverage (26/38, 6/7, 3/5).
* **Path B (Monte Carlo)** reruns the consistency and inference studies on
  mlsynth's own reimplementation of the paper's DGPs
  (:mod:`mlsynth.utils.si_helpers.simulation`), confirming SI-PCR is consistent
  only when the rank condition holds and that the bias-corrected CI's coverage
  rises toward the nominal 95% as :math:`T_0` grows.

.. note::

   The paper does not formally model **staggered adoption**; like the authors,
   ``mlsynth`` approximates it with a common pre-window and a fixed post-window
   (here 1999-2002). Donor states that adopted their policy after 1989 are,
   strictly, under control for part of that window — a limitation the paper
   flags in Section 6.1.

Core API
--------

.. automodule:: mlsynth.estimators.si
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SIConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``SI.fit()`` returns an
:class:`~mlsynth.utils.si_helpers.structures.SIResults`, whose ``arms`` maps each
intervention to an :class:`~mlsynth.utils.si_helpers.structures.SIArm` (donor
weights, the counterfactual, the ATT, the selected rank, and -- under bias
correction -- :math:`\hat\sigma`, the weight norm, and the confidence
intervals), alongside the prepared
:class:`~mlsynth.utils.si_helpers.structures.SIInputs`.

.. automodule:: mlsynth.utils.si_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

.. automodule:: mlsynth.utils.si_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.si_helpers.estimation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.si_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.si_helpers.simulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.si_helpers.plotter
   :members:
   :undoc-members:

References
----------

Agarwal, A., Shah, D., & Shen, D. (2026). "Synthetic Interventions: Extending
Synthetic Controls to Multiple Treatments." *Operations Research*
74(2):840-859. See [SI]_.

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods
for Comparative Case Studies." *Journal of the American Statistical
Association* 105(490):493-505.

Agarwal, A., Shah, D., Shen, D., & Song, D. (2021). "On Robustness of
Principal Component Regression." *Journal of the American Statistical
Association* 116(536):1731-1745.

Gavish, M., & Donoho, D. L. (2014). "The Optimal Hard Threshold for Singular
Values is :math:`4/\sqrt{3}`." *IEEE Transactions on Information Theory*
60(8):5040-5053.
