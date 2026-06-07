Spatial Synthetic Difference-in-Differences (SpSyDiD)
======================================================

.. currentmodule:: mlsynth

Overview
--------

SpSyDiD (Serenini, R., & Masek, F. (2024). *"Spatial Synthetic
Difference-in-Differences,"* SSRN 4736857) extends the Synthetic
Difference-in-Differences (SDID) estimator of
Arkhangelsky-Athey-Hirshberg-Imbens-Wager (2021) with a *spatial
spillover term*. The estimator separates two estimands that standard
SDID confounds when SUTVA is violated by geographic spillovers:

* :math:`\widehat{\tau}` -- the direct ATT on the directly-treated
  units (identical in form to standard SDID).
* :math:`\widehat{\tau}_s` -- the indirect / spillover coefficient per
  unit of neighbour-treatment exposure
  :math:`e_{it} = \sum_{j} w_{ij}\, d_{jt}`.

The implied population ATE follows from Serenini & Masek's eq. 14,

.. math::

   \widehat{\mathrm{ATE}} = \widehat{\tau} \cdot \bigl(1 + \overline{WD}\bigr),

where :math:`\overline{WD}` is the average exposure across the
directly + indirectly treated units in the post-period.

The user supplies a row-standardised :math:`N \times N` spatial
weight matrix :math:`\mathbf{W}`. Helpers in
:mod:`mlsynth.utils.spsydid_helpers.spatial` cover the standard
constructions:

* :func:`knn_weights` -- :math:`k`-nearest neighbours from coordinates.
* :func:`inverse_distance_weights` -- :math:`w_{ij} \propto 1/d_{ij}^p`
  with optional cutoff.
* :func:`contiguity_weights` -- queen / rook contiguity from an
  adjacency dictionary.

When :math:`\mathbf{W} = \mathbf{0}` (no spatial structure) or no donor has any
treated neighbour, SpSyDiD numerically reduces to plain SDID with
:math:`\widehat{\tau}_s = 0`.

When to Use This Method
-----------------------

Every difference-based estimator -- DiD, synthetic control, and plain
:doc:`sdid` -- rests on **SUTVA**: a control unit's outcome is unaffected
by anyone else's treatment. Geography routinely breaks this. When a policy
in the treated region leaks to its neighbours, those neighbours are exactly
the units a synthetic control wants to lean on, and the leakage corrupts
the comparison. Serenini & Masek (2024) make the bias explicit:

* **Spillovers onto units *inside* the donor pool** bias and render
  **inconsistent** the standard SDID ATT -- the synthetic control is built
  from partially-treated donors, so the "untreated" benchmark is itself
  moving with the treatment.
* **Spillovers *outside* the donor pool** leave the ATT identifiable but
  make the population **ATE** unidentified, because the indirect effect on
  exposed-but-excluded units is never measured.

SpSyDiD targets this regime directly. It adds a single **spatial exposure
term** :math:`e_{it} = \sum_{j} w_{ij}\, d_{jt}` to the doubly-weighted SDID
regression, so the estimator returns *two* numbers: the direct ATT
:math:`\widehat{\tau}` (same form as SDID) and the per-exposure indirect
coefficient :math:`\widehat{\tau}_s`. The population ATE then follows from
:math:`\widehat{\mathrm{ATE}} = \widehat{\tau}\,(1 + \overline{WD})` (eq. 14). Relative to
the older Spatial DiD of Delgado & Florax (2015), the synthetic weighting
sharpens identification of the **indirect** effect while keeping SDID's
robustness for the **direct** effect.

Reach for SpSyDiD whenever there is a plausible mechanism for the
treatment to *leak* from the directly-treated units to a subset of
the donor pool through spatial or structural proximity, **and** you can
supply a credible row-standardised weight matrix :math:`\mathbf{W}` encoding that
proximity. Typical examples:

* **Immigration policy with cross-border relocation.** Arizona's
  2007 LAWA legislation directly affected Arizona's noncitizen
  Hispanic population but also displaced workers to neighbouring
  states. SDID alone would either bias the ATT (if you include the
  spillover-affected states as controls) or be unable to estimate
  the spillover at all.
* **State tax changes with cross-border shopping.** A state sales-tax
  increase affects that state's revenue directly *and* leaks via
  cross-border shopping into neighbouring states.
* **Local advertising campaigns** with geographic spillovers across
  DMA boundaries.
* **Vaccine mandates** with cross-state mobility effects.

Do not use SpSyDiD when
^^^^^^^^^^^^^^^^^^^^^^^^

* **SUTVA holds / there is no spillover concern.** With
  :math:`\mathbf{W} = \mathbf{0}` or no treated neighbours, SpSyDiD reduces
  numerically to plain :doc:`sdid` with
  :math:`\widehat{\tau}_s = 0`; the extra exposure column just adds noise. Use
  :doc:`sdid` -- it is faster and more parsimonious.
* **You cannot defend a spatial weight matrix.** The whole identification
  of :math:`\widehat{\tau}_s` runs through :math:`\mathbf{W}`. If proximity is not the
  spillover channel (e.g., interference flows through an unobserved social
  or supply-chain network you cannot encode), a misspecified :math:`\mathbf{W}`
  buys biased indirect effects; consider :doc:`spillsynth`, which models
  spillover through donor membership rather than a fixed geographic kernel.
* **Interference is global or non-local.** SpSyDiD assumes exposure is a
  *local*, distance-decaying function of neighbours' treatment. General
  equilibrium effects that hit every unit equally are absorbed into the
  time effects and cannot be separated.
* **You only need the direct ATT and the donor pool is clean.** If the
  spillover-affected units can simply be *dropped* from the donor pool and
  the indirect effect is not of interest, plain :doc:`sdid` on the pruned
  pool is the simpler honest choice.
* **Distributional questions** (quantiles, tails) -- use :doc:`dsc`; or a
  **single treated unit with no spatial structure** -- use :doc:`tssc` /
  :doc:`fdid`.

Mathematical Formulation
------------------------

Setup
^^^^^

Let :math:`\mathcal{N} \coloneqq \{1, \dots, N\}` index the units and
:math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}` the periods (1-indexed);
the intervention takes effect **after** period :math:`T_0`, so the pre-period
is :math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` and the
post-period is :math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`,
with :math:`T_{\mathrm{post}} \coloneqq T - T_0`. Let :math:`y_{it}` be the
outcome, :math:`d_{it} \in \{0, 1\}` the direct-treatment indicator, and
:math:`\mathbf{W} \in \mathbb{R}_{\ge 0}^{N \times N}` a row-standardised
spatial weight matrix with zero diagonal. The *spillover exposure* of unit
:math:`i` at time :math:`t` is

.. math::

   e_{it} \coloneqq (\mathbf{W}\mathbf{d}_t)_i
         = \sum_{j \in \mathcal{N}} w_{ij}\, d_{jt} \in [0, 1],
   \qquad
   \mathbf{d}_t \coloneqq (d_{1t}, \dots, d_{Nt})^\top .

The estimator auto-partitions :math:`\mathcal{N}` into

* :math:`\mathcal{I}_{\mathrm{tr}}` -- directly treated units
  (:math:`d_{it} = 1` for some :math:`t`), with
  :math:`N_{\mathrm{tr}} \coloneqq |\mathcal{I}_{\mathrm{tr}}|`;
* :math:`\mathcal{I}_{\mathrm{sp}}` -- indirectly treated units
  (:math:`d_{it} = 0` for all :math:`t` but :math:`e_{it} > 0` for some
  :math:`t`), with :math:`N_{\mathrm{sp}} \coloneqq |\mathcal{I}_{\mathrm{sp}}|`;
* :math:`\mathcal{C}` -- pure controls (:math:`d_{it} = 0` and
  :math:`e_{it} = 0` for all :math:`t`).

Only :math:`\mathcal{C}` is used to fit the SDID unit / time weights.

Algorithm
^^^^^^^^^

**Step 1 -- SDID weights from pure controls.** Following
Arkhangelsky et al. (2021), fit the unit weights
:math:`\widehat{\boldsymbol{\omega}} \in \Delta^{|\mathcal{C}|}` and time
weights :math:`\widehat{\boldsymbol{\lambda}} \in \Delta^{T_0}` (each on the
unit simplex :math:`\Delta^{m} \coloneqq \{\mathbf{x} \in
\mathbb{R}_{\ge 0}^{m} : \|\mathbf{x}\|_1 = 1\}`) on :math:`\mathcal{C}` only.
The regularisation parameter is :math:`\zeta \coloneqq
T_{\mathrm{post}}^{1/4} \cdot \mathrm{sd}(\Delta \mathbf{y})`, the standard
deviation of the first-differenced pre-period donor outcomes.

**Step 2 -- assemble the full weight vector.** Set

.. math::

   \widehat{\omega}_i =
   \begin{cases}
       1 / N_{\mathrm{tr}}                & i \in \mathcal{I}_{\mathrm{tr}}, \\
       1 / N_{\mathrm{sp}}                & i \in \mathcal{I}_{\mathrm{sp}}, \\
       \widehat{\omega}_i^{\mathrm{SDID}} & i \in \mathcal{C}.
   \end{cases}

Time weights are SDID-fit for the pre-period and uniform
:math:`1 / T_{\mathrm{post}}` for the post-period.

**Step 3 -- augmented two-way FE WLS regression.** Solve

.. math::

   (\widehat{\tau}, \widehat{\tau}_s, \widehat{\mu},
    \widehat{\boldsymbol{\alpha}}, \widehat{\boldsymbol{\beta}})
   = \arg\min_{\tau, \tau_s, \mu, \boldsymbol{\alpha}, \boldsymbol{\beta}}
     \sum_{i \in \mathcal{N}} \sum_{t \in \mathcal{T}}
     \bigl[
        y_{it} - \mu - \alpha_i - \beta_t
        - \tau\, d_{it} - \tau_s\, e_{it}
     \bigr]^2
     \widehat{\omega}_i\, \widehat{\lambda}_t .

The augmented design jointly recovers the direct effect
:math:`\widehat{\tau}` (the ATT) and the spillover coefficient
:math:`\widehat{\tau}_s`.

**Step 4 -- combine.** With :math:`\overline{WD}` the average exposure over
:math:`\mathcal{I}_{\mathrm{tr}} \cup \mathcal{I}_{\mathrm{sp}}` in the
post-period, the indirect and total effects are

.. math::

   \widehat{\mathrm{AITE}} = \widehat{\tau}_s\, \overline{WD},
   \qquad
   \widehat{\mathrm{ATE}} = \widehat{\tau}\,(1 + \overline{WD}).

Identification assumptions
^^^^^^^^^^^^^^^^^^^^^^^^^^

A1. **No anticipation** -- units do not adjust outcomes in advance
of the treatment.

A2. **Parallel trends** -- in the absence of treatment, treated,
spillover, and control units would have followed similar trends,
conditional on unit and time fixed effects.

A3. **Additivity and linearity of spillovers** -- the potential
outcome of a unit depends linearly and additively on its own
treatment status and the treatment exposure of its neighbours,
captured by :math:`e_{it}`.

A4. **Limited interference** -- spillovers operate exclusively
through the structure defined by the exogenous :math:`\mathbf{W}`. No other
local or global interference mechanisms are assumed.

A5. **Synthetic-control transferability** -- the SDID synthetic
control built on the pure controls also approximates the
counterfactual trajectory for the indirectly-treated units. This
holds when spillover-affected units are spatially / structurally
similar to directly-treated units, which is typically the case in
geographic spillover settings (neighbours of treated states tend to
resemble treated states).

Connection to existing methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* When :math:`\mathbf{W} = \mathbf{0}` (no spatial structure), the spillover
  column vanishes and SpSyDiD reduces to plain SDID with
  :math:`\widehat{\tau}_s = 0`.
* When :math:`\widehat{\omega}_i = 1 / |\mathcal{C}|` for all
  controls (uniform weights), SpSyDiD reduces to the Spatial
  Difference-in-Differences estimator of Delgado & Florax (2015).
* When the panel is balanced + no spillover + non-trivial
  :math:`\mathbf{W}`, SpSyDiD's :math:`\widehat{\tau}` matches SDID's ATT.

Core API
--------

.. automodule:: mlsynth.estimators.spsydid
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SpSyDiDConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.spsydid_helpers.spatial
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spsydid_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spsydid_helpers.weights
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spsydid_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spsydid_helpers.structures
   :members:
   :undoc-members:

Example
-------

A self-contained one-draw Monte Carlo on a :math:`8 \times 8` spatial
grid. Six well-spaced units receive treatment of magnitude
:math:`\tau = 2.0`; their :math:`k = 4` neighbours absorb a spillover
of :math:`\tau_s = 1.0` per unit of exposure. SpSyDiD with the same
:math:`\mathbf{W}` recovers both estimates.

.. code-block:: python

   """One draw of a spatial spillover simulation."""

   import numpy as np
   import pandas as pd

   from mlsynth import SpSyDiD
   from mlsynth.utils.spsydid_helpers.spatial import knn_weights


   # ---------------------------------------------------------------------
   # 1. Lay out an 8x8 grid of units
   # ---------------------------------------------------------------------

   rng = np.random.default_rng(0)
   xs, ys = np.meshgrid(np.arange(8), np.arange(8))
   coords = np.column_stack([xs.flatten(), ys.flatten()])
   N = coords.shape[0]
   T_pre, T_post = 16, 8
   T = T_pre + T_post
   W = knn_weights(coords, k=4, row_standardized=True)


   # ---------------------------------------------------------------------
   # 2. Two-way FE DGP with planted direct + spillover effects
   # ---------------------------------------------------------------------

   tau_true = 2.0
   tau_s_true = 1.0

   unit_fe = rng.standard_normal(N) * 0.5
   time_fe = np.linspace(0.0, 1.0, T)
   Y0 = (
       unit_fe[:, None]
       + time_fe[None, :]
       + rng.standard_normal((N, T)) * 0.2
   )
   D = np.zeros((N, T), dtype=float)
   for u in (0, 7, 24, 39, 56, 63):
       D[u, T_pre:] = 1.0
   Y = Y0 + tau_true * D + tau_s_true * (W @ D)


   # ---------------------------------------------------------------------
   # 3. Long DataFrame
   # ---------------------------------------------------------------------

   rows = [
       {"unit": i, "time": t, "y": float(Y[i, t]), "D": float(D[i, t])}
       for i in range(N)
       for t in range(T)
   ]
   df = pd.DataFrame(rows)


   # ---------------------------------------------------------------------
   # 4. Fit SpSyDiD
   # ---------------------------------------------------------------------

   res = SpSyDiD({
       "df": df,
       "outcome": "y",
       "treat": "D",
       "unitid": "unit",
       "time": "time",
       "spatial_matrix": W,
   }).fit()


   # ---------------------------------------------------------------------
   # 5. Inspect the output
   # ---------------------------------------------------------------------

   print(f"true tau   = {tau_true:+.3f}    tau_hat   = {res.att:+.3f}")
   print(f"true tau_s = {tau_s_true:+.3f}    tau_s_hat = {res.aite:+.3f}")
   print(f"ATE        = {res.ate:+.3f}")
   print(f"partition  : {res.inputs.N_direct} direct, "
         f"{res.inputs.N_spillover} spillover, "
         f"{res.inputs.N_pure} pure controls")
   print(f"mean post-period exposure on treated union = "
         f"{res.metadata['mean_exposure_post_treated']:.3f}")

.. _spsydid-verification:

Verification (Path-B Monte Carlo)
---------------------------------

Serenini & Masek (2024) include an empirical example (the Arizona
2007 LAWA effect on noncitizen Hispanic share, Tables 8-11) but do
**not** release the CPS panel used to construct it -- their public
replication repo
(https://github.com/renanserenini/spatial_SDID) ships only the
simulation code and a BLS unemployment panel for two Monte Carlo
exercises. We therefore satisfy the Path-B contract by reproducing
those two simulation findings against the authors' own driver
(`functions_ssdid.py` in their repo), invoking
``SpSyDiD(config).fit()`` end-to-end on every replication.

The reference panels and adjacency matrices ship with mlsynth in
``basedata/``:

* ``state_unemployment.csv`` -- BLS monthly state unemployment 1976-2014.
* ``US_no_islands_matrix.gal`` -- queen-contiguity W for the 49
  contiguous states.
* ``spsydid_bls_county_subset.csv`` -- the BLS county-employment slice
  (2002-2004, states WY/OR/PA/AL) used in the county-level MC.
* ``spsydid_county_matrices.pkl`` -- per-state county adjacency
  matrices.

State-level Monte Carlo (40 rolling-window replications)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reproduces ``State_Level_Simulations.ipynb``: at each 3-year window
starting in 1975..2014, treat Arkansas (FIPS 5) only and inject
:math:`\text{ATT} = 25\%` of mean unemployment plus
:math:`\rho = 0.8` spillover via the queen-contiguity W. We compare
the authors' reference algorithm against ``SpSyDiD(config).fit()`` on
**the same 40 panels** to test for per-rep agreement.

.. code-block:: text

                          ref-mean    ref-sd    mlsynth-mean    mlsynth-sd
   ATT bias               +0.0187    0.3204         +0.0189        0.3229
   rho bias (tau_s/ATT)   +0.0596    0.9228         +0.0669        0.9965

   per-rep correlation:  ATT 0.9917      rho 0.9948

Both estimators recover the paper's headline finding: the **mean ATT
bias is essentially zero** (~0.019 against an ATT magnitude of ~1.5
percentage points). Per-replication, the two implementations agree to
~0.02 on every panel realisation; the small residual is the
unit-weight assignment for affected rows
(mlsynth: :math:`1/N_{sp}`; reference: mean of treated-unit SDID
weights). Both choices are valid downstream of the SDID weight QPs.

The driver is :file:`examples/spsydid/replicate_state_level_mc.py`;
run with ``python -m examples.spsydid.replicate_state_level_mc
--reps 40``.

County-level Monte Carlo (4 states x 200 reps)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reproduces ``Monte_Carlo_Simulations.ipynb``: for each of WY, OR, PA,
AL, randomly draw 10% of counties as directly treated (multiple
treated units per rep), inject :math:`\text{ATT} = -25\%` of mean
unemployment plus :math:`\rho = 0.5` spillover, fit
``SpSyDiD(config).fit()``, repeat. The four states span 23-67
counties, so the test is whether the SUTVA correction works across
panel sizes.

.. code-block:: text

   state  #counties  #treated   ATT bias mean   (sd)    AITE bias mean   (sd)
   WY        23         2          -0.003     0.260         -0.018     0.143
   OR        36         4          -0.023     0.225         -0.022     0.183
   PA        67         7          +0.034     0.246         +0.062     0.127
   AL        67         7          +0.028     0.228         -0.000     0.139

In every cell the absolute mean ATT bias is below 0.04 against an ATT
magnitude of ~-1.5 -- the spatial-DGP-induced bias of plain SDID is
cleanly removed by the SpSyDiD correction at the county scale.

The driver is :file:`examples/spsydid/replicate_county_level_mc.py`;
run with ``python -m examples.spsydid.replicate_county_level_mc
--reps 200``.

References
----------

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., &
Wager, S. (2021). "Synthetic Difference-in-Differences."
*American Economic Review* 111(12):4088-4118.

Delgado, M. S., & Florax, R. J. G. M. (2015). "Difference-in-Differences
Techniques for Spatial Data: Local Autocorrelation and Spatial
Interaction." *Economics Letters* 137:123-126.

Serenini, R., & Masek, F. (2024). "Spatial Synthetic
Difference-in-Differences." SSRN Working Paper 4736857.
