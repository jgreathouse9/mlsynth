.. _replication-spsc:

SPSC — Single Proxy Synthetic Control (Park & Tchetgen Tchetgen 2025)
=====================================================================

:Estimator: :doc:`../proximal` (method ``"SPSC"``) — :class:`mlsynth.PROXIMAL`
:Source: Park, C., & Tchetgen Tchetgen, E. J. (2025), *"Single Proxy Synthetic
   Control,"* Journal of Causal Inference 13(1), 20230079 [SPSC]_.
:Reference implementation: the authors' R package ``qkrcks0218/SPSC``
   (pinned at commit ``054f1fbb``).
:Replication type: Path B — the authors' interactive-fixed-effects Monte Carlo
   — and cross-validation against the R package on both empirical examples in
   the paper (California / Proposition 99 and the Panic of 1907).
:Status: Verified — the Monte Carlo geometry and both empirical examples
   reproduce the reference value-for-value.

Validation strategy
-------------------

SPSC views the donor units' outcomes as a single error-prone proxy of the
treated unit's treatment-free potential outcome and recovers the synthetic
control by a ridge-regularised GMM. The package exposes the detrend trend
(``detrend.ft``) and the treatment-effect basis (``att.ft``) as free choices;
mlsynth surfaces both through ``spsc_detrend_basis`` / ``spsc_detrend_degree``
and ``spsc_att_degree``, so the two empirical examples can be run in exactly the
parameterisation the authors used and checked against the R package run live on
the identical panel.

Path B — interactive-fixed-effects Monte Carlo
----------------------------------------------

The authors' README ships a toy interactive-fixed-effects DGP with a drifting
donor trend. mlsynth reproduces its geometry: both SPSC-DT (detrended) and
SPSC-NoDT recover the true ``ATT = 3`` essentially without bias, while only the
detrended estimator covers near nominal (the un-detrended one under-covers
because its sandwich SE cannot see the trend misspecification). Durable case
``spsc_ifem_mc``.

Cross-validation — California (Proposition 99)
----------------------------------------------

On ``basedata/smoking_data.csv`` (California plus 38 donor states, ``T0 = 18``)
the authors fit a linear detrend with a *linear-in-time* effect basis
(``att.ft = (1, t)``). mlsynth, configured to the same parameterisation
(``spsc_att_degree=1``, ``spsc_detrend_basis="poly"``, ridge ``lambda`` fixed at
:math:`10^{0}`), reproduces the reference effect path and its per-period
standard errors value-for-value:

.. list-table::
   :header-rows: 1
   :widths: 30 24 24

   * - Quantity
     - mlsynth
     - ``qkrcks0218/SPSC``
   * - effect path (1988 … 2000)
     - :math:`-4.845 \,\dots\, -35.284`
     - :math:`-4.845 \,\dots\, -35.284`
   * - per-period SE
     - :math:`0.0020 \,\dots\, 0.0235`
     - :math:`0.0020 \,\dots\, 0.0235`
   * - average ATT
     - :math:`-20.06`
     - :math:`-20.06`

The case runs the R package live and asserts the path and SE agree to solver
tolerance (``path_vs_ref`` and ``se_vs_ref`` are zero to five digits). Durable
case ``spsc_prop99``.

Cross-validation — Panic of 1907
--------------------------------

The paper's Example 2 takes the treated unit to be the average log stock price
of the two trusts the Panic struck — the Knickerbocker Trust and the Trust
Company of America — against a pool of trusts conjectured immune. On
``basedata/trust.dta`` (Knickerbocker = ID 34, Trust Co. of America = ID 57,
``normal`` trusts as donors, Panic after period 229) mlsynth's SPSC reproduces
the R package on the identical averaged-treated panel, with the ridge ``lambda``
fixed at :math:`10^{-2}` for a deterministic live run:

.. list-table::
   :header-rows: 1
   :widths: 30 24 24

   * - Variant
     - mlsynth ATT
     - ``qkrcks0218/SPSC``
   * - SPSC-NoDT
     - :math:`-0.8129`
     - :math:`-0.8129`
   * - SPSC-DT
     - :math:`-0.8035`
     - :math:`-0.8035`

Durable case ``spsc_panic``. Both empirical cases skip gracefully when
``Rscript`` or the SPSC clone is unavailable.
