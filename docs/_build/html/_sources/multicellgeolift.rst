.. _multicellgeolift:

MULTICELLGEOLIFT — multi-cell GeoLift analysis
==============================================

When to use
-----------

A multi-cell geo experiment runs several treatments at once — different
channels, budgets, or creative strategies — each on its own group of geos
("cells" :math:`A, B, \dots`), all measured against a shared pool of control
geos over the same window. Use ``MULTICELLGEOLIFT`` to measure each cell's
incremental effect and to compare the cells. It is the analysis analogue of
GeoLift's ``GeoLiftMultiCell`` (single-cell measurement is :doc:`geolift`).

Data model
----------

A unit-level cell-membership column plus a treatment-window indicator:

* ``cell_column_name`` — each geo's cell label (``"A"``, ``"B"``, …); blank /
  ``NaN`` (or an explicit ``control_label``) marks a control geo. The label
  is a property of the geo, so it is constant over that geo's rows.
* ``post_col`` — the (shared) ``0/1`` post-treatment window.

.. code-block:: python

   import pandas as pd
   from mlsynth import MULTICELLGEOLIFT

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "refs/heads/main/basedata/geolift_test_data.csv")
   df = pd.read_csv(url)                                   # GeoLift_Test: 40 mkts x 105 days
   dates = sorted(df["date"].unique())
   df["post"] = df["date"].isin(dates[90:]).astype(int)   # last 15 days = treatment window

   #   cell A -> social-media markets, cell B -> paid-search markets, blank = control
   cell = {"chicago": "A", "portland": "A", "atlanta": "B", "boston": "B"}
   df["cell"] = df["location"].map(cell).fillna("")        # blank = shared control pool

   res = MULTICELLGEOLIFT({
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "cell_column_name": "cell",   # "A"/"B"/... ; blank = control
       "post_col": "post",
       "fixed_effects": True,         # augsynth/GeoLift default
   }).fit()

   res.cells["A"].effects.att        # cell A's ATT (per unit)
   res.cells["A"].inference.p_value  # cell A's conformal p
   res.comparison                    # pairwise cross-cell rows
   res.winner                        # the cell that wins every comparison, or None

Notation
--------

This estimator is a thin multi-cell wrapper, so its symbols are GEOLIFT's
(:doc:`geolift`) applied once per cell. There are :math:`N` markets
:math:`\mathcal{N} \coloneqq \{1, \dots, N\}` over periods
:math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`, with the intervention taking
effect after :math:`T_0`, splitting :math:`\mathcal{T}` into the pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` and the
post-period :math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`. The
outcome of market :math:`j` at time :math:`t` is :math:`y_{jt}`, with market
series :math:`\mathbf{y}_j \in \mathbb{R}^{T}`.

Each cell :math:`c \in \{A, B, \dots\}` is a treated region
:math:`\mathcal{S}_c \subseteq \mathcal{N}` — a GEOLIFT design in its own right,
so :math:`\mathcal{S}_c` plays the canonical treated role through its aggregate
series :math:`\mathbf{y}^{\mathcal{S}_c}` (cf. GEOLIFT's treated set
:math:`\mathcal{S}`). The shared control pool is every market in no cell,
:math:`\mathcal{N}_0 \coloneqq \mathcal{N} \setminus \bigcup_c \mathcal{S}_c`
with cardinality :math:`N_0`, giving the donor matrix
:math:`\mathbf{Y}_0 \coloneqq [\mathbf{y}_j]_{j \in \mathcal{N}_0}`. Cell
:math:`c`'s donor pool excludes the other cells' markets,
:math:`\mathcal{N} \setminus \bigl(\mathcal{S}_c \cup \bigcup_{c' \neq c}
\mathcal{S}_{c'}\bigr) = \mathcal{N}_0`. The per-period effect for cell :math:`c`
is :math:`\tau_t \coloneqq y^{\mathcal{S}_c}_t - \widehat{y}^{\mathcal{S}_c}_t`
and its ATT is :math:`\widehat{\tau} \coloneqq |\mathcal{T}_2|^{-1}
\sum_{t \in \mathcal{T}_2} \tau_t`, as in :doc:`geolift`.

Assumptions
-----------

Each cell inherits GEOLIFT's per-cell identifying assumptions; the multi-cell
wrapper adds one cross-cell condition.

1. Pre-period synthesizability. Each cell's aggregate
   :math:`\mathbf{y}^{\mathcal{S}_c}` lies in (or near) the span / convex hull of
   the control pool over :math:`\mathcal{T}_1` (GEOLIFT's scaled-L2 imbalance
   :math:`\kappa(\mathcal{S}_c)` certifies it; see :doc:`geolift`).

   *Remark.* This is the prerequisite for a credible per-cell counterfactual,
   inspected cell by cell exactly as in the single-cell case.

2. Exchangeability under the null. The conformal test treats each cell's
   residual path as exchangeable under :math:`H_0` of no effect.

   *Remark.* The same all-period refit GEOLIFT uses to deliver this runs once per
   cell, so each cell's conformal p-value is read on the single-cell terms of
   :doc:`geolift`.

3. Placebo-window stationarity. Pre-period dynamics resemble the experiment
   window, so the design transports — the usual SC stability assumption, applied
   per cell.

   *Remark.* A regime change between :math:`\mathcal{T}_1` and
   :math:`\mathcal{T}_2` breaks the counterfactual for that cell even with a good
   pre-fit.

4. Cross-cell non-contamination. The other cells' markets are excluded from a
   cell's donor pool: cell :math:`c`'s synthetic control combines control geos
   only, never any :math:`\mathcal{S}_{c'}` for :math:`c' \neq c`.

   *Remark.* Other cells are treated (with a different treatment) and so
   contaminated; admitting them as donors would bias the counterfactual. This is
   GeoLift's ``filter(!location %in% other_cells)``.

Method
------

Per cell. Each cell is measured against the shared control pool with the
fixed-effect Augmented SCM + conformal inference of :doc:`geolift` — *crucially,
the other cells' geos are excluded from the donor pool*, because they are treated
(with a different treatment) and so contaminated (GeoLift's
``filter(!location %in% other_cells)``). So cell :math:`A`'s synthetic control is
a combination of control geos only, never cell :math:`B`'s.

Cross-cell winner. For each pair, a cell wins when its ATT confidence
interval lies strictly above the other's (GeoLift's non-overlapping-CI rule;
the ATT interval is the per-period conformal band averaged). The overall
``winner`` wins every pairwise comparison, else ``None``. This is deliberately
conservative: measuring each cell cleanly is well-powered, but *separating* two
cells needs the difference to clear both intervals — so overlapping CIs (no
declared winner) is common and correct, not a failure.

Result
------

``MULTICELLGEOLIFT.fit`` returns a
:class:`~mlsynth.utils.geolift_helpers.multicell.structures.MultiCellResults` (a
:class:`~mlsynth.config_models.DesignResult`): ``cells`` maps each label to its
:class:`EffectResult`, ``comparison`` is the pairwise table, ``winner`` the
overall winner, and ``report`` is the representative (winner / largest) cell.

Reading the results — per-cell plots and the comparison
-------------------------------------------------------

Each cell's report is a full GeoLift :class:`EffectResult`, so every single-cell
view (observed vs synthetic, the gap with its conformal band, the donor weights —
see :doc:`geolift`) works per cell. ``plot_multicell`` stacks the
observed-vs-synthetic panels, one row per cell:

.. code-block:: python

   import pandas as pd
   from mlsynth import MULTICELLGEOLIFT
   from mlsynth.utils.geolift_helpers.multicell.plotter import plot_multicell

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "refs/heads/main/basedata/geolift_test_data.csv")
   df = pd.read_csv(url)                                   # GeoLift_Test: 40 mkts x 105 days
   dates = sorted(df["date"].unique())
   df["post"] = df["date"].isin(dates[90:]).astype(int)   # last 15 days = treatment window
   cell = {"chicago": "A", "portland": "A", "atlanta": "B", "boston": "B"}
   df["cell"] = df["location"].map(cell).fillna("")        # blank = shared control pool

   res = MULTICELLGEOLIFT({
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "cell_column_name": "cell", "post_col": "post", "fixed_effects": True,
   }).fit()

   plot_multicell(res, show=True)                        # one panel per cell

   # per-cell numbers and views
   res.cells["A"].effects.att, res.cells["A"].inference.p_value
   res.cells["A"].time_series.estimated_gap              # cell A's gap path
   res.cells["A"].weights.donor_weights                  # cell A's controls

   # cross-cell comparison and the overall winner
   res.comparison        # [{cell_a, cell_b, att_a, att_b, att_diff, winner}, ...]
   res.winner            # the cell that wins every pairwise comparison, or None

A ``winner`` of ``None`` is the honest, common outcome: each cell is measured
well, but declaring one better needs its ATT interval to clear the other's
(GeoLift's non-overlapping-CI rule), which a single test rarely supports.

Not voodoo — one cell *is* the single-cell case
-----------------------------------------------

Multi-cell strictly generalizes single-cell: with one cell, every other unit is
a control (no other cells to exclude), so ``MULTICELLGEOLIFT`` makes the *same*
fit as the single-cell :doc:`geolift` realize — same treated set, same donor
pool, hence the same ATT, conformal p, and weights (pinned in
``test_multicell.py``):

.. code-block:: python

   import pandas as pd
   from mlsynth import MULTICELLGEOLIFT

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "refs/heads/main/basedata/geolift_test_data.csv")
   df = pd.read_csv(url)                                   # GeoLift_Test: 40 mkts x 105 days
   dates = sorted(df["date"].unique())
   df["post"] = df["date"].isin(dates[90:]).astype(int)   # last 15 days = treatment window

   # one cell A only; every other geo is a control (no other cells to exclude)
   df["cell"] = df["location"].map({"chicago": "A", "portland": "A"}).fillna("")

   res = MULTICELLGEOLIFT({
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "cell_column_name": "cell", "post_col": "post", "fixed_effects": True,
   }).fit()

   # one cell A, everyone else control  ==  single-cell GEOLIFT on {chicago, portland}
   res.cells["A"].effects.att        # 156.805165  (identical to the realize ATT)
   res.cells["A"].inference.p_value  # 0.006        (identical)
   res.winner                        # None — nothing to compare against

Verification
------------

Cross-validated against augsynth (the engine ``GeoLiftMultiCell`` wraps) on
the GeoLift_Test panel — cell A = {chicago, portland} (real effect), cell B =
{atlanta, boston} (placebo), the rest shared controls: the per-cell ATT matches
augsynth to the decimal (A ``156.84``, B ``119.38``), the conformal p-values
agree (A ``≈0.01``, B ``≈0.8``), and the donor-exclusion invariant holds (A never
uses B's markets). Durable case ``geolift_multicell``; unit tests
``mlsynth/tests/test_multicell.py``.

Core API
--------

.. autoclass:: mlsynth.MULTICELLGEOLIFT
   :members: fit

.. autoclass:: mlsynth.utils.geolift_helpers.multicell.config.MultiCellGeoLiftConfig
   :members:
