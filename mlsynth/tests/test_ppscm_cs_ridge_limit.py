"""The Callaway-Sant'Anna estimator is the infinite-ridge limit of PPSCM.

``docs/ppscm.rst`` states this as mathematics, so it is asserted here as
mathematics. Every number the page quotes has a test in this file; #473 exists
because a measured claim reached a doc page, an issue and a field description
with nothing asserting it, and was wrong.

The claims, in the order the page makes them:

1. The barycenter of the simplex minimises any strictly convex,
   permutation-symmetric penalty over it -- so the limit is the same whether
   the penalty is ridge, entropy or elastic net.
2. Partially pooled SCM's weights converge to that barycenter as ``lam``
   grows, at rate ``O(1/lam)``, for every ``nu``.
3. So does the estimate, to the Callaway-Sant'Anna estimate.
4. The departure from the barycenter is first order in ``1/lam`` along a fixed
   direction, tangent to the simplex, and ``nu`` sets that direction without
   moving the limit.
5. ``donor_weights="uniform"`` is that limit reached exactly, in one step and
   with no quadratic program.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM

try:                                             # the authority, where installed
    from diff_diff import CallawaySantAnna
    HAVE_DIFF_DIFF = True
except ImportError:                              # pragma: no cover - CI without it
    HAVE_DIFF_DIFF = False

needs_diff_diff = pytest.mark.skipif(
    not HAVE_DIFF_DIFF, reason="diff-diff not installed")

_REF_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "benchmarks" / "reference" / "ppscm_cs" / "reference.py")
_spec = importlib.util.spec_from_file_location("_ppscm_cs_reference", _REF_PATH)
REF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REF)


def panel(onsets=(10, 14, 18), k=3, n_units=50, n_per=30, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    assign = {u: g for i, g in enumerate(onsets) for u in range(i * k, (i + 1) * k)}
    rows = []
    for u in range(n_units):
        base, walk = rng.normal(10, 2), np.cumsum(rng.normal(0, 0.3, n_per))
        g = assign.get(u)
        for t in range(1, n_per + 1):
            on = g is not None and t >= g
            rows.append((u, t, scale * (base + walk[t - 1] + 0.5 * rng.normal()
                                        + 2.0 * on), int(on), g if g else 0))
    return pd.DataFrame(rows, columns=["id", "time", "y", "d", "first_treat"])


def fit(df, **over):
    cfg = {"df": df[["id", "time", "y", "d"]], "outcome": "y", "treat": "d",
           "unitid": "id", "time": "time", "display_graphs": False,
           "run_inference": False, "base_period": "pre_treatment",
           "donor_pool": "never_treated", **over}
    return PPSCM(cfg).fit()


def deviation_from_uniform(res):
    """``max_i |w_i - 1/N0|`` over the *full* donor set.

    ``donor_weights_by_cohort`` omits weights below 1e-8, so at small ``lam``,
    where the SCM solution is sparse, the stored vector is shorter than the
    donor pool. Dividing by its length would measure the deviation from a
    uniform distribution over the support that happened to survive, which is a
    different quantity and is small exactly when the weights are least uniform.
    Every omitted donor sits at zero and so contributes ``1/N0``.
    """
    stored = np.array(list(next(iter(res.donor_weights_by_cohort.values())).values()))
    n0 = int(res.metadata["n_control"])          # donor_pool="never_treated"
    dev = float(np.max(np.abs(stored - 1.0 / n0))) if stored.size else 1.0 / n0
    if stored.size < n0:
        dev = max(dev, 1.0 / n0)
    return dev


def dense_weights(res):
    """The stored weights, usable only where every donor is above the 1e-8 cut."""
    return np.array(list(next(iter(res.donor_weights_by_cohort.values())).values()))


def cs_oracle(df, L):
    """Callaway-Sant'Anna's estimate over PPSCM's cells, cohort-share weighted."""
    eff, _, inf = REF.group_time_att(df, "y", "id", "time", "first_treat")
    cohort = df.groupby("id")["first_treat"].first().to_numpy()
    cells = [(g, t) for (g, t) in eff if 0 <= t - g < L]
    pg = np.array([(cohort == g).sum() / cohort.shape[0] for g, _ in cells])
    est, _, _ = REF.aggregate(cells, eff, inf, cohort, weights=pg / pg.sum())
    return est


# =========================================================================== #
# 1. the barycenter is what the penalty selects
# =========================================================================== #
class TestTheBarycenter:
    """No estimator involved: a fact about the simplex."""

    @pytest.mark.parametrize("n", [2, 3, 5, 16, 41, 91])
    def test_uniform_minimises_the_squared_norm_over_the_simplex(self, n):
        bary = np.full(n, 1.0 / n)
        rng = np.random.default_rng(0)
        for _ in range(200):
            w = rng.dirichlet(np.full(n, 0.7))
            assert (w ** 2).sum() >= (bary ** 2).sum() - 1e-15

    @pytest.mark.parametrize("n", [3, 16, 41])
    def test_uniform_also_minimises_the_entropy_penalty(self, n):
        """BFR name entropy as an alternative penalty. Any strictly convex
        permutation-symmetric penalty has the same minimiser, so naming a
        different one does not move the limit."""
        neg_entropy = lambda w: float((w * np.log(np.where(w > 0, w, 1.0))).sum())
        bary = np.full(n, 1.0 / n)
        rng = np.random.default_rng(1)
        for _ in range(200):
            w = rng.dirichlet(np.full(n, 0.7))
            assert neg_entropy(w) >= neg_entropy(bary) - 1e-12

    def test_the_barycenter_is_interior_so_no_constraint_binds_there(self):
        """The rate is O(1/lam) and not O(1/sqrt(lam)) because the limit is in
        the relative interior: for large lam the non-negativity constraints are
        inactive and the path is smooth."""
        for n in (2, 5, 41):
            assert np.all(np.full(n, 1.0 / n) > 0)


# =========================================================================== #
# 2. the weights converge to it, at O(1/lam), for every nu
# =========================================================================== #
class TestTheRidgeLimit:
    @pytest.mark.parametrize("nu", ["auto", 0.0, 0.5, 1.0])
    def test_the_deviation_falls_by_a_decade_per_decade_of_lam(self, nu):
        df = panel()
        devs = {lam: deviation_from_uniform(
                    fit(df, donor_weights="scm", lam=lam, nu=nu))
                for lam in (1e6, 1e9, 1e12)}
        # The expansion is asymptotic, so the rate is asserted where it applies.
        # At lam = 1e3 the O(lam^-2) term is still visible (the 1e3 -> 1e6 ratio
        # is 933 at nu = 1), which is the expansion behaving as an expansion.
        for a, b in ((1e6, 1e9), (1e9, 1e12)):
            assert devs[a] / devs[b] == pytest.approx(1000.0, rel=0.05), (nu, a, b)

    def test_the_limit_is_reached_to_machine_precision(self):
        """The claim #473 corrects: there is no floor."""
        df = panel()
        assert deviation_from_uniform(fit(df, donor_weights="scm", lam=1e18)) < 1e-14

    @pytest.mark.parametrize("donor_pool", ["window", "never_treated"])
    @pytest.mark.parametrize("base_period", ["all_pre", "pre_treatment"])
    @pytest.mark.parametrize("scale", [1.0, 1e4])
    def test_no_configuration_has_a_floor(self, donor_pool, base_period, scale):
        """#473 asserted a floor at 3.5e-03 'at any nu'. It is absent from every
        combination of the conventions and both outcome scales."""
        df = panel(scale=scale)
        dev = deviation_from_uniform(fit(df, donor_weights="scm", lam=1e18,
                                        donor_pool=donor_pool,
                                        base_period=base_period))
        assert dev < 1e-14, f"{donor_pool}/{base_period}/{scale}: {dev:.2e}"

    @pytest.mark.parametrize("n_units,n_donors", [(25, 16), (50, 41), (100, 91)])
    def test_the_limit_holds_at_every_donor_count(self, n_units, n_donors):
        df = panel(n_units=n_units)
        res = fit(df, donor_weights="scm", lam=1e18)
        assert int(res.metadata["n_control"]) == n_donors
        assert deviation_from_uniform(res) < 1e-14


# =========================================================================== #
# 3. and so does the estimate, to Callaway-Sant'Anna's
# =========================================================================== #
class TestTheEstimateConverges:
    def test_the_ridge_path_reaches_the_callaway_santanna_estimate(self):
        df = panel()
        L = len(next(iter(fit(df, donor_weights="uniform").per_unit.values())).tau)
        oracle = cs_oracle(df, L)
        gaps = {}
        for lam in (1e3, 1e6, 1e9, 1e12):
            gaps[lam] = abs(float(fit(df, donor_weights="scm", lam=lam).att) - oracle)
        for a, b in ((1e3, 1e6), (1e6, 1e9), (1e9, 1e12)):
            assert gaps[a] / gaps[b] == pytest.approx(1000.0, rel=0.05), (a, b)
        assert gaps[1e12] < 1e-11

    def test_uniform_weights_are_that_limit_reached_exactly(self):
        """The setting is the limit in closed form: no quadratic program, and
        agreement with the oracle at machine precision instead of at whatever
        lam the caller thought to pick."""
        df = panel()
        res = fit(df, donor_weights="uniform")
        L = len(next(iter(res.per_unit.values())).tau)
        assert float(res.att) == pytest.approx(cs_oracle(df, L), abs=1e-12)
        assert deviation_from_uniform(res) < 1e-15


# =========================================================================== #
# 4. how it leaves the limit: first order in 1/lam, direction set by nu
# =========================================================================== #
class TestTheDepartureFromTheLimit:
    @staticmethod
    def _direction(df, lam, nu="auto"):
        w = dense_weights(fit(df, donor_weights="scm", lam=lam, nu=nu))
        return lam * (w - 1.0 / w.size)

    def test_the_scaled_departure_converges_to_a_fixed_vector(self):
        """``w_lam = w_bary + d / lam + O(lam^-2)``: multiplying the deviation
        by ``lam`` leaves something that stops moving."""
        df = panel()
        ds = [self._direction(df, lam) for lam in (1e4, 1e6, 1e8, 1e10)]
        for d in ds[1:]:
            cos = float(d @ ds[-1] / (np.linalg.norm(d) * np.linalg.norm(ds[-1])))
            assert cos == pytest.approx(1.0, abs=1e-6)
            assert np.linalg.norm(d) == pytest.approx(np.linalg.norm(ds[-1]), rel=1e-3)

    def test_the_departure_is_tangent_to_the_simplex(self):
        """The weights stay on the simplex, so the direction sums to zero."""
        df = panel()
        d = self._direction(df, 1e10)
        assert abs(float(d.sum())) < 1e-3 * float(np.linalg.norm(d))

    def test_nu_sets_the_direction_without_moving_the_limit(self):
        """Near the Callaway-Sant'Anna end of the path, nu chooses the direction
        in which partially pooled SCM departs from Callaway-Sant'Anna."""
        df = panel()
        norms = {nu: float(np.linalg.norm(self._direction(df, 1e10, nu)))
                 for nu in (0.0, 0.5, 1.0)}
        assert norms[0.0] < norms[0.5] < norms[1.0]
        assert norms[1.0] / norms[0.0] > 2.0, norms
        for nu in (0.0, 0.5, 1.0):
            assert deviation_from_uniform(
                fit(df, donor_weights="scm", lam=1e18, nu=nu)) < 1e-14, nu


# =========================================================================== #
# 5. the limit, against diff-diff itself
# =========================================================================== #
@needs_diff_diff
class TestTheLimitAgainstThePackage:
    """The vendored transcription is convenient; the package is the authority.

    Sections 2-4 measure the limit against ``reference.py``, which is code this
    repository wrote. That is enough to pin a rate, and not enough to claim the
    limit is Callaway-Sant'Anna's estimator: a transcription that drifted would
    still converge to itself. Here the target comes from ``diff-diff``'s own
    ``CallawaySantAnna`` fit, aggregated the way PPSCM aggregates, so the object
    at the end of the ridge path is the package's number.
    """

    @staticmethod
    def _package_att(df, L):
        r = CallawaySantAnna(control_group="never_treated", estimation_method="dr",
                             n_bootstrap=0).fit(df, outcome="y", unit="id",
                                                time="time", first_treat="first_treat")
        gt = {c: v["effect"] for c, v in r.group_time_effects.items()}
        cells = [(g, t) for (g, t) in gt if 0 <= t - g < L]
        cohort = df.groupby("id")["first_treat"].first().to_numpy()
        pg = np.array([(cohort == g).sum() for g, _ in cells], dtype=float)
        return float(np.array([gt[c] for c in cells]) @ (pg / pg.sum()))

    def test_the_ridge_path_reaches_the_package_estimate(self):
        df = panel()
        L = len(next(iter(fit(df, donor_weights="uniform").per_unit.values())).tau)
        target = self._package_att(df, L)
        gaps = {lam: abs(float(fit(df, donor_weights="scm", lam=lam).att) - target)
                for lam in (1e3, 1e6, 1e9, 1e12)}
        for a, b in ((1e3, 1e6), (1e6, 1e9), (1e9, 1e12)):
            assert gaps[a] / gaps[b] == pytest.approx(1000.0, rel=0.05), (a, b)
        assert gaps[1e12] < 1e-11

    @pytest.mark.parametrize("onsets,k", [((12,), 3), ((10, 14), 3),
                                          ((10, 14, 18), 3), ((10, 14, 18), 4)])
    def test_uniform_weights_equal_the_package_exactly(self, onsets, k):
        df = panel(onsets, k)
        res = fit(df, donor_weights="uniform")
        L = len(next(iter(res.per_unit.values())).tau)
        assert float(res.att) == pytest.approx(self._package_att(df, L), abs=1e-12)

    def test_the_transcription_and_the_package_agree_on_the_target(self):
        """If these ever part, the transcription is what to distrust."""
        df = panel()
        L = len(next(iter(fit(df, donor_weights="uniform").per_unit.values())).tau)
        assert cs_oracle(df, L) == pytest.approx(self._package_att(df, L), abs=1e-12)


# =========================================================================== #
# 6. the numbers the doc page quotes
# =========================================================================== #
class TestTheQuotedNumbers:
    """Each figure in the equivalence section of ``docs/ppscm.rst``."""

    def test_the_quoted_convergence_table(self):
        df = panel()
        L = len(next(iter(fit(df, donor_weights="uniform").per_unit.values())).tau)
        oracle = cs_oracle(df, L)
        quoted = {0.0: (1.0e-01, 2.6e-01), 1e6: (7.7e-07, 1.4e-07),
                  1e12: (7.7e-13, 1.4e-13)}
        for lam, (gap_q, dev_q) in quoted.items():
            res = fit(df, donor_weights="scm", lam=lam)
            gap = abs(float(res.att) - oracle)
            dev = deviation_from_uniform(res)
            assert gap == pytest.approx(gap_q, rel=0.1), (lam, gap)
            assert dev == pytest.approx(dev_q, rel=0.1), (lam, dev)

    def test_the_quoted_direction_norms(self):
        df = panel()
        for nu, quoted in ((0.0, 0.2846), (0.5, 0.6152), (1.0, 0.9541)):
            w = dense_weights(fit(df, donor_weights="scm", lam=1e10, nu=nu))
            d = 1e10 * (w - 1.0 / w.size)
            assert float(np.linalg.norm(d)) == pytest.approx(quoted, rel=1e-3), nu
