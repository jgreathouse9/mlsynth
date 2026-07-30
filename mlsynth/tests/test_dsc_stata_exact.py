"""DSC against the published numbers in the ``disco`` Stata Journal article.

Gunsilius & Van Dijcke's Stata Journal paper ships a replication repository
(`Davidvandijcke/disco_stata_journal`) with an anonymised dataset and a log
containing printed results. That makes it the only fully external reference for
mlsynth's ``DSC``: the R package's weights are a Monte Carlo draw (its
``DiSCo_weights_reg`` samples the quadrature points with ``runif``), so it
disagrees with *itself* across seeds by more than it disagrees with mlsynth and
cannot be matched exactly. The Stata implementation builds a deterministic
evenly spaced grid and consumes its seed only in the bootstrap, so it can.

The published run is::

    disco y_col id_col time_col, idtarget(2) t0(3) agg("quantileDiff") \\
        seed(12143) g(10) m(100) ci boots(300)

and its top-5 donor weights are the values pinned below.

Reproducing them to the published precision requires three things, and mlsynth
originally did none of them. In descending order of how much each one matters:

* the quadrature grid must be ``linspace(0, 1, M)`` with the endpoints
  *included* -- evaluating the sample minimum and maximum. Excluding them costs
  0.0869, by far the largest term;
* the weights must come from an exact QP rather than projected gradient (0.0047);
* the empirical quantile must be type 7 rather than type 1 (0.0026).

The third is worth a note. Both papers define the generalized inverse, which is
type 1, but *both* official implementations use type 7. mlsynth follows the
implementations here, because the point of this file is bit-level agreement with
a shipped reference; ``test_dsc_quantile_convention.py`` pins the distinction
itself so the choice stays visible rather than becoming folklore.

See issue #304 for the full investigation, including the several hypotheses that
turned out to be wrong.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

_DATA = (Path(__file__).resolve().parents[2] / "basedata" / "disco_tenure.parquet")

#: Verbatim from ``results/tenure_example.log`` in the replication repository.
#: Stata prints four decimals, so these are exact to 5e-5 and no tighter.
PUBLISHED_WEIGHTS = {
    "amazon": 0.2203,
    "autodesk": 0.1271,
    "cisco": 0.1066,
    "dell technologies": 0.0991,
    "slalom consulting": 0.0962,
}

#: The published command's settings.
TARGET_ID = 2
T0_PERIOD = 3          # Stata's t0(3): treatment in period 3, so pre = {1, 2}
M_POINTS = 100         # the example's m(100)


@pytest.fixture(scope="module")
def panel():
    if not _DATA.exists():  # pragma: no cover - vendored
        pytest.skip(f"missing {_DATA}")
    return pd.read_parquet(_DATA)


def _fit_weights(df):
    """Donor weights by company name, through the public API."""
    import warnings

    from mlsynth import DSC

    d = df.copy()
    d["treat"] = ((d.id_col == TARGET_ID) & (d.time_col >= T0_PERIOD)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = DSC({
            "df": d, "outcome": "y_col", "treat": "treat", "unitid": "id_col",
            "time": "time_col", "M": M_POINTS, "compute_inference": False,
            "display_graphs": False,
        }).fit()
    names = df.groupby("id_col").company_name.first()
    return {names[int(k)]: float(v) for k, v in res.donor_weights.items()}


class TestPublishedStataWeights:
    """The headline: mlsynth reproduces the printed table."""

    def test_every_published_weight_matches(self, panel):
        got = _fit_weights(panel)
        for name, expected in PUBLISHED_WEIGHTS.items():
            assert name in got, f"{name} absent from the donor pool"
            assert got[name] == pytest.approx(expected, abs=5e-5), (
                f"{name}: mlsynth {got[name]:.6f} vs published {expected}")

    def test_the_top_five_are_the_same_five_in_the_same_order(self, panel):
        """Ordering, not just values -- a permutation would pass value checks
        individually while reporting a different donor ranking."""
        got = _fit_weights(panel)
        top5 = sorted(got, key=got.get, reverse=True)[:5]
        assert top5 == list(PUBLISHED_WEIGHTS)

    def test_weights_are_a_simplex(self, panel):
        got = np.array(list(_fit_weights(panel).values()))
        assert got.min() >= -1e-12
        assert got.sum() == pytest.approx(1.0, abs=1e-9)


class TestTheSpecChoicesThatGetUsThere:
    """Each choice pinned separately, with the cost of getting it wrong.

    Without these, a future change could revert one of the three and the only
    signal would be the headline test failing with no indication of which.
    """

    def test_the_quadrature_grid_includes_its_endpoints(self):
        """The dominant term: 0.0869 if the endpoints are dropped."""
        from mlsynth.utils.dsc_helpers.quantiles import sample_quantile_grid

        g = sample_quantile_grid(M_POINTS, method="equidistant")
        assert g[0] == 0.0 and g[-1] == 1.0
        np.testing.assert_allclose(g, np.linspace(0.0, 1.0, M_POINTS),
                                   rtol=0, atol=1e-15)

    def test_the_default_grid_is_the_reference_grid(self):
        """Matching the shipped implementation is the default, not an opt-in."""
        from mlsynth.utils.dsc_helpers.quantiles import sample_quantile_grid

        np.testing.assert_allclose(sample_quantile_grid(64),
                                   sample_quantile_grid(64, method="equidistant"),
                                   rtol=0, atol=1e-15)

    def test_the_simplex_solver_is_exact(self):
        """0.0047 if projected gradient is used instead.

        FISTA matches an exact QP to 0.00 percent *on the objective* while its
        argmin differs by ~1e-3 -- so an objective-based check cannot detect
        this, and the reported quantity is the argmin.
        """
        cp = pytest.importorskip("cvxpy")
        from mlsynth.utils.dsc_helpers.weights import solve_simplex_weights

        rng = np.random.default_rng(0)
        D = rng.normal(size=(200, 12))
        y = D @ rng.dirichlet(np.ones(12)) + 0.01 * rng.normal(size=200)
        w = cp.Variable(12, nonneg=True)
        cp.Problem(cp.Minimize(cp.sum_squares(D @ w - y)),
                   [cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
        exact = np.clip(np.asarray(w.value).ravel(), 0, None)
        exact /= exact.sum()
        np.testing.assert_allclose(solve_simplex_weights(D, y), exact,
                                   rtol=0, atol=1e-6)
