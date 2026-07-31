"""SDID covariate matching against the authors' own ``Synth`` code.

de Brabander, Juodis & Miyazato Szini (2025) build their SDID-with-covariates
from two separate fits: ``omega`` comes from a ``Synth::synth`` call whose
predictor matrix stacks the demeaned pre-treatment outcomes on top of the
covariate means, and ``lambda`` from an outcome-only ``synthdid_estimate`` with
``zeta.omega = 0``. The covariates therefore enter *only* through ``omega``, and
only through Synth's nested ``V`` search.

So ``omega`` and ``V`` are what this file compares, not the ATT. An ATT
disagreement mixes the unit weights, the time weights and the bias correction
together and cannot say which of the three moved; the weights localise it.

Why the Brexit panel and not the ``#309`` fixture
-------------------------------------------------

Because the ``#309`` xsdid.mc panel cannot test this at all. There the treated
unit lies outside the donor convex hull -- at the last pre-period the demeaned
donors span [40.962, 57.561] and the treated sits at 39.968 -- so the simplex
solution is pinned at a vertex, ``V`` cannot move it, and the covariates
provably cannot bind whatever the matching scheme. That is the paper's own
convex-hull condition (their section 2.2) failing, and it is a property of the
fixture rather than of either implementation. The Brexit panel is the data the
specification was written for.

What is gated, and what is only recorded
----------------------------------------

Gated -- the specification. That the covariates' share of ``V`` rises as the
number of outcome rows falls, that the leading donors are the same donors in the
same order, and that the weight vectors correlate almost perfectly under the
scheme where the covariates actually bind. These follow from the estimator both
sides are implementing, so a disagreement is a defect.

Recorded, not gated -- the sparsity. ``Synth`` optimises with ``ipop``, an
interior-point routine, and returns all 23 donors with positive weight; mlsynth
solves the QP exactly and returns four to six. On the one-period scheme R's top
four donors carry 0.9615 and the remaining nineteen carry 0.0385, the smallest
ten between 2e-6 and 2.4e-5. That is interior-point tolerance spreading a
residue of mass, the same distinction already documented for ``synthdid`` in
``sdid_helpers/weights.py`` -- not a different estimand. Gating it would gate
mlsynth to a solver artefact.

One thing this file structurally cannot test, recorded so it is not mistaken
for coverage. The bundle hands over ``Ypre_demeaned.csv`` already demeaned, so
whether ``match_unit_weights`` demeans is invisible here -- re-centring a
zero-mean series is a no-op. Deleting the demeaning survives every assertion
below. It is covered instead by the corresponding mutant in
``test_sdid_covariate_methods.py``, which starts from raw outcomes. Feeding the
reference's own matrices is the right trade (rebuilding a reference's inputs in
the port is what produced a sign flip in an earlier replication), but it moves
that one step out of scope.

Gold from ``benchmarks/reference/brabander_sdid_match/reference.R``
(``Synth`` 1.1.10, R 4.3.3); CI has no R.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

_GOLD = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
         / "brabander_sdid_match")

#: mlsynth's name for each of the paper's three matching schemes.
_SCHEME = {"all": "all", "half": "half", "one": "last"}

#: Index of the United Kingdom among the 24 surviving countries (0-based).
_TREATED = 22


def _skip_if_missing():
    if not _GOLD.joinpath("summary.csv").exists():
        pytest.skip("brabander_sdid_match reference bundle not present")


@pytest.fixture(scope="module")
def gold():
    _skip_if_missing()
    return pd.read_csv(_GOLD / "summary.csv").set_index("scheme")


@pytest.fixture(scope="module")
def matrices():
    """The exact matrices ``Synth`` saw, read from the bundle.

    Deliberately not rebuilt from the ``.mat`` in Python. Rebuilding a
    reference's inputs in the port is what produced a sign flip in an earlier
    replication; one file, two readers.
    """
    _skip_if_missing()
    Ydm = pd.read_csv(_GOLD / "Ypre_demeaned.csv").to_numpy(dtype=float)
    cov = pd.read_csv(_GOLD / "covariate_means.csv", index_col=0).to_numpy(dtype=float)
    names = [l.strip() for l in (_GOLD / "country_names.txt").read_text().splitlines()
             if l.strip()]
    donors = [j for j in range(Ydm.shape[1]) if j != _TREATED]
    return {
        "Y0": Ydm[:, donors], "y1": Ydm[:, _TREATED],
        "Z0": cov[:, donors], "z1": cov[:, _TREATED],
        "donor_names": [names[j] for j in donors],
    }


def _fit(matrices, scheme):
    from mlsynth.utils.sdid_helpers.weights import match_unit_weights

    # zeta = 0: the reference takes omega from an unpenalised synth() and passes
    # zeta.omega = 0 to synthdid. See the note in match_unit_weights.
    return match_unit_weights(matrices["Y0"], matrices["y1"],
                              matrices["Z0"], matrices["z1"], 0.0,
                              pre_periods=_SCHEME[scheme])[1]


def _r_weights(scheme):
    return pd.read_csv(_GOLD / f"omega_{scheme}.csv")


class TestTheFixtureCanActuallyTestThis:
    """The premise the ``#309`` fixture failed, asserted here rather than assumed."""

    def test_the_treated_unit_is_inside_the_donor_hull(self, matrices):
        """The paper's convex-hull condition (section 2.2).

        Where it fails the simplex solution sits at a vertex and ``V`` is
        irrelevant, which is exactly why the xsdid.mc fixture could not test
        covariate matching. Checked on the last pre-period, the row the
        one-period scheme matches on.
        """
        row, treated = matrices["Y0"][-1], matrices["y1"][-1]
        assert row.min() <= treated <= row.max()

    def test_there_are_enough_donors_and_periods(self, gold):
        g = gold.loc["all"]
        assert int(g.n_donors) == 23
        assert int(g.n_outcome_rows) == 86

    def test_the_schemes_really_differ_in_size(self, gold):
        rows = [int(gold.loc[s].n_outcome_rows) for s in ("all", "half", "one")]
        assert rows == sorted(rows, reverse=True) and len(set(rows)) == 3


class TestTheCovariatesBindAsTheTheoryPredicts:
    """Kaul et al. (2022), reproduced by the reference as a gradient.

    Their result is that covariates are irrelevant once every pre-treatment
    outcome is a predictor. The reference shows it is not all-or-nothing: the
    covariates take 2.4 percent of ``V`` with all 86 rows, 8.7 percent with the
    later half, and 79.8 percent with the last row alone. That gradient is the
    evidence for ``match_pre_periods`` defaulting to ``"last"``.
    """

    def test_the_covariate_share_of_v_rises_as_rows_fall(self, gold):
        shares = [float(gold.loc[s].v_on_covariates) for s in ("all", "half", "one")]
        assert shares[0] < shares[1] < shares[2]

    def test_covariates_are_nearly_irrelevant_with_all_pre_periods(self, gold):
        assert float(gold.loc["all"].v_on_covariates) < 0.05

    def test_covariates_dominate_with_one_pre_period(self, gold):
        assert float(gold.loc["one"].v_on_covariates) > 0.5

    def test_the_outer_fit_degrades_as_rows_are_dropped(self, gold):
        """The cost side of the trade-off, so the default is not read as free."""
        losses = [float(gold.loc[s].loss_v) for s in ("all", "half", "one")]
        assert losses[0] < losses[1] < losses[2]


class TestMlsynthAgreesWhereItMust:
    """The donors that carry the weight, under the scheme that matters."""

    def test_the_leading_donors_are_the_same_three_in_the_same_order(self, matrices):
        w = _fit(matrices, "one")
        mine = [matrices["donor_names"][i] for i in np.argsort(-w)[:3]]
        theirs = list(_r_weights("one").sort_values("weight", ascending=False)
                      .donor.head(3))
        assert mine == theirs

    @pytest.mark.parametrize("donor,tol", [("Italy", 0.005), ("Iceland", 0.005)])
    def test_the_supporting_donors_match_closely(self, matrices, donor, tol):
        w = _fit(matrices, "one")
        mine = float(w[matrices["donor_names"].index(donor)])
        theirs = float(_r_weights("one").set_index("donor").weight[donor])
        assert mine == pytest.approx(theirs, abs=tol)

    def test_the_weight_vectors_correlate_almost_perfectly(self, matrices):
        w = _fit(matrices, "one")
        r = _r_weights("one").weight.to_numpy()
        assert np.corrcoef(r, w)[0, 1] > 0.99

    def test_agreement_is_best_where_the_covariates_bind(self, matrices):
        """Ordered, not just individually bounded.

        Correlation runs 0.636 / 0.969 / 0.998 across all / half / one. That it
        is *worst* under ``"all"`` is expected rather than alarming: that is the
        cell where ``V`` is least identified, since by Kaul the covariate rows
        carry almost no weight and the outcome rows are near-collinear.
        """
        corrs = [np.corrcoef(_r_weights(s).weight.to_numpy(),
                             _fit(matrices, s))[0, 1]
                 for s in ("all", "half", "one")]
        assert corrs[0] < corrs[1] < corrs[2]

    def test_mlsynth_weights_are_a_simplex(self, matrices):
        for s in ("all", "half", "one"):
            w = _fit(matrices, s)
            assert w.min() >= -1e-9
            assert w.sum() == pytest.approx(1.0, abs=1e-6)


class TestTheSparsityDifferenceIsRecordedNotGated:
    """Why the weight vectors are not compared elementwise.

    ``Synth`` optimises with ``ipop`` and returns every donor positive; mlsynth
    solves the QP exactly and returns a handful. Asserting agreement would hold
    mlsynth to an interior-point tolerance. Asserted here as the *shape* of the
    difference, so that if it ever becomes something else -- a genuine
    disagreement on which donors matter -- these fail.
    """

    def test_the_reference_spreads_a_small_residue_over_every_donor(self):
        w = _r_weights("one").weight.to_numpy()
        w = np.sort(w)[::-1]
        assert (w > 1e-8).sum() == 23           # every donor positive
        assert w[:4].sum() > 0.95               # but the top four carry it all
        assert w[4:].sum() < 0.05

    def test_mlsynth_is_sparse_by_comparison(self, matrices):
        w = _fit(matrices, "one")
        assert (w > 1e-8).sum() < 10

    def test_the_residue_is_too_small_to_be_a_disagreement_about_donors(self, matrices):
        """The donors mlsynth zeroes are ones the reference gives ~1e-5.

        This is the assertion that makes ignoring the sparsity legitimate: if
        mlsynth were dropping a donor the reference considered important, this
        would fail.
        """
        w = _fit(matrices, "one")
        r = _r_weights("one").weight.to_numpy()
        dropped = w <= 1e-8
        assert dropped.any()
        assert r[dropped].max() < 0.05
