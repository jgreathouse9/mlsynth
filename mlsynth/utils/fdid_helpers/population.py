r"""Population counterpart of the Forward DiD Web Appendix E DGPs.

:mod:`~mlsynth.utils.fdid_helpers.simulation` draws samples from the four
data-generating processes of Li (2023) Web Appendix E. This
module supplies the *population* objects those draws converge to: the
theoretical prediction variance :math:`V_U` and the theoretical forward
selection algorithm of Web Appendix D.

Both exist to give Propositions 2.2 and D.1 a computable benchmark. Those
propositions compare the subset :math:`\widehat{\mathcal{U}}` that the
empirical algorithm selects from a finite sample against the collection
:math:`\mathcal{U}^*` that the same algorithm would select if it knew the
true variances, and assert
:math:`\Pr(\widehat{\mathcal{U}} \subset \mathcal{U}^*) \to 1` as
:math:`T_1 \to \infty`. The empirical side is
:func:`~mlsynth.utils.fdid_helpers.estimation.forward_did_select`; the
theoretical side is :func:`theoretical_forward_selection` here.

The criterion
-------------

Web Appendix D defines the theoretical error variance of a control subset
:math:`U` as

.. math::

   V_U = \mathbb{E}\bigl[(y_{tr,t} - \bar y_{Ut} - \alpha_U)^2\bigr],
   \qquad \alpha_U = \mathbb{E}[y_{tr,t} - \bar y_{Ut}],

and the theoretical algorithm as the greedy forward search that minimises
:math:`V_U` at each step, taking the best of the :math:`N` nested subsets it
builds. The empirical algorithm is the same search on
:math:`\widehat V_U = T_1^{-1}\sum_t \hat v_{Ut}^2`, which is what maximising
pre-treatment :math:`R^2` amounts to for a fixed treated series.

Closed form for these DGPs
--------------------------

Under the Web Appendix E design the treated unit is
:math:`y_{tr,t} = a_0 + c_0 \mathbf{1}'f_t + \varepsilon_{tr,t}` and control
:math:`i` is :math:`y_{it} = 1 + c_{g(i)} \mathbf{1}'f_t + \varepsilon_{it}`,
with :math:`g(i) \in \{1, 2\}` the loading group, unit-variance independent
:math:`\varepsilon`, and factors independent of them. For a subset holding
:math:`n_1` members of group 1 and :math:`n_2` of group 2, write
:math:`n = n_1 + n_2` and
:math:`\bar c = (n_1 c_1 + n_2 c_2)/n`. Then

.. math::

   y_{tr,t} - \bar y_{Ut}
     = (a_0 - 1) + (c_0 - \bar c)\,\mathbf{1}'f_t
       + \varepsilon_{tr,t} - \bar\varepsilon_{Ut},

and since the factors are zero-mean, :math:`\alpha_U = a_0 - 1` for every
subset, leaving

.. math::

   V_U = (c_0 - \bar c)^2 \sigma_S^2 + 1 + \frac{1}{n},

with :math:`\sigma_S^2 = \operatorname{Var}(\mathbf{1}'f_t)` the sum of the
three factors' stationary variances. Two consequences shape everything
below: :math:`V_U` depends on the subset only through
:math:`(n_1, n_2)`, so members of a loading group are exchangeable and the
tie structure of :math:`\mathcal{U}^*` is described by count pairs; and the
criterion trades a loading gap against an averaging gain, which is what
makes the mismatched DGPs select a strict subset instead of the whole pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Dict, FrozenSet, Iterable, Optional, Sequence, Tuple

# Loading parameters (a_0, c_0, c_1, c_2), mirroring ``simulation._DGP_PARAMS``.
# Duplicated deliberately: importing them would couple the population module to
# the simulator's private surface, and the two must agree by test, not by
# reference -- ``test_prediction_variance_matches_monte_carlo`` is that test.
_DGP_PARAMS: Dict[int, Tuple[float, float, float, float]] = {
    1: (1.0, 1.0, 1.0, 1.0),
    2: (1.0, 1.0, 1.0, 2.0),
    3: (2.0, 1.0, 1.0, 1.0),
    4: (2.0, 1.0, 1.0, 2.0),
}

#: :math:`\sigma_S^2 = \operatorname{Var}(\mathbf{1}'f_t)`, the stationary
#: variance of the summed factor. ``f1`` is AR(1) with ``phi = 0.8``; ``f2`` is
#: ARMA(1,1) with ``phi = -0.6``, ``theta = 0.8``; ``f3`` is MA(2) with
#: ``theta = (0.9, 0.4)``. Unit-variance independent innovations drive the
#: three, so the variance of the sum is the sum of the variances.
FACTOR_SUM_VARIANCE: float = (
    1.0 / (1.0 - 0.8 ** 2)                                  # AR(1)
    + (1.0 + 2.0 * (-0.6) * 0.8 + 0.8 ** 2) / (1.0 - 0.6 ** 2)   # ARMA(1,1)
    + (1.0 + 0.9 ** 2 + 0.4 ** 2)                           # MA(2)
)

# Relative tolerance for calling two theoretical variances tied. The criterion
# is evaluated in closed form from small integers, so exact ties (the c1 == c2
# DGPs) land within a few ulp; anything looser would merge genuinely distinct
# states at large N, where the 1/n increments themselves get small.
_TIE_RTOL = 1e-12


def _params(dgp: int) -> Tuple[float, float, float, float]:
    if dgp not in _DGP_PARAMS:
        raise ValueError(f"dgp must be in {{1, 2, 3, 4}}; got {dgp}.")
    return _DGP_PARAMS[dgp]


def _group_sizes(N: int) -> Tuple[int, int]:
    """Group sizes for a pool of ``N`` controls.

    Mirrors ``simulate_fdid_sample``'s ``half = N // 2`` split, so an odd
    pool puts the extra unit in group 2. The two must agree: a mismatch
    would score the benchmark against the wrong controls.
    """
    if N < 1:
        raise ValueError(f"N must be a positive number of controls; got {N}.")
    half = N // 2
    return half, N - half


def group_counts(subset: Iterable[int], N: int) -> Tuple[int, int]:
    """Split donor indices into ``(n_group1, n_group2)``.

    Parameters
    ----------
    subset : iterable of int
        Donor indices, as positions into the ``(N, T)`` control array that
        :class:`~mlsynth.utils.fdid_helpers.simulation.FDIDSample` carries.
    N : int
        Size of the donor pool.

    Returns
    -------
    tuple of int
        How many of the indices fall in each loading group.
    """
    half, _ = _group_sizes(N)
    idx = list(subset)
    n1 = sum(1 for i in idx if i < half)
    return n1, len(idx) - n1


def prediction_variance(
    dgp: int, n_group1: int, n_group2: int, N: Optional[int] = None
) -> float:
    r"""Theoretical error variance :math:`V_U` of a control subset.

    Parameters
    ----------
    dgp : int
        Which Web Appendix E DGP (1-4).
    n_group1, n_group2 : int
        How many controls the subset draws from each loading group.
    N : int, optional
        Donor-pool size. When given, the counts are checked against the
        group sizes it implies.

    Returns
    -------
    float
        :math:`(c_0 - \bar c)^2 \sigma_S^2 + 1 + 1/n`.

    Raises
    ------
    ValueError
        If ``dgp`` is unknown, the subset is empty (:math:`V_U` divides by
        :math:`|U|`), or the counts exceed the pool.
    """
    _, c0, c1, c2 = _params(dgp)
    if n_group1 < 0 or n_group2 < 0:
        raise ValueError(
            f"subset counts must be non-negative; got ({n_group1}, {n_group2})."
        )
    n = n_group1 + n_group2
    if n == 0:
        raise ValueError(
            "the theoretical variance needs at least one control: V_U averages "
            "over |U| donors and is undefined for the empty subset."
        )
    if N is not None:
        size1, size2 = _group_sizes(N)
        if n_group1 > size1 or n_group2 > size2:
            raise ValueError(
                f"subset ({n_group1}, {n_group2}) exceeds the group sizes "
                f"({size1}, {size2}) implied by N = {N}."
            )
    c_bar = (n_group1 * c1 + n_group2 * c2) / n
    return (c0 - c_bar) ** 2 * FACTOR_SUM_VARIANCE + 1.0 + 1.0 / n


@dataclass(frozen=True)
class TheoreticalSelection:
    r"""What the theoretical forward selection algorithm selects.

    Attributes
    ----------
    dgp, N : int
        The design this describes.
    optimal_states : frozenset of (int, int)
        Every ``(n_group1, n_group2)`` count pair attaining the minimum of
        :math:`V` along the greedy path. :math:`\mathcal{U}^*` is the
        collection of subsets carrying one of these count pairs.
    variance : float
        :math:`V` at the optimum.
    path : tuple of (int, int, float)
        One representative state per greedy step, with its :math:`V`. Where
        a step admits tied states (which happens when the two loading groups
        are indistinguishable), the representative is the one holding most
        group-1 members; every tied state at a step carries the same
        :math:`V`, so the third element is unambiguous.
    reachable : tuple of frozenset
        All states reachable at each step, so a caller can see the tie
        structure the representative in ``path`` collapses.
    """

    dgp: int
    N: int
    optimal_states: FrozenSet[Tuple[int, int]]
    variance: float
    path: Tuple[Tuple[int, int, float], ...]
    reachable: Tuple[FrozenSet[Tuple[int, int]], ...]

    @property
    def counts(self) -> Tuple[int, int]:
        """The optimal ``(n_group1, n_group2)``.

        Raises
        ------
        ValueError
            When the optimum spans several count pairs, so no single pair
            describes it. All four Web Appendix E DGPs have one.
        """
        if len(self.optimal_states) != 1:
            raise ValueError(
                f"the optimum spans {len(self.optimal_states)} count pairs "
                f"({sorted(self.optimal_states)}); read optimal_states instead."
            )
        return next(iter(self.optimal_states))

    @property
    def unique(self) -> bool:
        r""":math:`\mathcal{U}^*` holds exactly one subset.

        True when the optimum is a single count pair *and* only one subset
        carries it -- which needs the count to exhaust its group, since
        otherwise the exchangeable members give
        :math:`\binom{N_1}{n_1}\binom{N_2}{n_2} > 1` tied subsets. This is
        the hypothesis Proposition 2.2 adds to Proposition D.1, and when it
        holds the claim sharpens to
        :math:`\Pr(\widehat{\mathcal{U}} = \mathcal{U}^*) \to 1`.
        """
        if len(self.optimal_states) != 1:
            return False
        size1, size2 = _group_sizes(self.N)
        n1, n2 = next(iter(self.optimal_states))
        return comb(size1, n1) * comb(size2, n2) == 1

    def contains(self, subset: Sequence[int]) -> bool:
        r"""Is ``subset`` a member of :math:`\mathcal{U}^*`?

        Members of a loading group are exchangeable under :math:`V`, so
        membership is decided entirely by the subset's group counts.
        """
        return group_counts(subset, self.N) in self.optimal_states


def theoretical_forward_selection(dgp: int, N: int) -> TheoreticalSelection:
    r"""Run the Web Appendix D theoretical forward selection algorithm.

    The greedy search of Web Appendix D, Steps 1-3, on the true variances:
    add the control that minimises :math:`V` at each step until all ``N``
    are in, then take the step whose :math:`V` is smallest. Ties are carried
    forward, not broken, so the returned ``optimal_states`` describes the
    whole collection :math:`\mathcal{U}^*`.

    Parameters
    ----------
    dgp : int
        Which Web Appendix E DGP (1-4).
    N : int
        Donor-pool size.

    Returns
    -------
    TheoreticalSelection

    Notes
    -----
    Because :math:`V` depends on a subset only through its group counts, the
    search runs over the :math:`(n_1, n_2)` lattice instead of over subsets,
    which keeps it :math:`O(N)` where an enumeration would be
    :math:`O(2^N)`. Every state reachable at a given step carries the same
    :math:`V`: with :math:`c_1 \ne c_2` the mean loading is strictly
    monotone in :math:`n_2`, so tied states coincide; with
    :math:`c_1 = c_2` the criterion depends on :math:`n` alone. Taking the
    minimum across branches at each step is therefore the same as letting
    each branch take its own.
    """
    size1, size2 = _group_sizes(N)
    _params(dgp)  # validate before any work

    states: FrozenSet[Tuple[int, int]] = frozenset({(0, 0)})
    path = []
    reachable = []
    for _ in range(N):
        candidates: Dict[Tuple[int, int], float] = {}
        for n1, n2 in states:
            if n1 < size1:
                nxt = (n1 + 1, n2)
                candidates[nxt] = prediction_variance(dgp, *nxt)
            if n2 < size2:
                nxt = (n1, n2 + 1)
                candidates[nxt] = prediction_variance(dgp, *nxt)
        best = min(candidates.values())
        states = frozenset(
            s for s, v in candidates.items()
            if v <= best + _TIE_RTOL * abs(best)
        )
        representative = max(states)          # prefer group-1-heavy states
        path.append((*representative, best))
        reachable.append(states)

    best_step = min(v for _, _, v in path)
    optimal = frozenset(
        s
        for step_states, (_, _, v) in zip(reachable, path)
        if v <= best_step + _TIE_RTOL * abs(best_step)
        for s in step_states
    )
    return TheoreticalSelection(
        dgp=dgp,
        N=N,
        optimal_states=optimal,
        variance=best_step,
        path=tuple(path),
        reachable=tuple(reachable),
    )
