"""Decision-table tests for ``VanillaSCConfig``'s cross-field rules.

``agents/agents_tests.md`` records two gaps as open; this file closes the
second one. Config validators are the place where a decision table earns its
keep: their conditions are *dependent*, and the classical selection guidance
(Jorgensen §10.5, Table 10.13) puts dependent variables straight at decision
tables, not at boundary or equivalence-class testing. Writing the table
out makes the impossible rule combinations explicit and shows which rows no
test reaches.

``VanillaSCConfig`` has the most cross-field rules in the library (nine
validators), and two of them are genuinely multi-condition.

Rule 1 -- ``_w_constr_is_exclusive``. ``w_constr`` routes the fit through
scpi's constrained program, which is a different estimator from the bilevel
predictor-weight engine and from user-supplied weights, so it clashes with
four other options:

======  ==========================  =========================================
symbol  condition                   set by
======  ==========================  =========================================
c1      ``w_constr`` is not None    ``w_constr={"name": "simplex"}``
c2      ``covariates`` truthy       ``covariates=["x"]``
c3      ``backend != "auto"``       ``backend="malo"``
c4      ``oracle_weights`` given    ``oracle_weights={"u1": 0.5, "u2": 0.5}``
c5      ``staggered_spec`` given    ``StaggeredSCMSpec(features=["y"])``
======  ==========================  =========================================

The action is to raise iff ``c1 and (c2 or c3 or c4 or c5)``, and -- the part
single-fault testing cannot reach -- the message must name *every* clash that
holds, not the first one found. Sixteen combinations of c2..c5 are tested
under c1, and the same sixteen with c1 false as the control half of the table:
without ``w_constr`` none of the four is a clash at all, which is what makes
the rule about ``w_constr`` and not about the four options individually.

Rule 2 -- ``_bias_correction_needs_predictors``. Two conditions, four rows,
one of which raises.

Exception surface. Constructing the config object directly raises pydantic's
``ValidationError``; the public path -- handing a dict to the estimator --
translates it to ``MlsynthConfigError`` in ``VanillaSC.__init__``. Both are
asserted, because the translation is what a user actually meets and it had no
test: the ``except ValidationError`` arm carried ``# pragma: no cover`` even
though any invalid dict config reaches it.
"""

from __future__ import annotations

import itertools

import pandas as pd
import pytest
from pydantic import ValidationError

from mlsynth import VanillaSC
from mlsynth.config_models import VanillaSCConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.vanillasc_helpers.config import StaggeredSCMSpec


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.DataFrame([
        {"unit": f"u{u}", "time": 2000 + t, "y": float(10 * u + t), "x": float(t),
         "treated": int(u == 0 and t >= 4)}
        for u in range(3) for t in range(8)
    ])


def _base(panel: pd.DataFrame) -> dict:
    return dict(df=panel, outcome="y", treat="treated", unitid="unit",
                time="time", display_graphs=False)


# Each clash condition: the kwargs that switch it on, and the fragment the
# message must contain when it does.
_CLASHES = {
    "covariates": (dict(covariates=["x"]), "covariates"),
    "backend": (dict(backend="malo"), "backend"),
    "oracle_weights": (dict(oracle_weights={"u1": 0.5, "u2": 0.5}), "oracle_weights"),
    "staggered_spec": (dict(staggered_spec=StaggeredSCMSpec(features=["y"])),
                       "staggered_spec"),
}

# All 2^4 combinations of the clash conditions.
_COMBOS = [
    combo
    for r in range(len(_CLASHES) + 1)
    for combo in itertools.combinations(sorted(_CLASHES), r)
]


def _kwargs_for(combo) -> dict:
    merged: dict = {}
    for name in combo:
        merged.update(_CLASHES[name][0])
    return merged


# ---------------------------------------------------------------------------
# Rule 1 -- w_constr exclusivity, both halves of the table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("combo", _COMBOS, ids=lambda c: "+".join(c) or "none")
def test_w_constr_clashes_name_every_condition_that_holds(panel, combo):
    """c1 true: raise iff any clash holds, and name all of them.

    The multiple-fault half. A validator that stopped at the first clash would
    pass every single-fault test and fail here, and a user acting on such a
    message would be sent back for a second and third round.
    """
    kwargs = dict(_base(panel), w_constr={"name": "simplex"}, **_kwargs_for(combo))

    if not combo:
        VanillaSCConfig(**kwargs)          # no clash: the rule permits it
        return

    with pytest.raises(ValidationError) as excinfo:
        VanillaSCConfig(**kwargs)
    message = str(excinfo.value)
    for name in combo:
        assert _CLASHES[name][1] in message, (
            f"{name} holds but is not named; the message reports only "
            f"a subset of the clashes: {message}"
        )


@pytest.mark.parametrize("combo", _COMBOS, ids=lambda c: "+".join(c) or "none")
def test_without_w_constr_none_of_the_four_is_a_clash(panel, combo):
    """c1 false: the control half.

    Every one of the sixteen combinations must construct. This is what makes
    the rule a statement about ``w_constr`` and not about the four options,
    and it is the half a test suite usually omits.
    """
    VanillaSCConfig(**dict(_base(panel), **_kwargs_for(combo)))


def test_w_constr_alone_is_permitted(panel):
    """The row where c1 holds and nothing else does."""
    VanillaSCConfig(**_base(panel), w_constr={"name": "simplex"})


# ---------------------------------------------------------------------------
# Rule 2 -- bias correction needs predictors, all four rows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bias_correct,with_covariates,raises",
    [
        (False, False, False),
        (False, True, False),
        (True, True, False),
        (True, False, True),      # the only row that raises
    ],
)
def test_bias_correction_decision_table(panel, bias_correct, with_covariates, raises):
    kwargs = dict(_base(panel), bias_correct=bias_correct)
    if with_covariates:
        kwargs["covariates"] = ["x"]

    if raises:
        with pytest.raises(ValidationError, match="needs covariates"):
            VanillaSCConfig(**kwargs)
    else:
        VanillaSCConfig(**kwargs)


# ---------------------------------------------------------------------------
# The exception surface a user actually meets
# ---------------------------------------------------------------------------

def test_dict_config_translates_validation_errors_to_mlsynth_config_error(panel):
    """The public path, and the arm that carried an unearned ``no cover``.

    ``VanillaSC(config_dict)`` funnels pydantic's ``ValidationError`` into
    ``MlsynthConfigError`` so callers catch one exception family. That arm is
    reachable from any invalid dict, so it is behaviour, not a defensive
    branch, and the original message has to survive the translation or the
    user loses the reason.
    """
    bad = dict(_base(panel), bias_correct=True)
    with pytest.raises(MlsynthConfigError, match="needs covariates"):
        VanillaSC(bad)


def test_dict_config_translation_covers_the_multi_clash_message(panel):
    """The translated error keeps every clash the validator named."""
    bad = dict(_base(panel), w_constr={"name": "simplex"},
               covariates=["x"], backend="malo")
    with pytest.raises(MlsynthConfigError) as excinfo:
        VanillaSC(bad)
    message = str(excinfo.value)
    assert "covariates" in message and "backend" in message


def test_a_valid_dict_config_is_accepted_by_the_same_path(panel):
    """The translation must not swallow a config that is simply fine."""
    assert VanillaSC(dict(_base(panel))) is not None
