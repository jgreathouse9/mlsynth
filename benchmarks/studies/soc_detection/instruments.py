"""The study's measuring instruments, in one place so they can be tested.

Five of them were wrong before they were tested, which is the reason they
live here. The cut count read the wrong column of SCIP's constraint table. The
``DetectAll`` column was read as the number of times a handler was asked, when
``nlhdlr.c`` increments it only when the handler *participates*, so a zero
there cannot tell "never asked" from "asked and declined". The first count of
negative squares used a regular expression that could not see one written as
``-((2-<x>))^2``. And its replacement assumed every row reads ``expr <= rhs``,
when presolve is free to flip a row to ``expr >= lhs`` -- a hypothesis property
found a case where it did, and the count came out two where the cone had one.
And it read only the ``(<x>)^2`` notation SCIP uses after presolve, when a row
as written spells the same square ``<x>*<x>``, so on every unpresolved row it
counted zero. Each is pinned in ``benchmarks/tests/test_soc_detection_ladder.py`` against
inputs whose answer is known.
"""
from __future__ import annotations

import os
import tempfile

HANDLERS = ("soc", "convex", "quadratic", "default")


def statistics_text(model) -> str:
    """SCIP's statistics as text.

    ``writeStatistics`` is used because ``printStatistics`` writes from C and
    never reaches a Python-level stdout redirect.
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tf:
        path = tf.name
    try:
        model.writeStatistics(path)
        return open(path).read()
    finally:
        os.unlink(path)


def parse_statistics(text: str) -> dict:
    """Participations per nonlinear handler, and the nonlinear cut counts.

    ``Detects`` is the last detection round and ``DetectAll`` the total over
    rounds; both count participations only (``nlhdlr.c``, the increment sits
    under ``if( *participating != SCIP_NLHDLR_METHOD_NONE )``).

    The cut counts come from the ``nonlinear`` row of the Constraints table,
    whose columns are Number MaxNumber #Separate #Propagate #EnfoLP #EnfoRelax
    #EnfoPS #Check #ResProp Cutoffs DomReds Cuts Applied Conss Children. The
    Number column can carry a ``+`` suffix, so fields are split, not counted
    in a pattern.
    """
    out: dict = {name: None for name in HANDLERS}
    out["nonlinear_cuts"] = out["nonlinear_applied"] = None
    section = None
    for line in text.split("\n"):
        head = line.split(":", 1)[0].strip()
        if not line.startswith(" ") and ":" in line:
            section = head
            continue
        if ":" not in line:
            continue
        fields = line.split(":", 1)[1].split()
        if section == "Nlhdlrs" and head in HANDLERS and len(fields) >= 2:
            out[head] = (int(fields[0]), int(fields[1]))
        elif (section == "Constraints" and head == "nonlinear"
              and len(fields) >= 13 and fields[0].rstrip("+").isdigit()):
            out["nonlinear_cuts"] = int(fields[11])
            out["nonlinear_applied"] = int(fields[12])
    return out


def handler_rows(model) -> dict:
    return parse_statistics(statistics_text(model))


def _split_terms(expr: str) -> list[tuple[str, str]]:
    """Top-level ``(sign, term)`` pairs of a SCIP expression string.

    Splits on ``+`` and ``-`` at parenthesis depth zero, except where the sign
    belongs to a number's exponent (``1e-05``).
    """
    terms, depth, start, sign = [], 0, 0, "+"
    s = expr.strip()
    i = 0
    # ``"" in "+-"`` is true, so test for a character before testing which.
    if s and s[0] in "+-":
        sign, start, i = s[0], 1, 1
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in "+-" and depth == 0:
            prev = s[i - 1] if i else ""
            prev2 = s[i - 2] if i > 1 else ""
            if not (prev in "eE" and prev2.isdigit()):
                piece = s[start:i].strip()
                if piece:
                    terms.append((sign, piece))
                sign, start = ch, i + 1
        i += 1
    piece = s[start:].strip()
    if piece:
        terms.append((sign, piece))
    return terms


def _body_and_sense(row: str) -> tuple[str, str]:
    """The expression of a written row and the side that bounds it.

    SCIP writes a row as ``expr <= rhs``, ``expr >= lhs``, ``lhs <= expr <=
    rhs`` or ``expr == rhs``. Returns the expression and ``"<="`` or ``">="``.
    A ranged row is read on its upper side and an equality as ``<=``: a cone
    is an upper bound on its left side, so that is the side that can hold one.
    """
    body = row.split(":", 1)[1] if ":" in row else row
    body = body.strip().rstrip(";").strip()
    if body.count("<=") == 2:
        return body.split("<=")[1], "<="
    for op in ("<=", ">=", "=="):
        if op in body:
            return body.split(op, 1)[0], (">=" if op == ">=" else "<=")
    return body, "<="


def _is_square(term: str) -> bool:
    """Whether a term is a square, in either notation SCIP writes.

    After presolve a square reads ``(<x>)^2`` or ``((2-<x>))^2``; in a row as
    written it reads ``<x>*<x>``. A leading coefficient, ``3.5*``, is allowed
    in both. A product of two different variables is bilinear, not a square.
    """
    if term.endswith("^2"):
        return True
    factors = [f.strip() for f in term.split("*")]
    while factors and not factors[0].startswith("<"):
        try:
            float(factors[0])
        except ValueError:
            return False
        factors = factors[1:]
    return len(factors) == 2 and factors[0] == factors[1] and factors[0].startswith("<")


def negative_squares(row: str) -> int:
    """How many top-level terms of a written row are minus a square, once the
    row is read as an upper bound.

    ``row`` is one ``[nonlinear]`` line as SCIP writes it. The expression is
    split into top-level terms and a term counts when it is a square and
    carries a minus sign in the ``<=`` orientation -- whether the base is a
    variable, ``(<x>)``, or a sum, ``((2-<x>))``, written ``^2`` or as
    ``<x>*<x>``, and with or without a coefficient. A row written ``expr >= lhs`` is the row ``-expr <= -lhs``, so
    there the squares that count are the ones written with a plus.
    """
    expr, sense = _body_and_sense(row)
    wanted = "-" if sense == "<=" else "+"
    return sum(1 for sign, term in _split_terms(expr)
               if sign == wanted and _is_square(term))


def presolved_rows(model) -> list[str]:
    """The ``[nonlinear]`` rows of a presolved model, as SCIP writes them."""
    with tempfile.NamedTemporaryFile("w+", suffix=".cip", delete=False) as tf:
        path = tf.name
    try:
        model.writeProblem(path, trans=True, verbose=False)
        return [line.strip() for line in open(path) if "[nonlinear]" in line]
    finally:
        os.unlink(path)
