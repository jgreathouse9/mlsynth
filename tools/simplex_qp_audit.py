"""Classify every cvxpy problem in the library by whether it is the simplex QP.

``bilevel/active_set.py::solve_simplex_qp`` minimises ``||A - Bw||^2`` over the
probability simplex ``{w >= 0, sum(w) = 1}``. A cvxpy site can be moved onto it
only if it solves that program. A site with an extra constraint, or one whose
weights may go negative, or one minimising something the solver cannot express,
is a different problem, and swapping the solver would change the answer.

Reading that off the source has been got wrong six times, in six ways, and
each one is a rule here.

A text window around each ``sum(w) == 1`` cannot see a constraint list
assembled in pieces, so the list is followed through its name, its ``+=`` and
its ``.append``.

An AST pass reading only the constraint list misses non-negativity declared as
``cp.Variable(n, nonneg=True)``, which is how most of this library writes it.
Both sources are read, and the counts they produce differ by more than a factor
of two.

Following a name by walking the whole module and keeping the last assignment
crosses function boundaries. SpSyDiD binds ``objective`` and ``constraints`` in
two functions, and the first solve was read as the second: wrong variable,
wrong constraints, wrong objective. DSC binds ``w`` in two, one declared
non-negative and one not, and the audit reported the sum-to-one solve as the
probability simplex -- "swap the solver here" about weights that may go
negative. Names are resolved inside the enclosing function, from assignments
above the call.

Matching any callable whose name ends in ``Problem`` matches a class.
LAXSCM caches compiled programs behind ``_PenalizedProblem`` and
``_RelaxedProblem``, and both constructors entered the audit as solves with
nothing to read.

A constraint appended inside an ``if`` holds on one branch. SHC appends
``w >= 0`` under one, so the feasible set is read whole for what it can
contain and without its conditional pieces for what it always contains.

The sixth is the one this module now spends most of its lines on. An objective
matched as text is matched on its syntax, and syntax is not the program.
``cp.norm(r, 2)`` and ``cp.sum_squares(r)`` have the same minimiser.
``quad_form(r, V)`` is a row scaling and ``quad_form(w, Q)`` a factorisation. A
ridge is extra design rows carrying no target. An intercept is profiled out by
centring. A penalty on a constraint the program already enforces is a constant
on the feasible set. Ten sites were filed as a different objective on those
grounds and all ten are the simplex least-squares program, checked against the
cvxpy solve they would replace (``agents/agents_simplex_audit.md``). So the
objective is split into a sum of terms, each term is matched against what the
solver can carry, and the reshaping the caller needs is recorded per site as
``transform``.

Run it directly for the classification, or import :func:`audit` for the rows.
``mlsynth/tests/test_simplex_qp_audit.py`` pins the eligible set and its
transforms, so a new cvxpy simplex problem has to be classified deliberately
instead of appearing in a count nobody re-derives.
"""
from __future__ import annotations

import ast
import pathlib
import re
from typing import Dict, List, NamedTuple, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1] / "mlsynth" / "utils"


class Site(NamedTuple):
    """One cvxpy problem carrying a sum-to-one constraint."""

    path: str
    lineno: int
    variable: str
    verdict: str          # see VERDICTS
    extras: Tuple[str, ...]
    constraints: str
    objective: str        # source of the Minimize argument, names resolved
    objective_kind: str   # "least-squares" | "other" | "unreadable"
    transform: str        # what the caller does to the data before the swap


VERDICTS = ("eligible", "wrong-objective", "unreadable",
            "extra-constraints", "no-nonnegativity")


# --------------------------------------------------------------------------- #
# Source-level algebra: splitting an expression into terms and factors.
# --------------------------------------------------------------------------- #

def _depths(s: str) -> List[int]:
    """Bracket depth before each character."""
    out, d = [], 0
    for ch in s:
        out.append(d)
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
    return out


def _balanced(s: str) -> bool:
    d = 0
    for ch in s:
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
            if d < 0:
                return False
    return d == 0


def _unwrap(s: str) -> str:
    """Strip whitespace and any redundant outer parentheses."""
    s = s.strip()
    while s.startswith("(") and s.endswith(")") and _balanced(s[1:-1]):
        s = s[1:-1].strip()
    return s


def _is_exponent(s: str, i: int) -> bool:
    """Whether ``s[i]`` is the sign of a float literal's exponent, as in ``1e-8``."""
    return i >= 2 and s[i - 1] in "eE" and s[i - 2].isdigit()


def _split_terms(s: str) -> List[Tuple[int, str]]:
    """Top-level sum, as ``(sign, term)`` pairs. A leading ``-`` signs its term."""
    s = _unwrap(s)
    d = _depths(s)
    out: List[Tuple[int, str]] = []
    sign, start = 1, 0
    for i, ch in enumerate(s):
        if d[i] or ch not in "+-" or _is_exponent(s, i):
            continue
        prev = s[:i].rstrip()
        if not prev or prev[-1] in "+-*/@(,":     # unary, or an operator's operand
            continue
        out.append((sign, s[start:i].strip()))
        sign, start = (1 if ch == "+" else -1), i + 1
    tail = s[start:].strip()
    if tail.startswith("-"):
        sign, tail = -sign, tail[1:].strip()
    if tail:
        out.append((sign, tail))
    return out


def _split_factors(s: str) -> List[str]:
    """Top-level product. ``lambd * sigma_y2 * cp.quad_form(w, Q)`` -> three."""
    s = _unwrap(s)
    d = _depths(s)
    out, start = [], 0
    for i, ch in enumerate(s):
        if d[i] or ch != "*" or s[i - 1: i] == "*" or s[i + 1: i + 2] == "*":
            continue
        out.append(s[start:i].strip())
        start = i + 1
    out.append(s[start:].strip())
    return [f for f in out if f]


def _peel(text: str, *names: str) -> Optional[str]:
    """Inner source of ``name(...)`` when ``text`` is exactly one such call."""
    text = _unwrap(text)
    for name in names:
        head = name + "("
        if text.startswith(head) and text.endswith(")") \
                and _balanced(text[len(head):-1]):
            return text[len(head):-1]
    return None


def _args(inner: str) -> List[str]:
    """Split a call's argument source on its top-level commas."""
    d = _depths(inner)
    out, start = [], 0
    for i, ch in enumerate(inner):
        if not d[i] and ch == ",":
            out.append(inner[start:i].strip())
            start = i + 1
    out.append(inner[start:].strip())
    return [a for a in out if a]


def _mentions(text: str, var: str) -> bool:
    return bool(var) and bool(re.search(rf"\b{re.escape(var)}\b", text))


# --------------------------------------------------------------------------- #
# What each term is, and what the solver has to be handed instead.
# --------------------------------------------------------------------------- #

_SQUARE = "square the objective, which has the same minimiser"
_METRIC = "scale the rows by the square root of the metric"
_FACTOR = "factor the Gram as R'R and take B = R"
_TARGET = "recovering the target from the linear term"
_IDENTITY = "augment the design with a multiple of the identity"
_SQRT_FACTOR = "augment the design with the penalty's square-root factor"
_SLACK = "the sum-to-one penalty is zero on the feasible set"
_CENTRE = "centre the design and the target to profile out the intercept"

# ``cp.norm(r)`` is the Euclidean norm; ``cp.norm(r, 2)`` says so explicitly.
# No other order shares a minimiser with the squared residual.
_EUCLIDEAN = {"", "2", "2.0", "'2'", '"2"'}


def _quadratic(core: str, var: str) -> Optional[Tuple[str, str]]:
    """A squared-residual term, as ``(shape, transform)``.

    ``shape`` is ``"fit"`` for a squared residual in the weights and ``"quad"``
    for a quadratic form in the weights alone, which is a fit when it stands
    by itself and a ridge when a fit is already present.
    """
    inner = _peel(core, "cp.sum_squares", "cp.SumSquares")
    if inner is not None:
        if _unwrap(inner) == var:
            return "quad", _IDENTITY
        # ``sum_squares(multiply(v, r))`` is the row scaling already written out.
        return "fit", (_METRIC if _peel(inner, "cp.multiply") else "")

    inner = _peel(core, "cp.norm2")
    if inner is None:
        inner = _peel(core, "cp.norm")
        if inner is not None:
            parts = _args(inner)
            order = parts[1] if len(parts) > 1 else ""
            if order.replace("p=", "").strip() not in _EUCLIDEAN:
                return None
            inner = parts[0]
    if inner is not None:
        return ("quad", _SQUARE) if _unwrap(inner) == var else ("fit", _SQUARE)

    inner = _peel(core, "cp.quad_form")
    if inner is not None:
        parts = _args(inner)
        if len(parts) < 2:
            return None                       # pragma: no cover - cvxpy needs two
        if _unwrap(parts[0]) == var:
            return ("quad", _FACTOR)
        if _mentions(parts[0], var):
            return ("fit", _METRIC)
    return None


def _is_linear(core: str, var: str) -> bool:
    """``coefficient @ var`` or ``var @ coefficient``: what ``linear`` carries."""
    core = _unwrap(core)
    d = _depths(core)
    at = [i for i, ch in enumerate(core) if ch == "@" and not d[i]]
    if len(at) != 1:
        return False
    left, right = _unwrap(core[:at[0]]), _unwrap(core[at[0] + 1:])
    return (right == var) != (left == var)


_SLACK_RE = re.compile(r"^cp\.sum\(\s*(?P<var>[\w.]+)\s*\)\s*-\s*1(\.0)?$")


def _is_slack(core: str, var: str) -> bool:
    """``(1'w - 1)^2``, which every feasible point sets to zero."""
    inner = _peel(core, "cp.square", "cp.sum_squares")
    if inner is None:
        return False
    m = _SLACK_RE.match(_unwrap(inner))
    return bool(m) and m.group("var") == var


def _classify_objective(text: str, var: str = "") -> Tuple[str, str]:
    """Read ``cp.Minimize(...)`` as a program, and say how to hand it over.

    Returns ``(kind, transform)``. ``kind`` is ``"least-squares"`` when the
    site solves ``min ||A - Bw||^2`` on the simplex, possibly plus a term
    linear in the weights; ``"unreadable"`` when the objective is a name with
    no binding to follow; ``"other"`` otherwise. ``transform`` names what the
    caller does to the data first, and is empty when nothing is needed.
    """
    inner = _peel(re.sub(r"\s+", " ", text).strip(), "cp.Minimize", "cp.minimize")
    if inner is None:
        return "other", ""
    inner = _unwrap(inner)
    if re.fullmatch(r"[A-Za-z_]\w*", inner):
        return "unreadable", ""
    listed = _peel(inner, "sum")
    if listed is not None:
        listed = _unwrap(listed)
        if listed.startswith(("[", "(")) and listed.endswith(("]", ")")):
            inner = " + ".join(_args(listed[1:-1]))

    fits: List[str] = []
    quads: List[str] = []
    steps: List[str] = []
    linear = False
    for sign, term in _split_terms(inner):
        factors = _split_factors(term)
        core = factors[-1] if factors else ""
        if _is_linear(core, var):
            linear = True
            continue
        if sign > 0 and _is_slack(core, var):
            steps.append(_SLACK)
            continue
        shape = _quadratic(core, var)
        if shape is None or sign < 0:
            return "other", ""            # concave, or a term nothing can carry
        kind, transform = shape
        (fits if kind == "fit" else quads).append(transform)

    if len(fits) > 1:
        # Two squared residuals are expressible, by stacking the designs. No
        # site writes it, so the answer stays conservative instead of naming a
        # transform nothing exercises.
        return "other", ""
    if not fits:
        if len(quads) != 1:
            return "other", ""
        steps.insert(0, quads[0] + (f", {_TARGET}" if linear else ""))
    else:
        steps = ([fits[0]] if fits[0] else []) + steps
        for q in quads:
            steps.append(_SQRT_FACTOR if q == _FACTOR else _IDENTITY)
    seen: List[str] = []
    for s in steps:
        if s and s not in seen:
            seen.append(s)
    return "least-squares", "; ".join(seen)


# --------------------------------------------------------------------------- #
# Reading the source: names resolved inside the function that binds them.
# --------------------------------------------------------------------------- #

def _enclosing(tree: ast.AST, lineno: int) -> ast.AST:
    """The innermost function containing ``lineno``, or the module."""
    best = tree
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and n.lineno <= lineno <= (n.end_lineno or n.lineno) \
                and (best is tree or n.lineno > best.lineno):
            best = n
    return best


def _cvxpy_objects(scope: ast.AST) -> Dict[str, Tuple[str, bool]]:
    """Names bound to a cvxpy ``Variable`` or ``Parameter``, with the nonneg flag.

    Both are leaves of an objective: substituting ``cp.Variable(N)`` for ``w``
    would destroy the expression the classifier is reading. Only a Variable is
    a candidate intercept, though -- mlSC's penalty grid is a ``cp.Parameter``
    and reading it as one put a centring step on a site that has no intercept.
    """
    out: Dict[str, Tuple[str, bool]] = {}
    for n in ast.walk(scope):
        if not (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)):
            continue
        func = ast.unparse(n.value.func)
        kind = next((k for k in ("Variable", "Parameter") if func.endswith(k)), "")
        if not kind:
            continue
        flag = any(k.arg == "nonneg" and getattr(k.value, "value", False) is True
                   for k in n.value.keywords)
        for t in n.targets:
            if isinstance(t, ast.Name):
                out[t.id] = (kind, out.get(t.id, ("", False))[1] or flag)
    return out


def _conditional_lines(scope: ast.AST) -> frozenset:
    """Lines inside an ``if``, whose statements run on one branch only."""
    out = set()
    for n in ast.walk(scope):
        if isinstance(n, ast.If):
            for stmt in list(n.body) + list(n.orelse):
                out.update(d.lineno for d in ast.walk(stmt)
                           if hasattr(d, "lineno"))
    return frozenset(out)


def _bindings(scope: ast.AST, src: str, before: int,
              skip: frozenset = frozenset()) -> Dict[str, str]:
    """Local names assigned above ``before``, mapped to their source.

    ``skip`` drops assignments on those lines. The audit reads the constraint
    list twice: once whole, for what the feasible set can contain, and once
    without the conditional pieces, for what it always contains.
    """
    out: Dict[str, str] = {}
    for n in ast.walk(scope):
        if getattr(n, "lineno", None) in skip:
            continue
        if isinstance(n, ast.Assign) and n.lineno < before:
            value = ast.get_source_segment(src, n.value) or ""
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = value
        elif isinstance(n, ast.AugAssign) and n.lineno < before \
                and isinstance(n.target, ast.Name):
            piece = ast.get_source_segment(src, n.value) or ""
            out[n.target.id] = (out.get(n.target.id, "") + " + " + piece).strip(" +")
        elif isinstance(n, ast.Call) and n.lineno < before and n.args \
                and isinstance(n.func, ast.Attribute) and n.func.attr == "append" \
                and isinstance(n.func.value, ast.Name):
            # ``objective_terms.append(penalty)``: a list assembled in pieces,
            # which mlSC uses to make its penalty term conditional. The audit
            # classifies the widest form, so the piece goes in unconditionally.
            held = out.get(n.func.value.id, "")
            piece = ast.get_source_segment(src, n.args[0]) or ""
            if held.endswith("]"):
                out[n.func.value.id] = f"{held[:-1].rstrip().rstrip(',')}, {piece}]"
    return out


_IDENT = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?![\w(])")
_MAX_EXPANSION = 4000


def _expand(text: str, bindings: Dict[str, str],
            leaves: Dict[str, Tuple[str, bool]]) -> str:
    """Substitute local names for their source, leaving cvxpy objects alone."""
    for _ in range(4):
        grown = _IDENT.sub(
            lambda m: (f"({bindings[m.group()]})"
                       if m.group() in bindings and m.group() not in leaves
                       else m.group()),
            text)
        if len(grown) > _MAX_EXPANSION:       # pragma: no cover - defensive:
            return text                       # a self-referential binding, as
        if grown == text:                     # in ``c = c + [w >= 0]``, grows
            return text                       # without converging
        text = grown
    return text                               # pragma: no cover - depth reached


# ``w <= 1`` is implied by ``w >= 0`` and ``sum(w) == 1``, so a site carrying it
# is still the probability simplex.
_REDUNDANT = re.compile(r"^\s*[\w.]+\s*<=\s*1(\.0)?\s*$")


def audit(root: pathlib.Path = ROOT) -> List[Site]:
    """Every cvxpy problem with a sum-to-one constraint, classified."""
    sites: List[Site] = []
    for path in sorted(root.rglob("*.py")):
        src = path.read_text()
        if "cp.Problem" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:                       # pragma: no cover - the tree
            continue                              # is the library's own source

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and ast.unparse(node.func).split(".")[-1] == "Problem"):
                continue
            arg: Optional[ast.AST] = None
            if len(node.args) >= 2:
                arg = node.args[1]
            else:
                for kw in node.keywords:
                    if kw.arg == "constraints":
                        arg = kw.value
            if arg is None:
                continue

            scope = _enclosing(tree, node.lineno)
            leaves = _cvxpy_objects(scope)
            bindings = _bindings(scope, src, node.lineno)
            certain = _bindings(scope, src, node.lineno,
                                skip=_conditional_lines(scope))

            def read(n: ast.AST, binds: Dict[str, str] = bindings) -> str:
                return re.sub(r"\s+", " ", _unwrap(_expand(
                    ast.get_source_segment(src, n) or "", binds, leaves)))

            flat = read(arg)
            always = read(arg, certain)
            obj_src = read(node.args[0]) if node.args else ""
            # A constraint list that arrives as an argument has no binding to
            # follow, so the audit cannot say this is not a simplex site.
            # fast_scm's ``_solve_qp_problem`` is one, and its callers do pass
            # simplex constraints; reading the whole module instead found them
            # in another function and reported them as this site's.
            if re.fullmatch(r"[A-Za-z_]\w*", flat):
                sites.append(Site(
                    str(path.relative_to(root.parent.parent)), node.lineno,
                    "?", "unreadable", (), flat, obj_src[:120], "unreadable", ""))
                continue
            if "== 1" not in flat:
                continue

            m = re.search(r"sum\((?:cp\.multiply\()?([\w.\[\]]+)", flat)
            var = m.group(1) if m else "?"
            # Read the objective after the variable is known: a linear term, a
            # ridge and a slack penalty are each defined against *this* variable.
            obj_kind, transform = _classify_objective(obj_src, var)
            base = var.split("[")[0].split(".")[-1]
            # Non-negativity has to hold on every path, so it is read off the
            # unconditional part: SHC appends ``w >= 0`` under an ``if``, and
            # counting that would report the site as the probability simplex
            # on the strength of a flag's default.
            nonneg = bool(re.search(rf"\b{re.escape(var)}\s*>=\s*0", always)) \
                or leaves.get(base, ("", False))[1]
            if obj_kind == "least-squares" and any(
                    n != base and kind == "Variable" and _mentions(obj_src, n)
                    for n, (kind, _) in leaves.items()):
                transform = "; ".join([_CENTRE] + ([transform] if transform else []))

            parts = [c.strip() for c in re.split(r",(?![^(\[]*[)\]])", flat.strip("[] "))
                     if c.strip()]
            extras = tuple(
                c for c in parts
                if not re.fullmatch(r"cp\.sum\(.*?\)\s*==\s*1", c)
                and not re.fullmatch(rf"{re.escape(var)}\s*>=\s*0", c)
                and not _REDUNDANT.match(c)
            )
            # The mirror of the non-negativity reading, and the one that would
            # cost more: a sum-to-one appended under an ``if`` leaves the
            # weights summing to nothing on the other branch. The site is
            # still listed, and the conditional constraint disqualifies it.
            if "== 1" not in always:
                extras += ("the sum-to-one constraint is conditional",)
            if not nonneg:
                verdict = "no-nonnegativity"
            elif extras:
                verdict = "extra-constraints"
            elif obj_kind == "least-squares":
                verdict = "eligible"
            elif obj_kind == "unreadable":
                verdict = "unreadable"
            else:
                verdict = "wrong-objective"
            sites.append(Site(
                str(path.relative_to(root.parent.parent)), node.lineno,
                var, verdict, extras, flat[:120], obj_src[:120], obj_kind,
                transform if verdict == "eligible" else ""))
    return sites


def main() -> None:  # pragma: no cover - the entry point, exercised by hand
    sites = audit()
    direct = [s for s in sites if s.verdict == "eligible" and not s.transform]
    reshaped = [s for s in sites if s.verdict == "eligible" and s.transform]

    print(f"\n[eligible, swap directly]: {len(direct)}")
    for s in direct:
        print(f"    {s.path}:{s.lineno}  on {s.variable}")
    print(f"\n[eligible, after reshaping the data]: {len(reshaped)}")
    for s in reshaped:
        print(f"    {s.path}:{s.lineno}  on {s.variable}\n        {s.transform}")
    for verdict, title in (
        ("wrong-objective", "the probability simplex, but a different objective"),
        ("unreadable", "the objective is not an expression at the call site"),
        ("extra-constraints", "simplex plus a constraint solve_simplex_qp cannot carry"),
        ("no-nonnegativity", "weights may go negative -- a different feasible set"),
    ):
        rows = [s for s in sites if s.verdict == verdict]
        print(f"\n[{verdict}] {title}: {len(rows)}")
        for s in rows:
            tail = (f"   extra: {list(s.extras)}" if s.extras
                    else f"   obj: {s.objective[:62]}" if verdict != "extra-constraints"
                    else "")
            print(f"    {s.path}:{s.lineno}  on {s.variable}{tail}")
    print(f"\n{len(sites)} cvxpy problems carry a sum-to-one constraint.")


if __name__ == "__main__":  # pragma: no cover
    main()
