"""Classify every cvxpy problem in the library by whether it is the simplex QP.

``bilevel/active_set.py::solve_simplex_qp`` minimises ``||A - Bw||^2`` over the
probability simplex ``{w >= 0, sum(w) = 1}``. A cvxpy site can be moved onto it
only if its feasible set is exactly that: a site with an extra constraint, or
one whose weights may go negative, is a different problem and swapping the
solver would change the answer, not the speed.

Reading that off the source is easy to get wrong, and this module exists
because it was got wrong twice. A text window around each ``sum(w) == 1``
cannot see a constraint list assembled in pieces. An AST pass that reads only
the constraint list misses non-negativity declared as
``cp.Variable(n, nonneg=True)``, which is how most of this library writes it.
Both sources have to be read, and the counts they produce differ by more than
a factor of two.

Run it directly for the classification, or import :func:`audit` for the rows.
``mlsynth/tests/test_simplex_qp_audit.py`` pins the eligible set so that a new
cvxpy simplex problem has to be classified deliberately instead of appearing
in a count nobody re-derives.
"""
from __future__ import annotations

import ast
import pathlib
import re
from typing import Dict, List, NamedTuple, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1] / "mlsynth" / "utils"


class Site(NamedTuple):
    """One cvxpy problem carrying a sum-to-one constraint."""

    path: str
    lineno: int
    variable: str
    verdict: str          # "eligible" | "extra-constraints" | "no-nonnegativity"
    extras: Tuple[str, ...]
    constraints: str
    objective: str        # source of the Minimize argument, name resolved
    objective_kind: str   # "least-squares" | "gram" | "other"


def _constraint_source(node: ast.AST, src: str, tree: ast.AST) -> str:
    """Source of a ``Problem``'s constraints argument, resolving a local name.

    A constraints list is often built into a variable and appended to before
    the ``Problem`` call, so following the name to its assignments is what
    makes the list visible at all.
    """
    if isinstance(node, (ast.List, ast.Tuple)):
        return ast.get_source_segment(src, node) or ""
    if isinstance(node, ast.Name):
        found = ""
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id == node.id:
                        found = ast.get_source_segment(src, n.value) or found
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) \
                    and n.target.id == node.id:
                found += " + " + (ast.get_source_segment(src, n.value) or "")
        return found
    return ast.get_source_segment(src, node) or ""


def _nonneg_variables(tree: ast.AST) -> Dict[str, bool]:
    """Names assigned a ``cp.Variable``, and whether it was declared nonneg."""
    out: Dict[str, bool] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) \
                and ast.unparse(n.value.func).endswith("Variable"):
            flag = any(k.arg == "nonneg" and getattr(k.value, "value", False) is True
                       for k in n.value.keywords)
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = out.get(t.id, False) or flag
    return out


# ``w <= 1`` is implied by ``w >= 0`` and ``sum(w) == 1``, so a site carrying it
# is still the probability simplex.
_REDUNDANT = re.compile(r"^\s*[\w.]+\s*<=\s*1(\.0)?\s*$")

# ``solve_simplex_qp`` minimises ``||A - Bw||^2`` and nothing else. A simplex
# constraint set is only half of eligibility: a site minimising an infinity
# norm, or a least-squares objective plus a ridge term, is a different program
# and swapping the solver would change its answer.
_LSQ = re.compile(r"^cp\.Minimize\(\s*cp\.sum_squares\([^()]*(?:\([^()]*\)[^()]*)*\)\s*\)$")
_GRAM = re.compile(r"quad_form")


def _classify_objective(text: str) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    if _LSQ.match(flat):
        return "least-squares"
    if _GRAM.search(flat):
        return "gram"
    return "other"


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
        nonneg_vars = _nonneg_variables(tree)

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and ast.unparse(node.func).endswith("Problem")):
                continue
            arg: ast.AST | None = None
            if len(node.args) >= 2:
                arg = node.args[1]
            else:
                for kw in node.keywords:
                    if kw.arg == "constraints":
                        arg = kw.value
            if arg is None:
                continue
            flat = re.sub(r"\s+", " ", _constraint_source(arg, src, tree))
            if "== 1" not in flat:
                continue
            obj_src = (_constraint_source(node.args[0], src, tree)
                       if node.args else "")
            obj_kind = _classify_objective(obj_src)

            m = re.search(r"sum\((?:cp\.multiply\()?([\w.\[\]]+)", flat)
            var = m.group(1) if m else "?"
            base = var.split("[")[0].split(".")[-1]
            nonneg = bool(re.search(rf"\b{re.escape(var)}\s*>=\s*0", flat)) \
                or nonneg_vars.get(base, False)

            parts = [c.strip() for c in re.split(r",(?![^(\[]*[)\]])", flat.strip("[] "))
                     if c.strip()]
            extras = tuple(
                c for c in parts
                if not re.fullmatch(r"cp\.sum\(.*?\)\s*==\s*1", c)
                and not re.fullmatch(rf"{re.escape(var)}\s*>=\s*0", c)
                and not _REDUNDANT.match(c)
            )
            verdict = ("eligible" if nonneg and not extras and
                       obj_kind == "least-squares"
                       else "wrong-objective" if nonneg and not extras
                       else "extra-constraints" if nonneg
                       else "no-nonnegativity")
            sites.append(Site(
                str(path.relative_to(root.parent.parent)), node.lineno,
                var, verdict, extras, flat[:120],
                re.sub(r"\s+", " ", obj_src)[:120], obj_kind))
    return sites


def main() -> None:  # pragma: no cover - the entry point, exercised by hand
    sites = audit()
    for verdict, title in (
        ("eligible", "simplex constraints AND a plain least-squares objective"),
        ("wrong-objective", "the probability simplex, but a different objective"),
        ("extra-constraints", "simplex plus a constraint solve_simplex_qp cannot carry"),
        ("no-nonnegativity", "weights may go negative -- a different feasible set"),
    ):
        rows = [s for s in sites if s.verdict == verdict]
        print(f"\n[{verdict}] {title}: {len(rows)}")
        for s in rows:
            tail = (f"   extra: {list(s.extras)}" if s.extras
                    else f"   obj: {s.objective[:62]}" if verdict == "wrong-objective"
                    else "")
            print(f"    {s.path}:{s.lineno}  on {s.variable}{tail}")
    print(f"\n{len(sites)} cvxpy problems carry a sum-to-one constraint.")


if __name__ == "__main__":  # pragma: no cover
    main()
