"""Targeted mutation test of the `force` decision points.

Coverage says a line ran. This asks the sharper question: if the line were
wrong, would a test fail? Each mutant is a single textual edit to one of the
places `force` decides something -- the two predicates, the demeaning branches
in Step 1, the intercept column in Step 2, the rank ceiling, and the config
alias. A mutant that SURVIVES is a hole in the suite.

Blind mutation over the whole package would mostly mutate arithmetic the
reference comparison already pins. These are the branches `force` introduced,
which nothing outside the unit tests guards.

Run it from the repo root::

    python tools/mutate_gsynth_force.py

It edits source files in place and restores them in a ``finally``, so do not
run it against a dirty tree. Exit status is 0 when every mutant is killed.

Two mutants were surviving when this was written, and both were real. Making
Step 1 sweep out period effects unconditionally moved ``force="none"`` from
4.1105 to 3.9784 with the suite still green, because nothing pinned that
setting against a reference. And dropping the intercept from the regressor
count relaxed the guard that refuses a rank the pre-period cannot identify,
which only bit exactly at the boundary. Two tests were added for those, not a
tolerance loosened.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IFE = REPO / "mlsynth/utils/gsynth_helpers/ife.py"
PIPE = REPO / "mlsynth/utils/gsynth_helpers/pipeline.py"
CFG = REPO / "mlsynth/utils/gsynth_helpers/config.py"

# (label, file, old, new)
MUTANTS = [
    # --- the two predicates ---
    ("has_unit_effects drops 'unit'", IFE,
     'return force in ("unit", "two-way")', 'return force in ("two-way",)'),
    ("has_unit_effects drops 'two-way'", IFE,
     'return force in ("unit", "two-way")', 'return force in ("unit",)'),
    ("has_unit_effects always True", IFE,
     'return force in ("unit", "two-way")', 'return True'),
    ("has_unit_effects always False", IFE,
     'return force in ("unit", "two-way")', 'return False'),
    ("has_time_effects drops 'time'", IFE,
     'return force in ("time", "two-way")', 'return force in ("two-way",)'),
    ("has_time_effects drops 'two-way'", IFE,
     'return force in ("time", "two-way")', 'return force in ("time",)'),
    ("has_time_effects always True", IFE,
     'return force in ("time", "two-way")', 'return True'),
    ("has_time_effects always False", IFE,
     'return force in ("time", "two-way")', 'return False'),
    ("predicates swapped", IFE,
     'return force in ("unit", "two-way")', 'return force in ("time", "two-way")'),

    # --- Step 1 demeaning branches ---
    ("Step 1 sweeps unit effects always", IFE,
     "    if has_unit_effects(force):\n        alpha_Y = YY.mean(axis=0)",
     "    if True:\n        alpha_Y = YY.mean(axis=0)"),
    ("Step 1 never sweeps unit effects", IFE,
     "    if has_unit_effects(force):\n        alpha_Y = YY.mean(axis=0)",
     "    if False:\n        alpha_Y = YY.mean(axis=0)"),
    ("Step 1 sweeps period effects always", IFE,
     "    if has_time_effects(force):\n        xi_Y = YY.mean(axis=1)",
     "    if True:\n        xi_Y = YY.mean(axis=1)"),
    ("Step 1 never sweeps period effects", IFE,
     "    if has_time_effects(force):\n        xi_Y = YY.mean(axis=1)",
     "    if False:\n        xi_Y = YY.mean(axis=1)"),
    ("returned alpha zeroed", IFE,
     "alpha=alpha_Y - alpha_X @ beta,", "alpha=np.zeros(N),"),
    ("returned xi zeroed", IFE,
     "xi=xi_Y - xi_X @ beta,", "xi=np.zeros(T),"),
    ("force validation removed", IFE,
     "    if force not in FORCE_OPTIONS:", "    if False:"),

    # --- Step 2 intercept column ---
    ("intercept column always appended", PIPE,
     "    if has_unit_effects(force):\n        return np.hstack",
     "    if True:\n        return np.hstack"),
    ("intercept column never appended", PIPE,
     "    if has_unit_effects(force):\n        return np.hstack",
     "    if False:\n        return np.hstack"),
    ("intercept keyed off time effects", PIPE,
     "    if has_unit_effects(force):\n        return np.hstack",
     "    if has_time_effects(force):\n        return np.hstack"),

    # --- rank ceiling and regressor count ---
    ("rank ceiling ignores the intercept", PIPE,
     "- (2 if has_unit_effects(force) else 1), 0)", "- 1, 0)"),
    ("rank ceiling always reserves one", PIPE,
     "- (2 if has_unit_effects(force) else 1), 0)", "- 2, 0)"),
    ("regressor count ignores the intercept", PIPE,
     "k = int(r) + (1 if has_unit_effects(force) else 0)", "k = int(r)"),
    ("unit_effects reported regardless", PIPE,
     "unit_effects = (loadings[:, -1] if has_unit_effects(force)\n                    else np.zeros(inputs.n_treated))",
     "unit_effects = loadings[:, -1]"),

    # --- config alias ---
    ("two_way alias inverted", CFG,
     'resolved = "two-way" if values.two_way else "time"',
     'resolved = "time" if values.two_way else "two-way"'),
    ("two_way alias always two-way", CFG,
     'resolved = "two-way" if values.two_way else "time"',
     'resolved = "two-way"'),
    ("both-given check removed", CFG,
     '            if "force" in values.model_fields_set:', "            if False:"),
]

TESTS = ["mlsynth/tests/test_gsynth.py", "-q", "-x", "-p", "no:cacheprovider",
         "--no-header", "-W", "ignore"]


def run_tests() -> bool:
    """True when the suite passes."""
    p = subprocess.run([sys.executable, "-m", "pytest", *TESTS],
                       cwd=REPO, capture_output=True, text=True, timeout=1800)
    return p.returncode == 0


def main() -> int:
    originals = {f: f.read_text() for f in (IFE, PIPE, CFG)}

    if not run_tests():
        print("baseline suite is RED; fix that before mutating")
        return 2
    print(f"baseline green; {len(MUTANTS)} mutants\n")

    survived = []
    try:
        for i, (label, path, old, new) in enumerate(MUTANTS, 1):
            src = originals[path]
            if old not in src:
                print(f"[{i:2d}/{len(MUTANTS)}] SKIP  {label} (pattern absent)")
                survived.append((label, "pattern absent"))
                continue
            path.write_text(src.replace(old, new, 1))
            try:
                passed = run_tests()
            finally:
                path.write_text(src)
            mark = "SURVIVED" if passed else "killed"
            if passed:
                survived.append((label, "tests still pass"))
            print(f"[{i:2d}/{len(MUTANTS)}] {mark:8s} {label}")
    finally:
        for f, src in originals.items():
            f.write_text(src)

    print(f"\n==== {len(MUTANTS) - len(survived)}/{len(MUTANTS)} killed ====")
    if survived:
        print("SURVIVING MUTANTS (holes in the suite):")
        for label, why in survived:
            print(f"  - {label}  [{why}]")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
