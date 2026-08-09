"""Run the semantic mutant catalogue in ``targets.toml``.

    python tools/mutation/run_mutants.py                  # every target
    python tools/mutation/run_mutants.py --target dataprep
    python tools/mutation/run_mutants.py --list

For each declared mutant: patch the module, confirm the patch actually landed,
run that target's scoped test command, and restore the module. A test command
that fails means the mutant was killed -- the suite noticed the defect. A
command that passes means it survived, which is a question and not a
verdict: either an assertion is too weak, or the mutant is equivalent to the
original. Both are printed; neither is silently tolerated.

Two properties matter more than the score.

The patch is verified to have applied. A ``find`` that no longer matches, or
that matches more than one site, is reported as an error and never as a
survivor. Mistaking one for the other reports confidence that was never
measured, which is worse than reporting none.

The module is restored on every path -- pass, fail, exception, timeout,
interrupt. The harness refuses to start against a dirty working tree for the
same reason, so a crashed run can always be told apart from an edit.

What belongs in the catalogue, and what does not, is documented in
``targets.toml``. In short: specific defects a reviewer can imagine, not
generic operator swaps, which cosmic-ray generates exhaustively once it is
installable again.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

try:                                        # Python 3.11+
    import tomllib
except ModuleNotFoundError:                 # pragma: no cover - 3.10 and below
    import tomli as tomllib                 # type: ignore[no-redef]

KILLED = "killed"
SURVIVED = "survived"
NOT_APPLIED = "not-applied"


@dataclass(frozen=True)
class Mutant:
    id: str
    find: str
    replace: str
    models: str


@dataclass(frozen=True)
class Target:
    name: str
    module_path: Path
    test_command: str
    timeout: float
    mutants: List[Mutant] = field(default_factory=list)


@dataclass(frozen=True)
class MutantResult:
    target: str
    mutant_id: str
    outcome: str
    detail: str


def load_targets(path: Path) -> List[Target]:
    """Parse ``targets.toml`` into ``Target`` objects."""
    data = tomllib.loads(Path(path).read_text())
    targets = []
    for entry in data.get("target", []):
        targets.append(Target(
            name=entry["name"],
            module_path=Path(entry["module-path"]),
            test_command=entry["test-command"],
            timeout=float(entry.get("timeout", 300.0)),
            mutants=[Mutant(id=m["id"], find=m["find"], replace=m["replace"],
                            models=m.get("models", ""))
                     for m in entry.get("mutant", [])],
        ))
    return targets


def apply_mutant(source: str, mutant: Mutant) -> str:
    """Return ``source`` with ``mutant`` applied, or raise if it cannot be.

    Raises
    ------
    ValueError
        If the pattern is absent, or occurs more than once. An ambiguous
        pattern would mutate an arbitrary site, so the result would not be the
        defect the catalogue describes.
    """
    count = source.count(mutant.find)
    if count == 0:
        raise ValueError(
            f"pattern not found in the module; it has probably been "
            f"refactored. Update or retire the mutant: {mutant.find!r}"
        )
    if count > 1:
        raise ValueError(
            f"pattern occurs {count} times; it must identify exactly one site, "
            f"otherwise the mutant applied is not the one described: "
            f"{mutant.find!r}"
        )
    return source.replace(mutant.find, mutant.replace, 1)


def read_source(module: Path):
    """Return ``(raw_bytes, normalized_text, newline)`` for ``module``.

    Restoring has to be byte-exact, and matching has to be newline-agnostic.
    Text-mode reads translate line endings on the way in but not on the way
    out, so a CRLF file round-tripped through ``read_text``/``write_text``
    comes back with every line ending rewritten -- 1032 of them in
    ``datautils.py``, which is CRLF throughout. The raw bytes are kept for the
    restore, the text is normalized so patterns written with ``\n`` match
    either convention, and the file's own convention is reapplied on write.
    """
    raw = module.read_bytes()
    text = raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    return raw, text.replace("\r\n", "\n"), newline


def write_source(module: Path, text: str, newline: str) -> None:
    """Write ``text`` back using ``newline`` as the line ending."""
    module.write_bytes(text.replace("\n", newline).encode("utf-8"))


def run_target(target: Target, root: Path) -> List[MutantResult]:
    """Run every mutant of ``target``, restoring the module each time.

    The unmutated suite is run first. Without that check a test command that
    is already failing -- a renamed file, a missing dependency, a typo in the
    command -- scores every mutant as killed and reports a perfect result
    measuring nothing at all.
    """
    module = Path(root) / target.module_path
    raw, original, newline = read_source(module)
    results: List[MutantResult] = []

    baseline = _score(target, Mutant("<baseline>", "", "", ""), root)
    if baseline.outcome != SURVIVED:
        return [MutantResult(
            target.name, "<baseline>", NOT_APPLIED,
            "the test command does not pass on the unmutated module, so no "
            f"mutant can be scored against it ({baseline.detail})")]

    for mutant in target.mutants:
        try:
            mutated = apply_mutant(original, mutant)
        except ValueError as exc:
            results.append(MutantResult(target.name, mutant.id, NOT_APPLIED, str(exc)))
            continue

        try:
            write_source(module, mutated, newline)
            # Belt and braces: the patch is verified in memory above, and the
            # file is re-read here so a write that silently did nothing cannot
            # be read as a survivor either.
            if module.read_bytes() == raw:          # pragma: no cover - defensive
                results.append(MutantResult(
                    target.name, mutant.id, NOT_APPLIED,
                    "the module on disk is unchanged after writing the mutant"))
                continue
            results.append(_score(target, mutant, root))
        finally:
            module.write_bytes(raw)

    return results


def purge_bytecode(module: Path) -> None:
    """Delete any cached bytecode for ``module``.

    CPython validates a ``.pyc`` against the source's size and its modification
    time *in whole seconds*. A mutant the same length as the code it replaces
    -- ``sum`` for ``max``, ``t1`` for ``t0`` -- therefore matches a cache
    written moments earlier by the baseline run, and the mutated source is
    never compiled. The suite passes, the harness records a survivor, and
    whether that happens depends on which side of a second boundary the two
    writes land. A mutation harness that reports nondeterministic false
    survivors is worse than none, so the cache is removed before every run and
    the child is told not to write a new one.
    """
    cache = module.parent / "__pycache__"
    if not cache.is_dir():
        return
    for stale in cache.glob(f"{module.stem}.*.pyc"):
        stale.unlink(missing_ok=True)


def _score(target: Target, mutant: Mutant, root: Path) -> MutantResult:
    """Run the scoped test command and classify the outcome."""
    purge_bytecode(Path(root) / target.module_path)
    # agents_tests.md makes derandomization mandatory for any run feeding a
    # mutation score: a flakily-killed mutant corrupts it. Left to the default
    # the suite would run the randomized `dev` profile and a mutant could be
    # killed by an unlucky draw instead of by an assertion.
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "MLSYNTH_HYPOTHESIS_PROFILE": "ci",
    }
    try:
        completed = subprocess.run(
            shlex.split(target.test_command), cwd=str(root), env=env,
            capture_output=True, text=True, timeout=target.timeout,
        )
    except subprocess.TimeoutExpired:
        return MutantResult(target.name, mutant.id, KILLED,
                            f"the test command timed out after {target.timeout}s")
    except (OSError, ValueError) as exc:
        return MutantResult(target.name, mutant.id, NOT_APPLIED,
                            f"the test command could not be run: {exc}")

    if completed.returncode == 0:
        return MutantResult(target.name, mutant.id, SURVIVED,
                            f"the suite passed with this mutant in place; {mutant.models}")
    return MutantResult(target.name, mutant.id, KILLED,
                        f"exit {completed.returncode}")


def _report(results: Sequence[MutantResult]) -> int:
    """Print a table and return the process exit code."""
    width = max((len(f"{r.target}/{r.mutant_id}") for r in results), default=20)
    for result in results:
        print(f"  {result.target + '/' + result.mutant_id:<{width}}  "
              f"{result.outcome:<12}  {result.detail[:96]}")

    killed = sum(r.outcome == KILLED for r in results)
    survived = [r for r in results if r.outcome == SURVIVED]
    broken = [r for r in results if r.outcome == NOT_APPLIED]

    print(f"\n{killed}/{len(results)} killed, {len(survived)} survived, "
          f"{len(broken)} could not be applied")

    if survived:
        print("\nSurvivors are a question, not a verdict. For each: is the "
              "assertion too weak, or is the mutant equivalent to the "
              "original? Record the answer either way.")
    if broken:
        print("\nMutants that could not be applied measure nothing. Fix the "
              "pattern or retire the mutant.")
    return 1 if (survived or broken) else 0


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", help="run only this target")
    parser.add_argument("--list", action="store_true",
                        help="list targets and mutants without running them")
    parser.add_argument("--targets-file",
                        default=str(Path(__file__).parent / "targets.toml"))
    args = parser.parse_args(argv)

    targets = load_targets(Path(args.targets_file))
    if args.target:
        targets = [t for t in targets if t.name == args.target]
        if not targets:
            parser.error(f"no target named {args.target!r}")

    if args.list:
        for target in targets:
            print(f"{target.name}  ({target.module_path})")
            for mutant in target.mutants:
                print(f"    {mutant.id}: {mutant.models}")
        return 0

    dirty = subprocess.run(["git", "status", "--porcelain", "--"] +
                           [str(t.module_path) for t in targets],
                           cwd=str(root), capture_output=True, text=True)
    if dirty.stdout.strip():
        print("Refusing to run: a target module has uncommitted changes, so a "
              "crashed run could not be told from an edit.\n" + dirty.stdout,
              file=sys.stderr)
        return 2

    results: List[MutantResult] = []
    for target in targets:
        print(f"\n{target.name}: {len(target.mutants)} mutants against "
              f"{target.module_path}")
        results.extend(run_target(target, root))
    return _report(results)


if __name__ == "__main__":
    raise SystemExit(main())
