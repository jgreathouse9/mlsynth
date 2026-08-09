# Mutation testing

Who tests the tests. Coverage says a line executed; a property test says an
invariant held over generated inputs; mutation testing asks the remaining
question — if this code were wrong, would anything fail?

```bash
python tools/mutation/run_mutants.py                 # every target
python tools/mutation/run_mutants.py --target dataprep
python tools/mutation/run_mutants.py --list          # what is catalogued, and why
```

Exit code is 0 only when every mutant was killed. Survivors and mutants that
could not be applied both exit non-zero, because both mean the run measured
less than it claims to.

## Two instruments, not one

`cosmic-ray` applies general operators over the parso syntax tree — replace a
binary operator, flip a comparison, delete a statement — exhaustively and
without imagination. It answers: is any assertion in this module weak?

`run_mutants.py` applies the short catalogue in `targets.toml`: specific
defects a reviewer thought plausible, at the one site where each would be
meaningful. It answers: would we notice *this*?

Neither subsumes the other, and the reason is the one `agents/agents_tests.md`
already gives. Mutation operators perturb the program syntactically, so they
generate faults of commission. Several of the catalogued mutants model
omission or higher-level logic errors: "report only the first clash" is a
statement insertion at one place, "exclude only this cohort's treated units" is
a name swap that would be noise applied anywhere else. cosmic-ray cannot
generate those without drowning the meaningful site, and it was never meant to.

So: generic operator swaps do not belong in `targets.toml`. They duplicate
cosmic-ray partially and worse.

### cosmic-ray is blocked upstream

It cannot currently be installed. Its `yattag` dependency ships a legacy
`setup.py` that modern setuptools rejects (`install_layout` was removed), so
the wheel fails to build. Nothing in this repo works around that.

`module-path`, `test-command` and `timeout` in `targets.toml` are cosmic-ray's
own configuration keys, so the targets already describe a cosmic-ray session:

```bash
python tools/mutation/emit_cosmic_ray_config.py --out build/cosmic-ray
cosmic-ray init build/cosmic-ray/dataprep.toml dataprep.sqlite
cosmic-ray exec dataprep.toml dataprep.sqlite
cr-report dataprep.sqlite
```

The emitter is tested and its output is valid TOML today, so when the upstream
packaging is fixed the blocker costs a `pip install` and not a redesign.

## Reading a result

A survivor is a question, not a verdict. Either an assertion is too weak, or
the mutant is equivalent to the original — and telling those apart is formally
undecidable, which is why the score is a diagnostic and 1.0 is not a target.
Record the answer either way; an accepted survivor gets a line in
`targets.toml` saying why, the same way `# pragma: no cover` records an
unreachable branch.

A mutant that could not be applied measures nothing. It usually means the code
was refactored and the pattern no longer matches. Fix the pattern or retire the
mutant — never leave it, because a permanently unapplied mutant is a claim of
coverage that no longer exists.

## What the harness guarantees

Four properties, each with a test in `mlsynth/tests/test_mutation_harness.py`,
and each of them a bug that was hit while building it.

The mutant is verified to have applied. A pattern that no longer matches, or
matches more than one site, is an error and never a survivor. A `sed` range
address that silently failed to match once produced a clean-looking survivor
during this work, which would have been recorded as "the tests cannot see this
defect" when the defect was never introduced.

The baseline is checked first. If the scoped suite does not pass on the
unmutated module, every mutant "fails" it too and the run reports a perfect
score measuring nothing. The target is reported unusable instead.

Cached bytecode is purged, and the child is told not to write more. CPython
validates a `.pyc` on the source's size and its mtime in whole seconds, so a
mutant the same length as the code it replaces — `sum` for `max`, `t1` for
`t0` — matches the cache the baseline run just wrote and never gets compiled.
The false survivor that produces depends on which side of a second boundary
the writes land, which is the worst kind of wrong.

The module is restored byte for byte. `datautils.py` is CRLF throughout; a
text-mode read translates line endings inbound and a text-mode write does not
put them back, so a naive round trip rewrites all 1032 of them. The mutant
would be gone and the diff would not.

The harness also refuses to start against a dirty working tree, so a crashed
run can always be told from an edit.

## Adding a target

```toml
[[target]]
name = "my-module"
module-path = "mlsynth/utils/my_module.py"
test-command = "python -m pytest mlsynth/tests/test_my_module.py -q -p no:cacheprovider"
timeout = 120.0

  [[target.mutant]]
  id = "short-kebab-case-id"
  find = "the exact source text, matching exactly one site"
  replace = "the defect"
  models = "what real mistake this stands in for, and why a reader should care"
```

`models` is not decoration. A mutant nobody can describe in those terms is
usually a generic operator swap, which belongs to cosmic-ray.

Two unit tests keep the catalogue honest: every `find` must still match its
module exactly once, and every test file named in a `test-command` must exist.
Both fail in CI, not at the next mutation run.

## Not a merge gate

The workflow runs weekly and on demand, never on a pull request. A mutation
score is a diagnostic to read, not a threshold to defend, and the run is far
too slow to sit in front of a merge.
