---
description: Diagnose a failure to its root cause and leave a five-why test ladder behind
argument-hint: "[failing test, traceback, wrong number, or estimator name]"
---

# Root-Cause Analysis (five whys, as tests)

Walk a failure from the symptom back to the cause, and write each step down as a
test. The contract is in `agents/agents_tests.md`, section "Root-Cause Analysis:
the five whys as a test ladder" -- read it before starting. This command is the
procedure; that section is the reasoning.

Use it whenever something is wrong: a red test, a number that disagrees with a
reference, a benchmark that moved, a user report.

## Steps

1. **State the incident in facts.** What was observed, on what input, against
   what expectation. No hypothesis yet. If the trigger is a number, get both
   numbers and their difference. Do not start from the docstring or the commit
   message -- read the executed path.

2. **Reproduce it minimally and deterministically.** Smallest panel, fixed seed,
   fewest estimators. If it will not reproduce, that is the first finding.

3. **Build the ladder.** For each rung, write the test *before* diagnosing the
   next one, and record whether it passes:

   | Rung | Question | Instrument |
   | --- | --- | --- |
   | 0 | Which reported quantity looks wrong? | smoke / Layer 4 |
   | 1 | Which other outputs moved with it? | example tests on the result |
   | 2 | Which term or branch produced them? | Layer 1 unit tests |
   | 3 | Which invariant is violated? | `hypothesis` property |
   | 4 | Which contract was never enforced? | edge / failure tests |
   | 5 | Would the suite have caught it? | mutation |

   Expect rung 0 to pass. That is normal and is the reason for the ladder.
   Follow every branch a why opens; two faults reaching the same rung is common.

4. **Confirm the bottom.** Both must answer *no*:
   - Would the failure still have occurred if this cause were absent?
   - Will it recur if this cause is corrected and nothing else changes?

   A *yes* means the ladder continues. The bottom is usually a fault of
   omission -- an invariant nobody asserted.

5. **Check the blast radius before scoping the fix.** `grep` the faulty helper's
   callers. A defect found in one estimator that lives in a shared helper is a
   shared-helper fix, which per `CLAUDE.md` lands on its own branch first, with
   the estimator work rebased onto it.

6. **Fix, then verify with data.** Red-to-green is not verification. Add the
   mutant that reintroduces the defect to `tools/mutation/targets.toml` with a
   `models` line naming what it stands for, and confirm it is killed. A
   surviving mutant means the assertion is too weak, or the mutant is
   equivalent -- record which, never leave it ambiguous.

7. **Close the loose ends.** Any docstring, comment, doc page or benchmark note
   that states the old behaviour, or states a *reason* that turned out not to be
   the cause, is itself a fault. Fix those in the same change.

## Output

Report the ladder, one line per rung, with pass/fail and the evidence. Then the
root cause, the two confirmation answers, the blast radius, and the mutant that
proves the fix. If a rung could not be answered, say so instead of skipping it.

## Anti-patterns

- Writing the fix from the incident report alone.
- Trusting a comment about what the code does over the code.
- Stopping at the first plausible story when the why had two answers.
- Declaring success on a green run with no mutant.
