---
description: Diagnose a failure to its root cause and leave a five-why test ladder behind
argument-hint: "[failing test, traceback, wrong number, or estimator name]"
---

# Root-Cause Analysis (five whys, as tests)

Walk a failure from the symptom back to its causes, and write each step down as a
test. The contract is in `agents/agents_tests.md`, section "Root-Cause Analysis:
the five whys as a test ladder" -- read it before starting. This command is the
procedure; that section is the reasoning.

Use it whenever something is wrong: a red test, a number that disagrees with a
reference, a benchmark that moved, a user report.

The ladder has two axes and both are mandatory.

* **Depth** -- the five whys. Symptom, then the stage that produced it, then the
  stage that produced *that*, down to an invariant nobody asserted.
* **Breadth** -- at each depth, the finite set of things that could have gone
  wrong at that stage. Enumerate the set, test each member, and clear them one
  at a time.

Depth alone finds *a* cause and stops. A wrong number usually has more than one,
and the first plausible story is the one that hides the rest.

## Steps

1. **State the incident in facts.** What was observed, on what input, against
   what expectation. No hypothesis yet. Get both numbers, their ratio and their
   difference. Name the estimand the reference number refers to: an average over
   a stated window is not a total, and a placebo is not an effect. Do not start
   from the docstring or the commit message -- read the executed path.

2. **Reproduce it minimally and deterministically.** Smallest panel, fixed seed,
   fewest estimators. If it will not reproduce, that is the first finding.

3. **Write the dependency chain.** Before descending, state what the reported
   number is a function of, in order, each stage depending only on the ones to
   its right. For an estimator that is usually:

   ```
   reported number  <-  aggregation  <-  effect series  <-  counterfactual
                    <-  weights  <-  objective + constraints  <-  code / data
   ```

   The chain is what makes breadth finite. Each link has a small, listable set
   of ways to be wrong, and clearing a link is a claim you can test.

4. **Descend the chain, clearing the cheapest link first.** At each link,
   enumerate the candidates *before* testing any of them, then test each against
   something with a known answer -- a simulation whose truth you set, or a
   reference implementation. Write down the result for every candidate,
   including the ones that pass: a cleared link is a finding, and it is what
   licenses moving on.

   The standing enumeration for the estimation link, in the order it is cheapest
   to check:

   | # | Candidate | How it is cleared |
   | --- | --- | --- |
   | 1 | An intercept is fitted where the formulation forbids one (or omitted where it is required) | read the built program, not the docstring |
   | 2 | The simplex constraint is missing or misstated (`sum(w) == 1`) | assert on the constraint list |
   | 3 | Non-negativity is absent, or strengthened to strict positivity | assert `w >= 0`, and that `w == 0` is attainable |
   | 4 | The objective is not the one the method specifies | solve the same panel under the correct objective and compare |
   | 5 | The solver returned without converging | assert `problem.status` is optimal |

   Only when every candidate at a link is cleared is that link ruled out and the
   next one down in scope. Clearing is valid only under a specification that
   gives the check power -- a design where two candidates coincide numerically
   cannot separate them, and a check that passes for that reason has not passed.

5. **Follow every branch, and count the causes.** A why with two answers is two
   descents. Keep going until each branch bottoms out, then state how many
   causes there are. A ladder reporting exactly one cause for a number that is
   wrong by a large factor should be treated as unfinished until the factor is
   accounted for. Where the faults are independent, their contributions
   multiply, and the product is the check: decompose the total error into one
   factor per link and confirm the factors reproduce it.

6. **Confirm the bottom.** Both must answer *no*, per cause:
   - Would the failure still have occurred if this cause were absent?
   - Will it recur if this cause is corrected and nothing else changes?

   A *yes* means the ladder continues. The bottom is usually a fault of
   omission -- an invariant nobody asserted.

7. **Check the blast radius before scoping the fix.** Two questions, and the
   first is nearly free. *How far downstream does it reach?* A fault damages
   only what is computed after it, so read the corruption's position in the
   chain: everything to its left is untouched by construction, and whole
   branches of the ladder retire without being measured. A value that is wrong
   is not the same as a value that did damage -- it is still a defect and still
   gets fixed, but it is not the explanation for a number. *How far sideways
   does it reach?* `grep` the faulty helper's callers. A defect found in one
   estimator that lives in a shared helper is a shared-helper fix, which per
   `CLAUDE.md` lands on its own branch first, with the estimator work rebased
   onto it.

8. **Fix, then verify with data.** Red-to-green is not verification. Add the
   mutant that reintroduces each defect to `tools/mutation/targets.toml` with a
   `models` line naming what it stands for, and confirm it is killed. A
   surviving mutant means the assertion is too weak, or the mutant is
   equivalent -- record which, never leave it ambiguous.

9. **Close the loose ends.** Any docstring, comment, doc page or benchmark note
   that states the old behaviour, or states a *reason* that turned out not to be
   the cause, is itself a fault. Fix those in the same change.

## The rungs, and what each one asks

| Rung | The why | The set to enumerate | Instrument |
| --- | --- | --- | --- |
| 0 | Which reported quantity looks wrong? | the reported scalars, and which estimand each claims to be | smoke / Layer 4 |
| 1 | Which other outputs moved with it? | every field on the result: weights, fit, series, diagnostics | example tests on the result |
| 2 | Which term or branch produced them? | the candidates at the failing link (table above) | Layer 1 unit tests |
| 3 | Which invariant is violated? | scale, permutation, duplication, feasibility, normalization | `hypothesis` property |
| 4 | Which contract was never enforced? | config validation, degenerate input, solver status | edge / failure tests |
| 5 | Would the suite have caught it? | one mutant per cause found | mutation |

Expect rung 0 to pass on a well-built estimator. That is normal and is the
reason for the ladder.

## Output

Report the chain and, for each link, every candidate with pass or fail and the
evidence. Then the causes, numbered, with the error factor each accounts for and
the product checked against the observed total. Then the two confirmation
answers per cause, the blast radius, and the mutant that proves each fix. If a
candidate could not be tested, say so instead of dropping it from the list.

## Anti-patterns

- Writing the fix from the incident report alone.
- Trusting a comment about what the code does over the code.
- Reporting one cause without accounting for the size of the discrepancy.
- Taking the reporter's hypothesis about which link is at fault as a starting
  point. It is evidence about where they looked, not about where the fault is.
- Confusing a value that is wrong with a value that did damage. Check its
  position in the chain before crediting it with anything.
- Descending past a link that was never actually cleared.
- Clearing a link on a design where the check had no power to fail.
- Declaring success on a green run with no mutant.
