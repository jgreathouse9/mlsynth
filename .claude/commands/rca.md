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

1. **State the incident in facts, the way a good Statalist post does.** A
   diagnosis is only as good as the report it starts from, and a vague report
   sends the first hour down the wrong branch. Before any hypothesis, write:

   - **The exact call.** The function or estimator and every argument that was
     passed, and the defaults that matter for the question (`gap=0` is the
     default and decides whether a proof is exact). Not "I ran the
     certificate"; `certify_staged(z, lo=2, hi=2)` with `z` the 174 pretest
     totals.
   - **Exactly what happened, in the words of the program.** "Raised
     `MlsynthEstimationError: no admissible pairs`", "returned `proved=True`
     with `gap_proved=-9.64`", "ran 30 minutes and was killed by `timeout`
     (exit 124) with no output". Say "crashed" only for a crash. Say "hung"
     only when you know it was not progressing; a long search is not a hang.
   - **Numbers, not adjectives.** Not "slow": 9.0 s. Not "many groups":
     1,012,347 surviving groups. Not "a big panel": 174 markets, 364 days.
     Both numbers being compared, their ratio and their difference.
   - **How the observation was made.** The command line, whether output went
     through a pipe (`| grep` and `| tail` buffer until the process ends, so
     "no output" may mean "not finished yet", not "nothing done"), what ran
     beside it on the machine, and the thread counts.
   - **Versions.** The commit SHA, the Python version, the versions of the
     libraries on the path (numpy, scipy, cvxpy, highspy, numba), and the CPU
     count. A report without them cannot be rerun.
   - **The data, shown.** Its shape and a summary (n, T, min, median, max, the
     ratio of the largest unit to the median), and a way to rebuild it: a seed
     and generator, a public file and the rows used, or a short extract. A
     private panel is never named or attached; it is described by its
     statistics and replaced by a synthetic one (step 2).
   - **The expectation and where it comes from.** The paper's table, a
     reference implementation's output, a brute-force oracle, or a property of
     the problem (no pairing beats the sorted pairing). Name the estimand each
     number refers to: an average over a stated window is not a total, and a
     placebo is not an effect.

   Do not start from the docstring or the commit message -- read the executed
   path. Correct the report as soon as a measurement contradicts it, and keep
   the correction in the write-up: a misreading that was undone is an
   eliminated hypothesis, and the next person will make it too.

2. **Reproduce it synthetically, small, against the current code, before
   anything else.** Build the smallest instance whose answer is known in
   advance and that shows the failure: ten units, not a hundred and seventy; a
   brute-force optimum, a planted effect, a closed-form answer, or a reference
   implementation's output on a public dataset. Put the suspected property in
   as a dial (a unit eight times the median; a group-size limit other than the
   default; the heuristic on and off) and turn it until the failure appears.

   - Commit the reproduction as a test on the fix branch, before the fix, and
     show it failing for the stated reason on the current code. It becomes the
     regression test the fix turns green. A reproduction that lives only in a
     scratch directory is lost with the container.
   - If it will not reproduce small, that is the first finding: the cause needs
     size, or the report is wrong. Say which, and grow the instance one knob at
     a time.
   - The large instance stays as the incident report; the descent happens on
     the small one, where each run takes seconds.

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

## When the descent stalls: instruments for a hard ladder

The steps above find most faults. Some faults hide below a link that keeps
"passing", or behind a fix that moves the cost somewhere else. These instruments
were what found the causes in the #965 RCA, where a hand-written interior-point
solver was 4 to 50% slower than Clarabel on the B&B beds. They are listed in the
order they were needed.

1. **Decompose the reported number into factors, and decompose it again after
   every fix.** Write the total as a product (or a sum) of measured factors:
   there, bed time = solves x iterations per solve x time per iteration, plus the
   time outside the solver. The product has to reproduce the measured ratio
   exactly, or a factor is missing. Re-run the decomposition after each candidate
   fix. A fix can shrink one factor and grow another: an exit rule that cut
   iterations by 60% left the bed totals unchanged, because the shallower points
   it returned added cut rounds outside the solver. Without the second
   decomposition that fix would have been reported as a success.

2. **Pin the measurement before trusting it.** Run each arm alone, with thread
   counts set for every library in every arm, and record the conditions next to
   the numbers. Re-measure the baseline under the same conditions. In #965 a
   10-20x per-iteration gap was the BLAS thread pool of one arm competing with
   other jobs on the machine; on a quiet machine it was 1.0x. A timing taken while
   other work runs is a hypothesis, not a measurement.

3. **Build a problem whose answer is known in advance, with the suspected
   property as a dial.** Plant the solution (a known optimum, a known effect, a
   known support), then add the property you suspect one knob at a time: there,
   the number of directions that lack strict complementarity, then the number of
   redundant active rows. Grow the toy until it reproduces the symptom. If a knob
   does not reproduce it, that hypothesis is eliminated, and that elimination is a
   finding. A `hypothesis` search over the toy's knobs finds cases the
   hand-picked ones miss. In #965 it found a second, independent cause (the
   starting point) on an instance with none of the suspected degeneracy.

4. **Run the reference side by side, and narrow the comparison in stages.**
   With a reference implementation available (another solver, the R package, the
   paper's code), compare at increasing granularity until the first point of
   disagreement is found.
   - Same input, both implementations: are the outputs different at all, and in
     which field?
   - Crossover arms: give the downstream pipeline one implementation's output A
     with the other's output B, both ways. The factor that moves with A is the one
     carrying the defect. In #965 the extra cut rounds followed the primal point,
     not the multipliers, which retired the multiplier branch in one run.
   - State handoff: start your implementation from the reference's intermediate
     state. If yours fails from the reference's good state, the defect is in your
     step. If it recovers, the defect is in your path. In #965 ours stalled from
     the reference's well-centred iterate on the first iteration, which ruled out
     every history-dependent candidate at once.
   - Per-block instrumentation at the failing step: log which component limits
     the step, residual or update, and descend into that one. There, it was the
     PSD dual step, at 4e-4 against 0.7 to 0.9 for every other block.

5. **Read the reference's source, then test the port before believing it.** A
   port is code you wrote, so it gets a test of its own invariant before it is
   used to clear a link. In #965 two ports were wrong. A termination rule was
   misread: the reference stops on a residual increase only below the full
   tolerance, and the port stopped on any uptick. A starting point had a sign
   error, caught only when its mutant survived and a test of its exact
   least-squares residual was added.

6. **Suspect the formula when the mathematics is right.** Two expressions equal
   in exact arithmetic can differ by orders of magnitude in floating point. On a
   degenerate or ill-conditioned instance, ask which side of an identity is well
   determined, and compute from that side. In #965 the dual step summed large
   terms that cancel on a degenerate face. Computing it from the primal step
   through the scaling (how CVXOPT does it) moved the attainable gap from 1e-5 to
   1e-9. Check what a stopping criterion is relative to: a backward error of 1e-16
   relative to |H||x| can sit beside an absolute error of 1e-6.

7. **Compare aggregates when trajectories are chaotic.** If a 1-ulp change
   (summation order, a thread count) changes a case's iteration count by +/-50%,
   single-case comparisons are noise. Decide on totals over many cases, and say so
   in the write-up.

8. **Ablate at the end.** With every fix in, remove each one in turn and
   re-measure. A fix whose removal changes nothing is a guard, not a cause, and
   the report should say which is which. In #965 the stall exit that first looked
   like the fix turned out to be downstream of the depth defect: with the depth
   fixed, removing it changed 375 iterations to 372.

9. **Leave the reproduction where it survives.** Scratch directories and
   containers are not durable. Put the exact configuration of every run that a
   conclusion rests on (inputs, seeds, ceilings, solver settings, thread counts,
   arms) in the write-up itself, so the run can be rebuilt from the issue alone.
   In #965 a container restart lost every scratch file, and item 1 could be rerun
   only because its configuration was in the issue comment.

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
- Declaring a fix from the factor it improved, without re-decomposing the total.
- Timing an arm while other work runs on the machine, or with thread counts left
  to each library's default.
- Clearing a link with a port of the reference that has no test of its own.
- Reporting a guard as a cause because it was the first change that helped.
- Leaving the only copy of a reproduction in a scratch directory.
