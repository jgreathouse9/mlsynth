# agents_unix.md

# The Unix Philosophy in `mlsynth`

This file settles which parts of the Unix philosophy `mlsynth` follows, which it
re-implements in a different medium, and which it refuses. It exists so that
"that isn't very Unix" stops being a matter of taste and becomes a citation.

The source is Raymond, *The Art of Unix Programming* (Addison-Wesley, 2003),
chapter 1, which collects McIlroy's 1978 summary, Pike's *Notes on C
Programming*, and the seventeen rules abstracted from what the Unix elders did.
Chapter and rule references below are to that book.

The thesis in one line:

> The Unix rules are claims about how to hold global complexity down. `mlsynth`
> adopts the claims and re-implements the 1978 mechanisms in the medium it
> actually has — Python packages, Pydantic models, and a typed result contract.

---

# The seventeen rules, and where each one stands

Four statuses. **binding** means the repository already enforces it somewhere and
this file only supplies the name. **adopted** means this file makes it a rule
where it was not one. **adapted** means the goal transfers and the mechanism does
not. **partly refused** means part of the rule conflicts with what an
econometrics library owes its user.

| # | Rule (§1.6) | Status | Where it lives here |
| --- | --- | --- | --- |
| 1 | Modularity — simple parts, clean interfaces | binding | one estimator = one package (`CLAUDE.md` invariant 4); `agents_utils.md` |
| 2 | Clarity is better than cleverness | binding | `agents_docs.md`; docstrings are publication-quality |
| 3 | Composition — design to be connected | adapted | the result contract is the universal interface; `mlsynth/spec.py` is the text boundary |
| 4 | Separation — policy from mechanism | adopted | see "Separation" below; the plotters are the open case |
| 5 | Simplicity — complexity only where you must | binding | match the nearest estimator before inventing (`CLAUDE.md` invariant 6) |
| 6 | Parsimony — a big program only by demonstration | adopted | `fit()` orchestrates; a dispatcher subpackage, never a new top-level estimator |
| 7 | Transparency — design for inspection | binding | `MethodDetailsResults`, `FitDiagnosticsResults` |
| 8 | Robustness — the child of transparency and simplicity | binding | the edge level in `agents_tests.md` |
| 9 | Representation — fold knowledge into data | binding | Pydantic configs and result models; dispatch by table |
| 10 | Least Surprise | binding | `CLAUDE.md` invariant 6 |
| 11 | Silence — say nothing when there is nothing surprising | adopted | see "Silence" below; 36 `print` calls are the open case |
| 12 | Repair — fail early and diagnosably | partly refused | the exception contract; Postel's half is refused, see below |
| 13 | Economy — programmer time over machine time | binding | `cvxpy` / `osqp` over hand-rolled solvers |
| 14 | Generation — write programs to write programs | binding | `hypothesis`, `parametrize`, `tools/gen_llms_txt.py`, `tools/mutation/targets.toml` |
| 15 | Optimization — prototype before polishing | binding | the demonstrate-first loop in `.claude/commands/replicate.md` |
| 16 | Diversity — distrust one true way | binding | four instruments (`agents_tests.md`); cross-implementation differential testing |
| 17 | Extensibility — design for the future | binding | standardized result sub-models; the versioned spec file |

McIlroy's three-sentence summary is the compression of all of it:

> Write programs that do one thing and do it well. Write programs to work
> together. Write programs to handle text streams, because that is a universal
> interface.

`mlsynth` takes the first two literally. The third is where the medium argument
starts.

---

# The universal interface is the result contract

McIlroy's case for text is a case for *one* interface that every tool speaks, so
that any tool can be replaced without disturbing its neighbours. Raymond gives
the reason in §1.6: "The simplicity of the text-stream interface enforces the
encapsulation of the tools." Text was the widest interface available between
Unix processes, and the limitation was the point — a format too poor to express
internal state cannot leak internal state.

`mlsynth` runs in one process, and the same discipline is available at a higher
level. The two-family result contract (`agents_results.md`) is the universal
interface: every `fit()` emits an `EffectResult` or a `DesignResult`, populated
from the same six sub-models. That is what makes `counterfactual_compare.py` and
`design_compare.py` possible at all — they consume the contract, not any
particular estimator, so an estimator can be swapped for another without either
one knowing.

The encapsulation argument survives the translation, and the type checker is
strictly stronger than a byte stream at enforcing it. What does not survive is
inspectability by general-purpose tools, and that is a real loss, paid for by
catching a misnamed field at construction instead of three stages downstream.

So: the interface between `mlsynth` stages is the typed contract, not text. Two
consequences bind.

1. Anything crossing an estimator boundary goes through the contract. A helper
   returning a bare tuple that a second estimator then unpacks positionally is
   the anti-pattern — that is the promiscuity §1.6 warns about, with none of
   text's poverty to prevent it.
2. Everything must still be able to *become* text at the outer boundary.
   `mlsynth/spec.py` is that boundary and is already built: `save_spec` writes
   every column name and method option to JSON or YAML, `load_spec` reads it back
   into a ready-to-fit estimator, and the `DataFrame` stays a runtime payload. An
   analysis specification is therefore version-controllable, diffable, and
   greppable — the Unix property, at the edge where it pays.

The benchmark layer takes the same shape and should keep it. A case computes and
returns a record; `benchmarks/run_benchmarks.py` and `benchmarks/compare.py`
decide what to display. A case that prints its own table cannot be diffed,
aggregated, or re-run under a different reporter.

---

# The rule that is partly refused: Postel's Prescription

Raymond pairs the Rule of Repair with Postel: "Be liberal in what you accept, and
conservative in what you send" (§1.6).

`mlsynth` refuses the first half, and does so deliberately. A liberal parser that
coerces a ragged panel, silently drops a unit with missing pre-periods, or
accepts an unrecognized keyword produces a number that looks like an estimate and
is not one. The cost of leniency in a text filter is a garbled line; the cost here
is an invalid causal claim that no downstream check will flag. So `extra="forbid"`
stands, validators fail early with `MlsynthConfigError` / `MlsynthDataError`, and
`dataprep` refuses what it cannot interpret.

McIlroy supplies the ground for this in the same section, qualifying Postel: "It
is the specifications that should be generous, not their interpretation." The
generosity belongs in what `dataprep` documents itself as accepting — long or
wide panels, staggered adoption, unbalanced input that `balance` can repair — and
not in what a validator will wave through.

The second half of Postel is kept whole, and strengthened: conservative in what
we send means the result contract, populated and typed.

Auto-correction survives inside the three conditions `agents_intro.md` already
sets — deterministic behavior, unambiguous intent, and a warning emitted. That
is the Rule of Repair proper: repair what can be repaired, and when repair is not
certain, fail at once and say why.

---

# Separation: policy from mechanism

The rule (§1.6): separate policy from mechanism, and interfaces from engines.
Raymond's justification is that the two change on different schedules, so fusing
them makes policy rigid and destabilizes the mechanism when policy moves. He adds
the consequence that matters here: separating them "make[s] it much easier to
write good tests for the mechanism".

`mlsynth`'s engine is estimation. Its policies are display, saving, formatting,
and verbosity. The estimator layer holds the first and should hold none of the
second.

The open case is plotting. Measured at `544c684`: 72 `plt.show()` calls live in
library code, 23 of them unconditional at the top level of a function, and 22
`plot_*` functions in `utils/*_helpers/plotter.py` build a figure, display it, and
return nothing. The figure — the mechanism's actual output — is unreachable. A
caller who wants the axes to compose a two-panel comparison, a test that wants to
assert on the number of lines drawn, and a notebook that wants to save without a
window all have nowhere to reach.

The standing rule for new and refactored plotting code:

- A plotter builds and returns its `Figure` (or `(fig, ax)`). That is mechanism.
- Displaying and saving are policy, applied by the caller from the config
  (`display_graphs`, `save`) or by `result.plot()`.
- A plotter that displays unconditionally has no seam, and a test of it can only
  assert that it did not raise.

The same split governs the rest: an estimator computes and returns; formatting a
table, choosing a filename, and deciding what to echo belong to whatever is
driving it.

---

# Silence, and the information it must not destroy

Two prescriptions from §1.6 combine into one rule here, and the combination is
what makes it more than a style preference.

> Rule of Silence: When a program has nothing surprising to say, it should say
> nothing.

> When filtering, never throw away information you don't need to.

Taken alone, the first invites deleting a diagnostic; the second forbids it. Taken
together they say where the information belongs: not on stdout, and not in the
bin — on the result object.

Measured at `544c684`, library code holds 36 `print` calls. They fall into three
kinds, with three different answers.

1. Diagnostics that carry real information. `laxscm_helpers/crossval.py` prints
   which solver failed and which fallback ran; `fast_scm_helpers/fast_scm_setup.py`
   prints which `post_col` it inferred. A user scripting over a hundred panels
   cannot recover any of this from stdout, and a test cannot assert on it. These
   become typed fields on the result (`MethodDetailsResults` is the natural home)
   or, where the condition is a genuine surprise the caller should act on, a
   `warnings.warn`.
2. Announcements of the expected. "Plot saved to: ..." appears at seven sites.
   The caller passed the filename; it learns nothing. Raymond's phrasing is that
   messages should obey a Rule of Most Surprise — be chatty only about what
   deviates from what was asked for. Delete these, and return the path.
3. Script output from `*_helpers/replication.py`. Those modules are drivers, and
   printing a Monte Carlo table is their job. The separation rule still applies:
   the computation returns the table, and a `main()` prints it. Then the same
   table can be asserted on by a test and consumed by a benchmark case.

The rule for new code: library code does not print. Estimation code emits results
and raises exceptions; drivers and reporters emit text.

---

# Compactness and parsimony: the size of a config and the size of a `fit()`

Compactness (§4.2) is the property that a design fits in a human head; Raymond's
practical test is whether an experienced user normally needs a manual, and his
rule of thumb comes from Miller's seven-plus-or-minus-two. Parsimony (§1.6) says
to write a big program only when it is clear by demonstration that nothing else
will do.

Measured at `544c684`: 78 configuration models, the largest carrying 52 `Field`
declarations (`CLUSTERSCConfig`), then 49 (`MAREXConfig`) and 38
(`VanillaSCConfig`). Across 75 estimators the median `fit()` body is 59 lines,
with a tail at 448 (`lexscm`), 238 (`ppscm`), 209 (`siv`), 205 (`clustersc`) and
201 (`sparse_sc`).

Neither number is a defect on its own, and Raymond is explicit that some domains
are too complex for a compact design to span. They are the diagnostic the rules
supply, and they mean three things for new work.

- A config field is a knob, and knobs multiply the states a reader must hold.
  Before adding one, check whether the value is derivable, whether an existing
  field covers it, and whether the estimator is really two estimators. A config
  that selects among genuinely different procedures is a dispatcher (`SPILLSYNTH`
  is the pattern), not a longer flag list.
- A compact working set beats a small total. Fifty fields with five that most
  users touch and forty-five with defaults nobody changes is semi-compact and
  fine; fifty fields that all interact is not. Document the working set.
- A `fit()` past roughly a hundred lines is doing helper work in the orchestrator.
  `agents_utils.md` already requires the split; the long tail above is the
  backlog. New estimators do not join it.

Detachment (§4.3) is the tool for the config case: before generalizing an
option, see how many of the accidental conditions of the paper that motivated it
can be dropped. Options survive that.

---

# Transparency and the SPOT rule

Transparency (§1.6, §6.6) is designing so the program can be seen to work.
Raymond's coding checklist gives the two questions that transfer directly.

- Static call depth: "how many levels of call might a human have to model
  mentally... If it's more than four, beware." `agents_utils.md`'s hierarchy
  (`FDID.fit()` → `fast_DID_selector` → `_forward_selection_loop` →
  `_select_best_donor` → `_r2_batch`) is four levels deep, which is the ceiling,
  not a floor to build on.
- Does the code have invariant properties that are strong and visible?
  `weights.sum() == 1`, feasibility, dimensional agreement between `y` and the
  donor matrix. These are the invariants the unit level asserts, and stating them
  in the docstring is what makes them visible.

Overprotectiveness is the failure mode §6.7 names: hiding internals in normal
operation is fine, making them inaccessible is not. The result contract is the
discoverability mechanism — every quantity an estimator computed and a user
might want to interrogate belongs on the result, reachable, and not recomputed
by the caller.

The SPOT rule (§4.2, after Kernighan): every piece of knowledge has a single,
unambiguous, authoritative representation. It is already the reason
per-estimator configs are being relocated next to their helpers and re-exported
from `config_models.py` — one definition, one import path for compatibility. Its
corollary for docs: where a page and the code would state the same fact, generate
one from the other (`tools/gen_llms_txt.py`) or link, and never restate.

---

# Representation, generation, and diversity

Representation (§1.6, after Pike's rule 5): fold knowledge into data so the logic
can be simple. Where a choice presents itself between a longer `if`/`elif` chain
and a table, the table wins — a dispatch dict from method name to helper, a
tuple of `(name, builder)` pairs driving a loop, a Pydantic model whose validators
carry the admissibility conditions. `mlsynth` is already built this way; the rule
is to keep choosing it when the chain looks shorter today.

Generation (§1.6): avoid hand-hacking. Three instances already run here —
`tools/gen_llms_txt.py` renders the agent-facing index from the code,
`tools/mutation/emit_cosmic_ray_config.py` renders a session config from
`targets.toml`, and `hypothesis` generates test inputs instead of a human
enumerating them. A hand-maintained list that duplicates something already in the
code is the smell; generate it.

Diversity (§1.6): distrust claims for one true way. `agents_tests.md`'s four
instruments are this rule, argued from Jorgensen's Figure 1.7 instead of from
temperament, and the cross-implementation differential contract is the same move
applied to estimation — two independent implementations of one program, compared
cell by cell, with neither treated as ground truth until it agrees.

---

# Applied to tests

The testing half of this doctrine lives in `agents_tests.md`, section "The Unix
rules, applied to tests". It is there and not here because that file is the
authoritative home for testing practice, and SPOT forbids saying it twice.

---

# The sweep

These are documentation-level checks, not enforced gates. Run them before opening
a PR that touches estimator, helper, or plotting code.

```bash
# Rule of Silence: library code does not print.
grep -rn "^\s*print(" mlsynth/ --include=*.py | grep -v /tests/

# Rule of Separation: a plotter that displays unconditionally has no seam.
python - <<'PY'
import ast, pathlib
for p in pathlib.Path("mlsynth").rglob("*.py"):
    if "/tests/" in str(p): continue
    for fn in [n for n in ast.walk(ast.parse(p.read_text()))
               if isinstance(n, ast.FunctionDef)]:
        for stmt in fn.body:
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                    and ast.unparse(stmt.value).strip() == "plt.show()"):
                print(f"{p}:{stmt.lineno} {fn.name}() displays unconditionally")
PY

# Parsimony: a fit() past ~100 lines is doing helper work.
python - <<'PY'
import ast, pathlib
for p in sorted(pathlib.Path("mlsynth/estimators").glob("*.py")):
    for n in ast.walk(ast.parse(p.read_text())):
        if isinstance(n, ast.FunctionDef) and n.name == "fit":
            span = n.end_lineno - n.lineno
            if span > 100:
                print(f"{span:5d}  {p.name}")
PY
```

The baseline at `544c684` is 36 prints, 23 unconditional displays, and 5 `fit()`
bodies over 200 lines. Those are a backlog, not a blocker: reducing them is its
own branch and its own scope, per the scope rule in `CLAUDE.md`. What binds now
is that new code does not add to the counts.

---

# Reference

Raymond, E. S. *The Art of Unix Programming*. Addison-Wesley, 2003. Chapter 1
(Philosophy), chapter 4 (Modularity), chapter 5 (Textuality), chapter 6
(Transparency), chapter 11 (Interfaces, "Silence Is Golden").

McIlroy, M. D. (1978), quoted in Raymond §1.6 and in Salus, *A Quarter Century of
Unix*.

Pike, R. *Notes on C Programming*, rules 1–6, quoted in Raymond §1.6.
