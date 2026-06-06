---
description: Cross-model review of the working diff before opening a PR
argument-hint: "[--base <ref>] [--backend codex|api]"
---

# AI Review (cross-model)

Have a *different* model audit the current diff against mlsynth's contract,
before a PR. Catches what single-model self-review misses.

## Steps
1. Collect the diff vs `--base` (default `origin/main`): `git diff`.
2. Send it to the configured reviewer (e.g. `OPENAI_API_KEY` via
   `.claude/scripts/openai_review.py`, or the Codex CLI) with these criteria:
   - **Correctness** — does the method match the cited paper/algorithm?
   - **Contract** — config/validators, `dataprep`, `BaseEstimatorResults`,
     `__all__` export, naming.
   - **Validation** — is there a replication/benchmark, and does it actually
     match the reference?
   - **Docs** — dedicated page, assumptions-with-remarks, runnable example.
   - **Tests** — coverage incl. error paths.
3. Triage findings: fix the real ones; record (don't blindly apply) the rest.

Single-model fallback if no reviewer backend is configured: run the same
checklist yourself, but flag that it's not an independent review.
