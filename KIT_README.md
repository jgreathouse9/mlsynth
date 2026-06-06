# mlsynth AI-infrastructure kit

Unzip at the **repo root**. Everything is additive except where noted; review
before committing. Mirrors the practices from a cousin DiD library (diff-diff).

## What's here (items 1-4)

1. **benchmarks/** — durable, re-runnable validation (Path A/B/cross-validation).
   `python benchmarks/run_benchmarks.py --all` (FDID Table-5 case works with no R).
2. **llms.txt + get_llm_guide()** — agent-facing index, generated from live
   docstrings by `tools/gen_llms_txt.py`. Files: `llms.txt`,
   `mlsynth/guides/llms.txt`, `mlsynth/_guides_api.py`, `mlsynth/guides/__init__.py`.
3. **CLAUDE.md** — operational agent guidance (commands + invariants), complements `agents/`.
4. **.claude/commands/** — `/paper-review`, `/replicate`, `/new-estimator`,
   `/ai-review`; plus an optional plan-gate hook (`.claude/hooks/` + `SETTINGS_SNIPPET.json`).

## Two tiny wiring steps (not done automatically, to avoid clobbering your files)

1. **Export `get_llm_guide`** — add to `mlsynth/__init__.py`:
   ```python
   from ._guides_api import get_llm_guide   # noqa: F401
   ```
   and add `"get_llm_guide"` to `__all__`.
2. **(optional) enable the plan gate** — merge `.claude/SETTINGS_SNIPPET.json`
   into `.claude/settings.json`.

Then regenerate the index: `python tools/gen_llms_txt.py` (already run once; the
shipped `llms.txt` reflects your current 45 estimators).

`MANIFEST.in` adds `mlsynth/guides/*.txt` as package data so `get_llm_guide()`
works after `pip install` (you already have `include_package_data=True`).
