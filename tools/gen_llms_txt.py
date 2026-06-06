#!/usr/bin/env python3
"""Generate mlsynth's llms.txt from live estimator docstrings + configs.

Introspects ``mlsynth.__all__``, reading the first docstring line of each
estimator and its config fields, so the agent-facing index never drifts from
the code. Run from the repo root:

    python tools/gen_llms_txt.py            # writes mlsynth/guides/llms.txt + ./llms.txt

The output follows the llms.txt convention (https://llmstxt.org): a short
project summary, then a flat, machine-readable index of estimators.
"""
from __future__ import annotations

import inspect
import textwrap
from pathlib import Path

import mlsynth
from mlsynth import config_models as cm

HEADER = """\
# mlsynth

> mlsynth is a strongly-typed Python library of synthetic-control and
> difference-in-differences estimators for causal inference with panel data.
> Every estimator exposes a Pydantic config and a `.fit()` that returns a
> standardized results object. Most are validated against the source paper's
> empirical result, Monte Carlo, or an authoritative reference implementation
> (see the Replications page).

## How to call any estimator

    from mlsynth import <ESTIMATOR>
    res = <ESTIMATOR>({"df": long_df, "outcome": "...", "treat": "...",
                       "unitid": "...", "time": "...", "display_graphs": False}).fit()

`df` is long-format (one row per unit-period); `treat` is a 0/1 indicator that
is 1 for the treated unit in post-treatment periods. Results expose
`.effects.att`, `.time_series`, `.inference`, etc. (see BaseEstimatorResults).

## Docs
- Full documentation: https://mlsynth.readthedocs.io
- Replications catalogue: https://mlsynth.readthedocs.io/en/latest/replications.html

## Estimators
"""


def first_doc_line(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        if line.strip():
            return line.strip().rstrip(".")
    return "(no description)"


def config_for(name: str):
    """Best-effort match of an estimator name to its *Config class."""
    for cand in (f"{name}Config", f"{name.upper()}Config", f"{name.title()}Config"):
        if hasattr(cm, cand):
            return getattr(cm, cand)
    return None


def main() -> None:
    names = [n for n in mlsynth.__all__ if n[:1].isupper()
             and inspect.isclass(getattr(mlsynth, n, None))]
    lines = [HEADER]
    for n in sorted(names):
        cls = getattr(mlsynth, n)
        desc = first_doc_line(cls)
        cfg = config_for(n)
        cfg_note = f"  (config: {cfg.__name__})" if cfg is not None else ""
        lines.append(f"- **{n}** — {desc}.{cfg_note}")
    text = "\n".join(lines) + "\n"

    root = Path(__file__).resolve().parents[1]
    (root / "mlsynth" / "guides").mkdir(parents=True, exist_ok=True)
    (root / "mlsynth" / "guides" / "llms.txt").write_text(text)
    (root / "llms.txt").write_text(text)
    print(f"wrote llms.txt ({len(names)} estimators, {len(text)} chars)")


if __name__ == "__main__":
    main()
