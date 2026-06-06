"""In-process access to mlsynth's agent/LLM guides.

    from mlsynth import get_llm_guide
    print(get_llm_guide())            # concise estimator index (llms.txt)

Variants map to files shipped under ``mlsynth/guides/``. Falls back to a clear
message if a variant isn't packaged (e.g. before ``tools/gen_llms_txt.py`` has
been run, or if package data wasn't included in the wheel).
"""
from __future__ import annotations

from importlib import resources

_VARIANTS = {
    "concise": "llms.txt",
    "full": "llms-full.txt",
    "practitioner": "llms-practitioner.txt",
}


def get_llm_guide(variant: str = "concise") -> str:
    """Return a bundled LLM guide as a string.

    Parameters
    ----------
    variant : {"concise", "full", "practitioner"}
        Which guide to return. ``"concise"`` (default) is the llms.txt
        estimator index.
    """
    fname = _VARIANTS.get(variant)
    if fname is None:
        raise ValueError(
            f"unknown guide variant {variant!r}; choose from {sorted(_VARIANTS)}."
        )
    try:
        return resources.files("mlsynth.guides").joinpath(fname).read_text()
    except (FileNotFoundError, ModuleNotFoundError):
        return (
            f"[mlsynth] guide '{variant}' ({fname}) is not packaged. "
            "Run `python tools/gen_llms_txt.py` and ensure mlsynth/guides/*.txt "
            "is included as package data (MANIFEST.in / package_data)."
        )
