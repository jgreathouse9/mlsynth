"""Root conftest: registers the suite's shard options.

``pytest_addoption`` is only honoured from a plugin or from a conftest at the
rootdir. ``mlsynth/tests/conftest.py`` is a subdirectory conftest, so it
registers the options only when the command line names a path inside that
directory -- ``pytest mlsynth/tests`` works and a bare ``pytest`` from the repo
root fails with ``unrecognized arguments: --num-shards``. CI runs the bare form,
so the registration has to live here.

Everything else about the suite stays in ``mlsynth/tests/conftest.py``; this file
exists for the one hook that has to be at the root.
"""

import pathlib
import sys

# ``_shard`` sits with the tests it selects, and the tests directory is not a
# package, so it is not importable until its directory is on the path. pytest
# adds that directory itself during collection, which is too late for an option
# that has to exist before the command line is parsed.
_TESTS = pathlib.Path(__file__).resolve().parent / "mlsynth" / "tests"
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from _shard import (  # noqa: E402,F401  (re-export: pytest reads these by name)
    pytest_addoption,
    pytest_collection_modifyitems,
)
