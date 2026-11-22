"""The study's solvers import highspy at module level (the master LP, the root MIP, the dual-stabilised root).
CI's cross-version jobs do not install it, so without it the study's tests are not collected, and the collection
of the rest of the suite (the sharding checks run ``pytest --collect-only`` on the whole tree) is not broken by
their import errors."""
import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("highspy") is not None else ["test_*.py"]
