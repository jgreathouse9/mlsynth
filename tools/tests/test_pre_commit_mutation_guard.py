"""The pre-commit guard has to tell a running mutation sweep from a mention of one.

``tools/mutation/run_mutants.py`` edits source files in place, so a commit made
while it runs can stage a planted mutant. That has shipped twice -- 4c836c51 and
591beb0b -- and in both cases nothing failed, because a mutant that is merely
present fails nothing by design.

The guard is only useful if it discriminates. A first version matched
``pgrep -f run_mutants.py``, which also matches the shell that launched the
runner and any editor with the path open, so it refused on a clean tree. A guard
that refuses every commit is worse than none: the override becomes reflex.

What the guard reads is the machine's process table, which the tests below both
read and write: ``holder`` puts a process there whose argv names the runner.
Under ``pytest -n auto`` four of these run at once, so a test asserting the hook
permits a commit saw another worker's holder and failed -- twice in every run of
``-n 4`` on main. They take a lock on the process table instead.
"""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX only, and the hook is a shell script
    fcntl = None

HOOK = Path(__file__).resolve().parents[2] / ".githooks" / "pre-commit"

#: Shared by every worker on this machine, because the process table is.
LOCK = Path(tempfile.gettempdir()) / "mlsynth-pre-commit-guard.lock"


def _run(env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    return subprocess.run([str(HOOK)], capture_output=True, text=True, env=env)


@contextlib.contextmanager
def _exclusive():
    if fcntl is None:  # pragma: no cover - see the import above
        yield
        return
    with open(LOCK, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


@pytest.fixture
def process_table():
    """Exclusive use of the process table the guard reads.

    Every test that plants a runner-named process, or asserts that none is
    there, takes this. The kernel releases the lock when the holder exits, so a
    worker that dies mid-test cannot leave it held. The rest of the suite is
    unaffected: only these tests contend.
    """
    with _exclusive():
        yield


@pytest.fixture
def holder(process_table):
    """A process whose argv genuinely names the runner, as the real one does."""
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "run_mutants.py"
        script.write_text("import time; time.sleep(30)\n")
        proc = subprocess.Popen([sys.executable, str(script), "--target", "x"])
        time.sleep(1.0)
        try:
            yield proc
        finally:
            os.kill(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)


def test_a_clean_tree_commits(process_table):
    assert _run().returncode == 0


def test_a_running_sweep_blocks_the_commit(holder):
    out = _run()
    assert out.returncode == 1
    assert "refusing to commit" in out.stderr
    assert "4c836c51" in out.stderr, "the message should name the incidents"


def test_a_shell_that_merely_mentions_the_runner_does_not_block(process_table):
    """The false positive that made the first version useless."""
    proc = subprocess.Popen(
        ["bash", "-c", "sleep 20  # python tools/mutation/run_mutants.py --target scm"])
    time.sleep(1.0)
    try:
        assert _run().returncode == 0
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)


def test_the_override_is_available_and_says_so(holder):
    out = _run({"MLSYNTH_ALLOW_COMMIT_DURING_MUTATION": "1"})
    assert out.returncode == 0
    assert "overridden" in out.stderr


def test_the_hook_is_executable():
    assert os.access(HOOK, os.X_OK), "a hook git cannot execute is a hook that does nothing"
