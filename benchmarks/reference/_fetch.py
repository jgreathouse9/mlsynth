"""Fetch a reference repo pinned at a commit, robust to a git-proxy that 403s.

The on-demand ``clone_*`` reference helpers normally ``git clone``. In sandboxed
environments the git proxy may refuse arbitrary clones (HTTP 403) while
``codeload.github.com`` tarballs of the *same pinned commit* remain reachable.
:func:`fetch_pinned_repo` tries git first and falls back to the codeload
tarball, so a benchmark only skips when neither path works.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from benchmarks.compare import BenchmarkSkipped


def _git_clone(repo_url: str, commit: str, dest: Path) -> bool:
    try:
        subprocess.run(["git", "-c", "credential.helper=", "clone", "--quiet", repo_url, str(dest)],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit],
                       check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        shutil.rmtree(dest, ignore_errors=True)
        return False


def _codeload(repo_url: str, commit: str, dest: Path) -> bool:
    m = re.search(r"github\.com[/:]+([^/]+)/(.+?)(?:\.git)?/?$", repo_url)
    if not m:
        return False
    url = f"https://codeload.github.com/{m.group(1)}/{m.group(2)}/tar.gz/{commit}"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tar = Path(tmp) / "repo.tar.gz"
            probe = subprocess.run(["curl", "-sL", "--max-time", "120", "-o", str(tar),
                                    "-w", "%{http_code}", url], capture_output=True, text=True)
            if probe.stdout.strip() != "200" or not tar.exists() or tar.stat().st_size == 0:
                return False
            subprocess.run(["tar", "xzf", str(tar), "-C", tmp], check=True, capture_output=True)
            tops = [p for p in Path(tmp).iterdir() if p.is_dir()]
            if len(tops) != 1:                    # codeload extracts a single <repo>-<commit>/ dir
                return False
            dest.mkdir(parents=True, exist_ok=True)
            for item in tops[0].iterdir():
                shutil.move(str(item), str(dest / item.name))
    except (OSError, subprocess.CalledProcessError):
        # A non-writable cache dir or a failed extraction skips gracefully, matching
        # _git_clone and the workflow contract (an unfetchable reference skips, not reds).
        shutil.rmtree(dest, ignore_errors=True)
        return False
    return True


def fetch_pinned_repo(repo_url: str, commit: str, dest) -> Path:
    """Ensure ``dest`` holds ``repo_url`` at ``commit``. git clone, else the
    codeload tarball; raise :class:`BenchmarkSkipped` if neither succeeds."""
    dest = Path(dest)
    if _git_clone(repo_url, commit, dest) or _codeload(repo_url, commit, dest):
        return dest
    raise BenchmarkSkipped(
        f"could not fetch reference repo {repo_url} @ {commit[:7]} "
        f"(git clone refused and codeload tarball unavailable)")
