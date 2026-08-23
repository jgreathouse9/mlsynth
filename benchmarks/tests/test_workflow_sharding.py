"""The benchmark workflow's shard count and its trigger guards.

Two invariants live in YAML, where nothing else checks them.

The shard count is written twice per job -- once as the matrix list the runners
fan out over, once as the ``NUM_SHARDS`` the driver slices with -- and the two
must agree. They are separated by thirty lines and a comment asking whoever
edits one to remember the other. If the matrix grows and ``NUM_SHARDS`` does
not, the extra shards re-run cases another shard already ran; if it shrinks and
``NUM_SHARDS`` does not, the tail of the registry is never run at all and the
suite goes green having silently skipped it. That is the failure this file
exists to prevent, because it is invisible in a passing log.

The trigger guards matter because the pull-request run shares a file with jobs
that must not see a pull request. ``validation-dashboard`` commits the rebuilt
dashboard back to the default branch; running it from a fork's PR is a write
this workflow should never make. The heavy reference jobs are excluded for cost
rather than safety, but the dashboard exclusion is load-bearing and is asserted
as such.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_WORKFLOW = (Path(__file__).resolve().parents[2]
             / ".github" / "workflows" / "benchmarks.yml")

#: Jobs that fan out over shards; each must keep its matrix and NUM_SHARDS in step.
SHARDED_JOBS = ("suite", "pr-suite")

#: Jobs that must never run for a pull request.
NON_PR_JOBS = ("suite", "validation-dashboard")


@pytest.fixture(scope="module")
def workflow():
    # GitHub's "on:" key parses as the boolean True in YAML 1.1; normalise it.
    doc = yaml.safe_load(_WORKFLOW.read_text())
    doc["on"] = doc.get("on", doc.get(True))
    return doc


@pytest.fixture(scope="module")
def jobs(workflow):
    return workflow["jobs"]


class TestTheShardCountIsWrittenConsistently:
    @pytest.mark.parametrize("job", SHARDED_JOBS)
    def test_the_job_exists(self, jobs, job):
        assert job in jobs

    @pytest.mark.parametrize("job", SHARDED_JOBS)
    def test_the_matrix_length_equals_num_shards(self, jobs, job):
        shards = jobs[job]["strategy"]["matrix"]["shard"]
        assert len(shards) == int(jobs[job]["env"]["NUM_SHARDS"])

    @pytest.mark.parametrize("job", SHARDED_JOBS)
    def test_the_shards_are_zero_based_and_contiguous(self, jobs, job):
        """``--shard`` is validated as ``0 <= shard < num_shards``."""
        shards = jobs[job]["strategy"]["matrix"]["shard"]
        assert sorted(shards) == list(range(len(shards)))

    @pytest.mark.parametrize("job", SHARDED_JOBS)
    def test_the_shards_partition_the_registry(self, jobs, job):
        """Round-robin shards must be disjoint and, unioned, exhaustive."""
        from benchmarks import registry
        from benchmarks.run_benchmarks import select_cases

        n = int(jobs[job]["env"]["NUM_SHARDS"])
        names = list(registry.CASES)
        seen: list[str] = []
        for shard in range(n):
            seen += select_cases(names, registry.NEEDS_REFERENCE,
                                 with_reference=True, shard=shard, num_shards=n)
        assert sorted(seen) == sorted(names)


class TestPullRequestsRunTheSuite:
    def test_pull_request_is_a_trigger(self, workflow):
        assert "pull_request" in workflow["on"]

    def test_the_pr_job_is_guarded_to_pull_requests(self, jobs):
        assert "pull_request" in jobs["pr-suite"]["if"]

    def test_the_pr_job_does_not_provision_r(self, jobs):
        """The R stack is the slow half; its cases self-skip without it."""
        steps = " ".join(str(s) for s in jobs["pr-suite"]["steps"])
        assert "r-base" not in steps

    def test_the_pr_job_still_reports(self, jobs):
        steps = " ".join(str(s) for s in jobs["pr-suite"]["steps"])
        assert "--report" in steps

    def test_the_pr_job_shards_more_finely_than_the_daily_one(self, jobs):
        """Without R the wall-clock is the driver, so buy it back with width."""
        assert (int(jobs["pr-suite"]["env"]["NUM_SHARDS"])
                >= int(jobs["suite"]["env"]["NUM_SHARDS"]))


class TestJobsThatMustNotSeeAPullRequest:
    @pytest.mark.parametrize("job", NON_PR_JOBS)
    def test_it_is_guarded_against_pull_requests(self, jobs, job):
        assert "pull_request" in jobs[job].get("if", "")

    def test_the_dashboard_never_runs_on_a_pull_request(self, jobs):
        """It commits the rebuilt dashboard back to the default branch."""
        guard = jobs["validation-dashboard"].get("if", "")
        assert "!=" in guard and "pull_request" in guard


class TestConcurrency:
    def test_a_pull_request_does_not_queue_behind_the_daily_run(self, workflow):
        assert "github.ref" in str(workflow["concurrency"]["group"])

    def test_a_superseded_pull_request_run_is_cancelled(self, workflow):
        assert "pull_request" in str(workflow["concurrency"]["cancel-in-progress"])


class TestTheTimeoutFitsTheSlowestShard:
    """Round-robin balances case count, not case cost.

    On the first pull-request run the eight shards' benchmark steps took 6, 17,
    8, 10, 19, 10, 14 minutes -- and shard 7, which drew most of the Monte Carlo
    cases, was still running when a 45-minute cap killed it. A killed shard is
    the worst outcome available: its cases neither pass nor fail, no report is
    written, and the run reports `cancelled` rather than red, so roughly an
    eighth of the suite goes unexamined while the PR looks merely unstable.

    The floor is set against the daily suite, which has run this registry to
    completion for months at 90 minutes with R provisioning on top. Adding heavy
    cases moves the slowest shard, not the average, so this is the number to
    revisit -- by raising the shard count -- when a shard next approaches it.
    """

    #: Minutes. Below this a heavy shard is at risk of being cancelled.
    FLOOR = 60

    @pytest.mark.parametrize("job", SHARDED_JOBS)
    def test_the_timeout_clears_the_floor(self, jobs, job):
        assert jobs[job]["timeout-minutes"] >= self.FLOOR

    def test_the_pr_timeout_is_not_tighter_than_the_daily_one(self, jobs):
        """The PR job runs the same cases on more shards but without R, so its
        per-shard work is lower -- never give it a tighter cap than the run
        that is known to complete."""
        assert jobs["pr-suite"]["timeout-minutes"] >= jobs["suite"]["timeout-minutes"]


class TestThePullRequestRunIsScopedToWhatCanMoveANumber:
    """A path filter, and the reason each entry is on the right side of it.

    ``pr-suite`` runs the whole case registry on every pull request. Most pull
    requests cannot move a pinned value: a docs page, a workflow edit, a change
    confined to the test suite. They still paid for the slowest shard, which has
    run past forty minutes on this repository.

    The filter is on the ``pull_request`` trigger, so it gates the workflow and
    not one job. That is safe here only because every other job in the file is
    already excluded from pull requests by its own ``if``, so a filtered-out pull
    request loses nothing that would otherwise have run.

    What has to be inside the filter is anything a benchmark case reads: the
    library, the cases and their captured references, and the dependency pins
    that decide which versions of both get installed. The workflow itself is in
    too, so a change to this file re-validates against the suite it drives.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def paths(workflow):
        return workflow["on"]["pull_request"]["paths"]

    def test_the_pull_request_trigger_is_filtered(self, paths):
        assert paths, "pr-suite runs the full registry; scope it to what can move it"

    @pytest.mark.parametrize("changed", [
        "mlsynth/utils/conformal/resample.py",   # a shared helper a band is drawn from
        "mlsynth/estimators/ppscm.py",           # an estimator a case fits
        "mlsynth/config_models.py",              # a config a case constructs
        "benchmarks/cases/pda_wheeler_lassosynth.py",   # a case and its pinned values
        "benchmarks/reference/ascm_mixtape/reference.json",  # a captured reference
        "benchmarks/run_benchmarks.py",          # the driver
        "pyproject.toml",                        # the dependency pins
        ".github/workflows/benchmarks.yml",      # this workflow
    ])
    def test_a_change_that_could_move_a_pinned_value_still_runs(self, paths, changed):
        assert _matches(paths, changed), changed

    @pytest.mark.parametrize("changed", [
        "docs/ppscm.rst",                        # prose
        "docs/index.rst",
        "README.md",
        "CLAUDE.md",
        "agents/agents_tests.md",
        "mlsynth/tests/test_ppscm.py",           # nothing imports the test suite
        "mlsynth/guides/llms.txt",               # generated agent docs, read by no case
        ".github/workflows/build.yml",           # a different workflow
    ])
    def test_a_change_that_cannot_move_one_does_not(self, paths, changed):
        assert not _matches(paths, changed), changed

    def test_a_mixed_change_still_runs(self, paths):
        """One qualifying file is enough; the filter is any-match, not all-match."""
        assert _matches(paths, "mlsynth/utils/datautils.py")
        assert not _matches(paths, "docs/choose.rst")

    def test_the_daily_run_is_not_filtered(self, workflow):
        """The schedule is the full gate and must not inherit the PR's scope."""
        schedule = workflow["on"]["schedule"]
        assert schedule and all("paths" not in entry for entry in schedule)


def _matches(patterns, path: str) -> bool:
    """GitHub's path filter: later negations override earlier matches.

    A leading ``!`` excludes. Patterns are evaluated in order and the last one to
    match decides, so ``mlsynth/**`` followed by ``!mlsynth/tests/**`` includes
    the library and excludes its test suite.
    """
    import fnmatch

    verdict = False
    for pattern in patterns:
        negated = pattern.startswith("!")
        glob = pattern[1:] if negated else pattern
        # GitHub's ** spans separators; fnmatch's * does not, so translate.
        if fnmatch.fnmatch(path, glob) or (
                glob.endswith("/**") and path.startswith(glob[:-2])):
            verdict = not negated
    return verdict
