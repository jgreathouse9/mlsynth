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
