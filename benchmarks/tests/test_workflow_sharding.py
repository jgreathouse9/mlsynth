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
SHARDED_JOBS = ("suite", "pr-suite", "case-deps")

#: Jobs that must never run for a pull request.
NON_PR_JOBS = ("suite", "validation-dashboard", "case-deps",
               "case-deps-commit")


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


class TestTheShardsAreSkippedWhenNothingCouldMoveANumber:
    """The paths filter on ``pr-suite``, and why each entry is where it is.

    The shards are the slow half of a pull request and most pull requests cannot
    move a pinned value, so they are gated on a ``dorny/paths-filter`` step --
    the same mechanism ``build.yml`` uses, asserted here for the same reason
    ``mlsynth/tests/test_benchmark_reference.py`` gives for that one: a skipped
    step reports success, and a skipped success is indistinguishable from a real
    one on the pull request page. Narrowing this list without deleting the
    dependency it covers is how a case stops being checked without anyone seeing
    it happen.

    The gate is a step and not a ``paths:`` on the trigger. A trigger-level
    filter stops the workflow, so the check never appears on the pull request;
    gating the step leaves the job reporting success.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def gate(jobs):
        steps = jobs["pr-suite"]["steps"]
        return next(s for s in steps if s.get("id") == "filter")

    @staticmethod
    @pytest.fixture(scope="class")
    def patterns(gate):
        """The path globs in the filter body.

        A rule may name a change type -- ``- added: '**'`` selects only files the
        diff creates -- so drop the type and keep the glob, which is the part
        that has to correspond to something in the repository.
        """
        out = []
        for line in gate["with"]["filters"].splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            body = line.lstrip("- ").strip()
            for kind in ("added", "modified", "deleted", "renamed"):
                if body.startswith(f"{kind}:"):
                    body = body.split(":", 1)[1].strip()
                    break
            out.append(body.strip("'"))
        return out

    def test_the_gate_uses_the_same_action_build_yml_does(self, gate):
        assert gate["uses"].startswith("dorny/paths-filter@")

    def test_the_expensive_step_is_the_one_gated(self, jobs):
        run = next(s for s in jobs["pr-suite"]["steps"]
                   if s["name"] == "Run benchmark suite shard")
        assert run["if"] == "steps.filter.outputs.benchmarked == 'true'"

    def test_the_trigger_itself_is_not_filtered(self, workflow):
        """A trigger-level filter would remove the check instead of passing it."""
        assert "paths" not in workflow["on"]["pull_request"]

    @pytest.mark.parametrize("directory", ["mlsynth/**", "benchmarks/**",
                                           "basedata/**"])
    def test_a_directory_a_case_reads_is_covered(self, patterns, directory):
        assert directory in patterns

    def test_basedata_is_covered_because_the_cases_read_it(self, patterns):
        """127 case files load fixtures from it; omitting it was the first bug here."""
        from pathlib import Path
        cases = Path(__file__).resolve().parents[1] / "cases"
        readers = sum("basedata" in f.read_text() for f in cases.glob("*.py"))
        assert readers > 50, readers
        assert "basedata/**" in patterns

    @pytest.mark.parametrize("excluded", ["!mlsynth/tests/**",
                                          "!benchmarks/tests/**"])
    def test_a_test_tree_no_case_imports_is_excluded(self, patterns, excluded):
        assert excluded in patterns

    def test_the_registry_really_loads_only_cases(self):
        """The claim the two exclusions rest on, checked and not assumed."""
        from benchmarks import registry
        assert {m.split(".")[1] for m in registry.CASES.values()} == {"cases"}

    def test_the_dependency_pins_are_covered(self, patterns):
        """They decide which versions a case gets installed against."""
        assert "requirements.txt" in patterns and "pyproject.toml" in patterns

    def test_every_listed_path_exists(self, patterns):
        """A pattern naming nothing is a filter entry that can never fire."""
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        for pattern in patterns:
            head = pattern.lstrip("!").split("*")[0].rstrip("/")
            assert (root / head).exists(), pattern


class TestTheDependencyMapIsWiredBothEnds:
    """The map is useless unless something writes it and something reads it."""

    def test_the_pull_request_shards_pass_the_diff(self, jobs):
        steps = jobs["pr-suite"]["steps"]
        run = next(s for s in steps if s["name"] == "Run benchmark suite shard")
        assert "--changed-from" in run["run"]

    def test_the_diff_comes_from_the_filter_and_not_from_git(self, jobs):
        """checkout is shallow here, so git cannot produce it.

        The job checks out at the default depth of one, which leaves no history
        and no merge base, so a three-dot ``git diff`` against the base branch
        fails outright -- it did, on every shard, the first time this was wired.
        dorny/paths-filter asks the API instead and is already in this job for
        the gate, so it can hand over the list it has already computed.
        """
        steps = {s["name"]: s for s in jobs["pr-suite"]["steps"]}
        collect = steps["Collect the changed files"]
        assert "git diff" not in collect.get("run", "")
        assert "steps.filter.outputs" in str(collect)

    def test_the_filter_reports_every_changed_path_not_only_the_gating_ones(self, jobs):
        """case_deps needs the whole diff to spot a file no manifest entry claims.

        The gating filter drops test trees and docs on purpose, and selecting on
        that alone would hide a path from the hole check. A second filter matching
        everything supplies the full list.
        """
        gate = next(s for s in jobs["pr-suite"]["steps"] if s.get("id") == "filter")
        assert gate["with"].get("list-files") == "json"
        assert "all:" in gate["with"]["filters"]

    def test_the_diff_is_collected_under_the_same_guard_as_the_run(self, jobs):
        """Collecting it when the shards will not run wastes a fetch."""
        steps = {s["name"]: s for s in jobs["pr-suite"]["steps"]}
        assert (steps["Collect the changed files"]["if"]
                == steps["Run benchmark suite shard"]["if"])

    def test_the_map_is_refreshed_off_the_pull_request(self, jobs):
        assert "pull_request" in jobs["case-deps"]["if"]
        run = " ".join(str(s) for s in jobs["case-deps"]["steps"])
        assert "tools/benchmark_deps.py --all" in run

    def test_refreshing_the_map_may_write_to_the_repository(self, jobs):
        """The write is the combine job's; the shards only measure."""
        assert jobs["case-deps-commit"]["permissions"]["contents"] == "write"

    def test_the_map_job_outlasts_a_full_registry_run(self, jobs):
        """It runs every case, so its timeout has to clear the daily suite's."""
        assert (int(jobs["case-deps"]["timeout-minutes"])
                >= int(jobs["suite"]["timeout-minutes"]))


class TestTheMapIsMeasuredInSlicesAndCommittedOnce:
    """``case-deps`` measures what every case executes, and the map it writes is
    what lets a pull request run only the cases its diff can reach.

    As one job it did not finish. Runs on 23, 24, 25 and 26 August each reached
    the 350-minute cap and were cancelled -- the 26th ran 07:36 to 13:26 -- so
    the map never grew past the five entries that shipped with it, and every
    pull request touching the library ran all 199 cases including the MIP and
    Monte Carlo ones. Slicing the measurement is what makes the map exist.

    Only the combine job commits. Four shards each rebasing and pushing the same
    file race, and the loser's slice is dropped without a failure anywhere.
    """

    def test_the_shards_measure_a_slice_each(self, jobs):
        step = next(s for s in jobs["case-deps"]["steps"]
                    if s["name"].startswith("Measure"))
        assert "--shard" in step["run"] and "--num-shards" in step["run"]

    def test_a_shard_writes_its_own_file(self, jobs):
        step = next(s for s in jobs["case-deps"]["steps"]
                    if s["name"].startswith("Measure"))
        assert "--out" in step["run"]

    def test_no_shard_commits(self, jobs):
        runs = " ".join(s.get("run", "") for s in jobs["case-deps"]["steps"])
        assert "git push" not in runs

    def test_the_combine_job_waits_for_every_shard(self, jobs):
        needs = jobs["case-deps-commit"]["needs"]
        assert "case-deps" in ([needs] if isinstance(needs, str) else needs)

    def test_the_combine_job_merges_the_slices(self, jobs):
        runs = " ".join(s.get("run", "") for s in jobs["case-deps-commit"]["steps"])
        assert "--combine" in runs

    def test_the_combine_job_is_the_one_that_commits(self, jobs):
        runs = " ".join(s.get("run", "") for s in jobs["case-deps-commit"]["steps"])
        assert "git push" in runs

    def test_the_combine_job_runs_even_when_a_shard_fails(self, jobs):
        """A dead shard must cost its own slice, not the other three."""
        assert "always()" in jobs["case-deps-commit"].get("if", "")

    def test_only_the_combine_job_can_write_to_the_repository(self, jobs):
        assert jobs["case-deps"].get("permissions", {}).get("contents") != "write"
        assert jobs["case-deps-commit"]["permissions"]["contents"] == "write"


class TestTheAddedSubsetReachesTheSelection:
    """A file the diff creates cannot have been executed by a pre-existing case,
    so it is not a hole in the map. The selection can only know that if the
    workflow tells it which files are new.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def gate(jobs):
        return next(s for s in jobs["pr-suite"]["steps"] if s.get("id") == "filter")

    def test_the_filter_asks_for_the_added_files(self, gate):
        assert "added:" in gate["with"]["filters"]

    def test_the_run_step_passes_them_to_the_driver(self, jobs):
        step = next(s for s in jobs["pr-suite"]["steps"]
                    if s["name"] == "Run benchmark suite shard")
        assert "--added-from" in step["run"]

    def test_the_added_set_never_arrives_without_the_diff(self, jobs):
        """``--added-from`` names a subset of ``--changed-from``; the driver
        refuses one without the other."""
        step = next(s for s in jobs["pr-suite"]["steps"]
                    if s["name"] == "Run benchmark suite shard")
        assert "--changed-from" in step["run"]
