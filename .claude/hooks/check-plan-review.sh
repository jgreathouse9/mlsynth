#!/bin/bash
# Optional PreToolUse hook for ExitPlanMode: refuse to approve a plan until a
# review of it exists. Plan-first, gated discipline (see diff-diff for prior art).
#
# Wire it by adding to .claude/settings.json:
#   { "hooks": { "PreToolUse": [
#       { "matcher": "ExitPlanMode",
#         "hooks": [{ "type": "command",
#                     "command": ".claude/hooks/check-plan-review.sh" }] } ] } }
#
# Convention: a plan at ~/.claude/plans/<name>.md is "reviewed" when a sibling
# ~/.claude/plans/<name>.review.md exists and is newer than the plan.
# PreToolUse hooks must print JSON to stdout and exit 0.
set -euo pipefail
plans_dir="${HOME}/.claude/plans"
deny() {
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":%s}}\n' \
    "$(printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
  exit 0
}
[ -d "$plans_dir" ] || exit 0                       # no plans dir -> nothing to gate
plan="$(ls -t "$plans_dir"/*.md 2>/dev/null | grep -v '\.review\.md$' | head -1 || true)"
[ -n "$plan" ] || exit 0
review="${plan%.md}.review.md"
if [ ! -f "$review" ]; then
  deny "Plan '$(basename "$plan")' has no review. Run /review-plan before approving."
fi
if [ "$plan" -nt "$review" ]; then
  deny "Plan '$(basename "$plan")' changed after its review. Re-run /review-plan."
fi
exit 0
