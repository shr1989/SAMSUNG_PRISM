#!/bin/bash
file=$(jq -r '.tool_input.file_path // empty')
[[ "$file" == *.py ]] || exit 0
root="$CLAUDE_PROJECT_DIR"
"$root/.venv/bin/ruff" format --quiet "$file"
if ! output=$("$root/.venv/bin/ruff" check --fix --quiet "$file" 2>&1); then
  echo "$output" >&2
  exit 2
fi
