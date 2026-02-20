#!/usr/bin/env bash
# Strict mypy check for pre-commit hook.
# Catches real type errors while filtering out import-related false positives
# from optional dependencies (gui_agents, browser_use, mcpadapt, etc.)
#
# Only fails on errors in files staged for commit (so pre-existing errors
# in untouched files don't block unrelated work).
#
# Filtered error codes:
#   import-not-found  — optional deps not installed
#   import-untyped    — optional deps without type stubs
#   [import]          — general import resolution failures
#   no-redef          — try/except fallback import patterns
#   "Source file found twice" — dual module name resolution (Jotty.core vs core)

# Get list of staged Python files
staged_files=$(git diff --cached --name-only --diff-filter=ACMR -- '*.py' 2>/dev/null)
if [ -z "$staged_files" ]; then
    exit 0
fi

# Build grep pattern from staged files
pattern=$(echo "$staged_files" | sed 's|^|^|' | paste -sd'|' -)

output=$(COLUMNS=500 mypy core/ apps/ sdk/ \
    --config-file=mypy.ini \
    --no-error-summary \
    --show-error-codes 2>&1 \
    | grep -v "import-not-found" \
    | grep -v "import-untyped" \
    | grep -v "\[import\]" \
    | grep -v "no-redef" \
    | grep -v "Source file found twice" \
    | grep -E "^(core|apps|sdk)/.*: error:" \
    | grep -E "$pattern")

if [ -n "$output" ]; then
    echo "$output"
    exit 1
else
    exit 0
fi
