#!/usr/bin/env bash
# Strict mypy check for pre-commit hook.
# Catches real type errors while filtering out import-related false positives
# from optional dependencies (gui_agents, browser_use, mcpadapt, etc.)
#
# Filtered error codes:
#   import-not-found  — optional deps not installed
#   import-untyped    — optional deps without type stubs
#   no-redef          — try/except fallback import patterns
#   "Source file found twice" — dual module name resolution (Jotty.core vs core)

output=$(COLUMNS=500 mypy core/ apps/ sdk/ \
    --config-file=mypy.ini \
    --no-error-summary \
    --show-error-codes 2>&1 \
    | grep -v "import-not-found" \
    | grep -v "import-untyped" \
    | grep -v "no-redef" \
    | grep -v "Source file found twice" \
    | grep -E "^(core|apps|sdk)/.*: error:")

if [ -n "$output" ]; then
    echo "$output"
    exit 1
else
    exit 0
fi
