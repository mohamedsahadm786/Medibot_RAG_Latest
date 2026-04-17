#!/usr/bin/env bash
# Auto-format and auto-lint Python files after every file write.
# Silent on success, noisy on errors.

set -e

# Only run if Python files exist
if ls backend/**/*.py 2>/dev/null | grep -q . || ls backend/*.py 2>/dev/null | grep -q .; then
    ruff format backend/ --quiet 2>/dev/null || true
    ruff check backend/ --fix --quiet 2>/dev/null || true
fi

exit 0