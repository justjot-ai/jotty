# Pre-commit Hooks - Preventing Issues Before They're Committed

## Overview

Pre-commit hooks automatically validate code **before** it gets committed to git, preventing syntax errors, import violations, and type issues from entering the codebase.

## What We Prevent

### ✅ Critical Issues (Will Block Commits)

1. **Python Syntax Errors**
   - Catches: Unterminated docstrings, invalid Python syntax
   - Example: Would have prevented the `coordination.py`, `lifecycle.py`, `resilience.py` syntax errors
   - Hook: `python-syntax-check`

2. **Security Vulnerabilities**
   - Catches: Hardcoded secrets, security anti-patterns
   - Tool: Bandit
   - Hook: `bandit`

3. **Critical Type Errors**
   - Catches: Syntax errors and import-not-found errors in type checking
   - Tool: mypy (critical errors only)
   - Hook: `mypy-critical`

4. **Architecture Violations**
   - Catches: Apps importing from core (bypassing SDK)
   - Tool: import-linter
   - Hook: `import-linter`

### ⚠️ Warnings (Should Fix, But Won't Block)

5. **Code Formatting**
   - Tools: black, isort, flake8
   - Auto-fixable with `pre-commit run --all-files`

6. **Debug Statements**
   - Catches: Leftover `breakpoint()`, `pdb.set_trace()`

7. **File Issues**
   - Trailing whitespace, missing newlines, large files

## Installation

Already installed! If you need to reinstall:

```bash
pip install pre-commit
pre-commit install
```

## Usage

### Automatic (Recommended)
Hooks run automatically on `git commit`. If they fail, the commit is blocked.

```bash
git add file.py
git commit -m "Fix bug"
# Hooks run automatically here
```

### Manual
Run on all files:
```bash
pre-commit run --all-files
```

Run specific hook:
```bash
pre-commit run python-syntax-check --all-files
pre-commit run mypy-critical --all-files
```

Run on specific files:
```bash
pre-commit run --files core/intelligence/memory/_consolidation_mixin.py
```

### Skip Hooks (Use Sparingly!)
Only in emergencies:
```bash
git commit --no-verify -m "Emergency fix"
```

## Hooks Configuration

See `.pre-commit-config.yaml` for full configuration.

### Critical Hooks

1. **python-syntax-check**
   - Validates Python AST
   - Catches syntax errors mypy might miss
   - **This would have prevented the recent protocol file syntax errors!**

2. **mypy-critical**
   - Only fails on syntax and import errors
   - Ignores attr-defined errors (we're fixing those)
   - Config: `mypy.ini`

3. **import-linter**
   - Enforces clean architecture boundaries
   - Apps must use SDK, not core directly
   - Config: `.importlinter`

4. **jotty-doctor**
   - Custom Jotty-specific checks
   - Import validation
   - Security scanning

## What Gets Checked

| Hook | Files | Fast? | Auto-fix? |
|------|-------|-------|-----------|
| python-syntax-check | All .py | ✅ Yes | ❌ No |
| mypy-critical | core/, apps/, sdk/ | ⚠️ Slow | ❌ No |
| import-linter | All .py | ⚠️ Slow | ❌ No |
| black | All .py | ✅ Yes | ✅ Yes |
| isort | All .py | ✅ Yes | ✅ Yes |
| flake8 | All .py | ✅ Yes | ⚠️ Partial |
| bandit | All .py | ⚠️ Slow | ❌ No |

## Performance

- **Fast hooks** (syntax, black, isort): < 1 second
- **Slow hooks** (mypy, import-linter, bandit): 10-30 seconds

To speed up commits, you can disable slow hooks temporarily:
```bash
SKIP=mypy-critical,import-linter,bandit git commit -m "Quick fix"
```

## Troubleshooting

### Hook fails but I can't see why
```bash
# Run with verbose output
pre-commit run python-syntax-check --all-files --verbose
```

### Hook passes locally but should fail
```bash
# Clean cache and re-run
pre-commit clean
pre-commit run --all-files
```

### Update hooks to latest versions
```bash
pre-commit autoupdate
```

## Examples

### Good: Syntax check catches error
```bash
$ git commit -m "Add feature"
Python AST Syntax Validation.............................................Failed
- hook id: python-syntax-check
- exit code: 1

SyntaxError: unterminated triple-quoted string literal
```
✅ **Prevented bad code from being committed!**

### Good: All checks pass
```bash
$ git commit -m "Add feature"
Python AST Syntax Validation.............................................Passed
mypy - Critical Errors Only..............................................Passed
Import Linter - Architecture Enforcement.................................Passed
[main abc1234] Add feature
 1 file changed, 10 insertions(+)
```
✅ **Clean code committed!**

## Impact

### Before Pre-commit Hooks
- ❌ Syntax errors committed to codebase
- ❌ Architecture violations (apps importing core)
- ❌ Type errors accumulating
- ❌ Discovered issues only during CI or at runtime

### After Pre-commit Hooks
- ✅ Syntax errors caught **before** commit
- ✅ Architecture enforced at commit time
- ✅ Critical type errors prevented
- ✅ Issues caught in **seconds**, not minutes/hours

## Maintenance

The hooks are configured in `.pre-commit-config.yaml`. To add new hooks:

1. Edit `.pre-commit-config.yaml`
2. Test: `pre-commit run <hook-id> --all-files`
3. Commit the updated config
4. Team members will get the new hooks automatically on next commit

## Status

✅ **Fully Operational** - All critical hooks working as of 2026-02-15

**Next Steps:**
1. Fix import-linter config for Jotty.apps.cli.commands.telegram_bot ignore pattern
2. Consider adding pylint or ruff for additional linting
3. Add hook to prevent commits with TODOs in critical files
