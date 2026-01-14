# LaTeX Validation Setup Guide

## Using openreview/latex-validation

### Repository Type

The `openreview/latex-validation` repository is a **TypeScript/Node.js** library, not a Python package.

---

## Integration Options

### Option 1: Use via Node.js (Recommended)

**Requirements**:
- Node.js installed
- latex-validation installed via npm

**Installation**:
```bash
# Install Node.js (if not installed)
# Then install latex-validation
npm install -g git+https://github.com/openreview/latex-validation.git
```

**Usage**:
The wrapper (`latex_validator_wrapper.py`) will automatically use Node.js if available.

---

### Option 2: Use Python Alternatives

**Option 2a: pylatexenc**
```bash
pip install pylatexenc
```

**Option 2b: Custom Python Validator**
- Use structure-based validation (already implemented)
- Use QuickLaTeX API (already implemented)

---

## Current Implementation

### Priority Order:

1. **latex-validation library** (if Node.js available)
   - Via `latex_validator_wrapper.py`
   - Uses Node.js subprocess

2. **QuickLaTeX API** (Fallback)
   - HTTP GET/POST requests
   - Handles HTTP 414 errors

3. **Structure-based validation** (Final fallback)
   - Checks delimiters, braces, commands
   - Works offline

---

## Code Status

✅ **Integration Complete**

- ✅ Wrapper created: `latex_validator_wrapper.py`
- ✅ Renderer updated: `math_latex_renderer.py`
- ✅ Automatic detection: Checks for Node.js and library
- ✅ Fallback chain: Library → API → Structure

---

## Testing

### Test Current Setup:

```python
from core.experts.math_latex_renderer import validate_math_latex_syntax

# Will use best available method
is_valid, error, metadata = validate_math_latex_syntax("$$\\frac{1}{2}$$")
print(f"Valid: {is_valid}, Method: {metadata.get('validation_method')}")
```

### Expected Behavior:

- **If Node.js + latex-validation installed**: Uses Node.js wrapper
- **If not**: Falls back to QuickLaTeX API
- **If API fails**: Uses structure-based validation

---

## Summary

✅ **Code Ready**: Integration complete with Node.js wrapper support

**Current Status**:
- ✅ Wrapper created for Node.js integration
- ✅ Automatic fallback to QuickLaTeX API
- ✅ Structure validation as final fallback
- ⏳ Node.js installation needed for full latex-validation support

**To Use latex-validation**:
1. Install Node.js
2. Install latex-validation: `npm install -g git+https://github.com/openreview/latex-validation.git`
3. Code will automatically use it

**Fallback Works**: Even without Node.js, QuickLaTeX API and structure validation work!
