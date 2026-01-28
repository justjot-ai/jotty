# API vs CLI Comparison

## Why Use API Instead of CLI?

### Issues with CLI:
1. **Hangs/Timeouts**: Claude CLI hangs on invalid options (--json-schema doesn't exist)
2. **Slow**: 120s timeouts, 3-10s per call
3. **Unreliable**: Subprocess issues, error handling problems
4. **No JSON Schema Support**: CLI doesn't support --json-schema option

### Benefits of API:
1. **Faster**: 0.5-2s per call (vs 3-10s for CLI)
2. **More Reliable**: Direct API calls, better error handling
3. **Native DSPy Support**: Uses DSPy's built-in Anthropic support
4. **Better JSON Schema**: Can enforce via prompts more effectively
5. **No Subprocess Issues**: Direct HTTP calls, no hanging

## Changes Made

### 1. Provider Priority Updated
**File**: `core/foundation/unified_lm_provider.py`
- **Before**: CLI providers first, API providers last
- **After**: API providers first (if API keys available), CLI as fallback

### 2. Direct DSPy LM for API Providers
**File**: `core/foundation/unified_lm_provider.py`
- **Before**: All providers through AISDKProviderLM (JustJot.ai API)
- **After**: API providers use `dspy.LM('anthropic/...')` directly (native DSPy)

### 3. Recursive System Updated
**File**: `recursive_self_improvement.py`
- **Before**: Always used CLI providers
- **After**: Prefers API providers if API keys available

## Usage

### With API Key:
```bash
export ANTHROPIC_API_KEY="your-key"
./start_recursive_with_api.sh
```

### Without API Key (CLI fallback):
```bash
./start_recursive_improvement.sh
```

## Expected Performance

- **API**: 0.5-2s per task type inference
- **CLI**: 3-10s per task type inference (or 120s timeout)

API is **5-10x faster** and more reliable!
