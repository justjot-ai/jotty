# Manual Test Scripts

Interactive test scripts for manual verification of components.

## Test Files

### `test_shared_components.py`
Test shared UI components (ChatInterface, renderers, state machine).

```bash
python tests/manual/test_shared_components.py
```

**Tests:**
- ChatInterface initialization
- Message rendering
- Status rendering
- State transitions
- Event processing
- Session management

### `test_telegram_shared.py`
Test Telegram-specific shared components.

```bash
python tests/manual/test_telegram_shared.py
```

**Tests:**
- Telegram message renderer
- Telegram status renderer
- Message formatting
- MarkdownV2 escaping
- Command handling

### `test_bot_simple.py`
Simple Telegram bot command handling test.

```bash
python tests/manual/test_bot_simple.py
```

**Tests:**
- Command parsing
- Command registry
- Mock message handling

### `test_sdk_manual.py`
Manual SDK testing (streaming, chat, workflow).

```bash
python tests/manual/test_sdk_manual.py
```

**Tests:**
- SDK initialization
- Streaming responses
- Chat execution
- Workflow execution
- Error handling

## Running All Manual Tests

```bash
# Run individual tests
python tests/manual/test_shared_components.py
python tests/manual/test_telegram_shared.py
python tests/manual/test_bot_simple.py
python tests/manual/test_sdk_manual.py
```

## Automated Tests

For automated unit/integration tests, see `tests/` directory:

```bash
pytest tests/
```

## See Also

- `docs/guides/TEST_ALL_PLATFORMS.md` - Complete platform testing guide
- `scripts/test_all.sh` - Platform status check script
