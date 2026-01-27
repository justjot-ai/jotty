# Web Application Testing Skill

Toolkit for interacting with and testing local web applications using Playwright.

## Description

This skill provides tools for testing local web applications using Playwright. Supports verifying frontend functionality, debugging UI behavior, capturing browser screenshots, and viewing browser logs.

## Tools

### `test_webapp_tool`

Test a local web application.

**Parameters:**
- `app_url` (str, required): URL of the web application (e.g., http://localhost:3000)
- `test_type` (str, optional): Type - 'screenshot', 'interaction', 'validation', 'full' (default: 'full')
- `actions` (list, optional): List of actions to perform
- `screenshot_path` (str, optional): Path to save screenshot
- `headless` (bool, optional): Run in headless mode (default: True)
- `wait_for_networkidle` (bool, optional): Wait for network idle (default: True)

**Returns:**
- `success` (bool): Whether test succeeded
- `screenshot_path` (str, optional): Path to screenshot
- `console_logs` (list, optional): Browser console logs
- `test_results` (dict): Test execution results
- `error` (str, optional): Error message if failed

## Usage Examples

### Take Screenshot

```python
result = await test_webapp_tool({
    'app_url': 'http://localhost:3000',
    'test_type': 'screenshot',
    'screenshot_path': 'screenshot.png'
})
```

### Test Interactions

```python
result = await test_webapp_tool({
    'app_url': 'http://localhost:5173',
    'test_type': 'interaction',
    'actions': [
        {'type': 'click', 'selector': 'button.submit'},
        {'type': 'fill', 'selector': 'input[name="email"]', 'value': 'test@example.com'}
    ]
})
```

## Dependencies

- `playwright`: For browser automation

## Important Notes

⚠️ **Always wait for networkidle** on dynamic apps before inspecting DOM
⚠️ **Use headless mode** for automated testing
⚠️ **Check if server is running** before testing
