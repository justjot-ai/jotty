# OAuth Automation Skill

Automate OAuth login flows for services like NotebookLM, Google services, etc.

## Description

This skill automates OAuth authentication flows using browser automation. It can handle:
- Google OAuth (used by NotebookLM)
- Custom OAuth providers
- Credential storage and management
- Headless/Docker environments


## Type
base


## Capabilities
- code

## Features

- Automated OAuth login flows
- Secure credential storage
- Session persistence
- Support for multiple OAuth providers
- Docker/headless compatible

## Usage

```python
from skills.oauth_automation.tools import oauth_login_tool

# Login with credentials
result = await oauth_login_tool({
    'provider': 'google',
    'email': 'user@example.com',
    'password': 'password',
    'profile_dir': '/path/to/profile',
    'headless': True
})
```

## Parameters

- `provider` (str, required): OAuth provider ('google', 'microsoft', etc.)
- `email` (str, required): Email address
- `password` (str, required): Password
- `profile_dir` (str, optional): Browser profile directory
- `headless` (bool, optional): Run in headless mode (default: True)
- `service_url` (str, optional): Target service URL (e.g., 'https://notebooklm.google.com')

## Security

- Credentials are never logged
- Uses secure browser automation
- Supports environment variables for credentials
- Session stored in browser profile

## Requirements

- Playwright for browser automation
- Credentials (provided securely)

## Triggers
- "oauth automation"
- "oauth"
- "authentication"
- "authorize"

## Category
workflow-automation
