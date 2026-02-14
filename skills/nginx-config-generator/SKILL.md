---
name: generating-nginx-config
description: "Generate Nginx reverse proxy and server block configurations. Pure Python templates. Use when the user wants to generate nginx config, reverse proxy, server block."
---

# Nginx Config Generator Skill

Generate Nginx reverse proxy and server block configurations. Pure Python templates. Use when the user wants to generate nginx config, reverse proxy, server block.

## Type
base

## Capabilities
- code
- devops

## Reference
For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Parse input parameters
- [ ] Step 2: Execute operation
- [ ] Step 3: Return results
```

## Triggers
- "nginx"
- "reverse proxy"
- "nginx config"
- "server block"
- "proxy pass"

## Category
development

## Tools

### nginx_reverse_proxy_tool
Generate Nginx reverse proxy configuration.

**Parameters:**
- `domain` (str, required): Domain name
- `upstream_port` (int, required): Backend port
- `ssl` (bool, optional): Enable SSL (default: true)
- `upstream_host` (str, optional): Backend host (default: 127.0.0.1)
- `websocket` (bool, optional): Enable WebSocket support (default: false)

**Returns:**
- `success` (bool)
- `config` (str): Nginx configuration

## Dependencies
None
