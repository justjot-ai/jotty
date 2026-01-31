# Jotty Web Gateway Deployment

## For jotty.justjot.ai on cmd.dev

### Quick Start (cmd.dev Web Terminal)

```bash
# 1. Navigate to Jotty directory
cd /var/www/sites/personal/stock_market/Jotty

# 2. Install web dependencies
pip install -r requirements-web.txt

# 3. Run the gateway
python web.py --port 8766
```

### Running Permanently

#### Option 1: Screen (Recommended for cmd.dev)
```bash
# Start in detached screen session
screen -dmS jotty python web.py --port 8766

# To attach to the session:
screen -r jotty

# To detach: Ctrl+A, then D
```

#### Option 2: nohup
```bash
nohup python web.py --port 8766 > jotty.log 2>&1 &
echo $! > jotty.pid

# To stop:
kill $(cat jotty.pid)
```

#### Option 3: Python Module
```bash
python -m Jotty.cli.gateway --port 8766
```

### Environment Variables

```bash
export JOTTY_HOST="0.0.0.0"      # Bind host
export JOTTY_PORT="8766"         # Bind port
export ANTHROPIC_API_KEY="..."   # For Claude API
export OPENAI_API_KEY="..."      # For OpenAI
export GROQ_API_KEY="..."        # For Groq (free tier)
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | PWA Chat Interface |
| `/ws` | WebSocket | Real-time chat |
| `/health` | GET | Health check |
| `/docs` | GET | API Documentation |
| `/webhook/telegram` | POST | Telegram webhook |
| `/webhook/slack` | POST | Slack webhook |
| `/webhook/discord` | POST | Discord webhook |
| `/webhook/whatsapp` | POST | WhatsApp webhook |
| `/message` | POST | Generic HTTP message |

### Testing

```bash
# Health check
curl http://localhost:8766/health

# Send message via HTTP
curl -X POST http://localhost:8766/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello Jotty!", "user_id": "test"}'
```

### Domain Setup (jotty.justjot.ai)

On cmd.dev, configure the domain to point to port 8766:
1. In cmd.dev settings, set up port forwarding for 8766
2. Configure DNS for jotty.justjot.ai → cmd.dev server
3. Enable SSL/TLS (cmd.dev handles this automatically)

### Channels Setup

#### Telegram
```bash
export TELEGRAM_TOKEN="your-bot-token"
# Set webhook: https://jotty.justjot.ai/webhook/telegram
```

#### Slack
```bash
export SLACK_SIGNING_SECRET="your-signing-secret"
# Events URL: https://jotty.justjot.ai/webhook/slack
```

#### Discord
```bash
export DISCORD_PUBLIC_KEY="your-public-key"
# Interactions URL: https://jotty.justjot.ai/webhook/discord
```

### Architecture

```
jotty.justjot.ai
       │
       ▼
┌─────────────────────────────────────────┐
│         UnifiedGateway (FastAPI)        │
│  ┌─────────────────────────────────┐    │
│  │  /ws (WebSocket)                │    │
│  │  /webhook/* (HTTP Webhooks)     │    │
│  │  /static/* (PWA)                │    │
│  └─────────────────────────────────┘    │
│                  │                      │
│                  ▼                      │
│           ChannelRouter                 │
│                  │                      │
│                  ▼                      │
│            JottyCLI                     │
│                  │                      │
│                  ▼                      │
│         SwarmManager / LeanExecutor     │
│         (Multi-Agent Orchestration)     │
└─────────────────────────────────────────┘
```
