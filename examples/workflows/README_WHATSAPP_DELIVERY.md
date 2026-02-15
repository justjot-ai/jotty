# WhatsApp #my-notes Delivery - Complete Guide

## Overview

This guide shows how to automatically deliver comprehensive olympiad-quality content to WhatsApp #my-notes via Bailey.

## Quick Start (Existing PDF)

If you already have the PDF and just want to send it to WhatsApp #my-notes:

```bash
cd /var/www/sites/personal/stock_market/Jotty
python3 examples/workflows/deliver_to_whatsapp.py
```

This will:
1. ✅ Auto-discover #my-notes group JID from Bailey
2. ✅ Send the PDF to WhatsApp
3. ✅ Update `.env.anthropic` with the JID for future runs

## Full Pipeline (Generate + Deliver)

To generate comprehensive content AND deliver to Telegram + WhatsApp:

```bash
cd /var/www/sites/personal/stock_market/Jotty
python3 examples/workflows/multiagent_with_whatsapp.py
```

This will:
1. ✅ Generate 10,000+ word comprehensive course using `olympiad_learning_swarm`
2. ✅ Create professional PDF (A4, 53 pages)
3. ✅ Send to Telegram
4. ✅ Auto-discover #my-notes JID and send to WhatsApp
5. ✅ Update `.env.anthropic` for future runs

## Prerequisites

### 1. Bailey WhatsApp Service

Bailey must be running for WhatsApp delivery:

```bash
# Start Bailey
docker-compose up -d baileys

# Verify it's running
curl http://localhost:3000/health

# List available groups
curl http://localhost:3000/groups
```

### 2. Environment Configuration

The `.env.anthropic` file should have:

```bash
# Bailey Configuration
BAILEYS_HOST=localhost
BAILEYS_PORT=3000

# WhatsApp Target (auto-discovered if not set)
WHATSAPP_CHANNEL=120363XXXXXXXXX@g.us  # Will be auto-filled
```

## How Auto-Discovery Works

1. Script checks `WHATSAPP_CHANNEL` in `.env.anthropic`
2. If it's a placeholder (`your-mynotes-channel-jid@g.us`), script calls Bailey API:
   ```bash
   GET http://localhost:3000/groups
   ```
3. Finds group with `subject: "my-notes"` (case-insensitive)
4. Extracts the `id` field (format: `120363XXXXXXXXX@g.us`)
5. Uses it for delivery
6. Updates `.env.anthropic` for future runs

## Manual JID Configuration

If you want to manually set the JID (bypassing auto-discovery):

1. Get your groups list:
   ```bash
   curl http://localhost:3000/groups | jq '.[] | {subject, id}'
   ```

2. Find #my-notes and copy the `id` value

3. Update `.env.anthropic`:
   ```bash
   WHATSAPP_CHANNEL=120363XXXXXXXXX@g.us  # Replace with actual ID
   ```

## Files Created

### Scripts

| File | Purpose |
|------|---------|
| `multiagent_with_whatsapp.py` | Generate content + deliver to Telegram + WhatsApp |
| `deliver_to_whatsapp.py` | Send existing PDF to WhatsApp only |
| `multiagent_comprehensive_working.py` | Original working version (Telegram only) |

### Output Files

| File | Description |
|------|-------------|
| `/home/coder/jotty/outputs/multiagent_systems_olympiad.md` | Markdown source (131 KB, 26,717 words) |
| `/home/coder/jotty/outputs/multiagent_systems_olympiad.pdf` | PDF output (123 KB, 53 pages) |

## Troubleshooting

### Bailey Not Running

**Error:** `Bailey service not responding`

**Fix:**
```bash
docker-compose up -d baileys
# Wait 10 seconds for initialization
curl http://localhost:3000/health
```

### Group Not Found

**Error:** `#my-notes group not found`

**Fix:**
```bash
# List all groups
curl http://localhost:3000/groups | jq '.[] | .subject'

# Check the exact name of your group
# Update the script if it's not exactly "my-notes"
```

### JID Format Issues

**Valid JID formats:**
- Group/Channel: `120363XXXXXXXXX@g.us`
- Direct message: `14155238886@s.whatsapp.net`

**Invalid:**
- `your-mynotes-channel-jid@g.us` (placeholder)
- Missing `@g.us` or `@s.whatsapp.net`

## Current Status

✅ **Content Generated**: 26,717 words, 53 pages, 123 KB PDF
✅ **Telegram Delivery**: Working (Message ID 18357)
⏳ **WhatsApp Delivery**: Ready (needs Bailey service running)

## Next Steps

1. **Start Bailey** (if not running):
   ```bash
   docker-compose up -d baileys
   ```

2. **Test WhatsApp Delivery**:
   ```bash
   python3 examples/workflows/deliver_to_whatsapp.py
   ```

3. **For Future Runs**:
   ```bash
   # Just run the full pipeline - it will use cached JID
   python3 examples/workflows/multiagent_with_whatsapp.py
   ```

## API Reference

### Bailey WhatsApp API

**Send Document:**
```bash
curl -X POST http://localhost:3000/send/media \
  -F "to=120363XXXXXXXXX@g.us" \
  -F "type=document" \
  -F "caption=Your caption here" \
  -F "file=@/path/to/file.pdf"
```

**List Groups:**
```bash
curl http://localhost:3000/groups
```

**Response Format:**
```json
[
  {
    "id": "120363XXXXXXXXX@g.us",
    "subject": "my-notes",
    "participants": [...]
  }
]
```

## Technical Details

### PDF Generation

The olympiad swarm generates PDFs using **WeasyPrint** (not Pandoc):

```python
from weasyprint import HTML

# Convert Markdown to HTML
html_content = markdown_to_html(markdown_content)

# Generate PDF
HTML(string=html_content).write_pdf(pdf_path)
```

### WhatsApp Delivery Flow

```
1. Check .env.anthropic for WHATSAPP_CHANNEL
2. If placeholder → Call Bailey GET /groups
3. Find group where subject == "my-notes"
4. Extract id field (JID)
5. POST /send/media with file + JID
6. Update .env.anthropic with JID
```

### Environment Variables Precedence

```python
whatsapp_jid = os.getenv('WHATSAPP_CHANNEL') or os.getenv('WHATSAPP_TO')
```

1. `WHATSAPP_CHANNEL` (preferred for groups)
2. `WHATSAPP_TO` (fallback)
3. Auto-discovery via Bailey API
4. Fail with helpful error message

## Complete Example Run

```bash
$ cd /var/www/sites/personal/stock_market/Jotty

# Start Bailey if needed
$ docker-compose up -d baileys

# Run full pipeline
$ python3 examples/workflows/multiagent_with_whatsapp.py

================================================================================
COMPREHENSIVE MULTI-AGENT SYSTEMS COURSE
Using Proven Olympiad Learning Swarm + WhatsApp #my-notes Delivery
================================================================================

✅ Using existing PDF: /home/coder/jotty/outputs/multiagent_systems_olympiad.pdf
   Size: 122.8 KB

📤 Sending to Telegram...
✅ Telegram: Message ID 18357

📤 Sending to WhatsApp #my-notes...
   🔍 Auto-discovering #my-notes JID...
   Target: 120363XXXXXXXXX@g.us
✅ WhatsApp: Message ID msg_abc123

💡 Updating .env.anthropic with discovered JID...
✅ Updated .env.anthropic for future runs

================================================================================
COMPLETE!
================================================================================
```

## Summary

This setup provides:
- ✅ **One-command delivery** to both Telegram and WhatsApp
- ✅ **Auto-discovery** of #my-notes group JID
- ✅ **Persistent configuration** via `.env.anthropic`
- ✅ **Comprehensive content** (10,000+ words, olympiad quality)
- ✅ **Professional PDFs** (A4, styled, multi-page)

Just run `python3 examples/workflows/multiagent_with_whatsapp.py` and it handles everything!
