# WhatsApp Skill

## Description

Dual-provider WhatsApp integration supporting both Baileys (open-source) and the official WhatsApp Business Cloud API.

Inspired by OpenClaw's approach: prefers Baileys (free, open-source) when available, falls back to Business API.


## Type
base

## Features

- Send text messages
- Send images with captions
- Send documents/files
- Send location pins
- Send template messages (Business API)
- Mark messages as read
- Dual-provider support (Baileys + Business API)

## Providers

### Baileys (Preferred)

Open-source WhatsApp Web API (like OpenClaw uses). Free, no Meta approval needed.

```bash
# Set either Baileys API server
export BAILEYS_HOST=localhost
export BAILEYS_PORT=3000

# Or session directory path
export BAILEYS_SESSION_PATH=/path/to/session
```

### WhatsApp Business API

Official Meta WhatsApp Business Cloud API. Requires Meta Business verification.

```bash
export WHATSAPP_PHONE_ID=your_phone_number_id
export WHATSAPP_TOKEN=your_access_token
```

## Tools

### send_whatsapp_message_tool

Send a text message via WhatsApp.

**Parameters:**
- `to` (str, required): Recipient phone number with country code (e.g., '14155238886')
- `message` (str, required): Message text
- `provider` (str, optional): "baileys", "business", or "auto" (default: "auto")
- `preview_url` (bool, optional): Enable URL preview (Business API only)

**Returns:**
- `success` (bool): Whether message was sent
- `message_id` (str): WhatsApp message ID
- `provider` (str): Provider used
- `error` (str, optional): Error message if failed

### send_whatsapp_media_tool

Send media (image, video, document) via WhatsApp.

**Parameters:**
- `to` (str, required): Recipient phone number
- `media_path` (str, required): Path to media file or URL
- `media_type` (str, optional): "image", "video", "audio", "document" (default: "image")
- `caption` (str, optional): Media caption
- `provider` (str, optional): Provider selection

**Returns:**
- `success` (bool): Whether media was sent
- `message_id` (str): WhatsApp message ID
- `provider` (str): Provider used

### get_whatsapp_status_tool

Get WhatsApp connection status.

**Parameters:**
- `provider` (str, optional): Provider to check status for

**Returns:**
- `success` (bool): Whether status check succeeded
- `connected` (bool): Whether WhatsApp is connected
- `phone` (str): Connected phone number
- `name` (str): Account name
- `provider` (str): Provider checked

### send_whatsapp_template_tool

Send a template message (Business API only).

**Parameters:**
- `to` (str, required): Recipient phone number
- `template_name` (str, required): Template name
- `language_code` (str, optional): Language code (default: 'en_US')
- `components` (list, optional): Template components for variables

### send_whatsapp_location_tool

Send a location pin.

**Parameters:**
- `to` (str, required): Recipient phone number
- `latitude` (float, required): Latitude
- `longitude` (float, required): Longitude
- `name` (str, optional): Location name
- `address` (str, optional): Location address

## Usage Examples

```python
from skills.whatsapp import (
    send_whatsapp_message,
    send_whatsapp_media,
    get_whatsapp_status
)

# Send a message (auto-selects provider)
send_whatsapp_message('+14155238886', 'Hello from Jotty!')

# Send with specific provider
send_whatsapp_message('+14155238886', 'Hello!', provider='baileys')

# Send an image
send_whatsapp_media('+14155238886', '/path/to/image.jpg', caption='Check this out!')

# Check status
status = get_whatsapp_status()
print(f"Connected: {status['connected']}")
```

## Provider Selection

The skill automatically selects providers based on availability:

1. If `BAILEYS_SESSION_PATH` or `BAILEYS_HOST` is set: Use Baileys
2. If `WHATSAPP_PHONE_ID` and `WHATSAPP_TOKEN` are set: Use Business API
3. If neither: Raise error with setup instructions

You can override this by specifying `provider` parameter in tool calls.

## Setting Up Baileys

1. Clone Baileys API: `git clone https://github.com/WhiskeySockets/Baileys`
2. Install: `npm install`
3. Start server: `node examples/example.ts`
4. Scan QR code with WhatsApp mobile app
5. Set `BAILEYS_HOST=localhost` and `BAILEYS_PORT=3000`

## Setting Up Business API

1. Create Meta Business account at business.facebook.com
2. Create WhatsApp Business App at developers.facebook.com
3. Add WhatsApp product to your app
4. Get Phone Number ID and generate permanent access token
5. Set environment variables

## Webhook Integration

The gateway at `/webhook/whatsapp` handles incoming WhatsApp messages.
Configure your Meta App's webhook URL to point to:

```
https://your-domain.com/webhook/whatsapp
```

Set verification token:
```bash
export WHATSAPP_VERIFY_TOKEN=jotty
```
