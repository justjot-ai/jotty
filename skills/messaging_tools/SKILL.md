---
name: messaging-tools
description: "Multi-channel message delivery (Telegram, WhatsApp, etc.). Use when the user wants to send a message or file to Telegram, WhatsApp, or multiple channels at once."
---

# Messaging Tools

Orchestrates telegram-sender, whatsapp, and other channel skills for multi-channel delivery.

## Type

composite

## Base Skills

- telegram-sender
- whatsapp

## Capabilities

- communicate
- messaging
- delivery

## Use When

- User wants to send a message or file to Telegram or WhatsApp
- User wants to deliver the same content to multiple channels
- User wants to notify via Telegram/WhatsApp after generating a report

## Tools

### send_to_telegram_tool

Send a message or file to Telegram.

**Parameters:** file_path (optional), message (optional), caption, chat_id, parse_mode (HTML/Markdown). At least one of file_path or message required.

### send_to_whatsapp_tool

Send a message or file to WhatsApp.

**Parameters:** to (required, phone with country code), file_path, message, caption, provider (auto/baileys/business). At least one of file_path or message required.

### send_to_all_channels_tool

Send to multiple channels at once.

**Parameters:** channels (required, e.g. ["telegram","whatsapp"]), file_path, message, caption, telegram_chat_id, whatsapp_to, whatsapp_provider.
