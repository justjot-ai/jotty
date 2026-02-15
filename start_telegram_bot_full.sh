#!/bin/bash
# Start Full-Featured Telegram Bot

cd /var/www/sites/personal/stock_market/Jotty

echo "================================"
echo "Starting Jotty Telegram Bot"
echo "Full CLI Feature Parity"
echo "================================"
echo ""
echo "Available commands:"
echo "  /start  /help  /status  /clear"
echo "  /session  /memory  /skill  /skills"
echo "  /agent  /agents  /swarm  /workflow"
echo "  /model  /config  /stats  /tokens  /cost"
echo ""
echo "Press Ctrl+C to stop"
echo "================================"
echo ""

python -m apps.telegram.bot_migrated_full
