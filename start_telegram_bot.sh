#!/bin/bash
# Start Telegram Bot with proper logging

cd /var/www/sites/personal/stock_market/Jotty

echo "Starting Telegram bot..."
echo "Press Ctrl+C to stop"
echo "================================"

python -m apps.telegram.bot_migrated
