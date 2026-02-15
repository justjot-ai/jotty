#!/bin/bash
# Restart Telegram Bot

echo "Stopping old Telegram bot..."
pkill -f "bot_migrated"
sleep 2

echo "Starting Telegram bot..."
./start_telegram_bot_full.sh
