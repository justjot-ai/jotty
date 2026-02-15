#!/bin/bash
# Restart Telegram Bot

echo "Stopping old Telegram bot..."
pkill -f "apps.telegram.bot"
sleep 2

echo "Starting Telegram bot..."
./start_telegram_bot.sh
