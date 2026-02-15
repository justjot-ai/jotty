#!/bin/bash
# Quick test all platforms

echo "================================"
echo "Testing All Jotty Platforms"
echo "================================"
echo ""

# Test TUI
echo "1️⃣  Testing TUI..."
timeout 3 python -m apps.cli 2>&1 | head -5
if [ $? -eq 124 ]; then
    echo "✅ TUI starts successfully"
else
    echo "❌ TUI failed to start"
fi
echo ""

# Test Telegram
echo "2️⃣  Checking Telegram bot..."
if ps aux | grep -q "apps.telegram.bot"; then
    echo "✅ Telegram bot is running"
    ps aux | grep "apps.telegram.bot" | grep -v grep | awk '{print "   PID: " $2}'
else
    echo "❌ Telegram bot is NOT running"
    echo "   Start with: ./start_telegram_bot.sh"
fi
echo ""

# Test Web backend
echo "3️⃣  Testing Web backend..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Web backend is running"
    echo "   Response:"
    curl -s http://localhost:8000/health | python -m json.tool | sed 's/^/   /'
else
    echo "❌ Web backend is NOT running"
    echo "   Start with: python apps/web/backend/server.py"
fi
echo ""

# Test Web frontend
echo "4️⃣  Checking Web frontend..."
if lsof -i :3000 > /dev/null 2>&1; then
    echo "✅ Frontend is running on port 3000"
else
    echo "⚠️  Frontend is NOT running"
    echo "   Start with: cd apps/web/frontend && npm start"
fi
echo ""

echo "================================"
echo "Summary"
echo "================================"
echo ""
echo "TUI:      python -m apps.cli"
echo "Telegram: ./start_telegram_bot.sh"
echo "Web:      python apps/web/backend/server.py"
echo "          cd apps/web/frontend && npm start"
echo ""
echo "See TEST_ALL_PLATFORMS.md for detailed testing guide"
echo "================================"
