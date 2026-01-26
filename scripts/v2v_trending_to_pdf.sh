#!/bin/bash
# V2V Trending Search → PDF → Telegram + reMarkable

set -e

echo "🔍 V2V Trending Search → PDF → Telegram + reMarkable"
echo "======================================================"
echo ""

# Default values
QUERY="${1:-trending topics}"
SEND_TELEGRAM="${SEND_TELEGRAM:-true}"
SEND_REMARKABLE="${SEND_REMARKABLE:-true}"

# Telegram credentials (from planmyinvesting.com)
export TELEGRAM_TOKEN="${TELEGRAM_TOKEN:-5228780618:AAE1W6XghhgnFtOGsUJfee_NRUssx32RyOU}"
export TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-810015653}"

echo "📝 Query: $QUERY"
echo "📱 Send to Telegram: $SEND_TELEGRAM"
echo "📱 Send to reMarkable: $SEND_REMARKABLE"
echo ""

cd "$(dirname "$0")/.."

python3 << EOF
import asyncio
import os
from pathlib import Path
from core.registry.skills_registry import get_skills_registry

async def run_workflow():
    registry = get_skills_registry()
    registry.init()
    
    composite_skill = registry.get_skill('v2v-to-pdf-telegram-remarkable')
    workflow_tool = composite_skill.tools.get('v2v_to_pdf_and_send_tool')
    
    result = await workflow_tool({
        'query': '$QUERY',
        'title': 'V2V Trending: $QUERY',
        'send_telegram': $SEND_TELEGRAM,
        'send_remarkable': $SEND_REMARKABLE,
        'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID')
    })
    
    if result.get('success'):
        print(f"\n✅ Success!")
        print(f"   PDF: {result.get('pdf_path')}")
        print(f"   Telegram: {'✅' if result.get('telegram_sent') else '❌'}")
        print(f"   reMarkable: {'✅' if result.get('remarkable_sent') else '❌'}")
        print(f"   Results: {result.get('results_count', 0)}")
    else:
        print(f"\n❌ Failed: {result.get('error')}")
        exit(1)

asyncio.run(run_workflow())
EOF
