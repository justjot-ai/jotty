#!/bin/bash
# Monitor slides generation every 2 minutes

SLIDES_DIR="/var/www/sites/personal/stock_market/Jotty/outputs/poodles_slides"
LOG_FILE="/tmp/poodles_slides.log"

echo "======================================"
echo "  Monitoring Poodles Slides (every 2 min)"
echo "======================================"
echo ""

check_status() {
    echo "🕐 Check at: $(date '+%H:%M:%S')"
    echo ""
    
    # Check if process is running
    if ps aux | grep -v grep | grep "paper2slides" > /dev/null; then
        echo "✅ Process: RUNNING"
        
        # Get stage info
        if [ -f "$SLIDES_DIR"/*/paper/normal/slides_academic_medium/state.json ]; then
            STAGE_INFO=$(cat "$SLIDES_DIR"/*/paper/normal/slides_academic_medium/state.json 2>/dev/null | python3 -c "import json,sys; data=json.load(sys.stdin); stages=data['stages']; print(f\"RAG: {stages['rag']}, Summary: {stages['summary']}, Plan: {stages['plan']}, Generate: {stages['generate']}\")" 2>/dev/null)
            echo "📊 Stages: $STAGE_INFO"
        fi
        
        # Get recent activity
        echo "📝 Recent activity:"
        tail -3 "$LOG_FILE" | grep -E "(Stage|STAGE|Layout|OCR|Predict|completed|failed)" | tail -2 || echo "   Processing..."
        
    else
        echo "⚠️  Process: STOPPED"
    fi
    
    echo ""
    
    # Check for slides
    if [ -f "$SLIDES_DIR"/*/paper/*/slides_academic_medium/*/slides.pdf ]; then
        SLIDES_PDF=$(find "$SLIDES_DIR" -name "slides.pdf" 2>/dev/null | head -1)
        SLIDES_COUNT=$(find "$(dirname "$SLIDES_PDF")" -name "slide_*.png" 2>/dev/null | wc -l)
        
        echo "🎉 ✨ SLIDES COMPLETE! ✨"
        echo "   📄 PDF: $SLIDES_PDF"
        echo "   🖼️  Total slides: $SLIDES_COUNT"
        ls -lh "$SLIDES_PDF"
        echo ""
        echo "======================================"
        exit 0
    fi
    
    echo "--------------------------------------"
    echo ""
}

# First check immediately
check_status

# Then check every 2 minutes, up to 10 times (20 minutes total)
for i in {1..10}; do
    sleep 120  # 2 minutes
    check_status
done

echo "Monitoring stopped after 20 minutes"
