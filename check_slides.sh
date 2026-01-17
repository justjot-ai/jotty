#!/bin/bash
# Quick status check for slides generation

echo "======================================"
echo "  Poodles Slides Generation Status"
echo "======================================"
echo ""

# Check if process is running
if ps aux | grep -v grep | grep "paper2slides" > /dev/null; then
    echo "✅ Process: RUNNING"

    # Get last few lines of log
    echo ""
    echo "📊 Recent activity:"
    tail -5 /tmp/poodles_slides.log | grep -E "(INFO|ERROR|STAGE)" || echo "   Processing..."

else
    echo "⚠️  Process: NOT RUNNING (might be finished or failed)"
fi

echo ""
echo "📁 Checking output directory..."

# Check for slides
SLIDES_DIR="/var/www/sites/personal/stock_market/Jotty/outputs/poodles_slides"
if [ -f "$SLIDES_DIR"/*/paper/*/slides_academic_medium/*/slides.pdf ]; then
    SLIDES_PDF=$(find "$SLIDES_DIR" -name "slides.pdf" 2>/dev/null | head -1)
    echo "🎉 SLIDES COMPLETE!"
    echo "   PDF: $SLIDES_PDF"

    # Count PNG slides
    SLIDES_COUNT=$(find "$(dirname "$SLIDES_PDF")" -name "slide_*.png" 2>/dev/null | wc -l)
    echo "   Total slides: $SLIDES_COUNT"

elif [ -f "$SLIDES_DIR"/*/paper/checkpoint_rag.json ]; then
    echo "⏳ Stage 1/4: RAG (PDF Parsing) - In Progress"

elif [ -f "$SLIDES_DIR"/*/paper/checkpoint_summary.json ]; then
    echo "⏳ Stage 2/4: Analysis - In Progress"

else
    echo "⏳ Starting up..."
fi

echo ""
echo "======================================"
echo "Full log: tail -f /tmp/poodles_slides.log"
echo "======================================"
