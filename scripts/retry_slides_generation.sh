#!/bin/bash
# Auto-retry slides generation with rate limit handling

LOG_FILE="/tmp/poodles_slides_auto_retry.log"
SLIDES_DIR="/var/www/sites/personal/stock_market/Jotty/outputs/poodles_slides"

echo "======================================" | tee -a "$LOG_FILE"
echo "  Auto-Retry Slides Generation" | tee -a "$LOG_FILE"
echo "  $(date)" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for i in {1..10}; do
    echo "🔄 Attempt $i - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

    # Run slides generation from Stage 4 (generate)
    cd /var/www/sites/personal/stock_market/Jotty/Paper2Slides
    timeout 300 python3 -m paper2slides \
        --input ../outputs/poodles_guide/Poodles_for_Dummies_A_Comprehensive_Guide_a4.pdf \
        --output slides \
        --length medium \
        --style academic \
        --from-stage generate \
        --output-dir ../outputs/poodles_slides \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=$?

    # Check if completed successfully
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ SUCCESS! All slides generated!" | tee -a "$LOG_FILE"

        # Count generated slides
        SLIDE_COUNT=$(find "$SLIDES_DIR" -name "slide_*.png" 2>/dev/null | wc -l)
        echo "📊 Total slides: $SLIDE_COUNT" | tee -a "$LOG_FILE"

        # Find PDF
        PDF_PATH=$(find "$SLIDES_DIR" -name "slides.pdf" 2>/dev/null | head -1)
        if [ -n "$PDF_PATH" ]; then
            echo "📄 PDF: $PDF_PATH" | tee -a "$LOG_FILE"
            ls -lh "$PDF_PATH" | tee -a "$LOG_FILE"
        fi

        exit 0
    fi

    # If not successful, check error and wait before retry
    if grep -q "rate.limit\|429" "$LOG_FILE"; then
        echo "⏳ Rate limit hit, waiting 75 seconds..." | tee -a "$LOG_FILE"
        sleep 75
    else
        echo "⚠️  Unknown error, waiting 30 seconds..." | tee -a "$LOG_FILE"
        sleep 30
    fi

    echo "" | tee -a "$LOG_FILE"
done

echo "❌ Failed after 10 attempts" | tee -a "$LOG_FILE"
exit 1
