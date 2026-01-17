#!/bin/bash
# Auto-complete Paper2Slides with rate limit retry
# Task ID: 696bb22a36fa588011c9c857

set -e

TASK_ID="696bb22a36fa588011c9c857"
LOG_FILE="/tmp/slides_generation_$(date +%Y%m%d_%H%M%S).log"
SLIDES_DIR="/var/www/sites/personal/stock_market/Jotty/outputs/poodles_slides"
MAX_ATTEMPTS=20
RATE_LIMIT_WAIT=75  # seconds

echo "======================================" | tee -a "$LOG_FILE"
echo "  Paper2Slides Auto-Completion" | tee -a "$LOG_FILE"
echo "  Task ID: $TASK_ID" | tee -a "$LOG_FILE"
echo "  Started: $(date)" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to count generated slides
count_slides() {
    find "$SLIDES_DIR" -name "slide_*.png" 2>/dev/null | wc -l
}

# Function to check if PDF exists
check_pdf() {
    find "$SLIDES_DIR" -name "slides.pdf" 2>/dev/null | head -1
}

# Initial count
INITIAL_COUNT=$(count_slides)
echo "📊 Initial slide count: $INITIAL_COUNT/8" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Main retry loop
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "🔄 Attempt $attempt/$MAX_ATTEMPTS - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

    # Change to Paper2Slides directory
    cd /var/www/sites/personal/stock_market/Jotty/Paper2Slides

    # Run slides generation from Stage 4 (generate)
    set +e  # Don't exit on error
    python3 -m paper2slides \
        --input ../outputs/poodles_guide/Poodles_for_Dummies_A_Comprehensive_Guide_a4.pdf \
        --output slides \
        --length medium \
        --style academic \
        --from-stage generate \
        --output-dir ../outputs/poodles_slides \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=$?
    set -e

    # Count current slides
    CURRENT_COUNT=$(count_slides)
    echo "📊 Current slide count: $CURRENT_COUNT/8" | tee -a "$LOG_FILE"

    # Check if all slides generated
    if [ $CURRENT_COUNT -ge 8 ]; then
        echo "✅ All 8 slides generated successfully!" | tee -a "$LOG_FILE"

        # Check for PDF
        PDF_PATH=$(check_pdf)
        if [ -n "$PDF_PATH" ]; then
            echo "📄 PDF created: $PDF_PATH" | tee -a "$LOG_FILE"
            ls -lh "$PDF_PATH" | tee -a "$LOG_FILE"

            echo "" | tee -a "$LOG_FILE"
            echo "======================================" | tee -a "$LOG_FILE"
            echo "  ✨ SUCCESS - All slides complete!" | tee -a "$LOG_FILE"
            echo "======================================" | tee -a "$LOG_FILE"

            exit 0
        else
            echo "⚠️  Slides generated but PDF not found, continuing..." | tee -a "$LOG_FILE"
        fi
    fi

    # Check if exit code was success
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Generation completed (exit code 0)" | tee -a "$LOG_FILE"

        # Verify PDF one more time
        PDF_PATH=$(check_pdf)
        if [ -n "$PDF_PATH" ]; then
            echo "📄 PDF found: $PDF_PATH" | tee -a "$LOG_FILE"
            exit 0
        fi
    fi

    # Check if we hit rate limit
    if grep -q "429\|rate.limit\|Rate limit" "$LOG_FILE" | tail -50; then
        echo "⏳ Rate limit detected, waiting $RATE_LIMIT_WAIT seconds..." | tee -a "$LOG_FILE"
        echo "   Next attempt will be at $(date -d "+$RATE_LIMIT_WAIT seconds" '+%H:%M:%S')" | tee -a "$LOG_FILE"
        sleep $RATE_LIMIT_WAIT
    else
        # Other error, shorter wait
        echo "⚠️  Non-rate-limit error, waiting 30 seconds..." | tee -a "$LOG_FILE"
        sleep 30
    fi

    echo "" | tee -a "$LOG_FILE"
done

# Failed after all attempts
echo "❌ Failed to complete after $MAX_ATTEMPTS attempts" | tee -a "$LOG_FILE"
echo "📊 Final slide count: $(count_slides)/8" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

exit 1
