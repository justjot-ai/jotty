#!/bin/bash
# Setup Paper2Slides Integration for Jotty
# This script installs Paper2Slides dependencies

set -e

echo "============================================"
echo "  Paper2Slides Integration Setup for Jotty"
echo "============================================"
echo ""

# Get the Python venv path
VENV_PYTHON="/var/www/sites/personal/stock_market/planmyinvesting.com/src/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found at $VENV_PYTHON"
    exit 1
fi

echo "Using Python: $VENV_PYTHON"
echo ""

# Install core dependencies for Paper2Slides
echo "📦 Installing Paper2Slides core dependencies..."
$VENV_PYTHON -m pip install --break-system-packages \
    huggingface_hub \
    lightrag-hku \
    tqdm \
    Pillow>=10.0.0 \
    reportlab>=4.0.0 \
    python-dotenv>=1.0.0

echo ""
echo "✅ Core dependencies installed!"
echo ""

# Note about mineru - it's large and has many dependencies
echo "⚠️  Note: mineru[core] is NOT installed (large dependency)"
echo "   Paper2Slides can work without it for basic conversions"
echo "   If needed, install manually: pip install mineru[core]==2.6.4"
echo ""

# Create .env configuration
ENV_FILE="Paper2Slides/paper2slides/.env"

if [ -f "$ENV_FILE" ]; then
    echo "✅ .env file already exists: $ENV_FILE"
else
    echo "📝 Creating .env configuration..."

    cat > "$ENV_FILE" << 'EOF'
# Paper2Slides Configuration for Jotty
# This uses Jotty's DirectClaudeCLI (no API key needed for local Claude CLI)

# RAG LLM - Using Claude CLI (will be configured by Jotty wrapper)
RAG_LLM_API_KEY="not-needed-using-claude-cli"
RAG_LLM_BASE_URL=""
RAG_LLM_MAX_TOKENS="16000"

# Image Generation - Configure one of these:
# Option 1: OpenRouter (recommended, multi-model support)
IMAGE_GEN_PROVIDER="openrouter"
IMAGE_GEN_API_KEY=""  # Add your OpenRouter API key here
IMAGE_GEN_BASE_URL="https://openrouter.ai/api/v1"
IMAGE_GEN_MODEL="google/gemini-flash-1.5-8b"
IMAGE_GEN_RESPONSE_MIME_TYPE="image/png"

# Option 2: Google Gemini (alternative)
# IMAGE_GEN_PROVIDER="google"
# IMAGE_GEN_API_KEY=""  # Add your Google API key here
# GOOGLE_GENAI_BASE_URL="https://generativelanguage.googleapis.com"
EOF

    echo "✅ Created .env file: $ENV_FILE"
    echo ""
    echo "⚠️  IMPORTANT: Edit $ENV_FILE and add your IMAGE_GEN_API_KEY"
    echo "   Get API key from:"
    echo "   - OpenRouter: https://openrouter.ai/keys"
    echo "   - Google Gemini: https://ai.google.dev/gemini-api/docs/api-key"
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Add IMAGE_GEN_API_KEY to: $ENV_FILE"
echo "2. Test with: python3 generate_guide_with_slides.py --topic 'Test'"
echo ""
