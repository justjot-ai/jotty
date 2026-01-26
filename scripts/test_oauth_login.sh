#!/bin/bash
# Test OAuth Login Script
# Run this on a machine with a display to test OAuth login with 2FA

set -e

echo "🔐 OAuth Login Test Script"
echo "=========================="
echo ""
echo "This script will test OAuth login for NotebookLM"
echo "Credentials will be kept private"
echo ""

# Set credentials (you can also use environment variables)
export OAUTH_EMAIL="setia.naveen@gmail.com"
export OAUTH_PASSWORD="N2Setia@india"

echo "📧 Email: $OAUTH_EMAIL"
echo "🔑 Password: [HIDDEN]"
echo ""

# Check if we have a display
if [ -z "$DISPLAY" ]; then
    echo "⚠️  No DISPLAY environment variable set"
    echo "   This script needs a display to show the browser for 2FA"
    echo ""
    echo "Options:"
    echo "  1. Run on a machine with X11 display"
    echo "  2. Use X11 forwarding: ssh -X user@host"
    echo "  3. Use xvfb: xvfb-run -a python3 scripts/oauth_login.py ..."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the OAuth login
echo "🚀 Starting OAuth login..."
echo ""

python3 scripts/oauth_login.py \
    --provider google \
    --service-url https://notebooklm.google.com \
    --no-headless \
    --profile-dir ~/.notebooklm_oauth_profile

echo ""
echo "✅ Test complete!"
echo ""
echo "Profile saved to: ~/.notebooklm_oauth_profile"
echo ""
echo "To use this profile:"
echo "  export OAUTH_PROFILE_DIR=~/.notebooklm_oauth_profile"
echo "  python3 scripts/notebooklm_pdf.py --file doc.md --profile-dir ~/.notebooklm_oauth_profile"
