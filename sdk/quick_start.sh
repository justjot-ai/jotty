#!/bin/bash
# Quick Start Script for SDK Generation

set -e

echo "🚀 Jotty Multi-Language SDK Generator"
echo "======================================"
echo ""

# Check if OpenAPI Generator is installed
if ! command -v openapi-generator-cli &> /dev/null; then
    echo "📦 Installing OpenAPI Generator CLI..."
    npm install -g @openapitools/openapi-generator-cli || {
        echo "❌ Failed to install OpenAPI Generator"
        echo "   Please install manually: npm install -g @openapitools/openapi-generator-cli"
        exit 1
    }
fi

# Generate OpenAPI spec
echo "📋 Generating OpenAPI specification..."
python sdk/openapi_generator.py sdk/openapi.json

# Generate SDKs
echo ""
echo "🔧 Generating SDKs for all languages..."
python sdk/generate_sdks.py

echo ""
echo "✅ Done! SDKs generated in sdk/generated/"
echo ""
echo "📚 Next steps:"
echo "   - Review generated SDKs: ls sdk/generated/"
echo "   - Test TypeScript SDK: cd sdk/generated/typescript-node && npm install"
echo "   - Test Python SDK: cd sdk/generated/python && pip install -e ."
echo "   - Read guide: cat sdk/MULTI_LANGUAGE_SDK_GUIDE.md"
