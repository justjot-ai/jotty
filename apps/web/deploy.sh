#!/bin/bash
# Jotty UI Deployment Script
# Standalone Next.js application deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Jotty UI Deployment${NC}"
echo -e "${GREEN}============================================${NC}"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    exit 1
fi

# Check Node version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}Error: Node.js 18+ required (found: $(node -v))${NC}"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
fi

# Build
echo -e "${YELLOW}Building Jotty UI...${NC}"
npm run build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}Deployment Options:${NC}"
    echo -e "1. Start production server: ${YELLOW}npm start${NC}"
    echo -e "2. Run development: ${YELLOW}npm run dev${NC}"
    echo -e "3. Deploy to server: Copy .next/ folder to server"
    echo -e "${GREEN}============================================${NC}"
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi
