#!/usr/bin/env python3
"""
Load tools and widgets from JSON files into Jotty registry.

Usage:
    python3 load_from_json.py <tools_json_file> <widgets_json_file>
"""

import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.registry.justjot_loader import load_justjot_tools_and_widgets


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 load_from_json.py <tools_json> <widgets_json>")
        sys.exit(1)
    
    tools_file = sys.argv[1]
    widgets_file = sys.argv[2]
    
    # Load JSON files
    with open(tools_file, 'r') as f:
        tools_data = json.load(f)
    
    with open(widgets_file, 'r') as f:
        widgets_data = json.load(f)
    
    # Load into registry
    load_justjot_tools_and_widgets(tools_data, widgets_data)
    print("✅ Loaded tools and widgets into Jotty registry")


if __name__ == '__main__':
    main()
