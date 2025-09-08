#!/bin/bash

echo "Fixing macOS security issues for Prometheus exporters..."
echo "=================================================="

# Remove quarantine attribute
echo "1. Removing quarantine attributes..."
xattr -dr com.apple.quarantine exporters/ 2>/dev/null

# Clear any extended attributes
echo "2. Clearing all extended attributes..."
xattr -cr exporters/ 2>/dev/null

# Option 1: Ad-hoc code signing (usually works)
echo "3. Ad-hoc signing the binaries..."
codesign --force --deep --sign - exporters/node_exporter
if [ -f exporters/process_exporter ]; then
    codesign --force --deep --sign - exporters/process_exporter
fi

echo ""
echo "Done! Try running: ./exporters/node_exporter --version"
echo ""
echo "If it still doesn't work, you may need to:"
echo "  1. Go to System Settings > Privacy & Security"
echo "  2. Look for a message about node_exporter being blocked"
echo "  3. Click 'Allow Anyway'"
echo ""
echo "Alternative: Run with explicit permission:"
echo "  sudo spctl --add exporters/node_exporter"
echo "  sudo spctl --add exporters/process_exporter"