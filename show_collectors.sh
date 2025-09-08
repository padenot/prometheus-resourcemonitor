#!/bin/bash
# Quick script to show which collectors are available on this system

echo "======================================"
echo "Node Exporter Collector Information"
echo "======================================"
echo ""

echo "Currently enabled collectors:"
echo "------------------------------"
python3 list_collectors.py --running 2>/dev/null | grep "✓" | head -20

echo ""
echo "To see all available collectors:"
echo "  python3 list_collectors.py"
echo ""
echo "To test which collectors work on this system:"
echo "  python3 list_collectors.py --test"
echo ""
echo "To use a custom configuration:"
echo "  python3 monitor_interactive.py --config config_minimal.yaml"