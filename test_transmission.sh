#!/bin/bash
# Simple test script for data transmission

echo "=== Testing Video Tunnel Transmission ==="
echo ""
echo "Creating test file..."
echo "Hello from Video Tunnel! This is a test message." > /tmp/test_input.txt
echo "Line 2 of test data" >> /tmp/test_input.txt
echo "Line 3 with some more content" >> /tmp/test_input.txt

echo "Test file created:"
cat /tmp/test_input.txt
echo ""
echo "File size: $(wc -c < /tmp/test_input.txt) bytes"
echo ""
echo "Now try running in two terminals:"
echo ""
echo "Terminal 1:"
echo "  uv run python -m video_tunnel listen 127.0.0.1 > /tmp/test_output.txt"
echo ""
echo "Terminal 2:"
echo "  cat /tmp/test_input.txt | uv run python -m video_tunnel connect 127.0.0.1"
echo ""
echo "After both complete, check the output:"
echo "  cat /tmp/test_output.txt"
