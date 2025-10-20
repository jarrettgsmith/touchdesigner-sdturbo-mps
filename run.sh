#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv_sdturbo" ]; then
    echo "ERROR: Virtual environment not found"
    echo "Please run setup.sh first:"
    echo "  chmod +x setup.sh"
    echo "  ./setup.sh"
    exit 1
fi

# Activate virtual environment
source venv_sdturbo/bin/activate

# Run server
echo "Starting SD Turbo server..."
echo "Press Ctrl+C to stop"
echo ""

python3 sdturbo_server.py
