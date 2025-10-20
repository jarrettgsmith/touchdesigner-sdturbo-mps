#!/bin/bash
set -e

echo "========================================="
echo "SD Turbo for TouchDesigner (macOS) Setup"
echo "========================================="
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "ERROR: This project is for macOS only (requires MPS and Syphon)"
    exit 1
fi

# Check Python version
echo "[1/6] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.9 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "  ✓ Found Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
if [ -d "venv_sdturbo" ]; then
    echo "  Virtual environment already exists. Removing..."
    rm -rf venv_sdturbo
fi

python3 -m venv venv_sdturbo
echo "  ✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "[3/6] Activating virtual environment..."
source venv_sdturbo/bin/activate
echo "  ✓ Activated"

# Upgrade pip
echo ""
echo "[4/6] Upgrading pip..."
pip install --upgrade pip
echo "  ✓ pip upgraded"

# Install PyTorch with MPS support
echo ""
echo "[5/6] Installing PyTorch (with MPS support)..."
echo "  This may take a few minutes..."
pip install torch torchvision torchaudio
echo "  ✓ PyTorch installed"

# Install project requirements
echo ""
echo "[6/6] Installing project dependencies..."
echo "  This may take a few minutes..."
pip install -r requirements.txt
echo "  ✓ Dependencies installed"

# Test MPS availability
echo ""
echo "Testing MPS availability..."
python3 -c "import torch; print('  ✓ MPS available!' if torch.backends.mps.is_available() else '  WARNING: MPS not available')"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To run the server:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source venv_sdturbo/bin/activate"
echo "  python3 sdturbo_server.py"
echo ""
echo "Note: First run will download the SD Turbo model (~2GB)"
echo ""
