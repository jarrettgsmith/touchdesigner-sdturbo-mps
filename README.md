# SD Turbo for TouchDesigner (macOS)

Real-time image-to-image generation using **Stable Diffusion Turbo** on **Apple Silicon** (MPS) with **TouchDesigner** integration via **Syphon** and **OSC**.

## Overview

This project enables real-time AI image generation in TouchDesigner using SD Turbo on Mac (M1/M2/M3). It uses:

- **SD Turbo**: Fast single-step diffusion model for real-time generation
- **MPS (Metal Performance Shaders)**: GPU acceleration on Apple Silicon
- **Syphon**: Low-latency video streaming between applications (macOS)
- **OSC**: Real-time control of prompts and generation parameters

## Features

- Real-time img2img at 512x512 resolution (~1-6 FPS on Apple Silicon)
- Live prompt changes via OSC
- Adjustable strength and inference steps
- Clean Python venv workflow
- Proven MPS implementation from working projects

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3) or Intel with AMD GPU
- **Python 3.9+**
- **TouchDesigner** (any recent version)
- ~2GB disk space for model download

## Installation

### 1. Clone or Download

```bash
cd /path/to/your/projects
# If you already have this folder, you're good to go
```

### 2. Run Setup

```bash
cd touchdesigner-sdturbo-mps
chmod +x setup.sh run.sh
./setup.sh
```

This will:
- Create a Python virtual environment
- Install PyTorch with MPS support
- Install all dependencies
- Verify MPS availability

**Note**: First run will download the SD Turbo model (~2GB)

## Usage

### 1. Start the Server

```bash
./run.sh
```

You should see:
```
======================================================================
SD Turbo Server → Syphon + OSC
======================================================================
[Device] Using: mps (torch.float16)
[SD Turbo] Loading model 'stabilityai/sd-turbo'...
[Syphon] Looking for input server 'TD Video Out'...
```

### 2. Setup TouchDesigner

Create the following network:

```
[Video Source] → [Syphon Out TOP] → "TD Video Out"

[Syphon In TOP] ← "SD Turbo Output"
    ↓
[Display]

[Text DAT] → [OSC Out DAT] → Port 7002
[OSC In CHOP] ← Port 7000 (for FPS stats)
```

#### Syphon Out TOP Settings:
- **Server Name**: `TD Video Out`
- **Resolution**: Any (will be resized to 512x512)

#### Syphon In TOP Settings:
- **Server Name**: `SD Turbo Output`

#### OSC Out DAT Settings:
- **Network Address**: `127.0.0.1:7002`

## OSC Control

Send OSC messages from TouchDesigner to control generation:

### Set Prompt
```python
# In a Text DAT:
/sd/prompt "a beautiful sunset over mountains"
```

### Set Strength (0.2 - 0.99)
```python
# How much to transform the input
# 0.2 = minimal changes (lower causes errors)
# 0.5 = balanced (default)
# 0.7 = strong transformation
# 0.9 = maximum prompt influence
/sd/strength 0.5
```

### Set Inference Steps (1-10)
```python
# More steps = better quality, slower
# 1-2 steps recommended for real-time
/sd/steps 2
```

### Switch Model
```python
# Available models:
# sd-turbo - Fastest, good quality (default, ~2GB)
# sdxl-turbo - Better quality, slower (~3-4GB)

/sd/model sdxl-turbo
/sd/model sd-turbo
```

**Note:** Model switching will pause generation briefly while the new model loads. First use will download the model.

### Reset to Defaults
```python
/sd/reset
```

## Expected Performance

**Apple Silicon (M1/M2/M3)** at 512x512, 2 steps:

| Model | FPS | Quality | Model Size | Native Res |
|-------|-----|---------|------------|------------|
| **sd-turbo** | 1-6 FPS | Good | ~2GB | 512x512 |
| **sdxl-turbo** | 0.5-3 FPS | Better | ~3-4GB | 1024x1024 |

**Comparison**:
- NVIDIA RTX A5000: ~15-20 FPS (with SD Turbo)
- MPS is ~3-10x slower than CUDA but enables Mac prototyping

**Recommendation**: Start with `sd-turbo` for best performance. Try `sdxl-turbo` if you want better quality and can accept lower FPS.

## Architecture

```
┌─────────────────┐   Syphon    ┌──────────────────────┐
│  TouchDesigner  │────Video───>│   Python Server      │
│                 │             │                      │
│  - Video Source │<───Video────│  - SD Turbo Model   │
│  - Syphon I/O   │             │  - MPS Acceleration │
│  - OSC Control  │────OSC─────>│  - Syphon I/O       │
│  - Parameters   │<───Stats────│  - OSC Control      │
└─────────────────┘             └──────────────────────┘
```

## Troubleshooting

### "Syphon server 'TD Video Out' not found"
1. Make sure TouchDesigner is running
2. Create a Syphon Out TOP with Server Name: `TD Video Out`
3. Connect a video source to it
4. Restart the Python server

### "MPS not available"
- Your Mac may not support MPS (requires macOS 12.3+ and Metal-capable GPU)
- The server will fall back to CPU (very slow)

### Low FPS
- Reduce resolution in TouchDesigner (512x512 is optimal)
- Use fewer inference steps (1-2 recommended)
- Lower strength (0.3-0.5)
- Close other GPU-intensive apps

### Model download fails
- Check internet connection
- May need to accept HuggingFace terms for some models
- Models are cached in `~/.cache/huggingface/`

## Project Structure

```
touchdesigner-sdturbo-mps/
├── sdturbo_server.py      # Main server (Syphon + OSC)
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script
├── run.sh                # Launch script
├── README.md             # This file
└── venv_sdturbo/         # Virtual environment (created by setup)
```

## Technical Details

### Why SD Turbo (not StreamDiffusion)?

After testing multiple approaches:
- **StreamDiffusion library**: Buggy on MPS, designed for CUDA
- **SD Turbo with AutoPipelineForImage2Image**: Works reliably on MPS
- This approach is proven from `streamdiffusion-mac/server_simple.py`

### Design Pattern

This project follows the successful pattern from:
- `touchdesigner-yolo` (Syphon + OSC)
- `touchdesigner-segment-anything` (bidirectional OSC)
- `touchdesigner-depth-anything` (clean venv workflow)

### Model Details

- **Model**: `stabilityai/sd-turbo`
- **Type**: Image-to-image diffusion
- **Pipeline**: `AutoPipelineForImage2Image`
- **Steps**: 1-4 (single-step distilled model)
- **Guidance**: 0.0 (guidance-free, required for SD Turbo)

## Future Improvements

- [ ] Add automatic mask generation mode
- [ ] Support for SDXL Turbo (larger model)
- [ ] ControlNet integration for better control
- [ ] Batch processing for higher throughput
- [ ] LCM (Latent Consistency Models) alternative

## Credits

- Based on working MPS implementation from `streamdiffusion-mac`
- Design pattern from successful YOLO/SAM/Depth Anything projects
- Uses Stability AI's SD Turbo model

## License

MIT
