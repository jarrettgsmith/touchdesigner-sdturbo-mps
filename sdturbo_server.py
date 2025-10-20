#!/usr/bin/env python3
"""
SD Turbo Server for TouchDesigner (macOS)
Real-time image-to-image generation using Stable Diffusion Turbo on Apple Silicon

Architecture:
- Receives video from TouchDesigner via Syphon
- Runs SD Turbo img2img inference on each frame
- Sends generated video back via Syphon
- Receives prompts and settings via OSC
"""

import cv2
import numpy as np
import torch
from PIL import Image
import syphon
from syphon.utils.numpy import copy_image_to_mtl_texture, copy_mtl_texture_to_image
from syphon.utils.raw import create_mtl_texture
import Metal
import time
from pythonosc import udp_client, dispatcher, osc_server
import threading
from diffusers import AutoPipelineForImage2Image
import signal
import sys

print("=" * 70)
print("SD Turbo Server → Syphon + OSC")
print("=" * 70)

# Global flag for clean shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\n\n[Shutdown] Interrupt signal received...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Configuration
WIDTH, HEIGHT = 512, 512
SYPHON_INPUT_NAME = "TD Video Out"  # Receive from TouchDesigner
SYPHON_OUTPUT_NAME = "SD Turbo Output"  # Send to TouchDesigner
OSC_RECEIVE_PORT = 7002  # Receive prompts/settings from TD
OSC_SEND_IP = "127.0.0.1"
OSC_SEND_PORT = 7000  # Send status to TD

# SD Turbo Configuration
AVAILABLE_MODELS = {
    "sd-turbo": "stabilityai/sd-turbo",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    # Note: Other models may require specific pipeline configurations
    # Only sd-turbo and sdxl-turbo are guaranteed to work with AutoPipelineForImage2Image
}
DEFAULT_MODEL = "sd-turbo"
MODEL_ID = AVAILABLE_MODELS[DEFAULT_MODEL]
INFERENCE_STEPS = 2  # Fast for real-time (1-4 steps)
GUIDANCE_SCALE = 0.0  # Turbo models require 0
STRENGTH = 0.5  # How much to transform (0.3-0.7 works well)
DEFAULT_PROMPT = ""  # Turbo models work without prompts

# Device selection (from working code)
def choose_device():
    print('\n[Device] Checking availability...')
    print('  CUDA available?', 'Yes' if torch.cuda.is_available() else "No")
    print('  MPS available?', 'Yes' if torch.backends.mps.is_available() else "No")

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f'  Using: {device} ({dtype})')
    return device, dtype

DEVICE, DTYPE = choose_device()

print(f"\n[Config] Video Input: Syphon '{SYPHON_INPUT_NAME}'")
print(f"[Config] Video Output: Syphon '{SYPHON_OUTPUT_NAME}'")
print(f"[Config] OSC Receive: Port {OSC_RECEIVE_PORT}")
print(f"[Config] OSC Send: {OSC_SEND_IP}:{OSC_SEND_PORT}")
print(f"[Config] Resolution: {WIDTH}x{HEIGHT}")
print(f"[Config] Steps: {INFERENCE_STEPS}, Strength: {STRENGTH}")

# Global state for settings (thread-safe)
class Settings:
    def __init__(self):
        self.prompt = DEFAULT_PROMPT
        self.strength = STRENGTH
        self.steps = INFERENCE_STEPS
        self.current_model = DEFAULT_MODEL
        self.model_changed = False
        self.lock = threading.RLock()

    def set_prompt(self, prompt):
        with self.lock:
            self.prompt = prompt

    def set_strength(self, strength):
        with self.lock:
            # Clamp between 0.2 and 0.99 to avoid edge cases
            # Below 0.2 causes zero denoising steps (empty tensor error)
            # At 1.0 can cause issues with img2img pipeline
            self.strength = max(0.2, min(0.99, strength))

    def set_steps(self, steps):
        with self.lock:
            self.steps = max(1, min(10, int(steps)))

    def set_model(self, model_name):
        with self.lock:
            if model_name in AVAILABLE_MODELS:
                self.current_model = model_name
                self.model_changed = True
                return True
            return False

    def consume_model_change(self):
        with self.lock:
            changed = self.model_changed
            self.model_changed = False
            return changed, self.current_model

    def get_all(self):
        with self.lock:
            return self.prompt, self.strength, self.steps

settings = Settings()

# OSC Handlers
def handle_prompt(unused_addr, prompt):
    """Set generation prompt"""
    print(f"\n[OSC] ✓ RECEIVED PROMPT MESSAGE")
    print(f"[OSC] Full prompt: '{prompt}'")
    settings.set_prompt(prompt)
    print(f"[OSC] Prompt set successfully\n")

def handle_strength(unused_addr, strength):
    """Set transformation strength (0.0-1.0)"""
    print(f"\n[OSC] ✓ RECEIVED STRENGTH: {strength:.2f}")
    settings.set_strength(strength)
    print(f"[OSC] Strength set successfully\n")

def handle_steps(unused_addr, steps):
    """Set inference steps (1-10)"""
    print(f"\n[OSC] ✓ RECEIVED STEPS: {int(steps)}")
    settings.set_steps(steps)
    print(f"[OSC] Steps set successfully\n")

def handle_reset(unused_addr):
    """Reset to defaults"""
    print(f"\n[OSC] ✓ RECEIVED RESET")
    settings.set_prompt(DEFAULT_PROMPT)
    settings.set_strength(STRENGTH)
    settings.set_steps(INFERENCE_STEPS)
    print(f"[OSC] Reset to defaults complete\n")

def handle_model(unused_addr, model_name):
    """Switch model"""
    print(f"\n[OSC] ✓ RECEIVED MODEL CHANGE: '{model_name}'")
    if settings.set_model(model_name):
        print(f"[OSC] Model will switch to '{model_name}' ({AVAILABLE_MODELS[model_name]})")
        print(f"[OSC] Note: Model will reload on next frame\n")
    else:
        available = ', '.join(AVAILABLE_MODELS.keys())
        print(f"[OSC] ERROR: Unknown model '{model_name}'")
        print(f"[OSC] Available models: {available}\n")

def handle_any(unused_addr, *args):
    """Catch-all for debugging"""
    print(f"\n[OSC DEBUG] Unknown message: {unused_addr} with args: {args}\n")

def create_osc_server():
    """Create OSC server for receiving controls"""
    disp = dispatcher.Dispatcher()
    disp.map("/sd/prompt", handle_prompt)
    disp.map("/sd/strength", handle_strength)
    disp.map("/sd/steps", handle_steps)
    disp.map("/sd/model", handle_model)
    disp.map("/sd/reset", handle_reset)
    disp.set_default_handler(handle_any)  # Catch all other messages

    server = osc_server.ThreadingOSCUDPServer(
        ("0.0.0.0", OSC_RECEIVE_PORT), disp
    )
    return server

def load_model(model_name):
    """Load a model by name"""
    model_id = AVAILABLE_MODELS[model_name]
    print(f"\n[Model] Loading '{model_name}' ({model_id})...")
    print("[Model] This may take a while on first run (downloading)...")

    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        safety_checker=None
    ).to(DEVICE)

    print(f"[Model] ✓ '{model_name}' loaded")

    # Warmup
    print(f"[Model] Warming up '{model_name}'...")
    dummy_img = Image.new('RGB', (WIDTH, HEIGHT), color=(128, 128, 128))
    generator = torch.manual_seed(42)

    try:
        _ = pipe(
            prompt="test",
            image=dummy_img,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            strength=STRENGTH,
            generator=generator,
            width=WIDTH,
            height=HEIGHT
        ).images[0]
        print(f"[Model] ✓ '{model_name}' warmup complete")
    except Exception as e:
        print(f"[Model] Warmup warning: {e}")
        print("[Model] Continuing anyway...")

    return pipe

def main():
    global running

    # Load initial model
    pipe = load_model(DEFAULT_MODEL)

    # Create Syphon client for input
    print(f"\n[Syphon] Looking for input server '{SYPHON_INPUT_NAME}'...")
    directory = syphon.SyphonServerDirectory()
    matching = directory.servers_matching_name(name=SYPHON_INPUT_NAME)

    if not matching:
        print(f"\n[ERROR] Syphon server '{SYPHON_INPUT_NAME}' not found!")
        print("\nTo fix:")
        print("1. Open TouchDesigner")
        print("2. Create a Syphon Out TOP")
        print(f"3. Set Server Name to: '{SYPHON_INPUT_NAME}'")
        print("4. Connect a video source")
        sys.exit(1)

    server_desc = matching[0]
    print(f"[Syphon] Found: '{server_desc.name}' from '{server_desc.app_name}'")
    syphon_in = syphon.SyphonMetalClient(server_desc)
    print("[Syphon] ✓ Input client created")

    # Create Syphon server for output
    print(f"[Syphon] Creating output server '{SYPHON_OUTPUT_NAME}'...")
    syphon_out = syphon.SyphonMetalServer(SYPHON_OUTPUT_NAME)
    print("[Syphon] ✓ Output server created")

    # Create Metal texture for output
    print(f"\n[Metal] Creating output texture ({WIDTH}x{HEIGHT})...")
    metal_device = Metal.MTLCreateSystemDefaultDevice()
    output_texture = create_mtl_texture(metal_device, WIDTH, HEIGHT)
    print("[Metal] ✓ Output texture created")

    # Create OSC client for sending status
    print(f"\n[OSC] Creating send client {OSC_SEND_IP}:{OSC_SEND_PORT}...")
    osc_send = udp_client.SimpleUDPClient(OSC_SEND_IP, OSC_SEND_PORT)
    print("[OSC] ✓ Send client created")

    # Create OSC server for receiving controls
    print(f"[OSC] Creating receive server on port {OSC_RECEIVE_PORT}...")
    osc_recv_server = create_osc_server()
    print("[OSC] ✓ Receive server created")

    # Start OSC server in background thread
    osc_thread = threading.Thread(target=osc_recv_server.serve_forever, daemon=True)
    osc_thread.start()
    print("[OSC] ✓ Receive server started")

    print("\n[Ready] Waiting for Syphon input...")
    print("=" * 70)
    print("TouchDesigner Setup:")
    print(f"  1. Syphon Out TOP - Server: '{SYPHON_INPUT_NAME}'")
    print(f"  2. Syphon In TOP - Server: '{SYPHON_OUTPUT_NAME}'")
    print(f"  3. OSC Out DAT - Port: {OSC_RECEIVE_PORT}")
    print(f"     Commands: /sd/prompt <text>")
    print(f"               /sd/strength <0.2-0.99>")
    print(f"               /sd/steps <1-10>")
    print(f"               /sd/model <sd-turbo/sdxl-turbo>")
    print(f"               /sd/reset")
    print(f"  4. OSC In CHOP - Port: {OSC_SEND_PORT}")
    print("=" * 70)
    print("Press Ctrl+C to quit\n")

    frame_count = 0
    start_time = time.time()
    last_output = None

    try:
        while running:
            # Check for model changes
            model_changed, new_model = settings.consume_model_change()
            if model_changed:
                print(f"\n[Model] Switching to '{new_model}'...")
                # Clean up old model
                del pipe
                if DEVICE == "mps":
                    torch.mps.empty_cache()
                import gc
                gc.collect()
                # Load new model
                pipe = load_model(new_model)
                print(f"[Model] ✓ Model switched to '{new_model}'\n")

            # Get frame from Syphon
            if syphon_in.has_new_frame:
                syphon_frame = syphon_in.new_frame_image

                if syphon_frame is not None:
                    # Convert Metal texture to numpy (BGRA format)
                    frame_bgra = copy_mtl_texture_to_image(syphon_frame)

                    # Swap B and R channels: BGRA → RGBA
                    frame_rgba = frame_bgra[:, :, [2, 1, 0, 3]]

                    # Flip for Syphon coordinate system
                    frame_rgba = cv2.flip(frame_rgba, 0)

                    # Drop alpha for RGB
                    frame_rgb = frame_rgba[:, :, :3]

                    # Resize if needed
                    if frame_rgb.shape[0] != HEIGHT or frame_rgb.shape[1] != WIDTH:
                        pil_image = Image.fromarray(frame_rgb, mode='RGB').resize(
                            (WIDTH, HEIGHT), Image.Resampling.LANCZOS
                        )
                    else:
                        pil_image = Image.fromarray(frame_rgb, mode='RGB')

                    # Get current settings
                    prompt, strength, steps = settings.get_all()

                    # Generate with error handling
                    try:
                        generator = torch.manual_seed(42)
                        output_image = pipe(
                            prompt=prompt or DEFAULT_PROMPT,
                            image=pil_image,
                            num_inference_steps=steps,
                            guidance_scale=GUIDANCE_SCALE,
                            strength=strength,
                            generator=generator,
                            width=WIDTH,
                            height=HEIGHT
                        ).images[0]
                    except Exception as e:
                        print(f"\n[ERROR] Generation failed with strength={strength}, steps={steps}")
                        print(f"[ERROR] {e}")
                        import traceback
                        traceback.print_exc()
                        # Use input image as fallback
                        output_image = pil_image
                        print("[ERROR] Using input image as fallback\n")

                    # Convert to numpy array (PIL outputs RGB)
                    output_rgb = np.array(output_image)

                    # Flip back for Syphon coordinate system
                    output_rgb = cv2.flip(output_rgb, 0)

                    # Add alpha channel: RGB → RGBA (Syphon out expects RGBA!)
                    alpha = np.ones((HEIGHT, WIDTH, 1), dtype=np.uint8) * 255
                    output_bgra = np.concatenate([output_rgb, alpha], axis=2)

                    # Copy to Metal texture and publish
                    copy_image_to_mtl_texture(output_bgra, output_texture)
                    syphon_out.publish_frame_texture(output_texture)

                    last_output = output_bgra.copy()

                    frame_count += 1

                    # Calculate and send FPS
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"[Stats] Frames: {frame_count}, FPS: {fps:.1f}")
                        osc_send.send_message("/sd/fps", fps)
                        osc_send.send_message("/sd/frames", frame_count)

            else:
                # No new frame, republish last output if we have one
                if last_output is not None:
                    copy_image_to_mtl_texture(last_output, output_texture)
                    syphon_out.publish_frame_texture(output_texture)

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[Shutdown] Ctrl+C received")
        running = False

    finally:
        print("\n[Shutdown] Cleaning up...")
        try:
            if 'osc_recv_server' in locals():
                osc_recv_server.shutdown()
            if 'osc_thread' in locals():
                osc_thread.join(timeout=1.0)
        except Exception as e:
            pass

        # Cleanup GPU memory
        try:
            import gc
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()
            print("[Shutdown] ✓ GPU memory cleared")
        except Exception as e:
            print(f"[Shutdown] Cleanup warning: {e}")

        if frame_count > 0:
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\n[Stats] Total frames: {frame_count}")
            print(f"[Stats] Total time: {elapsed:.1f}s")
            print(f"[Stats] Average FPS: {avg_fps:.1f}")

        print("[Shutdown] Done")
        sys.exit(0)

if __name__ == "__main__":
    main()
