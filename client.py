import asyncio
import cv2
import numpy as np
import websockets
import httpx
import argparse
import sys
import os
import pyvirtualcam

def parse_args():
    parser = argparse.ArgumentParser(description="Client for remote GPU accelerated face-swapping.")
    parser.add_argument("--server", type=str, default="localhost", help="IP address or domain of the remote GPU server.")
    parser.add_argument("--port", type=int, default=8000, help="Port of the remote GPU server.")
    parser.add_argument("--camera", type=int, default=0, help="Local webcam device index (default: 0).")
    parser.add_argument("--target", type=str, default="", help="Path to the local character image (JPEG/PNG) to swap with.")
    parser.add_argument("--width", type=int, default=640, help="Webcam capture width (default: 640).")
    parser.add_argument("--height", type=int, default=480, help="Webcam capture height (default: 480).")
    parser.add_argument("--fps", type=int, default=30, help="Webcam target FPS (default: 30).")
    parser.add_argument("--no-preview", action="store_true", help="Disable the local OpenCV preview window.")
    return parser.parse_args()

async def upload_target_face(server_ip, port, image_path):
    """Uploads the local target character image to the server."""
    if not os.path.exists(image_path):
        print(f"ERROR: Target image path does not exist: {image_path}")
        sys.exit(1)
        
    url = f"http://{server_ip}:{port}/set_target"
    print(f"Uploading target character face {image_path} to {url}...")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                response = await client.post(url, files=files)
                
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    print("SUCCESS:", result.get("message"))
                    return True
                else:
                    print("SERVER ERROR:", result.get("message"))
                    return False
            else:
                print(f"HTTP ERROR: Received status code {response.status_code}")
                return False
    except Exception as e:
        print(f"CONNECTION ERROR during upload: {e}")
        return False

async def streaming_loop(args):
    # Set up OpenCV webcam capture
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Retrieve actual resolution set by OpenCV
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if actual_fps <= 0:
        actual_fps = args.fps
        
    print(f"Webcam initialised: {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    if not cap.isOpened():
        print(f"ERROR: Could not open webcam index {args.camera}")
        sys.exit(1)
        
    # Set up virtual camera
    vcam = None
    try:
        vcam = pyvirtualcam.Camera(width=actual_width, height=actual_height, fps=actual_fps)
        print(f"Virtual camera active: {vcam.device}")
    except Exception as e:
        print("\n" + "="*60)
        print("WARNING: Could not initialize a virtual camera.")
        print(f"Detail: {e}")
        print("To output directly as a system camera feed:")
        print(" - On Windows: Install OBS Studio or 'OBS Virtual Camera'.")
        print(" - On Linux: Ensure 'v4l2loopback' is loaded: sudo modprobe v4l2loopback")
        print("The program will continue using local preview window only.")
        print("="*60 + "\n")
        
    websocket_url = f"ws://{args.server}:{args.port}/swap"
    print(f"Connecting to face-swapping stream at {websocket_url}...")
    
    # Establish persistent WebSocket connection
    try:
        async with websockets.connect(websocket_url, max_size=None) as ws:
            print("Connected to remote GPU swapper! Streaming started...")
            print("Press 'q' in the preview window to exit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from webcam.")
                    await asyncio.sleep(0.01)
                    continue
                    
                # Compress the raw frame to JPEG to minimize upload upload bandwidth and latency
                _, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                
                try:
                    # Send binary JPEG frame to cloud
                    await ws.send(encoded_img.tobytes())
                    
                    # Receive swapped binary JPEG frame from cloud
                    response_data = await ws.recv()
                    
                    # Decode swapped JPEG back to OpenCV frame
                    np_arr = np.frombuffer(response_data, dtype=np.uint8)
                    swapped_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if swapped_frame is None:
                        swapped_frame = frame  # Fallback to original frame if decoding failed
                        
                except Exception as e:
                    print(f"Stream frame transmission error: {e}")
                    swapped_frame = frame  # Fallback to original
                    
                # Write to the system's Virtual Camera (requires RGB format)
                if vcam is not None:
                    try:
                        rgb_frame = cv2.cvtColor(swapped_frame, cv2.COLOR_BGR2RGB)
                        vcam.send(rgb_frame)
                        vcam.sleep_until_next_frame()
                    except Exception as e:
                        print(f"Virtual camera write error: {e}")
                        
                # Render local preview window (requires BGR format)
                if not args.no_preview:
                    try:
                        cv2.imshow("Remote GPU Face-Swap Preview", swapped_frame)
                        # Use a tiny waitkey to allow OpenCV to pump window events
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except cv2.error:
                        print("\nWARNING: Local GUI preview window is not supported in this environment.")
                        print("Disabling preview and running in background-only mode.")
                        print("To run cleanly without GUI, use the '--no-preview' flag.")
                        args.no_preview = True
                else:
                    # If preview is disabled, run a tiny yield to prevent thread lock
                    await asyncio.sleep(0.001)
                    
    except Exception as e:
        print(f"\nConnection with server interrupted: {e}")
    finally:
        # Clean up resources
        cap.release()
        if vcam is not None:
            vcam.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Streaming closed. Cleanup complete.")

async def main():
    args = parse_args()
    
    # If a target character image is specified, upload it first via HTTP
    if args.target:
        success = await upload_target_face(args.server, args.port, args.target)
        if not success:
            print("WARNING: Target face upload failed. Streaming anyway (will not swap until a target is set).")
            
    # Start the real-time webcam stream
    await streaming_loop(args)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully via interrupt.")
