import asyncio
import cv2
import numpy as np
import websockets
import httpx
import argparse
import sys
import os
import pyvirtualcam
import threading
import time
from urllib.parse import urlparse
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import av
import fractions

def parse_args():
    parser = argparse.ArgumentParser(description="Client for remote GPU accelerated face-swapping.")
    parser.add_argument("--server", type=str, default="localhost", help="IP address or domain of the remote GPU server.")
    parser.add_argument("--port", type=int, default=8000, help="Port of the remote GPU server.")
    parser.add_argument("--secure", action="store_true", help="Use secure HTTPS/WSS connections.")
    parser.add_argument("--protocol", type=str, choices=["webrtc", "websockets"], default="websockets", help="Streaming protocol to use: webrtc or websockets (default: websockets).")
    parser.add_argument("--camera", type=int, default=0, help="Local webcam device index (default: 0).")
    parser.add_argument("--target", type=str, default="", help="Path to the local character image (JPEG/PNG) to swap with.")
    parser.add_argument("--width", type=int, default=640, help="Webcam capture width (default: 640).")
    parser.add_argument("--height", type=int, default=480, help="Webcam capture height (default: 480).")
    parser.add_argument("--fps", type=int, default=30, help="Webcam target FPS (default: 30).")
    parser.add_argument("--no-preview", action="store_true", help="Disable the local OpenCV preview window.")
    return parser.parse_args()

class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def get(self, prop):
        return self.cap.get(prop)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.started = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        self.cap.release()

async def upload_target_face(url, image_path):
    """Uploads the local target character image to the server."""
    if not os.path.exists(image_path):
        print(f"ERROR: Target image path does not exist: {image_path}")
        sys.exit(1)
        
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

async def websocket_loop(args, websocket_url):
    # Set up OpenCV webcam capture
    cap = ThreadedCamera(args.camera, args.width, args.height, args.fps).start()
    
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
        
    print(f"Connecting to face-swapping stream at {websocket_url}...")
    
    # Establish persistent WebSocket connection
    try:
        async with websockets.connect(websocket_url, max_size=None) as ws:
            print("Connected to remote GPU swapper! Streaming started...")
            print("Press 'q' in the preview window to exit.")
            
            frame_count = 0
            t_start = time.time()
            
            accum_grab = 0
            accum_compress = 0
            accum_net = 0
            accum_decode = 0
            accum_vcam = 0
            accum_preview = 0
            
            while True:
                t0 = time.time()
                ret, frame = cap.read()
                t1 = time.time()
                accum_grab += (t1 - t0)
                
                if not ret:
                    print("Failed to grab frame from webcam.")
                    await asyncio.sleep(0.01)
                    continue
                    
                t2 = time.time()
                # Compress the raw frame to JPEG to minimize upload upload bandwidth and latency
                _, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                t3 = time.time()
                accum_compress += (t3 - t2)
                
                try:
                    t4 = time.time()
                    # Send binary JPEG frame to cloud
                    await ws.send(encoded_img.tobytes())
                    
                    # Receive swapped binary JPEG frame from cloud
                    response_data = await ws.recv()
                    t5 = time.time()
                    accum_net += (t5 - t4)
                    
                    t6 = time.time()
                    # Decode swapped JPEG back to OpenCV frame
                    np_arr = np.frombuffer(response_data, dtype=np.uint8)
                    swapped_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    t7 = time.time()
                    accum_decode += (t7 - t6)
                    
                    if swapped_frame is None:
                        swapped_frame = frame  # Fallback to original frame if decoding failed
                        
                except Exception as e:
                    print(f"Stream frame transmission error: {e}")
                    swapped_frame = frame  # Fallback to original
                    
                t8 = time.time()
                # Write to the system's Virtual Camera (requires RGB format)
                if vcam is not None:
                    try:
                        rgb_frame = cv2.cvtColor(swapped_frame, cv2.COLOR_BGR2RGB)
                        vcam.send(rgb_frame)
                        vcam.sleep_until_next_frame()
                    except Exception as e:
                        print(f"Virtual camera write error: {e}")
                t9 = time.time()
                accum_vcam += (t9 - t8)
                
                t10 = time.time()
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
                    await asyncio.sleep(0.001)
                t11 = time.time()
                accum_preview += (t11 - t10)
                
                frame_count += 1
                if frame_count >= 30:
                    fps = frame_count / (time.time() - t_start)
                    print(f"\n--- FPS: {fps:.2f} ---")
                    print(f"Camera Grab:   {accum_grab/frame_count*1000:.2f}ms")
                    print(f"JPEG Compress: {accum_compress/frame_count*1000:.2f}ms")
                    print(f"Network RTT:   {accum_net/frame_count*1000:.2f}ms")
                    print(f"JPEG Decode:   {accum_decode/frame_count*1000:.2f}ms")
                    print(f"Virtual Cam:   {accum_vcam/frame_count*1000:.2f}ms")
                    print(f"GUI Preview:   {accum_preview/frame_count*1000:.2f}ms")
                    print("----------------\n")
                    
                    frame_count = 0
                    t_start = time.time()
                    accum_grab = 0
                    accum_compress = 0
                    accum_net = 0
                    accum_decode = 0
                    accum_vcam = 0
                    accum_preview = 0
                    
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

class OpenCVVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, cap, fps=30):
        super().__init__()
        self.cap = cap
        self.fps = fps
        self.time_base = fractions.Fraction(1, 90000)
        self.pts = 0

    async def recv(self):
        # Enforce frame rate timing on capture
        await asyncio.sleep(1.0 / self.fps)
        
        ret, frame = self.cap.read()
        if not ret:
            # Create a black dummy frame if capture fails
            h, w = 480, 640
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            
        # Convert OpenCV BGR frame to av.VideoFrame
        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        
        # Calculate PTS
        self.pts += int(90000 / self.fps)
        av_frame.pts = self.pts
        av_frame.time_base = self.time_base
        
        return av_frame

async def display_swapped_track(track, vcam, args, stop_event):
    try:
        while not stop_event.is_set():
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            
            # Write to system's Virtual Camera (requires RGB format)
            if vcam is not None:
                try:
                    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    vcam.send(rgb_frame)
                    vcam.sleep_until_next_frame()
                except Exception as e:
                    print(f"Virtual camera write error: {e}")
                    
            # Render local preview window (requires BGR format)
            if not args.no_preview:
                try:
                    cv2.imshow("Remote GPU Face-Swap Preview (WebRTC)", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User closed the preview window.")
                        stop_event.set()
                        break
                except cv2.error:
                    print("\nWARNING: Local GUI preview window is not supported in this environment.")
                    print("Disabling preview and running in background-only mode.")
                    args.no_preview = True
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error displaying swapped track: {e}")

async def webrtc_loop(args, offer_url):
    # Set up OpenCV webcam capture
    cap = ThreadedCamera(args.camera, args.width, args.height, args.fps).start()
    
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
        
    pc = RTCPeerConnection()
    
    # Add client's webcam track
    local_video = OpenCVVideoTrack(cap, fps=actual_fps)
    pc.addTrack(local_video)
    
    stop_event = asyncio.Event()
    
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            print("Received face-swapped video track from server over WebRTC!")
            asyncio.ensure_future(display_swapped_track(track, vcam, args, stop_event))
            
    # Create SDP offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    print(f"Connecting to face-swapping stream via WebRTC at {offer_url}...")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }
            response = await client.post(offer_url, json=payload)
            
            if response.status_code != 200:
                print(f"ERROR: Failed to establish WebRTC connection. Server returned {response.status_code}")
                return
                
            answer_data = response.json()
            answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
            await pc.setRemoteDescription(answer)
            print("WebRTC connection established successfully! Streaming started...")
            print("Press 'q' in the preview window to exit.")
            
            # Keep running until stop event is set (e.g. 'q' pressed in preview window)
            while not stop_event.is_set():
                await asyncio.sleep(0.1)
                
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"\nWebRTC session interrupted: {e}")
    finally:
        # Clean up resources
        print("Closing connection...")
        cap.release()
        await pc.close()
        if vcam is not None:
            vcam.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("WebRTC streaming closed. Cleanup complete.")

def resolve_urls(args):
    server_input = args.server
    
    # Check if a scheme is already present
    if "://" in server_input:
        parsed = urlparse(server_input)
        scheme = parsed.scheme
        host = parsed.netloc
    else:
        scheme = "https" if (args.secure or "cloudflare" in server_input) else "http"
        host = server_input

    # Strip any port from host if user specified port separately
    if ":" in host:
        host_parts = host.split(":")
        host = host_parts[0]
        port = int(host_parts[1])
    else:
        port = args.port

    # Build base HTTP and WebSocket URLs
    if (scheme == "https" and port == 443) or (scheme == "http" and port == 80) or ("cloudflare" in host):
        # Cloudflare tunnels do not expose custom ports on their public URLs, so we omit port
        http_url = f"{scheme}://{host}/set_target"
        ws_url = f"{'wss' if scheme == 'https' else 'ws'}://{host}/swap"
    else:
        http_url = f"{scheme}://{host}:{port}/set_target"
        ws_url = f"{'wss' if scheme == 'https' else 'ws'}://{host}:{port}/swap"
        
    return http_url, ws_url

async def main():
    args = parse_args()
    http_url, ws_url = resolve_urls(args)
    
    # If a target character image is specified, upload it first via HTTP
    if args.target:
        success = await upload_target_face(http_url, args.target)
        if not success:
            print("WARNING: Target face upload failed. Streaming anyway (will not swap until a target is set).")
            
    # Start the real-time webcam stream
    if args.protocol == "webrtc":
        offer_url = http_url.replace("/set_target", "/offer")
        await webrtc_loop(args, offer_url)
    else:
        await websocket_loop(args, ws_url)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully via interrupt.")
