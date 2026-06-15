import os
import urllib.request
import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse
import uvicorn
import argparse
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import av
from pydantic import BaseModel

import insightface
from insightface.app import FaceAnalysis

app = FastAPI(title="Remote GPU Face-Swap Service")

# Model configuration
MODEL_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
MODEL_PATH = "inswapper_128.onnx"

# Global state
target_face_object = None
face_analyser = None
swapper = None

def download_model_if_needed():
    """Downloads the inswapper_128.onnx model if not already present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading inswapper_128.onnx model from {MODEL_URL}...")
        try:
            # Create a custom opener to bypass some basic user-agent blocks if they exist
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please manually download the model from HuggingFace and place it in the server directory as 'inswapper_128.onnx'.")

@app.on_event("startup")
def startup_event():
    """Initialises models on application startup."""
    global face_analyser, swapper
    
    # Download the swapper model if missing
    download_model_if_needed()
    
    # Detect available ONNX runtime execution providers
    available_providers = ort.get_available_providers()
    print("Available ONNX Providers on server:", available_providers)
    
    providers = []
    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    print(f"Configuring models to use: {providers}")
    
    # Initialize InsightFace FaceAnalysis for face detection/landmark extraction
    # 'buffalo_l' is the standard high-quality face analysis pack
    face_analyser = FaceAnalysis(name='buffalo_l', providers=providers)
    # 640x640 is standard and handles webcam feeds beautifully
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    print("Face Analyser ready.")
    
    # Initialize the swapper model
    if os.path.exists(MODEL_PATH):
        try:
            if 'CUDAExecutionProvider' in available_providers:
                # Force swapper model to run on the GPU using a manual InferenceSession
                print("Forcing GPU execution for the Face Swapper model...")
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(MODEL_PATH, sess_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                swapper = insightface.model_zoo.get_model(MODEL_PATH, download=False, download_zip=False, session=sess)
            else:
                swapper = insightface.model_zoo.get_model(MODEL_PATH, download=False, download_zip=False)
            print("Face Swapper model loaded successfully.")
        except Exception as e:
            print(f"Error loading Face Swapper model: {e}")
    else:
        print("WARNING: inswapper_128.onnx not found. Swapping will not be available.")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Returns a simple web page monitoring the status of the server."""
    global target_face_object
    status = "LOADED" if target_face_object is not None else "NOT LOADED (Please upload a face image using the client)"
    available_providers = ort.get_available_providers()
    
    return f"""
    <html>
        <head>
            <title>GPU Face-Swap Server</title>
            <style>
                body {{ font-family: sans-serif; background-color: #121212; color: #e0e0e0; padding: 40px; }}
                h1 {{ color: #00ff88; }}
                .card {{ background-color: #1e1e1e; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
                .status-loaded {{ color: #00ff88; font-weight: bold; }}
                .status-empty {{ color: #ff3366; font-weight: bold; }}
                pre {{ background-color: #2e2e2e; padding: 15px; border-radius: 5px; color: #ffcc00; }}
            </style>
        </head>
        <body>
            <h1>Remote GPU Face-Swap Service</h1>
            <div class="card">
                <p><strong>Server Status:</strong> <span style="color: #00ff88">RUNNING</span></p>
                <p><strong>Target Character Face:</strong> <span class="{"status-loaded" if target_face_object is not None else "status-empty"}">{status}</span></p>
                <p><strong>Available ONNX Providers:</strong></p>
                <pre>{available_providers}</pre>
                <p><strong>Active Processing Providers:</strong></p>
                <pre>{'CUDAExecutionProvider (GPU accelerated)' if 'CUDAExecutionProvider' in available_providers else 'CPUExecutionProvider (Slow)'}</pre>
            </div>
        </body>
    </html>
    """

@app.post("/set_target")
async def set_target_endpoint(file: UploadFile = File(...)):
    """Receives an image, extracts face features, and sets it as the active face to swap to."""
    global target_face_object, face_analyser
    
    if face_analyser is None:
        return {"status": "error", "message": "Face analyser model is not initialised on the server."}
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Failed to decode the uploaded image file."}
            
        # Detect faces in the target image
        faces = face_analyser.get(img)
        if len(faces) == 0:
            return {"status": "error", "message": "No face could be detected in the provided image."}
            
        # Sort faces by bounding box size to pick the largest/main face
        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        target_face_object = faces[0]
        
        print("Successfully loaded new target face.")
        return {"status": "success", "message": "Target character face loaded and analysed successfully."}
        
    except Exception as e:
        print(f"Error loading target face: {e}")
        return {"status": "error", "message": str(e)}

pcs = set()

class FaceSwapVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        
        global target_face_object, face_analyser, swapper
        
        if target_face_object is None or swapper is None or face_analyser is None:
            return frame
            
        try:
            # Convert av.VideoFrame to OpenCV BGR image
            img = frame.to_ndarray(format="bgr24")
            
            # Run face-swapping
            faces = face_analyser.get(img)
            if len(faces) > 0:
                swapped_img = img.copy()
                for face in faces:
                    swapped_img = swapper.get(swapped_img, face, target_face_object, paste_back=True)
            else:
                swapped_img = img
                
            # Convert swapped BGR image back to av.VideoFrame
            new_frame = av.VideoFrame.from_ndarray(swapped_img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        except Exception as e:
            print(f"Error swapping frame in WebRTC track: {e}")
            return frame

class OfferModel(BaseModel):
    sdp: str
    type: str

@app.post("/offer")
async def webrtc_offer(params: OfferModel):
    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"WebRTC Connection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            print("WebRTC connection closed.")
            
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            print("Received client's video track over WebRTC.")
            swapped_track = FaceSwapVideoTrack(track)
            pc.addTrack(swapped_track)
            
    # Set remote description
    await pc.setRemoteDescription(offer)
    
    # Create SDP answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

@app.on_event("shutdown")
async def on_shutdown():
    # Close all peer connections on shutdown
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

@app.websocket("/swap")
async def websocket_endpoint(websocket: WebSocket):
    """Handles low-latency, real-time image swapping over a binary WebSocket connection."""
    global target_face_object, face_analyser, swapper
    
    await websocket.accept()
    print("New client connected for live face-swapping.")
    
    try:
        while True:
            # Receive compressed frame bytes from client
            data = await websocket.receive_bytes()
            
            # If no target face is loaded, return the frame unmodified
            if target_face_object is None or swapper is None or face_analyser is None:
                await websocket.send_bytes(data)
                continue
                
            # Decode JPEG frame
            np_arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                # If decoding failed, send original bytes back
                await websocket.send_bytes(data)
                continue
                
            # Detect faces on the webcam frame
            faces = face_analyser.get(img)
            
            if len(faces) > 0:
                # Swap every detected face with our target character face
                swapped_img = img.copy()
                for face in faces:
                    swapped_img = swapper.get(swapped_img, face, target_face_object, paste_back=True)
            else:
                swapped_img = img
                
            # Re-encode the swapped frame back to JPEG to minimize network payload size
            # Quality of 85 balances visual fidelity and payload size perfectly
            _, encoded_img = cv2.imencode('.jpg', swapped_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            
            # Send back the swapped frame as binary data
            await websocket.send_bytes(encoded_img.tobytes())
            
    except WebSocketDisconnect:
        print("Client disconnected from face-swapping feed.")
    except Exception as e:
        print(f"Error during WebSocket processing loop: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Remote GPU accelerated face-swapping server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind the server to (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Host on 0.0.0.0 so external clients can connect to the port
    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)
