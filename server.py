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
import time

import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align

def parse_args():
    parser = argparse.ArgumentParser(description="Remote GPU accelerated face-swapping server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind the server to (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000).")
    parser.add_argument("--det-size", type=int, nargs=2, default=[640, 480], help="Face detection size width height (default: 640 480). Smaller is faster.")
    # Use parse_known_args to be robust against extra arguments from wrappers
    args, _ = parser.parse_known_args()
    return args

# Parse arguments at module level so they are globally accessible during startup and runtime
args = parse_args()

app = FastAPI(title="Remote GPU Face-Swap Service")

# Model configuration
MODEL_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
MODEL_PATH = "inswapper_128.onnx"

# Global state
target_face_object = None
face_analyser = None
target_analyser = None
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
    global face_analyser, target_analyser, swapper
    
    # Download the swapper model if missing
    download_model_if_needed()
    
    # Detect available ONNX runtime execution providers
    available_providers = ort.get_available_providers()
    print("Available ONNX Providers on server:", available_providers)
    
    # Optimize CUDA options if GPU is available
    providers = []
    if 'CUDAExecutionProvider' in available_providers:
        # Enable CUDA provider with performance-enhancing options
        cuda_options = {
            'cudnn_conv_algo_search': 'DEFAULT', # Balances start speed and runtime speed
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'do_copy_in_default_stream': True
        }
        providers.append(('CUDAExecutionProvider', cuda_options))
    else:
        providers.append('CPUExecutionProvider')
    print(f"Configuring models to use: {providers}")
    
    # Initialize InsightFace FaceAnalysis for face detection/landmark extraction (live swapping)
    # We only load 'detection' for the webcam stream to minimize CPU/GPU load
    face_analyser = FaceAnalysis(name='buffalo_l', providers=providers, allowed_modules=['detection'])
    det_size_tuple = tuple(args.det_size)
    face_analyser.prepare(ctx_id=0, det_size=det_size_tuple)
    print(f"Face Analyser (detection-only, size {det_size_tuple}) ready.")
    
    # Initialize separate analyzer for processing the uploaded target face
    # This requires 'recognition' to extract target face embeddings
    target_analyser = FaceAnalysis(name='buffalo_l', providers=providers, allowed_modules=['detection', 'recognition'])
    target_analyser.prepare(ctx_id=0, det_size=(640, 640))
    print("Target Face Analyser (detection + recognition) ready.")
    
    # Initialize the swapper model
    if os.path.exists(MODEL_PATH):
        try:
            if 'CUDAExecutionProvider' in available_providers:
                # Force swapper model to run on the GPU using a manual InferenceSession
                print("Forcing GPU execution for the Face Swapper model...")
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(MODEL_PATH, sess_options, providers=['CUDAExecutionProvider'])
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
    global target_face_object, target_analyser
    
    if target_analyser is None:
        return {"status": "error", "message": "Target face analyser model is not initialised on the server."}
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Failed to decode the uploaded image file."}
            
        # Detect faces in the target image (requires detection + recognition, so we use target_analyser)
        faces = target_analyser.get(img)
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

def paste_back_frame(img, bgr_fake, aimg, M):
    """Pastes the swapped face BGR image back onto the original frame using affine transformation."""
    target_img = img
    fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
    fake_diff = np.abs(fake_diff).mean(axis=2)
    fake_diff[:2, :] = 0
    fake_diff[-2:, :] = 0
    fake_diff[:, :2] = 0
    fake_diff[:, -2:] = 0
    
    IM = cv2.invertAffineTransform(M)
    img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
    
    bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    
    img_white[img_white > 20] = 255
    fthresh = 10
    fake_diff[fake_diff < fthresh] = 0
    fake_diff[fake_diff >= fthresh] = 255
    img_mask = img_white
    
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if len(mask_h_inds) == 0 or len(mask_w_inds) == 0:
        return target_img
        
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))
    k = max(mask_size // 10, 10)
    
    kernel = np.ones((k, k), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    
    kernel = np.ones((2, 2), np.uint8)
    fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
    
    k = max(mask_size // 20, 5)
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    
    k = 5
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
    
    img_mask /= 255.0
    fake_diff /= 255.0
    
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    fake_merged = img_mask * bgr_fake + (1.0 - img_mask) * target_img.astype(np.float32)
    fake_merged = fake_merged.astype(np.uint8)
    return fake_merged


def process_batch(batch, face_analyser, swapper, target_face_object):
    """Processes a batch of frames: decodes, performs batched face-swapping, and re-encodes to JPEG."""
    decoded_frames = []
    original_datas = []
    
    # 1. Decode JPEG frames
    for frame_id, raw_bytes in batch:
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            decoded_frames.append(None)
        else:
            decoded_frames.append(img)
        original_datas.append(raw_bytes)
    
    # 2. Collect all face swap jobs across the decoded frames
    jobs = []
    for idx, img in enumerate(decoded_frames):
        if img is None:
            continue
        faces = face_analyser.get(img)
        if len(faces) > 0:
            for face in faces:
                # Align and crop face
                aimg, M = face_align.norm_crop2(img, face.kps, swapper.input_size[0])
                blob = cv2.dnn.blobFromImage(aimg, 1.0 / swapper.input_std, swapper.input_size,
                                              (swapper.input_mean, swapper.input_mean, swapper.input_mean), swapRB=True)
                # Prepare latent
                latent = target_face_object.normed_embedding.reshape((1, -1))
                latent = np.dot(latent, swapper.emap)
                latent /= np.linalg.norm(latent)
                
                jobs.append({
                    'frame_idx': idx,
                    'aimg': aimg,
                    'M': M,
                    'blob': blob,
                    'latent': latent
                })
                
    # 3. If we have jobs, perform batched GPU inference
    if len(jobs) > 0:
        try:
            # Concatenate blobs and latents along the batch axis (axis=0)
            blobs = np.concatenate([j['blob'] for j in jobs], axis=0)
            latents = np.concatenate([j['latent'] for j in jobs], axis=0)
            
            # Single GPU call for the whole batch of faces!
            preds = swapper.session.run(swapper.output_names, {
                swapper.input_names[0]: blobs,
                swapper.input_names[1]: latents
            })[0]
            
            # Post-process and paste each swapped face back to its corresponding frame
            for i, job in enumerate(jobs):
                frame_idx = job['frame_idx']
                img = decoded_frames[frame_idx]
                
                # Get the prediction for this face swap job
                img_fake = preds[i].transpose((1, 2, 0))
                bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
                
                # Paste back onto the frame
                decoded_frames[frame_idx] = paste_back_frame(img, bgr_fake, job['aimg'], job['M'])
        except Exception as e:
            print(f"Error during batched GPU face swapping: {e}")
            # On failure, we fallback to original frames
            
    # 4. Re-encode the processed frames to JPEG
    encoded_batch = []
    for idx, (frame_id, raw_bytes) in enumerate(batch):
        img = decoded_frames[idx]
        if img is None:
            encoded_batch.append((frame_id, raw_bytes))
        else:
            _, encoded_img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            encoded_batch.append((frame_id, encoded_img.tobytes()))
            
    return encoded_batch


def process_frame(img, face_analyser, swapper, target_face_object):
    """Processes a single frame: detects faces and performs swapping on them in-place with batched inference."""
    if target_face_object is None or swapper is None or face_analyser is None:
        return img
        
    faces = face_analyser.get(img)
    if len(faces) == 0:
        return img
        
    jobs = []
    for face in faces:
        aimg, M = face_align.norm_crop2(img, face.kps, swapper.input_size[0])
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / swapper.input_std, swapper.input_size,
                                      (swapper.input_mean, swapper.input_mean, swapper.input_mean), swapRB=True)
        latent = target_face_object.normed_embedding.reshape((1,-1))
        latent = np.dot(latent, swapper.emap)
        latent /= np.linalg.norm(latent)
        
        jobs.append({
            'aimg': aimg,
            'M': M,
            'blob': blob,
            'latent': latent
        })
        
    try:
        blobs = np.concatenate([j['blob'] for j in jobs], axis=0)
        latents = np.concatenate([j['latent'] for j in jobs], axis=0)
        
        preds = swapper.session.run(swapper.output_names, {
            swapper.input_names[0]: blobs,
            swapper.input_names[1]: latents
        })[0]
        
        for i, job in enumerate(jobs):
            img_fake = preds[i].transpose((1, 2, 0))
            bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
            img = paste_back_frame(img, bgr_fake, job['aimg'], job['M'])
    except Exception as e:
        print(f"Error during batched face-swapping in process_frame: {e}")
        # Fallback to sequential swapping on failure
        for face in faces:
            try:
                img = swapper.get(img, face, target_face_object, paste_back=True)
            except Exception:
                pass
                
    return img

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
            
            # Run face-swapping in a separate thread to keep the event loop responsive
            swapped_img = await asyncio.to_thread(process_frame, img, face_analyser, swapper, target_face_object)
                
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
    """Handles low-latency, real-time image swapping over a binary WebSocket connection using dynamic batching."""
    global target_face_object, face_analyser, swapper
    
    await websocket.accept()
    print("New client connected for live face-swapping.")
    
    # Decouple receiver and sender to enable dynamic batching
    input_queue = asyncio.Queue(maxsize=16)
    output_queue = asyncio.Queue(maxsize=16)
    
    async def receiver_task():
        frame_id = 0
        try:
            while True:
                data = await websocket.receive_bytes()
                await input_queue.put((frame_id, data))
                frame_id += 1
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"WebSocket Receiver error: {e}")
        finally:
            # Signal the batcher to stop
            await input_queue.put((None, None))

    async def sender_task():
        try:
            next_send_id = 0
            pending_sends = {}
            
            while True:
                frame_id, encoded_data = await output_queue.get()
                if frame_id is None:
                    break
                pending_sends[frame_id] = encoded_data
                output_queue.task_done()
                
                while next_send_id in pending_sends:
                    data_to_send = pending_sends.pop(next_send_id)
                    await websocket.send_bytes(data_to_send)
                    next_send_id += 1
        except Exception as e:
            print(f"WebSocket Sender error: {e}")

    async def batcher_task():
        MAX_BATCH_SIZE = 4
        BATCH_TIMEOUT = 0.005  # 5ms wait to dynamically collect incoming frames
        
        try:
            while True:
                # Wait for the first frame in the next batch
                frame_id, data = await input_queue.get()
                if frame_id is None:
                    await output_queue.put((None, None))
                    input_queue.task_done()
                    break
                
                batch = [(frame_id, data)]
                input_queue.task_done()
                
                # Dynamic batching: try to gather up to MAX_BATCH_SIZE frames
                start_time = time.time()
                while len(batch) < MAX_BATCH_SIZE:
                    time_remaining = BATCH_TIMEOUT - (time.time() - start_time)
                    if time_remaining <= 0:
                        break
                    try:
                        next_frame_id, next_data = await asyncio.wait_for(
                            input_queue.get(), 
                            timeout=max(0.001, time_remaining)
                        )
                        if next_frame_id is None:
                            # Re-add Sentinel for shutdown sequence
                            await input_queue.put((None, None))
                            break
                        batch.append((next_frame_id, next_data))
                        input_queue.task_done()
                    except asyncio.TimeoutError:
                        break
                
                # If target_face_object or models are not loaded, bypass immediately
                if target_face_object is None or swapper is None or face_analyser is None:
                    for fid, d in batch:
                        await output_queue.put((fid, d))
                    continue
                
                # Offload processing to thread pool to avoid blocking the event loop
                processed_batch = await asyncio.to_thread(
                    process_batch, 
                    batch, 
                    face_analyser, 
                    swapper, 
                    target_face_object
                )
                
                for fid, encoded_res in processed_batch:
                    await output_queue.put((fid, encoded_res))
                    
        except Exception as e:
            print(f"WebSocket Batcher error: {e}")
            await output_queue.put((None, None))

    # Run receiver, batcher, and sender concurrently
    tasks = [
        asyncio.create_task(receiver_task()),
        asyncio.create_task(batcher_task()),
        asyncio.create_task(sender_task())
    ]
    
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error in WebSocket handler tasks: {e}")
    finally:
        # Cancel any remaining tasks to ensure no dangling tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        print("Client disconnected from face-swapping feed.")

if __name__ == "__main__":
    # Host on 0.0.0.0 so external clients can connect to the port
    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)
