# 🚀 Remote GPU Real-Time Face-Swapping System

This project allows you to stream your local webcam feed to a **remote, cloud-based GPU server (e.g., RunPod, Vast.ai in Europe)**, perform real-time AI face-swapping with any fictional character image (via a single JPEG), and route the returned swapped feed directly into your system's virtual camera.

This allows you to go on video chat websites like **Omegle, Discord, or Zoom** as any Avatar of your choice—**completely powered by a remote GPU without requiring a graphics card on your local laptop!**

---

## 🛠️ System Architecture

```
                       ┌─────────────────────────────────────────┐
                       │            YOUR LOCAL COMPUTER          │
                       │   1. Capture webcam frame (OpenCV)      │
                       │   2. Upload target JPEG to server       │
                       │   3. Send frame stream via WebSockets   │
                       │   4. Receive & output to Virtual Camera │
                       └────────────┬────────────────▲───────────┘
                                    │                │
                        Websockets Upload        Websockets Download
                          ~15-40ms Ping            ~15-40ms Ping
                                    │                │
                       ┌────────────▼────────────────┴───────────┐
                       │          REMOTE CLOUD GPU SERVER        │
                       │   1. FastAPI Websocket Server           │
                       │   2. Face detection (buffalo_l)         │
                       │   3. ONNX GPU Face Swapper              │
                       └─────────────────────────────────────────┘
```

---

## ☁️ Server Deployment (Cloud GPU)

### 1. Rent a GPU Server in Europe
Use **RunPod.io** or **Vast.ai** to rent a GPU server.
* **Location Filter:** Select servers located in European regions (e.g., Frankfurt, Stockholm, Amsterdam, Paris) to keep your round-trip network latency under 40ms.
* **Recommended GPU:** **NVIDIA RTX 3080, 4070, 3090, or A4000/L4**.
* **Template selection:** Choose a **PyTorch with CUDA** template (e.g., PyTorch 2.0+, CUDA 11.8 or 12.1).
* **Port Mapping:** Ensure port **`8000`** is exposed/mapped externally on your container so the client can connect.

### 2. Install Server Dependencies
SSH into your cloud GPU server and run the following commands:
```bash
# Update container and install essential system dependencies
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Navigate to your workspace directory
cd /workspace # (or your container's default workspace)

# Download the server script and requirements file
# Install requirements
pip install -r requirements-server.txt
```

### 3. Run the Server
Launch the server using `uvicorn`:
```bash
python server.py
```
*Note: On first launch, the server will automatically download the 550MB `inswapper_128.onnx` face-swapping model from HuggingFace.*

---

## 💻 Local Client Setup (Your Computer)

### 1. Install a Virtual Camera Driver
For the face-swapped video to appear as a physical webcam inside your browser (for Omegle/Discord):
* **Windows:** Download and install [OBS Studio](https://obsproject.com/). Launch OBS once, click **"Start Virtual Camera"** in the bottom right corner, then stop it. This registers the system's virtual camera drivers.
* **Linux:** Install `v4l2loopback` and load the kernel module:
  ```bash
  sudo apt install v4l2loopback-dkms v4l2loopback-utils
  sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OBS Virtual Camera" exclusive_caps=1
  ```
* **macOS:** Install OBS Studio. It registers a virtual camera automatically.

### 2. Install Local Dependencies
Ensure you have **Python 3.8+** installed locally, then run:
```bash
pip install -r requirements-client.txt
```

### 3. Stream to your Remote GPU
Run the client script, specifying your remote GPU server's external IP address and the path to the JPEG of the character you want to become:

```bash
python client.py --server <REMOTE_GPU_IP_OR_DOMAIN> --port 8000 --target character.jpg
```

### Command Arguments Explained:
* `--server`: The public IP or domain of your rented cloud GPU.
* `--port`: The public mapped port (usually `8000` or the mapped external port provided by RunPod/Vast.ai).
* `--target`: Path to a high-resolution, front-facing image of the character you want to face-swap with.
* `--camera`: Webcam index (default `0`). Set to `1`, `2` if you have external webcams.
* `--width` & `--height`: Webcam capture resolution (default `640`x`480`). Keep it at default for the lowest possible latency and best performance.
* `--no-preview`: Disables the local popup video window (highly useful to save your local CPU/laptop battery).

---

## 🚀 Speed & Latency Optimization Tips

1. **Keep Capture Resolution Compact:** Capturing webcam frames at **640x480** is the sweet spot. A larger resolution (like 1080p) will require up to 4 times the bandwidth and significantly higher upload transmission latency, causing lag, with minimal visual improvement to the face-swapping area itself.
2. **Ping is King:** Always pick cloud servers in your closest geographical location. If you are in the UK/Europe, servers hosted by companies in Germany, the UK, Sweden, or France will result in <25ms network transmission overhead.
3. **JPEG Compression:** In `client.py`, the `cv2.imencode` quality is set to `80`. This provides an excellent balance: high visual clarity for facial landmark detection while reducing the data size per frame to a tiny fraction (often ~25-40KB per frame), meaning you can stream smoothly even on average home Wi-Fi connections.
