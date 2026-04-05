# PTZ AI Camera Dashboard (CSI + YOLO)

This app is now intentionally fixed to one runtime path:
- CSI camera ingest via native GStreamer (`nvarguscamerasrc` -> `appsink`)
- YOLO tracking inference (`model.track(..., persist=True)`)
- HTTP MJPEG stream for browser viewing
- Status API for fps/latency/inference/drop counters

## Project Structure

```text
ptz-ai-dashboard/
  app/
    config.py          # CSI pipeline + runtime settings
    pipeline.py        # capture thread + YOLO thread + latest frame buffering
    metrics.py         # fps/latency/temp counters
    main.py            # FastAPI routes and startup
    templates/
      index.html
    static/
      style.css
  requirements.txt
```

## Runtime Behavior

1. Capture thread
- Opens CSI GStreamer pipeline from `config.py`.
- Reads frames continuously.
- Keeps only the newest raw frame (single-slot buffer).

2. YOLO thread
- Pulls newest unprocessed frame only.
- Runs `model.track(frame, persist=True)`.
- Draws overlay using `results[0].plot()`.
- Publishes newest JPEG frame for `/stream.mjpg`.

3. Latest-frame-wins
- If inference is busy, older raw frames are overwritten.
- This avoids queue growth and keeps latency bounded.

## Dependencies

Python:
- fastapi
- uvicorn[standard]
- opencv-python
- numpy
- psutil
- jinja2
- ultralytics

System (Jetson):
- JetPack 5
- Working CSI camera
- GStreamer with `nvarguscamerasrc`
- Python GStreamer bindings: `python3-gi`, `gir1.2-gstreamer-1.0`

## First-Time Jetson Setup

```bash
cd /home/camera/ptz-ai-dashboard
sudo apt update
sudo apt install -y python3-venv python3-gi gir1.2-gstreamer-1.0

# Use system site packages so gi/GStreamer bindings are visible in venv.
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Environment Variables

```bash
export APP_HOST=0.0.0.0
export APP_PORT=8000

export FRAME_WIDTH=1280
export FRAME_HEIGHT=720
export FRAME_FPS=30
export JPEG_QUALITY=80

export CSI_SENSOR_ID=0
export CSI_FLIP_METHOD=2

export YOLO_MODEL_PATH=/path/to/yolo26n.engine
```

## Known good run

Use this exact block on Jetson terminal:

```bash
pkill -f uvicorn || true
pkill -f camera_test1.py || true
pkill -f gst-launch-1.0 || true

cd /home/camera/ptz-ai-dashboard
source .venv/bin/activate

export YOLO_MODEL_PATH="/home/camera/Desktop/share/yolo26n.engine"
export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0:/usr/lib/aarch64-linux-gnu/libgomp.so.1
export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=1000

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:
- on Jetson browser: `http://127.0.0.1:8000`
- on laptop (same network): `http://<jetson-ip>:8000`

## How to SSH from laptop
Get Jetson IP:

```bash
hostname -I
```

SSH from laptop:

```bash
ssh camera@<jetson-ip>
```

## Endpoints
- `GET /` dashboard UI
- `GET /stream.mjpg` live stream
- `GET /api/status` metrics JSON
