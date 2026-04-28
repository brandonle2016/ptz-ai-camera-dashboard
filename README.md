# PTZ AI Camera Dashboard (CSI + YOLO + H.264/WebRTC)

This app runs on Jetson and does:
- CSI capture via GStreamer (`nvarguscamerasrc`)
- YOLO inference (`ultralytics`, TensorRT engine supported)
- Latest-frame-wins processing (drops stale frames)
- H.264 output to MediaMTX, viewed in browser via WebRTC page
- Status API for capture fps, stream fps, latency, inference, drops, temp

Validated profile: 1280x720 at 60 FPS on Jetson (when camera and runtime support it).

## Runtime Paths

- Dashboard: `GET /`
- Status: `GET /api/status`
- Detections: `GET /api/detections`
- Media stream page (via MediaMTX): `http://<jetson-ip>:8889/ai_cam/` by default

## Project Layout

```text
ptz-ai-dashboard/
  app/
    config.py
    main.py
    metrics.py
    pipeline.py
    templates/index.html
    static/style.css
  requirements.txt
  README.md
```

## Jetson Setup

```bash
cd /home/camera/ptz-ai-camera-dashboard
sudo apt update
sudo apt install -y python3-venv python3-gi gir1.2-gstreamer-1.0

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Required Env Vars

```bash
export APP_HOST=0.0.0.0
export APP_PORT=8000

export FRAME_WIDTH=1280
export FRAME_HEIGHT=720
export FRAME_FPS=60

export STREAM_WIDTH=1280
export STREAM_HEIGHT=720
export STREAM_BITRATE=2500

export CSI_SENSOR_ID=0
export CSI_FLIP_METHOD=2

export YOLO_MODEL_PATH="/home/camera/Desktop/share/yolo26n.engine"

export MEDIAMTX_DIR="/home/camera/Downloads/mediamtx_v1.17.1_linux_arm64"
export MEDIAMTX_BIN="./mediamtx"
export MEDIAMTX_RTSP_PORT=8554
export MEDIAMTX_WEBRTC_PORT=8889
export STREAM_PATH="ai_cam"

export UDP_HOST=127.0.0.1
export UDP_PORT=5000
```

## Run

```bash
pkill -f uvicorn || true
pkill -f mediamtx || true
pkill -f "gst-launch-1.0 udpsrc" || true

cd /home/camera/ptz-ai-camera-dashboard
source .venv/bin/activate

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:
- Dashboard: `http://<jetson-ip>:8000`
- Direct stream page: `http://<jetson-ip>:8889/ai_cam/`

The "Detected Objects" panel in the dashboard is now fed by `/api/detections`.
The Track buttons are present in UI and intentionally no-op for now.

## How to SSH from laptop
Get Jetson IP:

```bash
hostname -I
```

SSH from laptop:

```bash
ssh camera@<jetson-ip>
```

## Notes

- If `mediamtx` path is different, update `MEDIAMTX_DIR`/`MEDIAMTX_BIN`.
- If CSI works in `gst-launch` but app fails, verify `python3-gi` and GStreamer plugins are installed.
