# PTZ AI Camera Dashboard (Prototype)

Minimal Jetson-oriented architecture for a browser-viewable PTZ AI demo.

## Goals Covered
- Live capture via GStreamer using `videotestsrc` (no camera hardware needed)
- Simulated AI overlay stage (replace later with Ultralytics YOLO26)
- Live browser stream over HTTP (MJPEG)
- Dashboard page with status panel (`fps`, `latency`, `temp`)
- Latest-frame-wins behavior (drops stale frames under load)

## Project Structure

```text
ptz-ai-dashboard/
  app/
    __init__.py
    config.py          # runtime settings + source pipeline switch
    main.py            # FastAPI app, routes, startup/shutdown
    metrics.py         # fps/latency/temp counters
    pipeline.py        # capture thread + AI thread + latest frame buffering
    static/
      style.css
    templates/
      index.html
  requirements.txt
  README.md
```

## Architecture (Current)

1. GStreamer capture thread
- Opens `cv2.VideoCapture(<gstreamer pipeline>, cv2.CAP_GSTREAMER)`.
- Default source is `videotestsrc is-live=true`.
- Writes only the newest frame into a shared buffer.

2. AI overlay thread
- Pulls newest unprocessed frame only.
- Simulates inference latency (`SIM_INFERENCE_MS`, default `45ms`).
- Draws demo bounding box/label onto frame.
- Encodes JPEG and publishes as newest output frame.

3. HTTP serving layer
- `GET /stream.mjpg`: multipart MJPEG stream.
- `GET /api/status`: JSON metrics.
- `GET /`: dashboard page.

4. Latest-frame-wins strategy
- Shared single-slot frame buffer (`_raw_latest`).
- If capture produces new frame before AI consumes previous, old one is dropped.
- This keeps latency bounded and avoids queue buildup.

## Dependencies

Python packages in `requirements.txt`:
- `fastapi`, `uvicorn[standard]`
- `opencv-python`
- `numpy`
- `psutil`
- `jinja2`

System deps (needed for OpenCV GStreamer pipeline):
- GStreamer runtime/plugins installed on host
- OpenCV build with GStreamer support

## Local Dev (Now, No Camera)

From project root:

```bash
cd ptz-ai-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open in browser:
- [http://localhost:8000](http://localhost:8000)

Optional environment variables:

```bash
export SOURCE_MODE=synthetic   # synthetic | test | csi
export FRAME_WIDTH=1280
export FRAME_HEIGHT=720
export FRAME_FPS=30
export JPEG_QUALITY=80
export SIM_INFERENCE_MS=45
```

## Jetson Notes (Later)

### 1) Switch capture source to CSI/MIPI
Set:

```bash
export SOURCE_MODE=csi
```

`config.py` already maps this to `nvarguscamerasrc`.

For desktop/mac demo with OpenCV builds that do not include GStreamer support, use:

```bash
export SOURCE_MODE=synthetic
```

### 2) Replace simulated AI with YOLO26
In `app/pipeline.py`, replace `_simulate_ai_overlay(...)` with:
- frame preprocessing for YOLO
- model inference call
- postprocess boxes/classes/confidence
- draw overlays onto frame

Keep the same latest-frame-wins behavior by preserving:
- single-slot raw frame buffer
- one inference worker processing newest frame only

### 3) Performance tuning on Jetson
- Reduce input resolution or inference interval if latency climbs.
- Consider NVMM zero-copy + hardware codecs where possible.
- If needed later, split ingest/inference/streaming into separate processes with shared memory.

## Endpoints
- `GET /` dashboard UI
- `GET /stream.mjpg` live video stream
- `GET /api/status` metrics JSON
