import logging
import threading
import time
import subprocess
import signal
from dataclasses import dataclass
from typing import Any, List, Optional

import cv2
import numpy as np

# Compatibility shim for older runtime code paths that still reference np.bool.
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

from ultralytics import YOLO

from .config import Settings
from .metrics import Metrics

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except Exception:
    Gst = None  # type: ignore

logger = logging.getLogger(__name__)

_CAPTURE_RETRY_SLEEP_S = 0.005
_AI_IDLE_SLEEP_S = 0.002
_ERROR_LOG_INTERVAL_S = 5.0


@dataclass
class FramePacket:
    seq: int
    captured_at: float
    frame: np.ndarray


class PipelineService:
    def __init__(self, settings: Settings, metrics: Metrics) -> None:
        self.settings = settings
        self.metrics = metrics

        self._gst_pipeline: Optional[Any] = None
        self._gst_sink: Optional[Any] = None
        self._running = threading.Event()
        self._threads: List[threading.Thread] = []

        self._raw_lock = threading.Lock()
        self._raw_latest: Optional[FramePacket] = None

        self._out_lock = threading.Lock()
        self._out_latest_jpeg: Optional[bytes] = None

        self._source_seq = 0
        self._last_processed_seq = -1
        self._yolo_model: Optional[Any] = None
        self._last_infer_error_ts = 0.0
        
        self._encode_lock = threading.Lock()
        self._encode_latest: Optional[tuple] = None
        self._video_writer: Optional[cv2.VideoWriter] = None

    def _init_ai_backend(self) -> None:
        self._yolo_model = YOLO(self.settings.yolo_model_path, task="detect")

    def _open_capture(self) -> None:
        if Gst is None:
            raise RuntimeError(
                "GStreamer Python bindings are unavailable. Install python3-gi and "
                "gir1.2-gstreamer-1.0 on Jetson."
            )

        Gst.init(None)
        pipeline = Gst.parse_launch(self.settings.gstreamer_pipeline)
        sink = pipeline.get_by_name("appsink0")
        if sink is None:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Could not find appsink0 in GStreamer pipeline.")

        pipeline.set_state(Gst.State.PLAYING)
        self._gst_pipeline = pipeline
        self._gst_sink = sink

    def start(self) -> None:
        self._start_background_services()
        self._init_ai_backend()
        self._open_capture()

        self._running.set()
        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True, name="capture-loop"),
            threading.Thread(target=self._ai_loop, daemon=True, name="ai-loop"),
            threading.Thread(target=self._encoder_loop, daemon=True, name="encoder-loop"),
        ]
        for thread in self._threads:
            thread.start()

    def _open_writer(self) -> None:
        """Blindly fires RTP packets to a local UDP port to prevent Python deadlocks."""
        out_pipeline = (
            "appsrc do-timestamp=true is-live=true ! "
            "video/x-raw, format=BGR ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "videoconvert ! "
            "video/x-raw, format=I420 ! "
            f"x264enc bitrate={self.settings.stream_bitrate_kbps} speed-preset=ultrafast tune=zerolatency ! "
            "rtph264pay config-interval=1 pt=96 ! "
            "udpsink host=127.0.0.1 port=5000 sync=false"
        )
        
        self._video_writer = cv2.VideoWriter(
            out_pipeline, 
            cv2.CAP_GSTREAMER, 
            0, 
            self.settings.fps, 
            (self.settings.stream_width, self.settings.stream_height)
        )
        self._video_writer = cv2.VideoWriter(
            out_pipeline, 
            cv2.CAP_GSTREAMER, 
            0, 
            self.settings.fps, 
            (self.settings.stream_width, self.settings.stream_height)
        )
        
        if not self._video_writer.isOpened():
            logger.error("Failed to open GStreamer H.264 VideoWriter!")
        
        self._video_writer = cv2.VideoWriter(
            out_pipeline, 
            cv2.CAP_GSTREAMER, 
            0, 
            self.settings.fps, 
            (self.settings.stream_width, self.settings.stream_height)
        )
        
        if not self._video_writer.isOpened():
            logger.error("Failed to open GStreamer H.264 VideoWriter!")
        
        if not self._video_writer.isOpened():
            logger.error("Failed to open GStreamer H.264 VideoWriter!")
            
    def _start_background_services(self) -> None:
        """Launches MediaMTX and the GStreamer Bridge automatically."""
        try:
            # 1. Start MediaMTX (Assumes the binary is in the path or same folder)
            # Adjust the path to where your mediamtx binary actually is
            self._mediamtx_proc = subprocess.Popen(
                ["./mediamtx"], 
                cwd="/home/camera/Downloads/mediamtx_v1.17.1_linux_arm64", # Use your actual path
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            logger.info("MediaMTX started in background.")

            # 2. Wait a moment for MediaMTX to open its ports
            time.sleep(2)

            # 3. Start the Bridge Command
            bridge_cmd = [
                "gst-launch-1.0", "udpsrc", "port=5000", 
                "caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=H264", 
                "!", "rtph264depay", "!", "h264parse", 
                "!", "rtspclientsink", "location=rtsp://127.0.0.1:8554/ai_cam", "protocols=tcp"
            ]
            self._bridge_proc = subprocess.Popen(bridge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("GStreamer Bridge started in background.")

        except Exception as e:
            logger.error(f"Failed to start background services: {e}")
            
    def stop(self) -> None:
        self._running.clear()
        
        if hasattr(self, '_bridge_proc'):
            self._bridge_proc.terminate()
        if hasattr(self, '_mediamtx_proc'):
            self._mediamtx_proc.terminate()
        
        for thread in self._threads:
            thread.join(timeout=1.5)
        self._threads.clear()
        
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        if self._gst_pipeline is not None and Gst is not None:
            self._gst_pipeline.set_state(Gst.State.NULL)
            self._gst_pipeline = None
            self._gst_sink = None

    #def latest_jpeg(self) -> Optional[bytes]:
     #   with self._out_lock:
     #       return self._out_latest_jpeg

    def _capture_loop(self) -> None:
        assert self._gst_sink is not None
        while self._running.is_set():
            sample = self._gst_sink.emit("try-pull-sample", 50_000_000)  # 50ms in nanoseconds
            if sample is None:
                time.sleep(_CAPTURE_RETRY_SLEEP_S)
                continue

            frame = self._sample_to_bgr_frame(sample)
            if frame is None:
                time.sleep(_CAPTURE_RETRY_SLEEP_S)
                continue

            self.metrics.mark_capture()
            self._publish_raw_frame(frame)

    def _ai_loop(self) -> None:
        while self._running.is_set():
            packet = self._next_frame_for_processing()
            if packet is None:
                time.sleep(_AI_IDLE_SLEEP_S)
                continue

            start = time.time()
            results = self._yolo_model.predict(packet.frame, imgsz=640, verbose=False)
            
            # overlay = self._run_yolo_track(packet.frame.copy())
            inference_ms = (time.time() - start) * 1000.0
            latency_ms = (time.time() - packet.captured_at) * 1000.0

            #encoded = self._encode_jpeg(overlay)
            with self._encode_lock:
            	self._encode_latest = (packet.frame, results)
            self.metrics.set_timing(latency_ms=latency_ms, inference_ms=inference_ms)

            #with self._out_lock:
            #    self._out_latest_jpeg = encoded

            #self.metrics.mark_output()
            #self.metrics.set_timing(latency_ms=latency_ms, inference_ms=inference_ms)
            
    def _encoder_loop(self) -> None:
        """Runs in a 3rd thread to keep the GPU fed."""
        self._open_writer()
       
        while self._running.is_set():
            
            # Grab the latest data from the AI loop safely
            with self._encode_lock:
                if self._encode_latest is None:
                    data = None
                else:
                    data = self._encode_latest
                    self._encode_latest = None  # Clear it so we don't re-encode the same frame
            
            # If no new frame from AI, sleep briefly
            if data is None:
                time.sleep(_AI_IDLE_SLEEP_S)
                continue

            frame, results = data
            
            # Plotting happens here, freeing up the AI thread
            if results and len(results) > 0:
                overlay = results[0].plot()
            else:
                overlay = frame
                
            overlay = cv2.resize(overlay, (self.settings.stream_width,self.settings.stream_height))

            # Push the frame directly to the GStreamer encoder
            if self._video_writer is not None:
            	self._video_writer.write(overlay)
            
            self.metrics.mark_output()

    def _next_frame_for_processing(self) -> Optional[FramePacket]:
        with self._raw_lock:
            if self._raw_latest is None:
                return None
            if self._raw_latest.seq == self._last_processed_seq:
                return None
            packet = self._raw_latest

        self._last_processed_seq = packet.seq
        return packet

    def _publish_raw_frame(self, frame: np.ndarray) -> None:
        packet = FramePacket(seq=self._source_seq, captured_at=time.time(), frame=frame)
        self._source_seq += 1

        with self._raw_lock:
            if self._raw_latest is not None:
                self.metrics.mark_drop()
            self._raw_latest = packet

    #def _encode_jpeg(self, frame: np.ndarray, quality: int = 50) -> bytes:
    #    ok, encoded = cv2.imencode(
     #       ".jpg",
     #       frame,
     #       [int(cv2.IMWRITE_JPEG_QUALITY), quality],
     #   )
      #  if not ok:
      #      return b""
      #  return encoded.tobytes()

    def _sample_to_bgr_frame(self, sample: Any) -> Optional[np.ndarray]:
        if Gst is None:
            return None

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if buffer is None or caps is None:
            return None

        structure = caps.get_structure(0)
        if structure is None:
            return None

        ok_w, width = structure.get_int("width")
        ok_h, height = structure.get_int("height")
        if not ok_w or not ok_h:
            return None

        ok_map, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok_map:
            return None

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3)).copy()
            return frame
        finally:
            buffer.unmap(map_info)

    def _run_yolo_track(self, frame: np.ndarray) -> np.ndarray:
        if self._yolo_model is None:
            return frame
        try:
            results = self._yolo_model.track(frame, persist=True, verbose=False)
            if results and len(results) > 0:
                return results[0].plot()
            return frame
        except Exception as track_err:
            # Some TensorRT engines fail in track() path; fallback to predict() for overlays.
            try:
                results = self._yolo_model.predict(frame, verbose=False)
                if results and len(results) > 0:
                    return results[0].plot()
                return frame
            except Exception as predict_err:
                now = time.time()
                if now - self._last_infer_error_ts > _ERROR_LOG_INTERVAL_S:
                    logger.warning("YOLO inference failed. track=%s predict=%s", track_err, predict_err)
                    self._last_infer_error_ts = now
                return frame
