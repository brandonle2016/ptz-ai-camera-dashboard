import logging
import threading
import time
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

        self._capture: Optional[cv2.VideoCapture] = None
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

    def _init_ai_backend(self) -> None:
        self._yolo_model = YOLO(self.settings.yolo_model_path, task="detect")

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.settings.gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if not capture.isOpened():
            raise RuntimeError(
                "Could not open CSI GStreamer pipeline. "
                "Verify camera wiring, nvarguscamerasrc, and runtime environment."
            )
        return capture

    def start(self) -> None:
        self._init_ai_backend()
        self._capture = self._open_capture()

        self._running.set()
        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True, name="capture-loop"),
            threading.Thread(target=self._ai_loop, daemon=True, name="ai-loop"),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._running.clear()
        for thread in self._threads:
            thread.join(timeout=1.5)
        self._threads.clear()

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def latest_jpeg(self) -> Optional[bytes]:
        with self._out_lock:
            return self._out_latest_jpeg

    def _capture_loop(self) -> None:
        assert self._capture is not None
        while self._running.is_set():
            ok, frame = self._capture.read()
            if not ok:
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
            overlay = self._run_yolo_track(packet.frame.copy())
            inference_ms = (time.time() - start) * 1000.0
            latency_ms = (time.time() - packet.captured_at) * 1000.0

            encoded = self._encode_jpeg(overlay)

            with self._out_lock:
                self._out_latest_jpeg = encoded

            self.metrics.mark_output()
            self.metrics.set_timing(latency_ms=latency_ms, inference_ms=inference_ms)

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

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality],
        )
        if not ok:
            return b""
        return encoded.tobytes()

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
