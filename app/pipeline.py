import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Settings
from .metrics import Metrics


@dataclass
class FramePacket:
    seq: int
    captured_at: float
    frame: np.ndarray


class PipelineService:
    def __init__(self, settings: Settings, metrics: Metrics) -> None:
        self.settings = settings
        self.metrics = metrics

        self._capture: cv2.VideoCapture | None = None
        self._running = threading.Event()
        self._threads: list[threading.Thread] = []

        self._raw_lock = threading.Lock()
        self._raw_latest: FramePacket | None = None

        self._out_lock = threading.Lock()
        self._out_latest_jpeg: bytes | None = None

        self._source_seq = 0
        self._last_processed_seq = -1

    def start(self) -> None:
        if self.settings.source_mode in {"test", "csi"}:
            self._capture = cv2.VideoCapture(self.settings.gstreamer_pipeline, cv2.CAP_GSTREAMER)
            if not self._capture.isOpened():
                raise RuntimeError(
                    "Could not open GStreamer pipeline. "
                    "Ensure gstreamer plugins are installed and OpenCV has GStreamer support. "
                    "For local demo without GStreamer, set SOURCE_MODE=synthetic."
                )

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

    def latest_jpeg(self) -> bytes | None:
        with self._out_lock:
            return self._out_latest_jpeg

    def _capture_loop(self) -> None:
        if self.settings.source_mode == "synthetic":
            self._synthetic_capture_loop()
            return

        assert self._capture is not None
        while self._running.is_set():
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.005)
                continue

            self.metrics.mark_capture()
            packet = FramePacket(seq=self._source_seq, captured_at=time.time(), frame=frame)
            self._source_seq += 1

            with self._raw_lock:
                if self._raw_latest is not None:
                    self.metrics.mark_drop()
                self._raw_latest = packet

    def _synthetic_capture_loop(self) -> None:
        frame_interval = 1.0 / max(self.settings.fps, 1)
        start_t = time.time()

        while self._running.is_set():
            now = time.time()
            t = now - start_t
            frame = np.zeros((self.settings.height, self.settings.width, 3), dtype=np.uint8)

            # Generate a moving pattern to mimic live motion for demos without camera/GStreamer.
            cx = int((np.sin(t * 1.3) * 0.4 + 0.5) * (self.settings.width - 1))
            cy = int((np.cos(t * 1.1) * 0.4 + 0.5) * (self.settings.height - 1))
            cv2.circle(frame, (cx, cy), max(min(self.settings.width, self.settings.height) // 16, 12), (40, 220, 180), -1)
            self.metrics.mark_capture()
            packet = FramePacket(seq=self._source_seq, captured_at=now, frame=frame)
            self._source_seq += 1

            with self._raw_lock:
                if self._raw_latest is not None:
                    self.metrics.mark_drop()
                self._raw_latest = packet

            sleep_for = frame_interval - (time.time() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _ai_loop(self) -> None:
        while self._running.is_set():
            packet = self._next_frame_for_processing()
            if packet is None:
                time.sleep(0.002)
                continue

            start = time.time()
            overlay = self._simulate_ai_overlay(packet.frame.copy(), packet.seq)
            inference_ms = (time.time() - start) * 1000.0
            latency_ms = (time.time() - packet.captured_at) * 1000.0

            encoded = cv2.imencode(
                ".jpg",
                overlay,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality],
            )[1].tobytes()

            with self._out_lock:
                self._out_latest_jpeg = encoded

            self.metrics.mark_output()
            self.metrics.set_timing(latency_ms=latency_ms, inference_ms=inference_ms)

    def _next_frame_for_processing(self) -> FramePacket | None:
        with self._raw_lock:
            if self._raw_latest is None:
                return None
            if self._raw_latest.seq == self._last_processed_seq:
                return None
            packet = self._raw_latest

        self._last_processed_seq = packet.seq
        return packet

    def _simulate_ai_overlay(self, frame: np.ndarray, seq: int) -> np.ndarray:
        # Simulate model runtime before replacing this with YOLO inference.
        time.sleep(max(self.settings.inference_ms, 0.0) / 1000.0)

        h, w = frame.shape[:2]
        box_w = max(w // 7, 80)
        box_h = max(h // 5, 60)

        x = int((seq * 17) % max(w - box_w, 1))
        y = int((seq * 9) % max(h - box_h, 1))

        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
        return frame
