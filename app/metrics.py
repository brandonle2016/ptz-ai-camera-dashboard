import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import psutil


@dataclass
class Metrics:
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _capture_ts: Deque[float] = field(default_factory=lambda: deque(maxlen=120), init=False)
    _output_ts: Deque[float] = field(default_factory=lambda: deque(maxlen=120), init=False)
    _latency_ms: float = 0.0
    _inference_ms: float = 0.0
    _dropped_frames: int = 0

    def mark_capture(self) -> None:
        with self._lock:
            self._capture_ts.append(time.time())

    def mark_output(self) -> None:
        with self._lock:
            self._output_ts.append(time.time())

    def set_timing(self, latency_ms: float, inference_ms: float) -> None:
        with self._lock:
            self._latency_ms = latency_ms
            self._inference_ms = inference_ms

    def mark_drop(self) -> None:
        with self._lock:
            self._dropped_frames += 1

    def _fps(self, timestamps: Deque[float]) -> float:
        if len(timestamps) < 2:
            return 0.0
        dt = timestamps[-1] - timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(timestamps) - 1) / dt

    def _read_temp_c(self) -> Optional[float]:
        try:
            temps = psutil.sensors_temperatures()
        except Exception:
            return None

        if not temps:
            return None

        for key in ("cpu_thermal", "soc_thermal", "thermal-fan-est", "coretemp"):
            if key in temps and temps[key]:
                return float(temps[key][0].current)

        for values in temps.values():
            if values:
                return float(values[0].current)

        return None

    def snapshot(self) -> dict:
        with self._lock:
            data = {
                "capture_fps": round(self._fps(self._capture_ts), 2),
                "stream_fps": round(self._fps(self._output_ts), 2),
                "latency_ms": round(self._latency_ms, 2),
                "inference_ms": round(self._inference_ms, 2),
                "dropped_frames": self._dropped_frames,
            }
        data["temp_c"] = self._read_temp_c()
        return data
