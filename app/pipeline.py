import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .config import Settings
from .metrics import Metrics

# Avoid FutureWarning from checking np.bool on newer NumPy.
if "bool" not in np.__dict__:
    np.bool = bool  # type: ignore[attr-defined]

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

        self._capture_pipeline: Optional[Any] = None
        self._capture_sink: Optional[Any] = None

        self._out_pipeline: Optional[Any] = None
        self._out_appsrc: Optional[Any] = None
        self._out_pts_ns = 0

        self._mediamtx_proc: Optional[subprocess.Popen] = None
        self._bridge_proc: Optional[subprocess.Popen] = None

        self._running = threading.Event()
        self._threads: List[threading.Thread] = []

        self._raw_lock = threading.Lock()
        self._raw_latest: Optional[FramePacket] = None

        self._encode_lock = threading.Lock()
        self._encode_latest: Optional[Tuple[np.ndarray, Any, List[Dict[str, Any]]]] = None
        self._detections_lock = threading.Lock()
        self._latest_detections: List[Dict[str, Any]] = []

        self._source_seq = 0
        self._last_processed_seq = -1
        self._yolo_model: Optional[Any] = None
        self._last_infer_error_ts = 0.0

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
            raise RuntimeError("Could not find appsink0 in capture pipeline.")

        state = pipeline.set_state(Gst.State.PLAYING)
        if state == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Could not open CSI capture GStreamer pipeline.")

        self._capture_pipeline = pipeline
        self._capture_sink = sink

    def _open_output_pipeline(self) -> None:
        if Gst is None:
            raise RuntimeError(
                "GStreamer Python bindings are unavailable. Install python3-gi and "
                "gir1.2-gstreamer-1.0 on Jetson."
            )

        Gst.init(None)
        common_src = (
            "appsrc name=outsrc is-live=true block=false do-timestamp=true format=time ! "
            f"video/x-raw,format=BGR,width={self.settings.stream_width},"
            f"height={self.settings.stream_height},framerate={self.settings.fps}/1 ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "videoconvert ! "
        )
        udp_tail = (
            "rtph264pay pt=96 config-interval=1 ! "
            f"udpsink host={self.settings.udp_host} port={self.settings.udp_port} sync=false"
        )

        candidates = [
            common_src
            + "video/x-raw,format=I420 ! "
            + f"x264enc bitrate={self.settings.stream_bitrate_kbps} speed-preset=ultrafast tune=zerolatency ! "
            + udp_tail,
            common_src
            + "video/x-raw,format=BGRx ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            + f"nvv4l2h264enc bitrate={self.settings.stream_bitrate_kbps * 1000} "
            + f"insert-sps-pps=1 idrinterval={self.settings.fps} iframeinterval={self.settings.fps} ! "
            + "h264parse ! "
            + udp_tail,
        ]

        for pipeline_str in candidates:
            try:
                pipeline = Gst.parse_launch(pipeline_str)
                appsrc = pipeline.get_by_name("outsrc")
                if appsrc is None:
                    pipeline.set_state(Gst.State.NULL)
                    continue

                state = pipeline.set_state(Gst.State.PLAYING)
                if state == Gst.StateChangeReturn.FAILURE:
                    pipeline.set_state(Gst.State.NULL)
                    continue

                self._out_pipeline = pipeline
                self._out_appsrc = appsrc
                self._out_pts_ns = 0
                logger.info("Opened output pipeline: %s", pipeline_str)
                return
            except Exception:
                logger.warning("Failed output pipeline candidate: %s", pipeline_str)

        raise RuntimeError("Could not open any H.264 output GStreamer pipeline.")

    def _start_background_services(self) -> None:
        try:
            self._mediamtx_proc = subprocess.Popen(
                [self.settings.mediamtx_bin],
                cwd=self.settings.mediamtx_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            time.sleep(2.0)
            if self._mediamtx_proc.poll() is not None:
                raise RuntimeError("MediaMTX exited during startup.")

            bridge_cmd = [
                "gst-launch-1.0",
                "udpsrc",
                f"port={self.settings.udp_port}",
                "caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96",
                "!",
                "rtph264depay",
                "!",
                "h264parse",
                "!",
                "rtspclientsink",
                (
                    f"location=rtsp://127.0.0.1:{self.settings.mediamtx_rtsp_port}/"
                    f"{self.settings.stream_path}"
                ),
                "protocols=tcp",
            ]
            self._bridge_proc = subprocess.Popen(
                bridge_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as err:
            self._stop_process(self._bridge_proc, "bridge")
            self._bridge_proc = None
            self._stop_process(self._mediamtx_proc, "mediamtx")
            self._mediamtx_proc = None
            raise RuntimeError(f"Failed to start background services: {err}") from err

    def start(self) -> None:
        try:
            self._start_background_services()
            self._init_ai_backend()
            self._open_output_pipeline()
            self._open_capture()
        except Exception:
            self.stop()
            raise

        self._running.set()
        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True, name="capture-loop"),
            threading.Thread(target=self._ai_loop, daemon=True, name="ai-loop"),
            threading.Thread(target=self._encoder_loop, daemon=True, name="encoder-loop"),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._running.clear()

        for thread in self._threads:
            thread.join(timeout=1.5)
        self._threads.clear()

        if self._out_pipeline is not None and Gst is not None:
            self._out_pipeline.set_state(Gst.State.NULL)
            self._out_pipeline = None
            self._out_appsrc = None

        if self._capture_pipeline is not None and Gst is not None:
            self._capture_pipeline.set_state(Gst.State.NULL)
            self._capture_pipeline = None
            self._capture_sink = None

        self._stop_process(self._bridge_proc, "bridge")
        self._bridge_proc = None
        self._stop_process(self._mediamtx_proc, "mediamtx")
        self._mediamtx_proc = None

    def _stop_process(self, proc: Optional[subprocess.Popen], name: str) -> None:
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                logger.debug("Failed to kill %s process", name)

    def _capture_loop(self) -> None:
        assert self._capture_sink is not None
        while self._running.is_set():
            sample = self._capture_sink.emit("try-pull-sample", 50_000_000)
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
            try:
                results = self._yolo_model.predict(packet.frame, imgsz=640, verbose=False)
            except Exception as err:
                now = time.time()
                if now - self._last_infer_error_ts > _ERROR_LOG_INTERVAL_S:
                    logger.warning("YOLO predict failed: %s", err)
                    self._last_infer_error_ts = now
                results = []

            inference_ms = (time.time() - start) * 1000.0
            latency_ms = (time.time() - packet.captured_at) * 1000.0
            detections = self._extract_detections(results, packet.captured_at)

            with self._encode_lock:
                self._encode_latest = (packet.frame, results, detections)
            with self._detections_lock:
                self._latest_detections = detections
            self.metrics.set_timing(latency_ms=latency_ms, inference_ms=inference_ms)

    def _encoder_loop(self) -> None:
        frame_duration_ns = int(1e9 / max(1, self.settings.fps))
        while self._running.is_set():
            with self._encode_lock:
                if self._encode_latest is None:
                    data = None
                else:
                    data = self._encode_latest
                    self._encode_latest = None

            if data is None:
                time.sleep(_AI_IDLE_SLEEP_S)
                continue

            frame, results, detections = data
            if results and len(results) > 0:
                # Keep YOLO box styling, but suppress default labels.
                overlay = results[0].plot(labels=False)
            else:
                overlay = frame.copy()

            overlay = self._draw_numbered_labels(overlay, detections)
            overlay = cv2.resize(
                overlay,
                (self.settings.stream_width, self.settings.stream_height),
            )
            self._push_out_frame(overlay, frame_duration_ns)
            self.metrics.mark_output()

    def _push_out_frame(self, frame: np.ndarray, frame_duration_ns: int) -> None:
        if Gst is None or self._out_appsrc is None:
            return

        frame_c = np.ascontiguousarray(frame)
        payload = frame_c.tobytes()

        buf = Gst.Buffer.new_allocate(None, len(payload), None)
        buf.fill(0, payload)
        buf.pts = self._out_pts_ns
        buf.dts = self._out_pts_ns
        buf.duration = frame_duration_ns
        self._out_pts_ns += frame_duration_ns

        ret = self._out_appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            logger.warning("push-buffer returned %s", ret)

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

    def latest_detections(self) -> List[Dict[str, Any]]:
        with self._detections_lock:
            return list(self._latest_detections)

    def _extract_detections(self, results: Any, captured_at: float) -> List[Dict[str, Any]]:
        if not results or len(results) == 0:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = getattr(result, "names", {}) or {}
        timestamp = time.strftime("%H:%M:%S", time.localtime(captured_at))
        raw: List[Dict[str, Any]] = []

        try:
            count = len(boxes)
        except Exception:
            return []

        for idx in range(count):
            try:
                cls_idx = int(boxes.cls[idx].item()) if boxes.cls is not None else -1
                conf = float(boxes.conf[idx].item()) if boxes.conf is not None else 0.0
                xyxy = boxes.xyxy[idx].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                base_label = str(names.get(cls_idx, cls_idx)).lower()
            except Exception:
                continue

            raw.append(
                {
                    "class_id": cls_idx,
                    "base_label": base_label,
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "timestamp": timestamp,
                }
            )

        # Assign deterministic per-class numbers left-to-right.
        raw.sort(key=lambda d: (d["base_label"], d["x1"], d["y1"]))
        per_class_count: Dict[str, int] = {}
        detections: List[Dict[str, Any]] = []
        for item in raw:
            base = item["base_label"]
            per_class_count[base] = per_class_count.get(base, 0) + 1
            ordinal = per_class_count[base]
            label = f"{base.capitalize()} {ordinal}"
            detections.append(
                {
                    "id": f"{base}-{ordinal}",
                    "label": label,
                    "confidence": item["confidence"],
                    "x1": item["x1"],
                    "y1": item["y1"],
                    "x2": item["x2"],
                    "y2": item["y2"],
                    "timestamp": item["timestamp"],
                }
            )

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def _draw_numbered_labels(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        overlay = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for det in detections:
            x1 = int(det["x1"])
            y1 = int(det["y1"])
            text = f'{det["label"]} {det["confidence"]:.2f}'

            (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
            top = max(0, y1 - th - 10)
            bottom = max(th + 10, y1)
            cv2.rectangle(overlay, (x1, top), (x1 + tw + 10, bottom), (28, 32, 38), -1)
            cv2.putText(overlay, text, (x1 + 5, bottom - 6), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        return overlay

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
            return np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3)).copy()
        finally:
            buffer.unmap(map_info)
