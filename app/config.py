from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "8000"))
    
    # Camera & AI Resolution (Keep this high)
    width: int = int(os.getenv("FRAME_WIDTH", "1280"))
    height: int = int(os.getenv("FRAME_HEIGHT", "720"))
    fps: int = int(os.getenv("FRAME_FPS", "60"))
    
    # Stream output resolution.
    stream_width: int = int(os.getenv("STREAM_WIDTH", "1280"))
    stream_height: int = int(os.getenv("STREAM_HEIGHT", "720"))
    stream_bitrate_kbps: int = int(os.getenv("STREAM_BITRATE", "2500"))
    
    # RTP output destination for encoded H.264.
    udp_host: str = os.getenv("UDP_HOST", "127.0.0.1")
    udp_port: int = int(os.getenv("UDP_PORT", "5000"))

    # MediaMTX settings.
    mediamtx_bin: str = os.getenv("MEDIAMTX_BIN", "./mediamtx")
    mediamtx_dir: str = os.getenv(
        "MEDIAMTX_DIR",
        "/home/camera/Downloads/mediamtx_v1.17.1_linux_arm64",
    )
    mediamtx_rtsp_port: int = int(os.getenv("MEDIAMTX_RTSP_PORT", "8554"))
    mediamtx_webrtc_port: int = int(os.getenv("MEDIAMTX_WEBRTC_PORT", "8889"))
    stream_path: str = os.getenv("STREAM_PATH", "ai_cam")
    
    yolo_model_path: str = os.getenv("YOLO_MODEL_PATH", "yolo26n.engine")
    sensor_id: int = int(os.getenv("CSI_SENSOR_ID", "0"))
    flip_method: int = int(os.getenv("CSI_FLIP_METHOD", "2"))
    @property
    def gstreamer_pipeline(self) -> str:
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){self.width}, height=(int){self.height}, "
            f"framerate=(fraction){self.fps}/1 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw, width=(int){self.width}, height=(int){self.height}, format=(string)BGRx ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! "
            "appsink name=appsink0 emit-signals=false drop=true max-buffers=1 sync=false"
        )
