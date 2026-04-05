from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "8000"))
    width: int = int(os.getenv("FRAME_WIDTH", "1280"))
    height: int = int(os.getenv("FRAME_HEIGHT", "720"))
    fps: int = int(os.getenv("FRAME_FPS", "30"))
    jpeg_quality: int = int(os.getenv("JPEG_QUALITY", "80"))
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
