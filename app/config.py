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
    
    # Stream Output Resolution (Add these to make the output smaller!)
    stream_width: int = int(os.getenv("STREAM_WIDTH", "1280"))
    stream_height: int = int(os.getenv("STREAM_HEIGHT", "720"))
    stream_bitrate_kbps: int = int(os.getenv("STREAM_BITRATE", "10000"))
    
    # Where to send the H.264 stream
    udp_host: str = os.getenv("UDP_HOST", "127.0.0.1")
    udp_port: int = int(os.getenv("UDP_PORT", "8554"))
    
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