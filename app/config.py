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
    inference_ms: float = float(os.getenv("SIM_INFERENCE_MS", "45"))

    source_mode: str = os.getenv("SOURCE_MODE", "synthetic")

    @property
    def gstreamer_pipeline(self) -> str:
        if self.source_mode == "csi":
            return (
                "nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
                f"framerate={self.fps}/1 ! "
                "nvvidconv ! video/x-raw, format=BGRx ! "
                "videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
            )

        return (
            "videotestsrc is-live=true pattern=ball ! "
            f"video/x-raw, width={self.width}, height={self.height}, framerate={self.fps}/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
