import asyncio
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from .config import Settings
from .metrics import Metrics
from .pipeline import PipelineService


settings = Settings()
metrics = Metrics()
pipeline = PipelineService(settings=settings, metrics=metrics)

app = FastAPI(title="PTZ AI Camera Dashboard", version="0.1.0")

base_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")


@app.on_event("startup")
async def on_startup() -> None:
    pipeline.start()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    pipeline.stop()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "PTZ AI Camera Dashboard",
        },
    )


@app.get("/api/status", response_class=JSONResponse)
async def status() -> JSONResponse:
    payload = metrics.snapshot()
    payload["source_mode"] = settings.source_mode
    return JSONResponse(payload)


@app.get("/stream.mjpg")
async def stream_mjpeg() -> StreamingResponse:
    async def generate():
        boundary = b"--frame\r\n"
        while True:
            frame = pipeline.latest_jpeg()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            yield (
                boundary
                + b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                + frame
                + b"\r\n"
            )
            await asyncio.sleep(0.001)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
