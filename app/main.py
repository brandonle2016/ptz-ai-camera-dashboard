from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
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


class TrackRequest(BaseModel):
    id: str = Field(min_length=1)


class ManualControlRequest(BaseModel):
    direction: str = Field(min_length=1)
    step: float = Field(default=4.0, gt=0.0, le=30.0)


@app.on_event("startup")
async def on_startup() -> None:
    pipeline.start()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    pipeline.stop()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    host = request.url.hostname or "127.0.0.1"
    stream_url = f"https://open-source-ai-camera.ga8ed.com/video/ai_cam/"
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "PTZ AI Camera Dashboard",
            "stream_url": stream_url,
        },
    )


@app.get("/api/status", response_class=JSONResponse)
async def status() -> JSONResponse:
    return JSONResponse(metrics.snapshot())


@app.get("/api/detections", response_class=JSONResponse)
async def detections() -> JSONResponse:
    return JSONResponse({"objects": pipeline.latest_detections()})


@app.post("/api/track", response_class=JSONResponse)
async def track_target(payload: TrackRequest) -> JSONResponse:
    ok = pipeline.select_tracking_target(payload.id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Detection '{payload.id}' not found")
    return JSONResponse({"ok": True, "tracking": pipeline.tracking_state()})


@app.post("/api/untrack-all", response_class=JSONResponse)
async def untrack_all() -> JSONResponse:
    pipeline.clear_tracking_target()
    return JSONResponse({"ok": True, "tracking": pipeline.tracking_state()})


@app.get("/api/track", response_class=JSONResponse)
async def tracking_state() -> JSONResponse:
    return JSONResponse(pipeline.tracking_state())


@app.post("/api/control/manual", response_class=JSONResponse)
async def manual_control(payload: ManualControlRequest) -> JSONResponse:
    ok = pipeline.manual_control(direction=payload.direction, step_deg=payload.step)
    if not ok:
        raise HTTPException(status_code=400, detail="direction must be one of: up, down, left, right")
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
